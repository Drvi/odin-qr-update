package blas

// ============================================================================
// Least squares (OLS): solve  min ||X*beta - y||  for beta.
//
// Two transforms, for the two regimes that occur. See docs/OLS_PLAN.md for the
// measurements and the reasoning; the short version:
//
//   ols_solve_dense  -- all m rows are resident. Householder QR, applies the
//                       reflectors straight to y. O(2mn^2) time, O(n) scratch.
//
//   Ols_Accum        -- rows arrive over time, or the work is chunked across
//                       frames. Folds rows into the (n+1)x(n+1) R factor of the
//                       augmented matrix [X | y] with Givens rotations.
//                       O(3n^2) per row, O(n^2) memory REGARDLESS OF m.
//
// Neither forms Q. beta depends on Q only through the n-vector Q^T*y, and Q is
// already stored implicitly as reflectors (dense) or folded into the triangle
// (accumulator). Materializing the m x m Q, as the old example code did, costs
// 489 MiB and 16.7 s at m = 8000 against 0.265 ms for the factorization itself.
//
// Neither routine allocates. All memory is caller-owned; the *_scratch
// procedures give the required sizes.
// ============================================================================

import "core:math"

// Rank-rejection threshold, applied as a ratio on the R diagonal:
// a system is rejected when min|R[j,j]| <= rcond * max|R[j,j]|.
//
// This is a cheap proxy for the condition number, not a condition estimate.
// See issues/005-rank-deficient-systems.md.
OLS_DEFAULT_RCOND :: 1.0e-12

Ols_Error :: enum {
	None = 0,
	Invalid_Dimension, // n < 1, ldx < n, count < 0, or a slice shorter than the stated shape
	Scratch_Too_Small, // caller buffer smaller than the matching *_scratch size
	Not_Enough_Rows, // fewer rows than predictors; the system is underdetermined
	Rank_Deficient, // R diagonal ratio below rcond; collinear predictors
	Non_Finite_Input, // NaN or +/-Inf in the input
}

// True for finite values only: false for NaN (all comparisons with NaN are
// false) and for both infinities. One compare on the common path.
@(private = "file")
is_finite :: #force_inline proc(v: f64) -> bool {
	return abs(v) <= max(f64)
}

// ============================================================================
// Transform A: resident data
// ============================================================================

// Scratch required by ols_solve_dense, in f64 elements.
ols_dense_scratch :: proc(n: int) -> int {
	return 2 * n + 1
}

// ols_solve_dense solves min ||X*beta - y|| by Householder QR.
//
// BATCH CONTRACT
//   x       in/out  f64[>= (m-1)*ldx + n], row-major, row stride ldx >= n.
//                   DESTROYED: overwritten with R and the reflectors.
//   y       in/out  f64[>= m]. DESTROYED: overwritten with Q^T*y.
//   beta    out     f64[>= n]. Written only when the call returns .None.
//   scratch temp    f64[>= ols_dense_scratch(n)]. Contents undefined on exit.
//   rcond   in      rank-rejection ratio, see OLS_DEFAULT_RCOND.
//   returns         rss = ||X*beta - y||^2, and an error code.
//
//   All buffers are caller-owned and only borrowed for the duration of the
//   call. Valid ranges: m >= n >= 1, all inputs finite.
//
// Cost: 2mn^2 - 2n^3/3 to factor, 4mn to apply Q^T, n^2 to back-substitute.
// Extra memory: 2n+1 f64, independent of m.
//
// x and y are destroyed because that is what makes the routine allocation-free;
// copy them first if you need them afterwards.
ols_solve_dense :: proc(
	m: int,
	n: int,
	x: []f64,
	ldx: int,
	y: []f64,
	beta: []f64,
	scratch: []f64,
	rcond: f64 = OLS_DEFAULT_RCOND,
) -> (
	rss: f64,
	err: Ols_Error,
) {
	if n < 1 || m < 0 || ldx < n {
		return 0, .Invalid_Dimension
	}
	if m < n {
		return 0, .Not_Enough_Rows
	}
	if len(x) < (m - 1) * ldx + n || len(y) < m || len(beta) < n {
		return 0, .Invalid_Dimension
	}
	if len(scratch) < ols_dense_scratch(n) {
		return 0, .Scratch_Too_Small
	}

	// Reject non-finite input before touching anything: a NaN would otherwise
	// propagate silently into beta.
	for i in 0 ..< m {
		base := i * ldx
		for j in 0 ..< n {
			if !is_finite(x[base + j]) {
				return 0, .Non_Finite_Input
			}
		}
		if !is_finite(y[i]) {
			return 0, .Non_Finite_Input
		}
	}

	tau := scratch[0:n]
	work := scratch[n:2 * n + 1]

	// X -> R + reflectors
	dgeqrf(m, n, x, ldx, tau, work)

	// y -> Q^T*y, applying the reflectors in place. No m x m Q.
	ols_apply_qt(m, n, x, ldx, tau, y, 1, 1, work)

	// Rank check on the R diagonal before dividing by it.
	dmax, dmin := 0.0, max(f64)
	for j in 0 ..< n {
		d := abs(x[j * ldx + j])
		dmax = max(dmax, d)
		dmin = min(dmin, d)
	}
	if dmax == 0.0 || dmin <= rcond * dmax {
		return 0, .Rank_Deficient
	}

	// The tail of Q^T*y that R cannot reach is exactly the residual.
	rss = 0.0
	for i in n ..< m {
		rss += y[i] * y[i]
	}

	for i in 0 ..< n {
		beta[i] = y[i]
	}
	dtrsv(.Upper, .No_Trans, .Non_Unit, n, x, ldx, beta, 1)

	return rss, .None
}

// ols_apply_qt overwrites C with Q^T*C, where Q comes from dgeqrf.
//
// Q = H(0)*H(1)*...*H(k-1), so Q^T = H(k-1)*...*H(0) and the reflectors are
// applied in forward order. Q is never formed: this is O(m*n*ncols) time and
// O(ncols) scratch, against O(m^2) time and memory for dorgqr followed by a
// multiply.
//
//   a     in/out  the factored matrix from dgeqrf, row-major, stride lda.
//                 Restored to its input state before returning.
//   tau   in      f64[>= min(m,n)] from dgeqrf.
//   c     in/out  f64, m x ncols, row-major, stride ldc. Overwritten.
//   work  temp    f64[>= ncols].
ols_apply_qt :: proc(
	m: int,
	n: int,
	a: []f64,
	lda: int,
	tau: []f64,
	c: []f64,
	ldc: int,
	ncols: int,
	work: []f64,
) {
	k := min(m, n)
	for i in 0 ..< k {
		if tau[i] == 0.0 {
			continue
		}
		// The reflector is the subcolumn below the diagonal with an implicit 1
		// on it. Plant the 1, apply, restore.
		aii := a[i * lda + i]
		a[i * lda + i] = 1.0
		dlarf(.Left, m - i, ncols, a[i * lda + i:], lda, tau[i], c[i * ldc:], ldc, work)
		a[i * lda + i] = aii
	}
}

// ============================================================================
// Transform B: streaming / frame-chunked data
// ============================================================================

// Ols_Accum holds the R factor of the augmented matrix [X | y].
//
// Flat struct, no hidden allocation, no destructor: `tri` and `row` are views
// into one caller-owned block handed to ols_accum_init. Copying the struct
// aliases that block rather than duplicating it.
//
// Layout of `tri`, (n+1) x (n+1) row-major with stride ld = n+1, upper
// triangular, everything below the diagonal held at zero:
//
//     tri[0:n, 0:n]  R factor of X
//     tri[0:n, n]    Q^T*y, first n components
//     tri[n, n]      +/- ||X*beta - y||
//
// nrows is public: progress across frames is nrows / your_total. There is
// deliberately no step function or phase enum -- per-row cost is uniform and
// the final solve is O(n^2), so the caller's own loop is the state machine.
Ols_Accum :: struct {
	n:     int, // predictors
	ld:    int, // row stride of tri, always n+1
	nrows: int, // rows absorbed so far
	tri:   []f64, // (n+1)*(n+1)
	row:   []f64, // n+1, scratch for the row being folded in
}

// Scratch required by ols_accum_init, in f64 elements. One contiguous block.
ols_accum_scratch :: proc(n: int) -> int {
	return (n + 1) * (n + 1) + (n + 1)
}

// ols_accum_init prepares an accumulator over a caller-owned block.
//
// The block must be at least ols_accum_scratch(n) f64 and must stay alive and
// untouched by the caller for as long as the accumulator is used -- typically
// across many frames. It is zeroed here.
ols_accum_init :: proc(acc: ^Ols_Accum, n: int, scratch: []f64) -> Ols_Error {
	if n < 1 {
		return .Invalid_Dimension
	}
	if len(scratch) < ols_accum_scratch(n) {
		return .Scratch_Too_Small
	}
	tri_len := (n + 1) * (n + 1)
	acc.n = n
	acc.ld = n + 1
	acc.tri = scratch[0:tri_len]
	acc.row = scratch[tri_len:tri_len + n + 1]
	ols_accum_reset(acc)
	return .None
}

// ols_accum_reset clears the accumulator back to zero observations, keeping
// the same block. Use this to start a new fit without reallocating.
ols_accum_reset :: proc(acc: ^Ols_Accum) {
	for i in 0 ..< len(acc.tri) {
		acc.tri[i] = 0.0
	}
	for i in 0 ..< len(acc.row) {
		acc.row[i] = 0.0
	}
	acc.nrows = 0
}

// ols_accum_rows folds a batch of rows into the accumulator.
//
// BATCH CONTRACT
//   acc   in/out  accumulator; acc.tri and acc.nrows are updated.
//   x     in      f64[>= (count-1)*ldx + n], row-major, row stride ldx >= n.
//                 Read only, never modified.
//   y     in      f64[>= count]. Read only.
//   count in      rows in this batch, >= 0. count == 0 is a no-op.
//   returns       how many rows were absorbed, and an error code.
//
//   Rows are validated one at a time immediately before being folded in, so a
//   bad row costs nothing extra in cache and leaves the accumulator valid:
//   on .Non_Finite_Input the returned count is the index of the offending row
//   and everything before it has been absorbed. A NaN cannot be removed from
//   the triangle once folded in, which is why this is checked rather than
//   propagated.
//
// Cost: 3n(n+1) flops per row. Memory: none beyond the block from init --
// in particular the caller need never hold all m rows at once.
//
// Access pattern: `tri` is walked as n+1 contiguous runs, shortening by one
// each step, and is small enough to stay resident (n = 5 is 288 bytes; n = 100
// is 80 KiB). The input batch is read linearly, once. The `row[j] == 0` test
// is the only data-dependent branch and is essentially never taken on dense
// input, so it predicts well; it is kept because it makes structurally sparse
// rows cheap and is a no-op otherwise.
ols_accum_rows :: proc(
	acc: ^Ols_Accum,
	x: []f64,
	ldx: int,
	y: []f64,
	count: int,
) -> (
	absorbed: int,
	err: Ols_Error,
) {
	n := acc.n
	if n < 1 {
		return 0, .Invalid_Dimension
	}
	if count < 0 || ldx < n {
		return 0, .Invalid_Dimension
	}
	if count == 0 {
		return 0, .None
	}
	if len(x) < (count - 1) * ldx + n || len(y) < count {
		return 0, .Invalid_Dimension
	}

	for i in 0 ..< count {
		base := i * ldx

		// Validate and stage in one pass; the row is in cache either way.
		for j in 0 ..< n {
			v := x[base + j]
			if !is_finite(v) {
				return i, .Non_Finite_Input
			}
			acc.row[j] = v
		}
		yv := y[i]
		if !is_finite(yv) {
			return i, .Non_Finite_Input
		}
		acc.row[n] = yv

		ols_absorb_row(acc.tri, acc.ld, acc.row, n)
		acc.nrows += 1
	}
	return count, .None
}

// ols_absorb_row folds one staged row into the triangle with n+1 Givens
// rotations, zeroing it as it goes. `row` is left all zeros.
@(private = "file")
ols_absorb_row :: proc(tri: []f64, ld: int, row: []f64, n: int) {
	for j in 0 ..< n + 1 {
		g := row[j]
		if g == 0.0 {
			continue
		}
		c, s, r := dlartg(tri[j * ld + j], g)
		tri[j * ld + j] = r
		row[j] = 0.0

		// Rotate the rest of triangle row j against the rest of the staged row.
		rest := n - j
		if rest > 0 {
			drot(rest, tri[j * ld + j + 1:], 1, row[j + 1:], 1, c, s)
		}
	}
}

// ols_accum_solve extracts the coefficients.
//
//   beta  out  f64[>= n]. Written only when the call returns .None.
//
// Safe to call at any point, including mid-stream: it reports
// .Not_Enough_Rows until n rows have been absorbed and .Rank_Deficient while
// the predictors seen so far are collinear. The accumulator is not modified,
// so streaming can continue afterwards.
//
// Cost: n^2. Does not touch the input data.
ols_accum_solve :: proc(
	acc: ^Ols_Accum,
	beta: []f64,
	rcond: f64 = OLS_DEFAULT_RCOND,
) -> Ols_Error {
	n, ld := acc.n, acc.ld
	if n < 1 {
		return .Invalid_Dimension
	}
	if len(beta) < n {
		return .Invalid_Dimension
	}
	if acc.nrows < n {
		return .Not_Enough_Rows
	}

	dmax, dmin := 0.0, max(f64)
	for j in 0 ..< n {
		d := abs(acc.tri[j * ld + j])
		dmax = max(dmax, d)
		dmin = min(dmin, d)
	}
	if dmax == 0.0 || dmin <= rcond * dmax {
		return .Rank_Deficient
	}

	for i in 0 ..< n {
		beta[i] = acc.tri[i * ld + n]
	}
	dtrsv(.Upper, .No_Trans, .Non_Unit, n, acc.tri, ld, beta, 1)
	return .None
}

// ============================================================================
// Iterating on the model without re-reading the data
// ============================================================================
//
// The triangle is a complete summary of the fit: T'T = A'A for A = [X | y].
// Two consequences, and they are what make model experimentation cheap.
//
//   1. Selecting columns commutes with the Gram product, so the R factor of any
//      SUBSET of the predictors can be built from the triangle alone --
//      ols_accum_select. Accumulate every candidate predictor once, then every
//      sub-model costs O(p*k^2) and never touches the m rows again.
//
//   2. The p+1 rows of the triangle span the same row space as the data that
//      produced them, so two triangles over disjoint row sets combine by
//      folding one into the other -- ols_accum_merge. Accumulate per data
//      segment once, then any union of segments costs O(p^3).
//
// Together those cover both axes of "fit the right model to the right data":
// which predictors, and which observations. Every experiment after the first
// pass is independent of m.

// ols_accum_select builds the sub-model over `keep` from a superset fit.
//
// BATCH CONTRACT
//   dst   out  accumulator with dst.n == len(keep). Reset and overwritten.
//              Must not alias src.
//   src   in   the superset fit. Not modified.
//   keep  in   src-predictor indices to retain, each in [0, src.n), no
//              duplicates. Order is free, so this also reorders predictors.
//              Column i of the sub-model is column keep[i] of the superset.
//
// dst.nrows is set to src.nrows: the sub-model sees the same observations,
// just fewer predictors. dst's RSS is the sub-model's RSS, so candidate models
// can be ranked straight out of ols_accum_rss with no further work.
//
// Cost: (src.n + 1) row folds into a (k+1) triangle, so O(src.n * k^2) --
// INDEPENDENT OF m. Re-accumulating the sub-model from raw data instead costs
// O(m * k^2), so this is cheaper by a factor of about m / src.n.
ols_accum_select :: proc(dst: ^Ols_Accum, src: ^Ols_Accum, keep: []int) -> Ols_Error {
	if dst.n < 1 || src.n < 1 {
		return .Invalid_Dimension
	}
	if len(keep) != dst.n || dst.n > src.n {
		return .Invalid_Dimension
	}
	// Aliasing would corrupt src as dst is reset and rewritten.
	if raw_data(dst.tri) == raw_data(src.tri) {
		return .Invalid_Dimension
	}
	// Duplicates would produce an exactly singular sub-model, which is a
	// caller mistake rather than a property of the data. Reject it here so it
	// is not mistaken for .Rank_Deficient later. O(k^2) on a small k.
	for a in 0 ..< len(keep) {
		if keep[a] < 0 || keep[a] >= src.n {
			return .Invalid_Dimension
		}
		for b in a + 1 ..< len(keep) {
			if keep[a] == keep[b] {
				return .Invalid_Dimension
			}
		}
	}

	ols_accum_reset(dst)

	// Each row of the source triangle, restricted to the kept columns plus the
	// response, is a valid pseudo-observation for the sub-model.
	for i in 0 ..< src.n + 1 {
		base := i * src.ld
		for j in 0 ..< dst.n {
			dst.row[j] = src.tri[base + keep[j]]
		}
		dst.row[dst.n] = src.tri[base + src.n]
		ols_absorb_row(dst.tri, dst.ld, dst.row, dst.n)
	}

	dst.nrows = src.nrows
	return .None
}

// ols_accum_merge folds src into dst, giving the fit over both row sets.
//
//   dst  in/out  accumulator; must have the same n as src. Must not alias src.
//   src  in      not modified.
//
// The two accumulators must cover DISJOINT observations -- merging overlapping
// sets double-counts the shared rows, silently and without any way to detect
// it here. dst.nrows becomes the sum.
//
// Cost: (n+1) row folds, so O(n^3), independent of how many rows either side
// absorbed. Merging is exact, not an approximation: the result matches
// accumulating every row into one accumulator, up to rotation ordering.
//
// This is what makes data experimentation cheap. Accumulate one triangle per
// segment -- per level, per session, per cohort -- and any union of segments
// costs O(n^3) instead of a fresh pass over the rows.
ols_accum_merge :: proc(dst: ^Ols_Accum, src: ^Ols_Accum) -> Ols_Error {
	if dst.n < 1 || dst.n != src.n {
		return .Invalid_Dimension
	}
	if raw_data(dst.tri) == raw_data(src.tri) {
		return .Invalid_Dimension
	}

	for i in 0 ..< src.n + 1 {
		base := i * src.ld
		for j in 0 ..< src.n + 1 {
			dst.row[j] = src.tri[base + j]
		}
		ols_absorb_row(dst.tri, dst.ld, dst.row, dst.n)
	}

	dst.nrows += src.nrows
	return .None
}

// ols_accum_rss returns ||X*beta - y||^2 over the rows absorbed so far.
//
// Free: the augmented factorization leaves the residual norm sitting on
// tri[n, n], so this needs no pass over the data and no prior call to
// ols_accum_solve.
ols_accum_rss :: proc(acc: ^Ols_Accum) -> f64 {
	if acc.n < 1 {
		return 0.0
	}
	e := acc.tri[acc.n * acc.ld + acc.n]
	return e * e
}

// ols_flops_per_row is the per-row cost of ols_accum_rows, for callers
// converting a frame budget into a batch size.
//
// Multiply by your measured flops-per-second: for the frame budget B seconds
// and rate F, batch = B*F / ols_flops_per_row(n). Measure F on the target
// machine; do not assume it.
ols_flops_per_row :: proc(n: int) -> int {
	return 3 * n * (n + 1)
}

// ols_residual_norm returns ||X*beta - y|| computed directly from the data,
// for checking a solution against the input it came from.
//
//   x  in  f64[>= (m-1)*ldx + n], row-major, stride ldx >= n. Read only.
//   y  in  f64[>= m]. Read only.
//
// Cost: 2mn, one linear pass. This recomputes from the data on purpose -- the
// residual reported by the solvers comes out of the factorization instead, so
// the two agreeing is a real check.
ols_residual_norm :: proc(m, n: int, x: []f64, ldx: int, beta: []f64, y: []f64) -> f64 {
	if m < 1 || n < 1 || ldx < n {
		return 0.0
	}
	acc := 0.0
	for i in 0 ..< m {
		base := i * ldx
		p := 0.0
		for j in 0 ..< n {
			p += x[base + j] * beta[j]
		}
		d := p - y[i]
		acc += d * d
	}
	return math.sqrt_f64(acc)
}
