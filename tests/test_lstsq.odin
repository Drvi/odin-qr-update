package tests

// Least-squares tests. Verifies the done-criteria in docs/OLS_PLAN.md section 6.
//
// Ground truth is numpy.linalg.lstsq (LAPACK) on the repository's own real
// data, examples/X_initial.csv + examples/y_initial.csv. Generated, do not
// hand-edit the arrays.

import "core:fmt"
import "core:math"
import "core:mem"
import "../src/blas"

REAL_M :: 20
REAL_N :: 5
real_x := [REAL_M * REAL_N]f64{
	1.00000000000000000e+00, 1.80696520626320889e+00, -2.00286407072816486e+00, -6.73314191532263018e-01, -2.27941591134884991e-01,
	1.00000000000000000e+00, -5.45015499141784585e-02, -4.31732891589146184e-01, -1.19017203695454898e+00, 1.72222864118495994e-01,
	1.00000000000000000e+00, -8.96776626880892658e-01, 1.23601307527518389e+00, -4.12325374515255874e-02, -5.21110069294932643e-01,
	1.00000000000000000e+00, 1.66103701017477001e+00, 6.02798106095064234e-01, -5.32091914098378305e-01, 2.24970018182077708e-01,
	1.00000000000000000e+00, -1.97874329697118906e-01, -2.81996745495467993e-01, 1.64021191582314513e-01, -1.44171087809297305e+00,
	1.00000000000000000e+00, 2.30105492504340287e-01, 1.17521243604356701e-01, -1.70377613195158595e-01, -7.02225165849471367e-01,
	1.00000000000000000e+00, 5.20378214727840738e-01, -5.45157761349789882e-01, -9.14304512966550798e-01, -1.50780672175817299e+00,
	1.00000000000000000e+00, -1.22302949366602398e+00, -2.83366716249365325e-01, 6.59632894632580569e-01, 1.66517292256850391e+00,
	1.00000000000000000e+00, -5.04490489433986472e-02, -1.86149461663608212e-01, -5.98320598999032782e-01, -1.18965501244372107e+00,
	1.00000000000000000e+00, -1.29664522288130701e+00, -2.60576862278575794e-01, -2.87170828722559779e-01, 9.99133707948695937e-01,
	1.00000000000000000e+00, -5.66056029108767289e-01, -7.12608445821623970e-01, 1.45145608409401894e+00, -5.39096322326081179e-01,
	1.00000000000000000e+00, -6.67073868415202741e-03, -1.43457563812222899e+00, -1.84674047786557094e+00, -9.16380261127881846e-01,
	1.00000000000000000e+00, 1.04058498434399405e+00, 5.81590168984240319e-01, 4.34082844883449115e-01, -5.72028059736093882e-03,
	1.00000000000000000e+00, -8.87897163962972247e-01, -5.38432380377212971e-01, -1.74283389766377295e+00, 1.50176123331625710e+00,
	1.00000000000000000e+00, 2.32149037983016004e-01, -1.36183641234462499e-01, -1.59885647051280300e+00, 1.04785035390173409e+00,
	1.00000000000000000e+00, -7.21928066734303098e-02, 5.04454722838971525e-01, 8.66322888054649048e-02, 4.50858424093556087e-01,
	1.00000000000000000e+00, 8.78586100015962379e-01, -1.87358656464988793e+00, 3.96189275527507601e-02, -8.73298397222637135e-01,
	1.00000000000000000e+00, -8.99117565472470237e-01, 2.35770800194150393e-01, -2.47387303650559387e-01, -1.46613057779397105e+00,
	1.00000000000000000e+00, -2.99224079059578818e+00, 2.41753287152915597e+00, -1.25240718263184198e-01, -5.66002250176695254e-01,
	1.00000000000000000e+00, -4.76580757933917909e-02, -5.18747230070099263e-01, 1.13168363241087394e-01, -6.00646125489620153e-01,
}
real_y := [REAL_M]f64{
	-2.42484874524200711e+00,
	2.60740913921959105e+00,
	2.97471512669853100e+00,
	5.79660846466808710e-01,
	-4.84703750074401085e-01,
	1.11872362978182305e+00,
	-1.51813365433644010e+00,
	4.68294059226219694e+00,
	1.54878840001781204e+00,
	5.24595966114646739e+00,
	1.43903666772483696e+00,
	1.59544752832341707e-01,
	5.02599314354152726e-01,
	4.61803753347725809e+00,
	3.66750667197709479e+00,
	2.91080504847655819e+00,
	-1.24501086727536192e+00,
	1.61679011979567799e+00,
	6.70525779343277328e+00,
	1.12232978338764000e+00,
}
truth_beta := [REAL_N]f64{
	1.89716706451919981e+00,
	-1.34505710023176062e+00,
	6.51590911962780073e-01,
	-2.68730046077009954e-01,
	1.22555665161290550e+00,
}
truth_rss :: 3.87978484064556417e+00

// ============================================================================
// Criterion 1: both transforms reproduce the ground truth on the real data
// ============================================================================

test_ols_dense_ground_truth :: proc() -> bool {
	fmt.println("Testing ols_solve_dense against numpy ground truth...")

	x := real_x // copy: the routine destroys its input
	y := real_y
	beta: [REAL_N]f64
	scratch := make([]f64, blas.ols_dense_scratch(REAL_N))
	defer delete(scratch)

	rss, err := blas.ols_solve_dense(REAL_M, REAL_N, x[:], REAL_N, y[:], beta[:], scratch)
	if err != .None {
		fmt.printf("  FAILED: err = %v\n", err)
		return false
	}
	for i in 0 ..< REAL_N {
		if abs(beta[i] - truth_beta[i]) > 1e-12 {
			fmt.printf("  FAILED: beta[%d] = %.17e want %.17e\n", i, beta[i], truth_beta[i])
			return false
		}
	}
	if abs(rss - truth_rss) > 1e-12 {
		fmt.printf("  FAILED: rss = %.17e want %.17e\n", rss, truth_rss)
		return false
	}
	fmt.println("  PASSED")
	return true
}

test_ols_accum_ground_truth :: proc() -> bool {
	fmt.println("Testing Ols_Accum against numpy ground truth...")

	scratch := make([]f64, blas.ols_accum_scratch(REAL_N))
	defer delete(scratch)
	acc: blas.Ols_Accum
	if e := blas.ols_accum_init(&acc, REAL_N, scratch); e != .None {
		fmt.printf("  FAILED: init %v\n", e)
		return false
	}

	n, err := blas.ols_accum_rows(&acc, real_x[:], REAL_N, real_y[:], REAL_M)
	if err != .None || n != REAL_M {
		fmt.printf("  FAILED: absorbed %d/%d err %v\n", n, REAL_M, err)
		return false
	}

	beta: [REAL_N]f64
	if e := blas.ols_accum_solve(&acc, beta[:]); e != .None {
		fmt.printf("  FAILED: solve %v\n", e)
		return false
	}
	for i in 0 ..< REAL_N {
		if abs(beta[i] - truth_beta[i]) > 1e-12 {
			fmt.printf("  FAILED: beta[%d] = %.17e want %.17e\n", i, beta[i], truth_beta[i])
			return false
		}
	}
	if abs(blas.ols_accum_rss(&acc) - truth_rss) > 1e-12 {
		fmt.printf("  FAILED: rss = %.17e want %.17e\n", blas.ols_accum_rss(&acc), truth_rss)
		return false
	}

	// The free residual from the factorization must match one recomputed from
	// the data itself.
	direct := blas.ols_residual_norm(REAL_M, REAL_N, real_x[:], REAL_N, beta[:], real_y[:])
	if abs(direct * direct - truth_rss) > 1e-12 {
		fmt.printf("  FAILED: recomputed rss %.17e\n", direct * direct)
		return false
	}
	fmt.println("  PASSED")
	return true
}

// ============================================================================
// Criterion 3: the accumulator result does not depend on batch size
// ============================================================================

test_ols_accum_chunk_invariance :: proc() -> bool {
	fmt.println("Testing Ols_Accum chunk-size invariance...")

	solve_in_chunks :: proc(chunk: int, beta: []f64) -> blas.Ols_Error {
		scratch := make([]f64, blas.ols_accum_scratch(REAL_N))
		defer delete(scratch)
		acc: blas.Ols_Accum
		blas.ols_accum_init(&acc, REAL_N, scratch)

		for acc.nrows < REAL_M {
			take := min(chunk, REAL_M - acc.nrows)
			_, e := blas.ols_accum_rows(
				&acc,
				real_x[acc.nrows * REAL_N:],
				REAL_N,
				real_y[acc.nrows:],
				take,
			)
			if e != .None {
				return e
			}
		}
		return blas.ols_accum_solve(&acc, beta)
	}

	ref: [REAL_N]f64
	if e := solve_in_chunks(REAL_M, ref[:]); e != .None {
		fmt.printf("  FAILED: reference solve %v\n", e)
		return false
	}

	for chunk in ([]int{1, 3, 7}) {
		got: [REAL_N]f64
		if e := solve_in_chunks(chunk, got[:]); e != .None {
			fmt.printf("  FAILED: chunk %d -> %v\n", chunk, e)
			return false
		}
		for i in 0 ..< REAL_N {
			// Folding rows one at a time performs exactly the same rotations in
			// exactly the same order regardless of how they are grouped, so this
			// is bit-identical, not merely close.
			if got[i] != ref[i] {
				fmt.printf(
					"  FAILED: chunk %d beta[%d] = %.17e want %.17e\n",
					chunk,
					i,
					got[i],
					ref[i],
				)
				return false
			}
		}
	}
	fmt.println("  PASSED")
	return true
}

// ============================================================================
// Criterion 2: the two transforms agree with each other
// ============================================================================

test_ols_transforms_agree :: proc() -> bool {
	fmt.println("Testing dense and accumulator paths agree...")

	// Larger, independently generated problem so this is not just a rerun of
	// the 20x5 case.
	M :: 400
	N :: 7
	x := make([]f64, M * N);defer delete(x)
	y := make([]f64, M);defer delete(y)

	state: u64 = 12345
	nextf :: proc(s: ^u64) -> f64 {
		s^ = s^ * 6364136223846793005 + 1442695040888963407
		return f64((s^ >> 11) & 0x1FFFFFFFFFFFFF) / f64(0x1FFFFFFFFFFFFF) * 2.0 - 1.0
	}
	for i in 0 ..< M {
		x[i * N] = 1.0
		for j in 1 ..< N {
			x[i * N + j] = nextf(&state)
		}
		s := 0.0
		for j in 0 ..< N {
			s += x[i * N + j] * f64(j + 1) * 0.3
		}
		y[i] = s + 0.05 * nextf(&state)
	}

	// Accumulator path
	as := make([]f64, blas.ols_accum_scratch(N));defer delete(as)
	acc: blas.Ols_Accum
	blas.ols_accum_init(&acc, N, as)
	blas.ols_accum_rows(&acc, x, N, y, M)
	beta_acc: [N]f64
	if e := blas.ols_accum_solve(&acc, beta_acc[:]); e != .None {
		fmt.printf("  FAILED: accum solve %v\n", e)
		return false
	}

	// Dense path (destroys its inputs, so give it copies)
	xd := make([]f64, M * N);defer delete(xd)
	yd := make([]f64, M);defer delete(yd)
	copy(xd, x)
	copy(yd, y)
	ds := make([]f64, blas.ols_dense_scratch(N));defer delete(ds)
	beta_den: [N]f64
	rss_den, e2 := blas.ols_solve_dense(M, N, xd, N, yd, beta_den[:], ds)
	if e2 != .None {
		fmt.printf("  FAILED: dense solve %v\n", e2)
		return false
	}

	for i in 0 ..< N {
		if abs(beta_acc[i] - beta_den[i]) > 1e-10 {
			fmt.printf(
				"  FAILED: beta[%d] accum %.17e dense %.17e\n",
				i,
				beta_acc[i],
				beta_den[i],
			)
			return false
		}
	}
	if abs(blas.ols_accum_rss(&acc) - rss_den) > 1e-10 {
		fmt.printf("  FAILED: rss accum %.17e dense %.17e\n", blas.ols_accum_rss(&acc), rss_den)
		return false
	}

	// Both must match a residual recomputed straight from the untouched data.
	rn := blas.ols_residual_norm(M, N, x, N, beta_acc[:], y)
	if abs(rn * rn - rss_den) > 1e-10 {
		fmt.printf("  FAILED: recomputed rss %.17e vs %.17e\n", rn * rn, rss_den)
		return false
	}
	fmt.println("  PASSED")
	return true
}

// ============================================================================
// Criterion 4: every boundary policy in the plan is provoked
// ============================================================================

test_ols_boundaries :: proc() -> bool {
	fmt.println("Testing OLS boundary policies...")

	scratch := make([]f64, blas.ols_accum_scratch(REAL_N));defer delete(scratch)
	dscratch := make([]f64, blas.ols_dense_scratch(REAL_N));defer delete(dscratch)
	beta: [REAL_N]f64

	// -- Scratch_Too_Small
	{
		acc: blas.Ols_Accum
		small := make([]f64, blas.ols_accum_scratch(REAL_N) - 1);defer delete(small)
		if e := blas.ols_accum_init(&acc, REAL_N, small); e != .Scratch_Too_Small {
			fmt.printf("  FAILED: short scratch -> %v\n", e)
			return false
		}
	}
	// -- Invalid_Dimension: n < 1
	{
		acc: blas.Ols_Accum
		if e := blas.ols_accum_init(&acc, 0, scratch); e != .Invalid_Dimension {
			fmt.printf("  FAILED: n=0 -> %v\n", e)
			return false
		}
	}
	// -- Invalid_Dimension: ldx < n
	{
		acc: blas.Ols_Accum
		blas.ols_accum_init(&acc, REAL_N, scratch)
		if _, e := blas.ols_accum_rows(&acc, real_x[:], REAL_N - 1, real_y[:], 2);
		   e != .Invalid_Dimension {
			fmt.printf("  FAILED: ldx<n -> %v\n", e)
			return false
		}
	}
	// -- Not_Enough_Rows: solving before n rows have arrived
	{
		acc: blas.Ols_Accum
		blas.ols_accum_init(&acc, REAL_N, scratch)
		blas.ols_accum_rows(&acc, real_x[:], REAL_N, real_y[:], 3)
		if e := blas.ols_accum_solve(&acc, beta[:]); e != .Not_Enough_Rows {
			fmt.printf("  FAILED: 3 rows -> %v\n", e)
			return false
		}
	}
	// -- Non_Finite_Input: rows before the bad one are kept, accumulator valid
	{
		acc: blas.Ols_Accum
		blas.ols_accum_init(&acc, REAL_N, scratch)
		bad := real_x
		bady := real_y
		bad[4 * REAL_N + 2] = math.nan_f64()
		got, e := blas.ols_accum_rows(&acc, bad[:], REAL_N, bady[:], REAL_M)
		if e != .Non_Finite_Input || got != 4 || acc.nrows != 4 {
			fmt.printf("  FAILED: NaN row -> got %d nrows %d err %v\n", got, acc.nrows, e)
			return false
		}
		// The triangle must be untainted: absorbing the remaining good rows
		// still yields a finite, solvable system.
		if _, e2 := blas.ols_accum_rows(
			&acc,
			real_x[5 * REAL_N:],
			REAL_N,
			real_y[5:],
			REAL_M - 5,
		); e2 != .None {
			fmt.printf("  FAILED: resume after NaN -> %v\n", e2)
			return false
		}
		if e3 := blas.ols_accum_solve(&acc, beta[:]); e3 != .None {
			fmt.printf("  FAILED: solve after NaN recovery -> %v\n", e3)
			return false
		}
		for i in 0 ..< REAL_N {
			if beta[i] != beta[i] {
				fmt.println("  FAILED: NaN reached beta")
				return false
			}
		}
	}
	// -- Non_Finite_Input on the dense path, with Inf rather than NaN
	{
		bx := real_x
		by := real_y
		by[7] = math.inf_f64(1)
		if _, e := blas.ols_solve_dense(
			REAL_M,
			REAL_N,
			bx[:],
			REAL_N,
			by[:],
			beta[:],
			dscratch,
		); e != .Non_Finite_Input {
			fmt.printf("  FAILED: Inf y -> %v\n", e)
			return false
		}
	}
	// -- Rank_Deficient: duplicate a predictor to make X collinear
	{
		M :: 12
		N :: 3
		x := make([]f64, M * N);defer delete(x)
		y := make([]f64, M);defer delete(y)
		for i in 0 ..< M {
			v := f64(i) * 0.37 - 2.0
			x[i * N + 0] = 1.0
			x[i * N + 1] = v
			x[i * N + 2] = v // exact copy of column 1
			y[i] = 3.0 * v + 1.0
		}
		as := make([]f64, blas.ols_accum_scratch(N));defer delete(as)
		acc: blas.Ols_Accum
		blas.ols_accum_init(&acc, N, as)
		blas.ols_accum_rows(&acc, x, N, y, M)
		b3: [N]f64
		if e := blas.ols_accum_solve(&acc, b3[:]); e != .Rank_Deficient {
			fmt.printf("  FAILED: collinear accum -> %v\n", e)
			return false
		}
		ds := make([]f64, blas.ols_dense_scratch(N));defer delete(ds)
		if _, e := blas.ols_solve_dense(M, N, x, N, y, b3[:], ds); e != .Rank_Deficient {
			fmt.printf("  FAILED: collinear dense -> %v\n", e)
			return false
		}
	}
	// -- m < n on the dense path
	{
		x := make([]f64, 2 * 5);defer delete(x)
		y := make([]f64, 2);defer delete(y)
		if _, e := blas.ols_solve_dense(2, 5, x, 5, y, beta[:], dscratch);
		   e != .Not_Enough_Rows {
			fmt.printf("  FAILED: m<n -> %v\n", e)
			return false
		}
	}
	// -- count == 0 is a no-op, not an error
	{
		acc: blas.Ols_Accum
		blas.ols_accum_init(&acc, REAL_N, scratch)
		if got, e := blas.ols_accum_rows(&acc, real_x[:], REAL_N, real_y[:], 0);
		   e != .None || got != 0 || acc.nrows != 0 {
			fmt.printf("  FAILED: count=0 -> got %d err %v\n", got, e)
			return false
		}
	}
	fmt.println("  PASSED")
	return true
}

// ============================================================================
// Regression tests for the numerical bugs fixed alongside this work
// ============================================================================

test_dnrm2_scaling :: proc() -> bool {
	fmt.println("Testing dnrm2 over/underflow scaling...")

	// Blue's constants must satisfy these exact powers of two; a sign slip in
	// the exponents silently flushes small vectors to zero.
	if blas.NRM2_TSML != math.pow_f64(2, -511) ||
	   blas.NRM2_TBIG != math.pow_f64(2, 486) ||
	   blas.NRM2_SSML != math.pow_f64(2, 537) ||
	   blas.NRM2_SBIG != math.pow_f64(2, -538) {
		fmt.println("  FAILED: Blue's algorithm constants are wrong")
		return false
	}

	cases := [][3]f64 {
		{3.0, 4.0, 5.0},
		{3.0e-100, 4.0e-100, 5.0e-100}, // underflows if ssml/tsml are wrong
		{3.0e200, 4.0e200, 5.0e200}, // overflows without scaling
		{1.0e-160, 0.0, 1.0e-160},
	}
	for c in cases {
		v := []f64{c[0], c[1]}
		got := blas.dnrm2(2, v, 1)
		if abs(got - c[2]) > 1e-12 * c[2] {
			fmt.printf("  FAILED: dnrm2([%.3e,%.3e]) = %.6e want %.6e\n", c[0], c[1], got, c[2])
			return false
		}
	}
	fmt.println("  PASSED")
	return true
}

test_row_major_lda_checks :: proc() -> bool {
	fmt.println("Testing dgemv/dger accept tall row-major matrices...")

	// A tall matrix with lda = n is the normal OLS shape. Validating lda
	// against m (the Fortran column-major rule) rejected all of these.
	a := []f64{1, 2, 3, 4, 5, 6, 7, 8} // 4x2, row-major, lda=2
	x := []f64{1, 1}
	y := []f64{0, 0, 0, 0}
	blas.dgemv(.No_Trans, 4, 2, 1.0, a, 2, x, 1, 0.0, y, 1)
	want := []f64{3, 7, 11, 15}
	for i in 0 ..< 4 {
		if abs(y[i] - want[i]) > 1e-12 {
			fmt.printf("  FAILED: dgemv y = %v want %v\n", y, want)
			return false
		}
	}

	g := []f64{0, 0, 0, 0, 0, 0, 0, 0}
	u := []f64{1, 2, 3, 4}
	v := []f64{10, 20}
	blas.dger(4, 2, 1.0, u, 1, v, 1, g, 2)
	wantg := []f64{10, 20, 20, 40, 30, 60, 40, 80}
	for i in 0 ..< 8 {
		if abs(g[i] - wantg[i]) > 1e-12 {
			fmt.printf("  FAILED: dger = %v want %v\n", g, wantg)
			return false
		}
	}

	// dlarf is built on both, so it was a silent no-op for tall C.
	c := []f64{1, 0, 0, 1, 0, 0, 0, 0}
	vv := []f64{1, 0, 0, 0}
	w := []f64{0, 0}
	blas.dlarf(.Left, 4, 2, vv, 1, 2.0, c, 2, w)
	if abs(c[0] - (-1.0)) > 1e-12 {
		fmt.printf("  FAILED: dlarf c[0] = %.6e want -1\n", c[0])
		return false
	}
	fmt.println("  PASSED")
	return true
}

test_dgeqrf_extreme_scale :: proc() -> bool {
	fmt.println("Testing dgeqrf on extreme-magnitude data...")

	// Squaring entries of this size overflows to Inf; dlapy2 avoids it.
	for scale in ([]f64{1.0e200, 1.0e-200}) {
		a := []f64{scale, scale * 0.5, scale, scale * 0.25}
		tau := []f64{0, 0}
		work := []f64{0, 0}
		blas.dgeqrf(2, 2, a, 2, tau, work)
		for v in a {
			if v != v || abs(v) > max(f64) {
				fmt.printf("  FAILED: scale %.0e produced %v\n", scale, a)
				return false
			}
		}
		expect := scale * math.sqrt_f64(2.0)
		if abs(abs(a[0]) - expect) > 1e-10 * expect {
			fmt.printf("  FAILED: |R[0,0]| = %.6e want %.6e\n", abs(a[0]), expect)
			return false
		}
	}
	fmt.println("  PASSED")
	return true
}
// All 31 non-empty predictor subsets of the real 20x5 data.
// Ground truth from numpy.linalg.lstsq. Generated; do not hand-edit.
// Fields: mask (bit i = predictor i kept), k, rss, then k betas in
// increasing predictor order, zero-padded to 5.
Subset_Truth :: struct { mask: int, k: int, rss: f64, beta: [5]f64 }
SUBSET_TRUTH := [31]Subset_Truth{
	{ 1, 1, 1.09305950731535319e+02, {1.79137040320616725e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00}},
	{ 2, 1, 8.33543566852987823e+01, {-1.97270804289350510e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00}},
	{ 4, 1, 1.49642376571022425e+02, {1.09712975037115856e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00}},
	{ 8, 1, 1.62560275721744659e+02, {-8.44487766070021739e-01, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00}},
	{16, 1, 1.55407534090944011e+02, {9.82835449490391366e-01, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00}},
	{ 3, 2, 3.67649495055484934e+01, {1.53954513376579594e+00, -1.78516971818153047e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00}},
	{ 5, 2, 6.83955707425052850e+01, {2.04761970901275703e+00, 1.45998629213401077e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00}},
	{ 9, 2, 1.09295677768703456e+02, {1.78144941388789735e+00, -2.82672199924407161e-02, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00}},
	{17, 2, 6.97978320509897685e+01, {2.12716079954227899e+00, 1.49381121248028514e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00}},
	{ 6, 2, 8.33469011983154502e+01, {-1.96216883368683392e+00, 2.24998384488582261e-02, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00}},
	{10, 2, 6.67059657173935676e+01, {-2.04094792222110000e+00, -1.04581387818703209e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00}},
	{18, 2, 7.22654524563500900e+01, {-1.90399311401862059e+00, 7.73524456368599234e-01, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00}},
	{12, 2, 1.27311796607060458e+02, {1.38071700857799118e+00, -1.24962495283803188e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00}},
	{20, 2, 1.34955500751740374e+02, {1.02015792816432205e+00, 8.89388926596517249e-01, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00}},
	{24, 2, 1.45098077225504483e+02, {-8.20527817806849136e-01, 9.66178826806529112e-01, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00}},
	{ 7, 3, 3.23056764897202626e+01, {1.68061765529784668e+00, -1.49916508357593292e+00, 5.73895696577674141e-01, 0.00000000000000000e+00, 0.00000000000000000e+00}},
	{11, 3, 3.49115949447949276e+01, {1.39889173582030502e+00, -1.82734658538473260e+00, -3.83802392160850381e-01, 0.00000000000000000e+00, 0.00000000000000000e+00}},
	{19, 3, 9.82410107341190830e+00, {1.84012338491998806e+00, -1.63797560578373735e+00, 1.24479355700179850e+00, 0.00000000000000000e+00, 0.00000000000000000e+00}},
	{13, 3, 6.62580752489426317e+01, {1.91376098488855129e+00, 1.53086176731227619e+00, -4.16838442558730815e-01, 0.00000000000000000e+00, 0.00000000000000000e+00}},
	{21, 3, 3.27304620076842099e+01, {2.35503906376014527e+00, 1.39147591965002637e+00, 1.42109170012908548e+00, 0.00000000000000000e+00, 0.00000000000000000e+00}},
	{25, 3, 6.92788862284253213e+01, {2.20331143799011775e+00, 2.02620747940022494e-01, 1.51621691052426111e+00, 0.00000000000000000e+00, 0.00000000000000000e+00}},
	{14, 3, 6.53693872946457901e+01, {-1.90014648678495601e+00, 3.11385683930985646e-01, -1.12329315268458041e+00, 0.00000000000000000e+00, 0.00000000000000000e+00}},
	{22, 3, 7.22612250171360415e+01, {-1.91187642564371885e+00, -1.69601151171921843e-02, 7.74211370581925573e-01, 0.00000000000000000e+00, 0.00000000000000000e+00}},
	{26, 3, 5.64237168529014994e+01, {-1.97310595671780575e+00, -1.02064155291418190e+00, 7.45207789918793218e-01, 0.00000000000000000e+00, 0.00000000000000000e+00}},
	{28, 3, 1.14255389305396434e+02, {1.29781768487652993e+00, -1.20448160879932287e+00, 8.39504435873787092e-01, 0.00000000000000000e+00, 0.00000000000000000e+00}},
	{15, 4, 2.93877595735770392e+01, {1.51983480641483171e+00, -1.51654681430997362e+00, 6.46500370874330099e-01, -4.87430118326309059e-01, 0.00000000000000000e+00}},
	{23, 4, 4.74673238581670365e+00, {1.99425145493209333e+00, -1.33094653949034614e+00, 6.12604349265916337e-01, 1.25945534309503171e+00, 0.00000000000000000e+00}},
	{27, 4, 9.49087666364173010e+00, {1.77462486369735473e+00, -1.65859461405865138e+00, -1.64663603330981484e-01, 1.22345048017995151e+00, 0.00000000000000000e+00}},
	{29, 4, 3.23567038775825822e+01, {2.29391602729728872e+00, 1.42244012262914188e+00, -1.76128133385949404e-01, 1.39999733069710941e+00, 0.00000000000000000e+00}},
	{30, 4, 5.54608602488324536e+01, {-1.85456196711568544e+00, 2.64691442353514494e-01, -1.08692656863627080e+00, 7.32648314663321809e-01, 0.00000000000000000e+00}},
	{31, 5, 3.87978484064556550e+00, {1.89716706451919981e+00, -1.34505710023176062e+00, 6.51590911962780073e-01, -2.68730046077009954e-01, 1.22555665161290550e+00}},
}

// ============================================================================
// Model iteration: select a sub-model without re-reading the data
// ============================================================================

// Build the full 5-predictor fit once; every sub-model comes from it.
build_full :: proc(scratch: []f64, acc: ^blas.Ols_Accum) -> bool {
	if blas.ols_accum_init(acc, REAL_N, scratch) != .None {
		return false
	}
	_, e := blas.ols_accum_rows(acc, real_x[:], REAL_N, real_y[:], REAL_M)
	return e == .None
}

test_ols_select_all_subsets :: proc() -> bool {
	fmt.println("Testing ols_accum_select over all 31 predictor subsets...")

	full_scratch := make([]f64, blas.ols_accum_scratch(REAL_N));defer delete(full_scratch)
	full: blas.Ols_Accum
	if !build_full(full_scratch, &full) {
		fmt.println("  FAILED: could not build superset fit")
		return false
	}

	for truth in SUBSET_TRUTH {
		// Decode the mask into predictor indices.
		keep: [REAL_N]int
		k := 0
		for i in 0 ..< REAL_N {
			if truth.mask & (1 << uint(i)) != 0 {
				keep[k] = i
				k += 1
			}
		}
		if k != truth.k {
			fmt.printf("  FAILED: mask %d decoded to %d indices\n", truth.mask, k)
			return false
		}

		sub_scratch := make([]f64, blas.ols_accum_scratch(k));defer delete(sub_scratch)
		sub: blas.Ols_Accum
		if e := blas.ols_accum_init(&sub, k, sub_scratch); e != .None {
			fmt.printf("  FAILED: init k=%d -> %v\n", k, e)
			return false
		}
		if e := blas.ols_accum_select(&sub, &full, keep[:k]); e != .None {
			fmt.printf("  FAILED: select mask %d -> %v\n", truth.mask, e)
			return false
		}

		// The sub-model must see the same observations as its parent.
		if sub.nrows != REAL_M {
			fmt.printf("  FAILED: mask %d nrows %d want %d\n", truth.mask, sub.nrows, REAL_M)
			return false
		}

		beta: [REAL_N]f64
		if e := blas.ols_accum_solve(&sub, beta[:k]); e != .None {
			fmt.printf("  FAILED: solve mask %d -> %v\n", truth.mask, e)
			return false
		}
		for j in 0 ..< k {
			if abs(beta[j] - truth.beta[j]) > 1e-10 {
				fmt.printf(
					"  FAILED: mask %d beta[%d] = %.17e want %.17e\n",
					truth.mask,
					j,
					beta[j],
					truth.beta[j],
				)
				return false
			}
		}
		// RSS drives model ranking, so it has to be right too.
		if abs(blas.ols_accum_rss(&sub) - truth.rss) > 1e-9 * max(1.0, truth.rss) {
			fmt.printf(
				"  FAILED: mask %d rss = %.17e want %.17e\n",
				truth.mask,
				blas.ols_accum_rss(&sub),
				truth.rss,
			)
			return false
		}
	}
	fmt.printf("  PASSED (%d subsets vs numpy)\n", len(SUBSET_TRUTH))
	return true
}

test_ols_select_matches_direct :: proc() -> bool {
	fmt.println("Testing select-from-superset == accumulate-that-subset...")

	full_scratch := make([]f64, blas.ols_accum_scratch(REAL_N));defer delete(full_scratch)
	full: blas.Ols_Accum
	build_full(full_scratch, &full)

	// Including a reordering, which select allows.
	cases := [][]int{{0, 2, 4}, {1, 3}, {4, 0}, {0, 1, 2, 3, 4}, {3}}
	for keep in cases {
		k := len(keep)

		sub_scratch := make([]f64, blas.ols_accum_scratch(k));defer delete(sub_scratch)
		sub: blas.Ols_Accum
		blas.ols_accum_init(&sub, k, sub_scratch)
		if e := blas.ols_accum_select(&sub, &full, keep); e != .None {
			fmt.printf("  FAILED: select %v -> %v\n", keep, e)
			return false
		}

		// Gather those columns out of the raw data and fit them directly.
		gx := make([]f64, REAL_M * k);defer delete(gx)
		for i in 0 ..< REAL_M {
			for j in 0 ..< k {
				gx[i * k + j] = real_x[i * REAL_N + keep[j]]
			}
		}
		dir_scratch := make([]f64, blas.ols_accum_scratch(k));defer delete(dir_scratch)
		direct: blas.Ols_Accum
		blas.ols_accum_init(&direct, k, dir_scratch)
		blas.ols_accum_rows(&direct, gx, k, real_y[:], REAL_M)

		bs := make([]f64, k);defer delete(bs)
		bd := make([]f64, k);defer delete(bd)
		if blas.ols_accum_solve(&sub, bs) != .None || blas.ols_accum_solve(&direct, bd) != .None {
			fmt.printf("  FAILED: solve %v\n", keep)
			return false
		}
		for j in 0 ..< k {
			if abs(bs[j] - bd[j]) > 1e-10 {
				fmt.printf("  FAILED: %v beta[%d] select %.17e direct %.17e\n", keep, j, bs[j], bd[j])
				return false
			}
		}
		if abs(blas.ols_accum_rss(&sub) - blas.ols_accum_rss(&direct)) > 1e-9 {
			fmt.printf("  FAILED: %v rss differs\n", keep)
			return false
		}
		if sub.nrows != direct.nrows {
			fmt.printf("  FAILED: %v nrows %d vs %d\n", keep, sub.nrows, direct.nrows)
			return false
		}
	}
	fmt.println("  PASSED")
	return true
}

// ============================================================================
// Model iteration: merge data segments
// ============================================================================

test_ols_merge :: proc() -> bool {
	fmt.println("Testing ols_accum_merge...")

	// Reference: everything in one accumulator.
	ref_scratch := make([]f64, blas.ols_accum_scratch(REAL_N));defer delete(ref_scratch)
	ref: blas.Ols_Accum
	build_full(ref_scratch, &ref)
	ref_beta: [REAL_N]f64
	blas.ols_accum_solve(&ref, ref_beta[:])

	check :: proc(
		label: string,
		got: ^blas.Ols_Accum,
		want_beta: []f64,
		want_rss: f64,
		want_rows: int,
	) -> bool {
		if got.nrows != want_rows {
			fmt.printf("  FAILED: %s nrows %d want %d\n", label, got.nrows, want_rows)
			return false
		}
		b: [REAL_N]f64
		if e := blas.ols_accum_solve(got, b[:]); e != .None {
			fmt.printf("  FAILED: %s solve %v\n", label, e)
			return false
		}
		for j in 0 ..< REAL_N {
			if abs(b[j] - want_beta[j]) > 1e-10 {
				fmt.printf("  FAILED: %s beta[%d] %.17e want %.17e\n", label, j, b[j], want_beta[j])
				return false
			}
		}
		if abs(blas.ols_accum_rss(got) - want_rss) > 1e-9 {
			fmt.printf("  FAILED: %s rss %.17e want %.17e\n", label, blas.ols_accum_rss(got), want_rss)
			return false
		}
		return true
	}

	// Split into two disjoint halves and merge.
	SPLIT :: 7
	sa := make([]f64, blas.ols_accum_scratch(REAL_N));defer delete(sa)
	sb := make([]f64, blas.ols_accum_scratch(REAL_N));defer delete(sb)
	a, b: blas.Ols_Accum
	blas.ols_accum_init(&a, REAL_N, sa)
	blas.ols_accum_init(&b, REAL_N, sb)
	blas.ols_accum_rows(&a, real_x[:], REAL_N, real_y[:], SPLIT)
	blas.ols_accum_rows(&b, real_x[SPLIT * REAL_N:], REAL_N, real_y[SPLIT:], REAL_M - SPLIT)

	if e := blas.ols_accum_merge(&a, &b); e != .None {
		fmt.printf("  FAILED: merge -> %v\n", e)
		return false
	}
	if !check("merge(a,b)", &a, ref_beta[:], truth_rss, REAL_M) {
		return false
	}

	// Merging the other way round must agree; rotation order differs, so this
	// is a tolerance check rather than bit-equality.
	sc := make([]f64, blas.ols_accum_scratch(REAL_N));defer delete(sc)
	sd := make([]f64, blas.ols_accum_scratch(REAL_N));defer delete(sd)
	c, d: blas.Ols_Accum
	blas.ols_accum_init(&c, REAL_N, sc)
	blas.ols_accum_init(&d, REAL_N, sd)
	blas.ols_accum_rows(&c, real_x[:], REAL_N, real_y[:], SPLIT)
	blas.ols_accum_rows(&d, real_x[SPLIT * REAL_N:], REAL_N, real_y[SPLIT:], REAL_M - SPLIT)
	if e := blas.ols_accum_merge(&d, &c); e != .None {
		fmt.printf("  FAILED: reverse merge -> %v\n", e)
		return false
	}
	if !check("merge(b,a)", &d, ref_beta[:], truth_rss, REAL_M) {
		return false
	}

	// Three-way, merged pairwise.
	s1 := make([]f64, blas.ols_accum_scratch(REAL_N));defer delete(s1)
	s2 := make([]f64, blas.ols_accum_scratch(REAL_N));defer delete(s2)
	s3 := make([]f64, blas.ols_accum_scratch(REAL_N));defer delete(s3)
	p1, p2, p3: blas.Ols_Accum
	blas.ols_accum_init(&p1, REAL_N, s1)
	blas.ols_accum_init(&p2, REAL_N, s2)
	blas.ols_accum_init(&p3, REAL_N, s3)
	blas.ols_accum_rows(&p1, real_x[:], REAL_N, real_y[:], 5)
	blas.ols_accum_rows(&p2, real_x[5 * REAL_N:], REAL_N, real_y[5:], 5)
	blas.ols_accum_rows(&p3, real_x[10 * REAL_N:], REAL_N, real_y[10:], REAL_M - 10)
	blas.ols_accum_merge(&p2, &p3)
	blas.ols_accum_merge(&p1, &p2)
	if !check("3-way", &p1, ref_beta[:], truth_rss, REAL_M) {
		return false
	}

	// An empty accumulator is an identity element.
	se := make([]f64, blas.ols_accum_scratch(REAL_N));defer delete(se)
	sf := make([]f64, blas.ols_accum_scratch(REAL_N));defer delete(sf)
	full2, empty: blas.Ols_Accum
	build_full(se, &full2)
	blas.ols_accum_init(&empty, REAL_N, sf)
	if e := blas.ols_accum_merge(&full2, &empty); e != .None {
		fmt.printf("  FAILED: merge empty -> %v\n", e)
		return false
	}
	if !check("merge(full,empty)", &full2, ref_beta[:], truth_rss, REAL_M) {
		return false
	}

	fmt.println("  PASSED")
	return true
}

test_ols_merge_then_select :: proc() -> bool {
	fmt.println("Testing merge then select (the full iteration loop)...")

	// Two data segments, combined, then a sub-model chosen -- and the answer
	// must still match numpy for that subset over all 20 rows.
	SPLIT :: 12
	sa := make([]f64, blas.ols_accum_scratch(REAL_N));defer delete(sa)
	sb := make([]f64, blas.ols_accum_scratch(REAL_N));defer delete(sb)
	a, b: blas.Ols_Accum
	blas.ols_accum_init(&a, REAL_N, sa)
	blas.ols_accum_init(&b, REAL_N, sb)
	blas.ols_accum_rows(&a, real_x[:], REAL_N, real_y[:], SPLIT)
	blas.ols_accum_rows(&b, real_x[SPLIT * REAL_N:], REAL_N, real_y[SPLIT:], REAL_M - SPLIT)
	blas.ols_accum_merge(&a, &b)

	for truth in SUBSET_TRUTH {
		keep: [REAL_N]int
		k := 0
		for i in 0 ..< REAL_N {
			if truth.mask & (1 << uint(i)) != 0 {
				keep[k] = i
				k += 1
			}
		}
		ss := make([]f64, blas.ols_accum_scratch(k));defer delete(ss)
		sub: blas.Ols_Accum
		blas.ols_accum_init(&sub, k, ss)
		if e := blas.ols_accum_select(&sub, &a, keep[:k]); e != .None {
			fmt.printf("  FAILED: select mask %d -> %v\n", truth.mask, e)
			return false
		}
		beta: [REAL_N]f64
		if e := blas.ols_accum_solve(&sub, beta[:k]); e != .None {
			fmt.printf("  FAILED: solve mask %d -> %v\n", truth.mask, e)
			return false
		}
		for j in 0 ..< k {
			if abs(beta[j] - truth.beta[j]) > 1e-9 {
				fmt.printf("  FAILED: mask %d beta[%d] %.17e want %.17e\n", truth.mask, j, beta[j], truth.beta[j])
				return false
			}
		}
	}
	fmt.println("  PASSED")
	return true
}

test_ols_iteration_boundaries :: proc() -> bool {
	fmt.println("Testing select/merge boundary policies...")

	fs := make([]f64, blas.ols_accum_scratch(REAL_N));defer delete(fs)
	full: blas.Ols_Accum
	build_full(fs, &full)

	s3 := make([]f64, blas.ols_accum_scratch(3));defer delete(s3)
	sub: blas.Ols_Accum
	blas.ols_accum_init(&sub, 3, s3)

	// keep length must match dst.n
	if e := blas.ols_accum_select(&sub, &full, []int{0, 1}); e != .Invalid_Dimension {
		fmt.printf("  FAILED: short keep -> %v\n", e)
		return false
	}
	// index out of range
	if e := blas.ols_accum_select(&sub, &full, []int{0, 1, REAL_N}); e != .Invalid_Dimension {
		fmt.printf("  FAILED: oob index -> %v\n", e)
		return false
	}
	if e := blas.ols_accum_select(&sub, &full, []int{0, -1, 2}); e != .Invalid_Dimension {
		fmt.printf("  FAILED: negative index -> %v\n", e)
		return false
	}
	// duplicates are a caller error, distinct from .Rank_Deficient
	if e := blas.ols_accum_select(&sub, &full, []int{0, 2, 2}); e != .Invalid_Dimension {
		fmt.printf("  FAILED: duplicate index -> %v\n", e)
		return false
	}
	// aliasing
	if e := blas.ols_accum_select(&full, &full, []int{0, 1, 2, 3, 4}); e != .Invalid_Dimension {
		fmt.printf("  FAILED: select aliased -> %v\n", e)
		return false
	}
	if e := blas.ols_accum_merge(&full, &full); e != .Invalid_Dimension {
		fmt.printf("  FAILED: merge aliased -> %v\n", e)
		return false
	}
	// merge with mismatched n
	if e := blas.ols_accum_merge(&sub, &full); e != .Invalid_Dimension {
		fmt.printf("  FAILED: merge n mismatch -> %v\n", e)
		return false
	}
	// selecting more predictors than the parent has
	s9 := make([]f64, blas.ols_accum_scratch(REAL_N + 1));defer delete(s9)
	big: blas.Ols_Accum
	blas.ols_accum_init(&big, REAL_N + 1, s9)
	if e := blas.ols_accum_select(&big, &full, []int{0, 1, 2, 3, 4, 0}); e != .Invalid_Dimension {
		fmt.printf("  FAILED: widen -> %v\n", e)
		return false
	}

	// A sub-model that is rank deficient still reports Rank_Deficient, not a
	// select error: predictor 0 is the intercept, and a constant column plus
	// nothing else is fine, but two copies is what we reject above. Here use a
	// genuinely collinear parent instead.
	M :: 10
	NP :: 3
	x := make([]f64, M * NP);defer delete(x)
	yy := make([]f64, M);defer delete(yy)
	for i in 0 ..< M {
		v := f64(i)
		x[i * NP + 0] = 1.0
		x[i * NP + 1] = v
		x[i * NP + 2] = 2.0 * v // collinear with column 1
		yy[i] = v
	}
	ps := make([]f64, blas.ols_accum_scratch(NP));defer delete(ps)
	par: blas.Ols_Accum
	blas.ols_accum_init(&par, NP, ps)
	blas.ols_accum_rows(&par, x, NP, yy, M)

	cs := make([]f64, blas.ols_accum_scratch(2));defer delete(cs)
	col: blas.Ols_Accum
	blas.ols_accum_init(&col, 2, cs)
	blas.ols_accum_select(&col, &par, []int{1, 2}) // both collinear columns
	bb: [2]f64
	if e := blas.ols_accum_solve(&col, bb[:]); e != .Rank_Deficient {
		fmt.printf("  FAILED: collinear sub-model -> %v\n", e)
		return false
	}
	// but a non-degenerate sub-model of the same parent solves fine
	cs2 := make([]f64, blas.ols_accum_scratch(2));defer delete(cs2)
	ok2: blas.Ols_Accum
	blas.ols_accum_init(&ok2, 2, cs2)
	blas.ols_accum_select(&ok2, &par, []int{0, 1})
	if e := blas.ols_accum_solve(&ok2, bb[:]); e != .None {
		fmt.printf("  FAILED: good sub-model -> %v\n", e)
		return false
	}

	fmt.println("  PASSED")
	return true
}
// Design matrices with prescribed condition numbers, for checking that
// both solvers degrade gracefully. Built as U*diag(s)*V^T with
// geometrically spaced singular values. Ground truth is numpy lstsq.
COND_M :: 40
COND_N :: 5
COND_CASES :: 4
cond_values := [COND_CASES]f64{1.0e+02, 1.0e+04, 1.0e+06, 1.0e+08}
cond_x := [COND_CASES * COND_M * COND_N]f64{
	-3.70355572946670045e-03,
	-5.23391138050690631e-03,
	5.60599113927159973e-03,
	2.14552896499778118e-03,
	-1.60964327242107547e-02,
	-1.11838800966330476e-01,
	1.48658372357204110e-01,
	1.54462785002106515e-02,
	5.13227955917600932e-03,
	1.13755626315920517e-02,
	5.56365410591406900e-02,
	-7.35596908713840592e-02,
	1.86949296373832355e-02,
	-3.33065251797667059e-03,
	-1.18532107623198872e-02,
	7.70357275447965950e-02,
	-9.91229323794461936e-02,
	-3.99628430618630057e-02,
	1.47422792726779883e-02,
	5.49287817818500668e-02,
	-2.24136629288468364e-01,
	2.57651223053914469e-01,
	-4.71606991009210844e-02,
	1.73166352342271311e-02,
	-1.25903194865094378e-02,
	-2.28947467909197261e-02,
	3.62049583752218004e-02,
	-8.82463393294381365e-02,
	2.18633243729404216e-02,
	1.00932668604626088e-01,
	-1.89509212871670690e-01,
	2.12701926493030324e-01,
	-4.53178755885004364e-02,
	1.83562338762806865e-02,
	3.59698247414342621e-03,
	-9.29574590060749106e-02,
	1.19513416467481953e-01,
	6.79067943032432651e-03,
	5.42921449553337900e-03,
	1.03535224913736262e-02,
	5.80183406095335485e-03,
	-2.41504931529476705e-02,
	-1.78172497092298904e-02,
	9.35268916496955101e-05,
	-1.77143055051121670e-02,
	-1.87573772388434840e-01,
	2.14560613968571540e-01,
	1.40574885151730847e-02,
	3.25737057077203325e-03,
	-3.96350067127882108e-02,
	9.38038635567109980e-02,
	-1.00865697098542498e-01,
	-2.87980654084887995e-02,
	1.83200192249832099e-03,
	5.48476373869262110e-02,
	8.93616016775499872e-02,
	-8.92970937896143263e-02,
	1.42330505538140195e-02,
	-1.00215191455349725e-02,
	1.27585916027790468e-02,
	2.22196668386438478e-02,
	-2.84895849386228271e-02,
	-8.65114302879320049e-03,
	6.45728830167090814e-03,
	2.27284177264630016e-02,
	-2.14539426733635352e-02,
	2.75005625403426830e-02,
	4.67194089048153585e-02,
	-3.21785760758904393e-03,
	-2.41872980674561755e-02,
	7.47583800389689385e-02,
	-8.62053662387157887e-02,
	-6.25776778378947102e-02,
	1.03142757765448990e-02,
	8.26141251850321529e-02,
	7.80803884967315592e-02,
	-1.00805156699005985e-01,
	-9.20061176840866399e-03,
	-2.70119936849379412e-03,
	1.24072246725162837e-02,
	-4.89149262348286124e-02,
	6.46915678819213330e-02,
	-7.40589754543851449e-03,
	4.95620647091155140e-03,
	1.61748553345373094e-02,
	-1.31639244900505908e-01,
	1.59212131421680697e-01,
	-2.02291404336979661e-02,
	4.42025078814087344e-03,
	-9.40988838744906582e-03,
	-2.08095446281740017e-03,
	6.57681227080676766e-04,
	1.32504506719129094e-02,
	-7.34178660904547149e-03,
	-3.38529176345991473e-02,
	6.91017423173997530e-02,
	-7.75650004520814507e-02,
	-2.60430567881156094e-02,
	1.31582619105728785e-02,
	6.22345627956099740e-02,
	-2.71008766449203006e-02,
	5.53167296327332011e-02,
	-3.05977008422328028e-02,
	2.24785153316888571e-03,
	3.83143185144835738e-02,
	-7.20719714440721249e-02,
	9.08072213964826735e-02,
	8.76234122783402063e-03,
	2.28159334841652886e-03,
	-4.58889586645554095e-03,
	7.74129259248724499e-02,
	-1.09466334414828087e-01,
	5.41920764714814377e-03,
	-5.62760778419451116e-03,
	-3.33079385968331337e-02,
	-1.27044253522610201e-01,
	1.46688101457021103e-01,
	-1.81178237991165211e-02,
	4.48011620901933322e-03,
	-2.63228375450564461e-02,
	5.71404262814582953e-03,
	-1.46518819280958262e-02,
	-1.68976668475042738e-02,
	1.24231179581025877e-02,
	2.49021678330879279e-02,
	4.34031107053995252e-02,
	-4.08947116178189768e-02,
	-4.53222536778488880e-02,
	1.56811858159741149e-02,
	1.05142126588130283e-01,
	-1.06094776717703176e-01,
	1.12986354574270684e-01,
	1.97456986883644409e-02,
	2.94540828766935844e-03,
	-3.34319924923676959e-02,
	1.70236114416373957e-01,
	-2.07181149439650603e-01,
	7.66035893284951978e-03,
	-6.89599730133525889e-03,
	3.91510626471399097e-03,
	1.28872130871178742e-01,
	-1.56182584519625106e-01,
	-5.80954516240308853e-03,
	4.63729058378125310e-05,
	2.66649307528077396e-02,
	-1.50727708093514656e-01,
	1.77798699361694457e-01,
	2.22829249304969876e-02,
	-3.24626390217279751e-03,
	-6.10312960921526246e-02,
	-8.45781876929762588e-02,
	9.79435231112839866e-02,
	-2.63891361610121382e-02,
	9.51008025156777764e-03,
	4.33261768949762347e-03,
	-3.29298701972483357e-02,
	4.97686506199196618e-02,
	-6.10240217172859645e-02,
	1.03830239776041919e-02,
	4.68028512681795822e-02,
	-1.22382951880751656e-01,
	1.53873626323832141e-01,
	2.56642499557059450e-02,
	5.86086638959799379e-03,
	7.69773337488577521e-04,
	-6.98525794044506654e-02,
	1.03638665796208229e-01,
	-4.78050736948316402e-02,
	1.55478171063753827e-02,
	8.32699554112958512e-02,
	-9.18015540068238128e-02,
	1.02494318506122492e-01,
	-2.42669310746977847e-03,
	5.53144532264485990e-03,
	-2.94686434734133251e-02,
	-1.47211150451997369e-01,
	1.64840157297695328e-01,
	2.01217939669593726e-02,
	-9.42898714940959256e-04,
	-6.84105030162983768e-02,
	-4.94620072004524833e-02,
	6.87710848754927995e-02,
	-8.87849594221822372e-03,
	6.30786336177022698e-03,
	2.86339347418513891e-02,
	6.67800635333939008e-03,
	-3.80522499901755473e-03,
	4.74428025986193616e-02,
	-1.14603666217477102e-02,
	-4.22614463204251806e-02,
	-1.60916513157456892e-01,
	1.97777854937001951e-01,
	3.21916043227916601e-02,
	-5.89843703511902647e-04,
	-3.28522498257233647e-02,
	9.38867433635764692e-02,
	-1.12753572110928679e-01,
	4.94462766876775095e-02,
	-1.51676831780357922e-02,
	-2.04302174872631505e-02,
	-8.49793944857944950e-05,
	-8.16368429878890664e-04,
	2.55074669821460896e-03,
	-6.34631200085218009e-04,
	-4.32833177424456293e-03,
	-1.17342895921862270e-01,
	1.41403918589661809e-01,
	-1.07737985031715463e-03,
	4.62501389251159488e-03,
	-1.09760671460174698e-02,
	5.81489634253770579e-02,
	-7.00864039445476261e-02,
	5.77188638029423225e-03,
	-3.29402637139906713e-03,
	1.26298610636333634e-03,
	8.24363044852087501e-02,
	-9.71883451113348396e-02,
	-1.13405759255719978e-02,
	3.12027196390495811e-04,
	2.57901019921579974e-02,
	-2.19434879661292909e-01,
	2.61207779960047348e-01,
	-1.20222807670235850e-02,
	9.99889977103083326e-03,
	-2.06450720136683655e-02,
	-2.21837988662314731e-02,
	2.98637381229465322e-02,
	-2.68504276116626542e-02,
	7.56280242754087031e-03,
	3.07526533225414532e-02,
	-1.82350894447062456e-01,
	2.17370882073588123e-01,
	-1.26429271559231528e-02,
	9.12146141235514740e-03,
	-1.32518466529369366e-02,
	-9.56586458837074272e-02,
	1.15140108302360211e-01,
	-2.04369520434420767e-03,
	4.02704208124143957e-03,
	-8.11158369401057157e-03,
	1.25848096906484592e-02,
	-1.64319954017023359e-02,
	-1.03987007652367631e-03,
	-5.32594921973386475e-04,
	-1.11618183529917417e-03,
	-1.84035218437757997e-01,
	2.18457950579252991e-01,
	2.55218817657881228e-03,
	5.51433657815669255e-03,
	-2.99314063890964294e-02,
	9.08323958518785640e-02,
	-1.06432569755475617e-01,
	-8.79307857376876445e-03,
	-7.64929678274472416e-04,
	2.53078752220133249e-02,
	8.16182441320199847e-02,
	-9.64034529617392860e-02,
	3.20396272900936707e-03,
	-3.49412845720694027e-03,
	9.94053515106616860e-03,
	2.41488474403593611e-02,
	-2.82030276330508187e-02,
	-3.54252864526555461e-03,
	2.97835303201095941e-04,
	8.82968278614173520e-03,
	-2.30386894689597679e-02,
	2.70795810757746558e-02,
	1.02250316416990696e-02,
	-1.40197029651317425e-03,
	-1.28828744502469065e-02,
	7.68963180362359083e-02,
	-8.92636429347280630e-02,
	-1.80954243203932073e-02,
	2.01218616763636574e-03,
	3.38588750558262361e-02,
	8.17892880710514025e-02,
	-9.76514333918026556e-02,
	-1.10549230508453535e-03,
	-2.52652567835387346e-03,
	1.21994405525942557e-02,
	-5.07029381624226616e-02,
	6.13938795837978002e-02,
	-4.15549734628586735e-03,
	2.88048185918082095e-03,
	-6.18958683613223719e-04,
	-1.32521734264793689e-01,
	1.58064525603612255e-01,
	-5.44425137363809073e-03,
	5.54063312298131837e-03,
	-1.38330196115942512e-02,
	-2.99349542901023759e-03,
	2.37218088619712587e-03,
	5.95496364948555016e-03,
	-1.66471521398568810e-03,
	-9.70040310031453426e-03,
	6.95033448982888824e-02,
	-8.09225763702747930e-02,
	-9.87589370644077411e-03,
	5.81880390017080053e-04,
	2.47706236837571848e-02,
	-3.58059737652848195e-02,
	4.47973532354853585e-02,
	-1.00332573621695429e-02,
	3.60657242699909423e-03,
	8.56103569313056666e-03,
	-7.39924446080151288e-02,
	8.85539687761933147e-02,
	3.85343967025064151e-04,
	2.51932579869232212e-03,
	-9.60711322010106110e-03,
	8.29885329453574072e-02,
	-1.01050651786484688e-01,
	6.25715096475348495e-03,
	-4.67120625557231137e-03,
	1.39420818638929638e-04,
	-1.25663183732204004e-01,
	1.48982177867871463e-01,
	-3.20047780175012828e-03,
	4.63749083090089255e-03,
	-1.72878226031127416e-02,
	1.04537960513608266e-02,
	-1.20468111861369957e-02,
	-5.67589078005150977e-03,
	1.33301931037093204e-03,
	8.86355498548363659e-03,
	4.35651865807129487e-02,
	-4.81625727521899291e-02,
	-1.82753559913245121e-02,
	3.73626461275307440e-03,
	3.39980861237226609e-02,
	-1.00690857941986026e-01,
	1.18880813698632459e-01,
	4.64688065852465922e-03,
	2.32045256141713650e-03,
	-2.03925699358311896e-02,
	1.71262234284645071e-01,
	-2.04777396953250201e-01,
	4.71735416514401217e-03,
	-6.95212512704418244e-03,
	1.78547150729262034e-02,
	1.30561687681085364e-01,
	-1.55304563067242163e-01,
	-1.03282439521064167e-03,
	-3.97773017940246387e-03,
	2.04203762876713837e-02,
	-1.52002672382600540e-01,
	1.79626221700467659e-01,
	7.08418536919994905e-03,
	3.04543948725608535e-03,
	-3.24090988157867341e-02,
	-8.27495563121558586e-02,
	9.87785177618925453e-02,
	-7.05189863324197797e-03,
	4.41825475999556034e-03,
	-4.70370078097197535e-03,
	-3.56913078388767097e-02,
	4.42732199185639913e-02,
	-1.64596390042343887e-02,
	5.05370186914235429e-03,
	1.37725290478229768e-02,
	-1.24861214805191661e-01,
	1.49830193622091740e-01,
	1.59380945690121810e-03,
	4.30001528960553391e-03,
	-1.53326478878356067e-02,
	-7.46225203809507548e-02,
	9.26654073034911679e-02,
	-1.88933864137270022e-02,
	7.60176094666778056e-03,
	1.69389604337072570e-02,
	-8.97174611771301422e-02,
	1.05884536916126162e-01,
	5.67966766787996857e-04,
	2.70196703282870329e-03,
	-1.58366377110069310e-02,
	-1.44902365755363194e-01,
	1.70522618402230541e-01,
	7.53257878428361619e-03,
	2.66610367550644545e-03,
	-3.31666949532510302e-02,
	-5.21365183177593811e-02,
	6.36828488455702391e-02,
	-5.76318526769040076e-03,
	3.44628896961914804e-03,
	2.32301675439765959e-03,
	2.84929501032257478e-03,
	-4.44734577596917098e-03,
	1.30164856165078009e-02,
	-3.31426914247803924e-03,
	-1.46097139964424961e-02,
	-1.63980686603911935e-01,
	1.95374099914370186e-01,
	5.57119362010554536e-03,
	4.27752882790870780e-03,
	-2.80650307218526275e-02,
	9.31453566086286877e-02,
	-1.11713351357879437e-01,
	1.30916413215518106e-02,
	-6.19509628000206730e-03,
	-5.23156964667878662e-04,
	1.25394176174922170e-04,
	-3.22242737441162642e-04,
	9.09805898832858153e-04,
	-2.46041907534548716e-04,
	-1.27497448097126814e-03,
	-1.17848532686672405e-01,
	1.40571337024929105e-01,
	-1.65968504659320211e-03,
	4.36848920315596051e-03,
	-1.46348925868144790e-02,
	5.82323875537044816e-02,
	-6.95720707789116294e-02,
	2.18954423733968770e-03,
	-2.48495180618411209e-03,
	5.70243348670963074e-03,
	8.26746606532790806e-02,
	-9.79698174323032217e-02,
	-2.90714208378842383e-03,
	-1.98390723232569257e-03,
	1.58071143880599181e-02,
	-2.19016645113544017e-01,
	2.61040351134857629e-01,
	-4.84631742735580021e-03,
	8.44941319588639501e-03,
	-2.62116858563206760e-02,
	-2.21879426644658317e-02,
	2.75730499246233905e-02,
	-8.52352485669079576e-03,
	2.92804132002783988e-03,
	7.78118610830081520e-03,
	-1.81955119625628131e-01,
	2.17001167821059077e-01,
	-4.91399798977371852e-03,
	7.25598454636054279e-03,
	-2.05775980942594014e-02,
	-9.59755915680576832e-02,
	1.14497368918273876e-01,
	-1.64765562924633368e-03,
	3.62837566485171674e-03,
	-1.16015359194122716e-02,
	1.30818621121818138e-02,
	-1.57448535000402380e-02,
	2.22324642200905364e-04,
	-5.30091493631087941e-04,
	1.19311601009822817e-03,
	-1.83963420180378423e-01,
	2.18878263455142108e-01,
	-4.88842643260227060e-04,
	6.20858530693450209e-03,
	-2.63133397667707353e-02,
	9.06610078570271871e-02,
	-1.07489175822639713e-01,
	-2.21373265520857961e-03,
	-2.41922069310228881e-03,
	1.62480157432526445e-02,
	8.12270811675106369e-02,
	-9.67113109923153358e-02,
	1.33878185558460559e-03,
	-3.01349880448001806e-03,
	1.03911313637224303e-02,
	2.41646065902139662e-02,
	-2.85798383016210092e-02,
	-1.02845902551137233e-03,
	-5.22732530543669038e-04,
	4.94974902672094819e-03,
	-2.33045211895264170e-02,
	2.74626121529051651e-02,
	2.65848564331090578e-03,
	1.21592826236408661e-04,
	-6.51364265802583352e-03,
	7.69403451701248381e-02,
	-9.08209691269137170e-02,
	-5.08193139205707217e-03,
	-1.23685449666954979e-03,
	1.78239458614886867e-02,
	8.19575941035060179e-02,
	-9.75737858212798320e-02,
	3.41425472057447783e-04,
	-2.80682233133368233e-03,
	1.14497693600143901e-02,
	-5.08682663687753181e-02,
	6.08089532258046447e-02,
	-1.80846631791313430e-03,
	2.16142865390698364e-03,
	-4.95862197246785437e-03,
	-1.32470279156632065e-01,
	1.57864853017282147e-01,
	-2.45911535975569962e-03,
	4.99490236894779231e-03,
	-1.63640060352120871e-02,
	-2.89709107706166572e-03,
	3.11208000280378816e-03,
	2.02291595305810045e-03,
	-4.44183880447013648e-04,
	-3.21324811106594305e-03,
	6.94003899465990143e-02,
	-8.20909342462878000e-02,
	-2.83232360671330013e-03,
	-1.53938716686643595e-03,
	1.40445440348787639e-02,
	-3.61658083397611574e-02,
	4.35596867909662025e-02,
	-3.48823184471084777e-03,
	2.09755431894240348e-03,
	-6.60740671302983084e-04,
	-7.41681785162160156e-02,
	8.83560026336371596e-02,
	-5.70459711433464814e-04,
	2.61436004310263738e-03,
	-9.95949407332558795e-03,
	8.34636628154600269e-02,
	-9.98313663955656683e-02,
	2.95769916592786329e-03,
	-3.55772876143914570e-03,
	7.98811376440471468e-03,
	-1.25410195260567875e-01,
	1.49279905508979627e-01,
	-1.53930343536396591e-03,
	4.51022394697318894e-03,
	-1.66857974681880634e-02,
	1.06212462861352136e-02,
	-1.24063781817206238e-02,
	-1.71764384491869147e-03,
	9.84010189935125956e-05,
	3.78477619466914116e-03,
	4.32190220486616539e-02,
	-5.04423199934649347e-02,
	-5.88383136668157655e-03,
	1.33501115469281470e-04,
	1.43990903871926181e-02,
	-1.00542030169073635e-01,
	1.19478443314898930e-01,
	7.33552651265059297e-04,
	3.13969545475736595e-03,
	-1.56730548101572317e-02,
	1.71400062274310983e-01,
	-2.04296135970377407e-01,
	2.77508701200396019e-03,
	-6.38949830890585876e-03,
	2.14049120218559562e-02,
	1.30623615220709061e-01,
	-1.55450868220107424e-01,
	5.69489275053595956e-04,
	-4.46079441323887077e-03,
	1.84060393468907929e-02,
	-1.51946732877951574e-01,
	1.80512465197778804e-01,
	1.29814956152744386e-03,
	4.67303675519750283e-03,
	-2.40693072167658430e-02,
	-8.25755418601279434e-02,
	9.85226183753971779e-02,
	-2.60161448147132887e-03,
	3.38441452619068963e-03,
	-8.89133356313464782e-03,
	-3.56545383535462404e-02,
	4.30742370580489906e-02,
	-5.17563418969905521e-03,
	2.48916262163965561e-03,
	1.29480592540136687e-03,
	-1.25279780302271659e-01,
	1.49299766885510998e-01,
	-8.93099015594852295e-04,
	4.42042203350375123e-03,
	-1.67092252703191849e-02,
	-7.50756477592639943e-02,
	9.03849847247598370e-02,
	-6.82462348478730333e-03,
	4.27512408899287030e-03,
	-1.76839158498092093e-03,
	-8.94783549210769191e-02,
	1.06385154154963329e-01,
	-2.24241043655859794e-04,
	2.99658016763440938e-03,
	-1.30285879933732980e-02,
	-1.44638521194868142e-01,
	1.71725394823652566e-01,
	1.61758153725102870e-03,
	4.33963282249823050e-03,
	-2.35407637073586189e-02,
	-5.24125347854515894e-02,
	6.27720539805089600e-02,
	-2.43975776849829004e-03,
	2.38596188505999559e-03,
	-4.25487201947816803e-03,
	2.64340358616355119e-03,
	-3.63223038774273827e-03,
	3.92172687632804697e-03,
	-1.08445764912942950e-03,
	-4.53827987247528916e-03,
	-1.64290777628944840e-01,
	1.95475217102700383e-01,
	2.93582053858532023e-04,
	5.38175423722541540e-03,
	-2.41578949977092443e-02,
	9.29255123680214901e-02,
	-1.11062644519221376e-01,
	4.45918191391222150e-03,
	-4.19530057675419725e-03,
	8.09253494632097836e-03,
	1.43119534003587317e-04,
	-2.17014077428076530e-04,
	2.99702672419767217e-04,
	-8.31985477028644922e-05,
	-3.81472646207881743e-04,
	-1.17900590943754713e-01,
	1.40442412026406699e-01,
	-1.37882466307013662e-03,
	4.25577291289972771e-03,
	-1.54447931860105126e-02,
	5.82396019128063325e-02,
	-6.94238477596344217e-02,
	1.08619047821577786e-03,
	-2.20449186144156244e-03,
	7.12556407746837723e-03,
	8.26774848070079976e-02,
	-9.82902098613200187e-02,
	-3.43745797532460763e-04,
	-2.64289513694814220e-03,
	1.25658324804582486e-02,
	-2.18983515775097171e-01,
	2.60865799242636143e-01,
	-2.97782301338302526e-03,
	7.99978579046370722e-03,
	-2.82731204894669137e-02,
	-2.22115112720589897e-02,
	2.68200681346375189e-02,
	-2.83482732399674940e-03,
	1.46630052550362111e-03,
	4.37678794006765820e-04,
	-1.81932101090826526e-01,
	2.16769841698101817e-01,
	-2.75879192653850349e-03,
	6.72040051591698325e-03,
	-2.31133436908940934e-02,
	-9.60111509393630908e-02,
	1.14377703363758132e-01,
	-1.20900029554151791e-03,
	3.48729840643361398e-03,
	-1.24716723363477045e-02,
	1.31287145336525524e-02,
	-1.56596964601391428e-02,
	2.06197125039476905e-04,
	-4.91148228332753374e-04,
	1.61274783156524883e-03,
	-1.83958787583163835e-01,
	2.18999251744843471e-01,
	-1.41012979153179083e-03,
	6.44155386927160178e-03,
	-2.51396865761323272e-02,
	9.06399482708131615e-02,
	-1.07790253781674172e-01,
	-8.89309777238465909e-05,
	-2.97039363272272447e-03,
	1.34187889382044161e-02,
	8.12004283469474203e-02,
	-9.67059354197533222e-02,
	9.51862874907523100e-04,
	-2.92640937467184351e-03,
	1.06903749041251522e-02,
	2.41563673436638732e-02,
	-2.87052724887620035e-02,
	-1.67349167658207697e-04,
	-7.53908394967762346e-04,
	3.76842984178241353e-03,
	-2.33288750167707538e-02,
	2.76679423742043386e-02,
	6.41061666955736902e-04,
	6.08380032821666012e-04,
	-4.22271479583472692e-03,
	7.69272093006257429e-02,
	-9.13456726711954353e-02,
	-1.07198930883548732e-03,
	-2.26451118951804175e-03,
	1.26763185380922884e-02,
	8.19699027525458007e-02,
	-9.75948743700152110e-02,
	6.79312656069425732e-04,
	-2.88448959503637137e-03,
	1.11240204885990172e-02,
	-5.08882323895629360e-02,
	6.06637249631371694e-02,
	-9.33034509308323283e-04,
	1.92366119927884138e-03,
	-6.23135735882127985e-03,
	-1.32462787569728319e-01,
	1.57781568946947393e-01,
	-1.66329020604036727e-03,
	4.80427448032320296e-03,
	-1.72718903805673223e-02,
	-2.87629772779802095e-03,
	3.32526935206708951e-03,
	6.35498520135850914e-04,
	-7.09154270076799961e-05,
	-1.26715159623360242e-03,
	6.93731386656507160e-02,
	-8.24425035944288537e-02,
	-4.40824368756032416e-04,
	-2.17566067904825843e-03,
	1.07693719636393712e-02,
	-3.61951457193509935e-02,
	4.32477233343906142e-02,
	-1.35814932587208855e-03,
	1.54763854190805624e-03,
	-3.52506755516032418e-03,
	-7.41854953053924376e-02,
	8.83413668065137353e-02,
	-7.03933049592883522e-04,
	2.63423030380463552e-03,
	-9.94510115481153645e-03,
	8.35143954770137670e-02,
	-9.95649503703925909e-02,
	1.54512743273554779e-03,
	-3.16207737751410776e-03,
	1.01922015358504297e-02,
	-1.25381963102700361e-01,
	1.49303417690022616e-01,
	-1.30770711225148824e-03,
	4.47691465844772714e-03,
	-1.67136572399028682e-02,
	1.06216451791692149e-02,
	-1.25671820927921018e-02,
	-4.68962555700232588e-04,
	-2.29473122440982785e-04,
	2.16919011424613825e-03,
	4.31584011438881018e-02,
	-5.10892681635042173e-02,
	-1.60610403222814324e-03,
	-1.00624943558105871e-03,
	8.46799752980484030e-03,
	-1.00533269261783367e-01,
	1.19638035205289317e-01,
	-4.56041587506626973e-04,
	3.43914203526025355e-03,
	-1.41487502162264342e-02,
	1.71416494520660101e-01,
	-2.04177102830482293e-01,
	2.05581722038738470e-03,
	-6.19504006649135210e-03,
	2.24475933716045510e-02,
	1.30623874173436016e-01,
	-1.55514859945174194e-01,
	1.07014207641191591e-03,
	-4.59155245428877548e-03,
	1.77620158500147868e-02,
	-1.51928698613510299e-01,
	1.80785861686199339e-01,
	-6.14561349585493921e-04,
	5.17645322661219789e-03,
	-2.14925247307249830e-02,
	-8.25638167256475541e-02,
	9.83887606268945558e-02,
	-1.36533987790534708e-03,
	3.07877055123874453e-03,
	-1.03447790852904068e-02,
	-3.56543578164781050e-02,
	4.26639011883287386e-02,
	-1.85326223446620020e-03,
	1.65487612921859741e-03,
	-2.83103943997310833e-03,
	-1.25327128299570156e-01,
	1.49248534896925716e-01,
	-1.18895861738413356e-03,
	4.45218985855647901e-03,
	-1.67809130262239307e-02,
	-7.51357535542054034e-02,
	8.97631740330073619e-02,
	-2.70375718274855998e-03,
	3.18456348434666744e-03,
	-7.45465301441806515e-03,
	-8.94540555609345811e-02,
	1.06482043240614555e-01,
	-6.58746733628639376e-04,
	3.12334786249560082e-03,
	-1.22812226502887530e-02,
	-1.44603869254042255e-01,
	1.72045774316489403e-01,
	-4.50292708680592865e-04,
	4.89086226714299452e-03,
	-2.06455067769838567e-02,
	-5.24457768233330263e-02,
	6.25515598699662168e-02,
	-1.15448397403851975e-03,
	2.03343001272924531e-03,
	-6.15979078222529761e-03,
	2.63716610848146755e-03,
	-3.30653872366849615e-03,
	1.23652296771741672e-03,
	-4.05246313309233920e-04,
	-1.20981083985071108e-03,
	-1.64316713192850100e-01,
	1.95597680874069230e-01,
	-1.06030466889854721e-03,
	5.70496594981760499e-03,
	-2.26862101708341091e-02,
	9.29108053071739798e-02,
	-1.10783955166146308e-01,
	2.01117730731026909e-03,
	-3.58658669249511128e-03,
	1.10292608913191165e-02,
}
cond_y := [COND_CASES * COND_M]f64{
	4.01489051110290249e-02,
	-4.03099900995434857e-01,
	2.19890588151358424e-01,
	2.17132836874989132e-01,
	-6.92184926656179367e-01,
	-2.25236667586282524e-01,
	-5.87899558708971037e-01,
	-3.27831236410048132e-01,
	7.20465742007765086e-02,
	-5.40439609125506037e-01,
	2.04359381947611329e-01,
	2.25870757595289989e-01,
	6.01524140288492298e-02,
	-2.64680030529542178e-02,
	1.22900463285717701e-01,
	2.48375500702470664e-01,
	-1.91393931163198389e-01,
	-4.32802575493492048e-01,
	3.19830060981104169e-02,
	1.57332865649127934e-01,
	-2.03759954933974835e-01,
	-2.35577141250754601e-01,
	3.32132082705096365e-01,
	-3.76555455464118505e-01,
	2.64831063886541360e-02,
	-8.14147677057553733e-03,
	-2.63210953035852457e-01,
	5.61869275020873249e-01,
	3.98474297037435365e-01,
	-4.13376664465651911e-01,
	-2.71629428475461621e-01,
	-2.02033256651957205e-01,
	-4.00869982761378474e-01,
	-3.79293881642858988e-01,
	-2.37206289856537939e-01,
	-3.67043471284022094e-01,
	-2.15469931544366639e-01,
	6.70214798333761824e-02,
	-4.92867361537315585e-01,
	3.29258259757702287e-01,
	7.41224598425539120e-03,
	-3.70350964876861477e-01,
	1.89432250076553149e-01,
	2.33392364295205185e-01,
	-6.86897410252280571e-01,
	-1.18777068977431732e-01,
	-5.76173292601280140e-01,
	-3.02710486334215934e-01,
	4.50068138495871573e-02,
	-5.58235369773621426e-01,
	2.59045165929267629e-01,
	2.50634321997852849e-01,
	6.64300065547555607e-02,
	-5.69666846158124254e-02,
	2.01624076320585321e-01,
	2.50660754055514912e-01,
	-1.65999639275118738e-01,
	-4.14001751719548117e-01,
	4.79590537308964039e-03,
	1.91001444524250746e-01,
	-1.32438810748978508e-01,
	-2.28939068505690874e-01,
	2.73997190976026095e-01,
	-3.85384127219379924e-01,
	2.24128090561915294e-02,
	9.09625019870944490e-02,
	-2.98577263316144093e-01,
	5.35538221615563859e-01,
	3.98091563496294376e-01,
	-4.49962387515041085e-01,
	-2.63522115552529690e-01,
	-1.37965039653490107e-01,
	-3.87825931626944787e-01,
	-2.72003389605779855e-01,
	-2.69341639657380227e-01,
	-4.24431447881775004e-01,
	-1.75528911177582320e-01,
	3.02239344776186851e-02,
	-4.97013736448144838e-01,
	3.05316691593671641e-01,
	2.40072130433810526e-03,
	-3.64762736082833328e-01,
	1.82462863132139108e-01,
	2.47497985015082189e-01,
	-6.78855846778208649e-01,
	-8.44835270059831889e-02,
	-5.65779229824127983e-01,
	-2.97507118882801846e-01,
	4.13025556947391390e-02,
	-5.63868823875684599e-01,
	2.72902917074884510e-01,
	2.50690307610031138e-01,
	7.18169971506359001e-02,
	-6.67661146034514708e-02,
	2.25595720030714597e-01,
	2.51679986849395187e-01,
	-1.59467610012171135e-01,
	-4.09897302273960040e-01,
	-4.62278617714359866e-03,
	2.06480517506648420e-01,
	-1.17745332448008166e-01,
	-2.28383094428938421e-01,
	2.61948894642283436e-01,
	-3.86179829033591770e-01,
	2.91952350081474167e-02,
	1.19963356005874391e-01,
	-3.06203674769927048e-01,
	5.30102969829542747e-01,
	4.00818973126849554e-01,
	-4.62200764378228879e-01,
	-2.57432448859408247e-01,
	-1.18864270920404996e-01,
	-3.86001665028160468e-01,
	-2.43778887949158030e-01,
	-2.73826636901864540e-01,
	-4.38950216718557606e-01,
	-1.65635774554918597e-01,
	1.54247269120813841e-02,
	-5.02712512327578187e-01,
	2.92555095207879368e-01,
	1.04825912020807023e-03,
	-3.63540277103307885e-01,
	1.80330050114866747e-01,
	2.52309557094731407e-01,
	-6.75795929752710856e-01,
	-7.35275331932155191e-02,
	-5.62020467537436508e-01,
	-2.96201361822745202e-01,
	4.06584342726773393e-02,
	-5.65627950042657868e-01,
	2.77136922779202877e-01,
	2.50273141370117858e-01,
	7.35688276158402710e-02,
	-7.01848100471072489e-02,
	2.33274464643202784e-01,
	2.52160311823468319e-01,
	-1.57562309058649652e-01,
	-4.08536319582537999e-01,
	-7.52105024799050188e-03,
	2.11355007568903214e-01,
	-1.13438762135517596e-01,
	-2.28401799477996909e-01,
	2.58640916463762638e-01,
	-3.86140567564421389e-01,
	3.15800297832712837e-02,
	1.28812990914333680e-01,
	-3.08498519047101016e-01,
	5.28541727186420562e-01,
	4.01770305225783664e-01,
	-4.66038919044031552e-01,
	-2.55268269906205048e-01,
	-1.12697386832265281e-01,
	-3.85891517598236311e-01,
	-2.35279480788457618e-01,
	-2.74955693940338863e-01,
	-4.43279894067650171e-01,
	-1.62787313857918559e-01,
	1.04675987147612257e-02,
	-5.04899165083255164e-01,
	2.88181764985929401e-01,
}
cond_beta := [COND_CASES * COND_N]f64{
	1.00008098022079794e+00,
	-1.99993423827098060e+00,
	4.99986540574510696e-01,
	3.00000217832131710e+00,
	-1.50002421036974276e+00,
	1.00349685316441284e+00,
	-1.99736771729247176e+00,
	4.99728613470487193e-01,
	3.00466733641514239e+00,
	-1.50143967436619641e+00,
	1.32157899731282757e+00,
	-1.75954552940630049e+00,
	4.91595401490622530e-01,
	3.49607719907656866e+00,
	-1.63089948556265307e+00,
	-8.05131232904900429e+00,
	-8.71221804012251511e+00,
	5.06425901831166647e-01,
	-1.24580118217708478e+01,
	2.29938044404704645e+00,
}
cond_rss := [COND_CASES]f64{3.29389184420655405e-11, 3.06409170115322901e-11, 3.05550910860840003e-11, 3.43764020982943043e-11}

// ============================================================================
// Conditioning: closes the gap flagged in docs/OLS_RESULTS.md
// ============================================================================

test_ols_conditioning :: proc() -> bool {
	fmt.println("Testing both paths on cond(X) = 1e2 .. 1e8...")

	// A backward-stable solver loses roughly cond(X) * eps of relative
	// accuracy. At cond = 1e8 that is ~1e-8, so the tolerance has to scale
	// with the conditioning rather than being a fixed constant. The point of
	// the test is that neither path does materially WORSE than that bound, and
	// that the Givens accumulator tracks the Householder path.
	for c in 0 ..< COND_CASES {
		cond := cond_values[c]
		xb := cond_x[c * COND_M * COND_N:][:COND_M * COND_N]
		yb := cond_y[c * COND_M:][:COND_M]
		tb := cond_beta[c * COND_N:][:COND_N]

		tol := max(1e-9, cond * 1e-13)

		// Accumulator
		as_ := make([]f64, blas.ols_accum_scratch(COND_N));defer delete(as_)
		acc: blas.Ols_Accum
		blas.ols_accum_init(&acc, COND_N, as_)
		if _, e := blas.ols_accum_rows(&acc, xb, COND_N, yb, COND_M); e != .None {
			fmt.printf("  FAILED: cond %.0e accumulate %v\n", cond, e)
			return false
		}
		ba: [COND_N]f64
		if e := blas.ols_accum_solve(&acc, ba[:]); e != .None {
			fmt.printf("  FAILED: cond %.0e accum solve %v\n", cond, e)
			return false
		}

		// Dense (destroys input, so copy)
		xd := make([]f64, COND_M * COND_N);defer delete(xd)
		yd := make([]f64, COND_M);defer delete(yd)
		copy(xd, xb)
		copy(yd, yb)
		ds := make([]f64, blas.ols_dense_scratch(COND_N));defer delete(ds)
		bd: [COND_N]f64
		_, e2 := blas.ols_solve_dense(COND_M, COND_N, xd, COND_N, yd, bd[:], ds)
		if e2 != .None {
			fmt.printf("  FAILED: cond %.0e dense solve %v\n", cond, e2)
			return false
		}

		worst_a, worst_d, worst_ad := 0.0, 0.0, 0.0
		for j in 0 ..< COND_N {
			scale := max(1.0, abs(tb[j]))
			worst_a = max(worst_a, abs(ba[j] - tb[j]) / scale)
			worst_d = max(worst_d, abs(bd[j] - tb[j]) / scale)
			worst_ad = max(worst_ad, abs(ba[j] - bd[j]) / scale)
		}
		fmt.printf(
			"    cond %.0e: accum %.2e, dense %.2e, apart %.2e (tol %.1e)\n",
			cond,
			worst_a,
			worst_d,
			worst_ad,
			tol,
		)
		if worst_a > tol || worst_d > tol {
			fmt.println("  FAILED: relative error exceeds the stability bound")
			return false
		}
		// Neither path may be materially worse than the other -- this is the
		// disproof condition from the plan.
		if worst_ad > tol {
			fmt.println("  FAILED: the two paths disagree beyond the bound")
			return false
		}
	}
	fmt.println("  PASSED")
	return true
}

// ============================================================================
// ols_apply_qt against the explicit Q it replaced
// ============================================================================

test_ols_apply_qt_vs_explicit_q :: proc() -> bool {
	fmt.println("Testing ols_apply_qt == dorgqr followed by an explicit Q^T*b...")

	// The whole design rests on being able to skip forming Q, so check the
	// shortcut against the long way round on a tall matrix.
	M :: 30
	N :: 6
	state: u64 = 999
	nf :: proc(s: ^u64) -> f64 {
		s^ = s^ * 6364136223846793005 + 1442695040888963407
		return f64((s^ >> 11) & 0x1FFFFFFFFFFFFF) / f64(0x1FFFFFFFFFFFFF) * 2.0 - 1.0
	}

	a0 := make([]f64, M * N);defer delete(a0)
	b0 := make([]f64, M);defer delete(b0)
	for i in 0 ..< M * N {a0[i] = nf(&state)}
	for i in 0 ..< M {b0[i] = nf(&state)}

	// Path 1: factor, then apply reflectors straight to b.
	a1 := make([]f64, M * N);defer delete(a1)
	copy(a1, a0)
	tau := make([]f64, N);defer delete(tau)
	work := make([]f64, M + N);defer delete(work)
	blas.dgeqrf(M, N, a1, N, tau, work)
	b1 := make([]f64, M);defer delete(b1)
	copy(b1, b0)
	blas.ols_apply_qt(M, N, a1, N, tau, b1, 1, 1, work)

	// Path 2: build the m x m Q, then multiply.
	q := make([]f64, M * M);defer delete(q)
	blas.dorgqr(M, N, a1, N, tau, q, M, work)
	b2 := make([]f64, M);defer delete(b2)
	for i in 0 ..< M {
		s := 0.0
		for j in 0 ..< M {
			s += q[j * M + i] * b0[j] // Q^T[i,j] = Q[j,i]
		}
		b2[i] = s
	}

	worst := 0.0
	for i in 0 ..< M {
		worst = max(worst, abs(b1[i] - b2[i]))
	}
	if worst > 1e-11 {
		fmt.printf("  FAILED: max |apply_qt - explicit| = %.3e\n", worst)
		return false
	}

	// apply_qt must leave the factored matrix exactly as it found it, since
	// callers keep using it as R afterwards.
	a2 := make([]f64, M * N);defer delete(a2)
	copy(a2, a0)
	blas.dgeqrf(M, N, a2, N, tau, work)
	b3 := make([]f64, M);defer delete(b3)
	copy(b3, b0)
	blas.ols_apply_qt(M, N, a2, N, tau, b3, 1, 1, work)
	for i in 0 ..< M * N {
		if a1[i] != a2[i] {
			fmt.printf("  FAILED: apply_qt did not restore a[%d]\n", i)
			return false
		}
	}

	// Q^T must preserve the norm.
	n0, n1 := 0.0, 0.0
	for i in 0 ..< M {
		n0 += b0[i] * b0[i]
		n1 += b1[i] * b1[i]
	}
	if abs(math.sqrt_f64(n0) - math.sqrt_f64(n1)) > 1e-11 {
		fmt.printf("  FAILED: norm not preserved, %.17e vs %.17e\n", n0, n1)
		return false
	}

	fmt.printf("  PASSED (max diff %.2e)\n", worst)
	return true
}

// ============================================================================
// Degenerate but legal shapes
// ============================================================================

test_ols_degenerate_shapes :: proc() -> bool {
	fmt.println("Testing degenerate shapes: n=1, exact fit, m==n, duplicates...")

	// n = 1: fit y = b*x through the origin.
	{
		s := make([]f64, blas.ols_accum_scratch(1));defer delete(s)
		acc: blas.Ols_Accum
		blas.ols_accum_init(&acc, 1, s)
		x := []f64{1, 2, 3, 4}
		y := []f64{2, 4, 6, 8} // exactly 2x
		blas.ols_accum_rows(&acc, x, 1, y, 4)
		b: [1]f64
		if e := blas.ols_accum_solve(&acc, b[:]); e != .None {
			fmt.printf("  FAILED: n=1 solve %v\n", e)
			return false
		}
		if abs(b[0] - 2.0) > 1e-12 {
			fmt.printf("  FAILED: n=1 beta %.17e want 2\n", b[0])
			return false
		}
		// An exact fit leaves only rounding residue on the triangle corner.
		// That is squared to form RSS, so the bound is relative to ||y||^2
		// rather than exactly zero.
		ynorm2 := 0.0
		for v in y {ynorm2 += v * v}
		if blas.ols_accum_rss(&acc) > 1e-25 * ynorm2 {
			fmt.printf("  FAILED: exact fit rss = %.3e, ||y||^2 = %.3e\n", blas.ols_accum_rss(&acc), ynorm2)
			return false
		}
	}

	// m == n: square, exactly determined, residual zero.
	{
		N :: 3
		x := []f64{1, 0, 0, 0, 1, 0, 0, 0, 1}
		y := []f64{5, -3, 7}
		s := make([]f64, blas.ols_accum_scratch(N));defer delete(s)
		acc: blas.Ols_Accum
		blas.ols_accum_init(&acc, N, s)
		blas.ols_accum_rows(&acc, x, N, y, N)
		b: [N]f64
		if e := blas.ols_accum_solve(&acc, b[:]); e != .None {
			fmt.printf("  FAILED: m==n solve %v\n", e)
			return false
		}
		for j in 0 ..< N {
			if abs(b[j] - y[j]) > 1e-12 {
				fmt.printf("  FAILED: identity beta[%d] %.17e want %.17e\n", j, b[j], y[j])
				return false
			}
		}
		ynorm2 := 0.0
		for v in y {ynorm2 += v * v}
		if blas.ols_accum_rss(&acc) > 1e-25 * ynorm2 {
			fmt.printf("  FAILED: square rss = %.3e, ||y||^2 = %.3e\n", blas.ols_accum_rss(&acc), ynorm2)
			return false
		}
	}

	// Repeating every observation must not move the coefficients, only double
	// the residual sum. Exercises the accumulator's scale invariance.
	{
		s1 := make([]f64, blas.ols_accum_scratch(REAL_N));defer delete(s1)
		s2 := make([]f64, blas.ols_accum_scratch(REAL_N));defer delete(s2)
		one, twice: blas.Ols_Accum
		blas.ols_accum_init(&one, REAL_N, s1)
		blas.ols_accum_init(&twice, REAL_N, s2)
		blas.ols_accum_rows(&one, real_x[:], REAL_N, real_y[:], REAL_M)
		blas.ols_accum_rows(&twice, real_x[:], REAL_N, real_y[:], REAL_M)
		blas.ols_accum_rows(&twice, real_x[:], REAL_N, real_y[:], REAL_M)

		b1, b2: [REAL_N]f64
		blas.ols_accum_solve(&one, b1[:])
		if e := blas.ols_accum_solve(&twice, b2[:]); e != .None {
			fmt.printf("  FAILED: duplicated solve %v\n", e)
			return false
		}
		for j in 0 ..< REAL_N {
			if abs(b1[j] - b2[j]) > 1e-11 {
				fmt.printf("  FAILED: duplication moved beta[%d]: %.17e vs %.17e\n", j, b1[j], b2[j])
				return false
			}
		}
		if abs(blas.ols_accum_rss(&twice) - 2.0 * blas.ols_accum_rss(&one)) > 1e-9 {
			fmt.printf(
				"  FAILED: duplicated rss %.17e want %.17e\n",
				blas.ols_accum_rss(&twice),
				2.0 * blas.ols_accum_rss(&one),
			)
			return false
		}
		if twice.nrows != 2 * REAL_M {
			fmt.printf("  FAILED: duplicated nrows %d\n", twice.nrows)
			return false
		}
	}

	// Rows arriving in reverse order must give the same fit.
	{
		s := make([]f64, blas.ols_accum_scratch(REAL_N));defer delete(s)
		rev: blas.Ols_Accum
		blas.ols_accum_init(&rev, REAL_N, s)
		row: [REAL_N]f64
		for i := REAL_M - 1; i >= 0; i -= 1 {
			for j in 0 ..< REAL_N {row[j] = real_x[i * REAL_N + j]}
			blas.ols_accum_rows(&rev, row[:], REAL_N, []f64{real_y[i]}, 1)
		}
		b: [REAL_N]f64
		blas.ols_accum_solve(&rev, b[:])
		for j in 0 ..< REAL_N {
			if abs(b[j] - truth_beta[j]) > 1e-10 {
				fmt.printf("  FAILED: reversed beta[%d] %.17e want %.17e\n", j, b[j], truth_beta[j])
				return false
			}
		}
	}

	fmt.println("  PASSED")
	return true
}

// ============================================================================
// The library must never allocate
// ============================================================================

// Every entry point exercised with context.allocator AND context.temp_allocator
// set to mem.panic_allocator, and every buffer a stack array. Any hidden
// allocation anywhere in the call tree aborts the process rather than quietly
// working because the default allocator happened to be available.
//
// This is the enforceable form of the guarantee: callers supply all memory,
// so the library is usable from an arena, a frame allocator, or a fixed
// budget with no heap at all.
test_ols_no_allocation :: proc() -> bool {
	fmt.println("Testing the whole OLS API under mem.panic_allocator...")

	saved_alloc := context.allocator
	saved_temp := context.temp_allocator
	defer {
		context.allocator = saved_alloc
		context.temp_allocator = saved_temp
	}
	context.allocator = mem.panic_allocator()
	context.temp_allocator = mem.panic_allocator()

	N :: REAL_N

	// Stack scratch, sized by the library's own size procedures. These are
	// compile-time constants here only because N is; ols_accum_scratch(N) is
	// the general form.
	accum_buf: [(N + 1) * (N + 1) + (N + 1)]f64
	sub_buf: [(N + 1) * (N + 1) + (N + 1)]f64
	other_buf: [(N + 1) * (N + 1) + (N + 1)]f64
	dense_buf: [2 * N + 1]f64
	beta: [N]f64

	if len(accum_buf) < blas.ols_accum_scratch(N) || len(dense_buf) < blas.ols_dense_scratch(N) {
		fmt.println("  FAILED: stack scratch smaller than the size procedures require")
		return false
	}

	// --- accumulator path
	acc: blas.Ols_Accum
	if e := blas.ols_accum_init(&acc, N, accum_buf[:]); e != .None {
		fmt.printf("  FAILED: init %v\n", e)
		return false
	}
	if _, e := blas.ols_accum_rows(&acc, real_x[:], N, real_y[:], REAL_M); e != .None {
		fmt.printf("  FAILED: accumulate %v\n", e)
		return false
	}
	if e := blas.ols_accum_solve(&acc, beta[:]); e != .None {
		fmt.printf("  FAILED: solve %v\n", e)
		return false
	}
	for j in 0 ..< N {
		if abs(beta[j] - truth_beta[j]) > 1e-12 {
			fmt.printf("  FAILED: beta[%d] wrong under panic allocator\n", j)
			return false
		}
	}
	_ = blas.ols_accum_rss(&acc)
	_ = blas.ols_flops_per_row(N)
	_ = blas.ols_residual_norm(REAL_M, N, real_x[:], N, beta[:], real_y[:])

	// --- select
	sub: blas.Ols_Accum
	keep := [3]int{0, 2, 4}
	if e := blas.ols_accum_init(&sub, 3, sub_buf[:]); e != .None {
		fmt.printf("  FAILED: sub init %v\n", e)
		return false
	}
	if e := blas.ols_accum_select(&sub, &acc, keep[:]); e != .None {
		fmt.printf("  FAILED: select %v\n", e)
		return false
	}
	if e := blas.ols_accum_solve(&sub, beta[:3]); e != .None {
		fmt.printf("  FAILED: sub solve %v\n", e)
		return false
	}

	// --- merge
	other: blas.Ols_Accum
	blas.ols_accum_init(&other, N, other_buf[:])
	blas.ols_accum_rows(&other, real_x[:], N, real_y[:], REAL_M)
	if e := blas.ols_accum_merge(&other, &acc); e != .None {
		fmt.printf("  FAILED: merge %v\n", e)
		return false
	}
	blas.ols_accum_reset(&other)

	// --- dense path, on stack copies of the input it destroys
	xd: [REAL_M * N]f64 = real_x
	yd: [REAL_M]f64 = real_y
	rss, e2 := blas.ols_solve_dense(REAL_M, N, xd[:], N, yd[:], beta[:], dense_buf[:])
	if e2 != .None {
		fmt.printf("  FAILED: dense %v\n", e2)
		return false
	}
	if abs(rss - truth_rss) > 1e-12 {
		fmt.printf("  FAILED: dense rss under panic allocator\n")
		return false
	}

	// --- the underlying BLAS/QR entry points the solvers rest on
	tau: [N]f64
	work: [REAL_M + N]f64
	xq: [REAL_M * N]f64 = real_x
	bq: [REAL_M]f64 = real_y
	blas.dgeqrf(REAL_M, N, xq[:], N, tau[:], work[:])
	blas.ols_apply_qt(REAL_M, N, xq[:], N, tau[:], bq[:], 1, 1, work[:])
	blas.dtrsv(.Upper, .No_Trans, .Non_Unit, N, xq[:], N, bq[:], 1)
	_ = blas.dnrm2(REAL_M, bq[:], 1)
	_ = blas.ddot(N, bq[:], 1, beta[:], 1)

	fmt.println("  PASSED (no allocation reached the allocator)")
	return true
}

// ============================================================================
// ols_accum_rows_gather: the one-pass path for changes the triangle cannot serve
// ============================================================================

test_ols_gather_equals_plain :: proc() -> bool {
	fmt.println("Testing gather with identity cols == ols_accum_rows...")

	// With every column selected in order and nothing dropped, the general
	// path must reproduce the specialised one bit for bit: same rows, same
	// order, same rotations.
	a_buf := make([]f64, blas.ols_accum_scratch(REAL_N));defer delete(a_buf)
	b_buf := make([]f64, blas.ols_accum_scratch(REAL_N));defer delete(b_buf)
	plain, gath: blas.Ols_Accum
	blas.ols_accum_init(&plain, REAL_N, a_buf)
	blas.ols_accum_init(&gath, REAL_N, b_buf)

	blas.ols_accum_rows(&plain, real_x[:], REAL_N, real_y[:], REAL_M)

	cols := [REAL_N]int{0, 1, 2, 3, 4}
	got, e := blas.ols_accum_rows_gather(
		&gath, real_x[:], REAL_N, real_y[:], 0, REAL_M, cols[:], nil,
	)
	if e != .None || got != REAL_M {
		fmt.printf("  FAILED: gather returned %d, %v\n", got, e)
		return false
	}
	for i in 0 ..< len(plain.tri) {
		if plain.tri[i] != gath.tri[i] {
			fmt.printf("  FAILED: tri[%d] %.17e vs %.17e\n", i, plain.tri[i], gath.tri[i])
			return false
		}
	}
	if plain.nrows != gath.nrows {
		fmt.println("  FAILED: nrows differ")
		return false
	}
	fmt.println("  PASSED (bit-identical)")
	return true
}

test_ols_gather_column_subset :: proc() -> bool {
	fmt.println("Testing gather column subsets against numpy...")

	// Every subset, built by a data pass rather than by select. Both routes
	// must land on the same numbers the ground truth has.
	for truth in SUBSET_TRUTH {
		keep: [REAL_N]int
		k := 0
		for i in 0 ..< REAL_N {
			if truth.mask & (1 << uint(i)) != 0 {keep[k] = i;k += 1}
		}
		buf := make([]f64, blas.ols_accum_scratch(k));defer delete(buf)
		acc: blas.Ols_Accum
		blas.ols_accum_init(&acc, k, buf)
		if _, e := blas.ols_accum_rows_gather(
			&acc, real_x[:], REAL_N, real_y[:], 0, REAL_M, keep[:k], nil,
		); e != .None {
			fmt.printf("  FAILED: mask %d -> %v\n", truth.mask, e)
			return false
		}
		beta: [REAL_N]f64
		if e := blas.ols_accum_solve(&acc, beta[:k]); e != .None {
			fmt.printf("  FAILED: mask %d solve %v\n", truth.mask, e)
			return false
		}
		for j in 0 ..< k {
			if abs(beta[j] - truth.beta[j]) > 1e-10 {
				fmt.printf("  FAILED: mask %d beta[%d] %.17e want %.17e\n",
					truth.mask, j, beta[j], truth.beta[j])
				return false
			}
		}
		if abs(blas.ols_accum_rss(&acc) - truth.rss) > 1e-9 * max(1.0, truth.rss) {
			fmt.printf("  FAILED: mask %d rss\n", truth.mask)
			return false
		}
	}
	fmt.println("  PASSED")
	return true
}

test_ols_gather_row_exclusion :: proc() -> bool {
	fmt.println("Testing gather row exclusion and chunking...")

	drop := [?]int{0, 3, 4, 11, 19}

	// Reference: physically build the table without those rows.
	kept := REAL_M - len(drop)
	rx := make([]f64, kept * REAL_N);defer delete(rx)
	ry := make([]f64, kept);defer delete(ry)
	w := 0
	for i in 0 ..< REAL_M {
		skip := false
		for d in drop {
			if d == i {skip = true;break}
		}
		if skip {continue}
		for j in 0 ..< REAL_N {rx[w * REAL_N + j] = real_x[i * REAL_N + j]}
		ry[w] = real_y[i]
		w += 1
	}
	ref_buf := make([]f64, blas.ols_accum_scratch(REAL_N));defer delete(ref_buf)
	ref: blas.Ols_Accum
	blas.ols_accum_init(&ref, REAL_N, ref_buf)
	blas.ols_accum_rows(&ref, rx, REAL_N, ry, kept)
	ref_beta: [REAL_N]f64
	if e := blas.ols_accum_solve(&ref, ref_beta[:]); e != .None {
		fmt.printf("  FAILED: reference solve %v\n", e)
		return false
	}

	cols := [REAL_N]int{0, 1, 2, 3, 4}

	// Same thing via exclusion, at several chunk sizes. Absolute row indices
	// mean drop_rows is reused unchanged across chunks.
	for chunk in ([]int{1, 3, 7, REAL_M}) {
		buf := make([]f64, blas.ols_accum_scratch(REAL_N));defer delete(buf)
		acc: blas.Ols_Accum
		blas.ols_accum_init(&acc, REAL_N, buf)

		pos := 0
		for pos < REAL_M {
			take := min(chunk, REAL_M - pos)
			if _, e := blas.ols_accum_rows_gather(
				&acc, real_x[:], REAL_N, real_y[:], pos, take, cols[:], drop[:],
			); e != .None {
				fmt.printf("  FAILED: chunk %d -> %v\n", chunk, e)
				return false
			}
			pos += take
		}
		if acc.nrows != kept {
			fmt.printf("  FAILED: chunk %d absorbed %d want %d\n", chunk, acc.nrows, kept)
			return false
		}
		// Excluding rows mid-stream must give exactly the same rotations as
		// never having presented them.
		for i in 0 ..< len(ref.tri) {
			if ref.tri[i] != acc.tri[i] {
				fmt.printf("  FAILED: chunk %d tri[%d] %.17e vs %.17e\n",
					chunk, i, ref.tri[i], acc.tri[i])
				return false
			}
		}
	}
	fmt.println("  PASSED (bit-identical across chunk sizes 1/3/7/all)")
	return true
}

test_ols_gather_boundaries :: proc() -> bool {
	fmt.println("Testing gather boundary policies...")

	buf := make([]f64, blas.ols_accum_scratch(3));defer delete(buf)
	acc: blas.Ols_Accum
	blas.ols_accum_init(&acc, 3, buf)
	ok := [3]int{0, 2, 4}

	// cols length must equal acc.n
	if _, e := blas.ols_accum_rows_gather(
		&acc, real_x[:], REAL_N, real_y[:], 0, 2, []int{0, 1}, nil,
	); e != .Invalid_Dimension {
		fmt.printf("  FAILED: short cols -> %v\n", e)
		return false
	}
	// out-of-range and duplicate columns
	if _, e := blas.ols_accum_rows_gather(
		&acc, real_x[:], REAL_N, real_y[:], 0, 2, []int{0, 1, REAL_N}, nil,
	); e != .Invalid_Dimension {
		fmt.printf("  FAILED: oob col -> %v\n", e)
		return false
	}
	if _, e := blas.ols_accum_rows_gather(
		&acc, real_x[:], REAL_N, real_y[:], 0, 2, []int{0, 2, 2}, nil,
	); e != .Invalid_Dimension {
		fmt.printf("  FAILED: duplicate col -> %v\n", e)
		return false
	}
	// negative first, negative count
	if _, e := blas.ols_accum_rows_gather(
		&acc, real_x[:], REAL_N, real_y[:], -1, 2, ok[:], nil,
	); e != .Invalid_Dimension {
		fmt.printf("  FAILED: negative first -> %v\n", e)
		return false
	}
	if _, e := blas.ols_accum_rows_gather(
		&acc, real_x[:], REAL_N, real_y[:], 0, -1, ok[:], nil,
	); e != .Invalid_Dimension {
		fmt.printf("  FAILED: negative count -> %v\n", e)
		return false
	}
	// reading past the end of the table
	if _, e := blas.ols_accum_rows_gather(
		&acc, real_x[:], REAL_N, real_y[:], 0, REAL_M + 1, ok[:], nil,
	); e != .Invalid_Dimension {
		fmt.printf("  FAILED: overrun -> %v\n", e)
		return false
	}
	// count == 0 is a no-op
	if got, e := blas.ols_accum_rows_gather(
		&acc, real_x[:], REAL_N, real_y[:], 0, 0, ok[:], nil,
	); e != .None || got != 0 || acc.nrows != 0 {
		fmt.printf("  FAILED: count=0 -> %d %v\n", got, e)
		return false
	}
	// an unsorted exclusion list is rejected, not silently misapplied
	if _, e := blas.ols_accum_rows_gather(
		&acc, real_x[:], REAL_N, real_y[:], 0, REAL_M, ok[:], []int{5, 5},
	); e != .Invalid_Dimension {
		fmt.printf("  FAILED: duplicate drop -> %v\n", e)
		return false
	}
	blas.ols_accum_reset(&acc)
	if _, e := blas.ols_accum_rows_gather(
		&acc, real_x[:], REAL_N, real_y[:], 0, REAL_M, ok[:], []int{9, 2},
	); e != .Invalid_Dimension {
		fmt.printf("  FAILED: unsorted drop -> %v\n", e)
		return false
	}
	// a NaN aborts without poisoning the triangle
	{
		bad := real_x
		blas.ols_accum_reset(&acc)
		bad[6 * REAL_N + 4] = math.nan_f64() // column 4 is selected by `ok`
		got, e := blas.ols_accum_rows_gather(
			&acc, bad[:], REAL_N, real_y[:], 0, REAL_M, ok[:], nil,
		)
		if e != .Non_Finite_Input || got != 6 || acc.nrows != 6 {
			fmt.printf("  FAILED: NaN -> got %d nrows %d %v\n", got, acc.nrows, e)
			return false
		}
		// A NaN in a column that is NOT selected must be invisible.
		blas.ols_accum_reset(&acc)
		clean := real_x
		clean[6 * REAL_N + 1] = math.nan_f64() // column 1 is not in `ok`
		if got2, e2 := blas.ols_accum_rows_gather(
			&acc, clean[:], REAL_N, real_y[:], 0, REAL_M, ok[:], nil,
		); e2 != .None || got2 != REAL_M {
			fmt.printf("  FAILED: unselected NaN leaked -> %d %v\n", got2, e2)
			return false
		}
	}
	fmt.println("  PASSED")
	return true
}

// ============================================================================
// ols_accum_drop_cols: the Givens route to a sub-model
// ============================================================================

test_ols_drop_cols :: proc() -> bool {
	fmt.println("Testing ols_accum_drop_cols against select and numpy...")

	full_scratch := make([]f64, blas.ols_accum_scratch(REAL_N));defer delete(full_scratch)
	full: blas.Ols_Accum
	if !build_full(full_scratch, &full) {
		fmt.println("  FAILED: superset")
		return false
	}
	dscratch := make([]f64, blas.ols_accum_drop_scratch(REAL_N));defer delete(dscratch)

	// Every subset, reached by dropping the complement. Must match the same
	// numpy ground truth that select is checked against.
	worst := 0.0
	for truth in SUBSET_TRUTH {
		keep: [REAL_N]int
		drop: [REAL_N]int
		k, d := 0, 0
		for i in 0 ..< REAL_N {
			if truth.mask & (1 << uint(i)) != 0 {
				keep[k] = i
				k += 1
			} else {
				drop[d] = i
				d += 1
			}
		}
		sub_scratch := make([]f64, blas.ols_accum_scratch(k));defer delete(sub_scratch)
		sub: blas.Ols_Accum
		blas.ols_accum_init(&sub, k, sub_scratch)
		if e := blas.ols_accum_drop_cols(&sub, &full, drop[:d], dscratch); e != .None {
			fmt.printf("  FAILED: mask %d -> %v\n", truth.mask, e)
			return false
		}
		if sub.nrows != REAL_M {
			fmt.printf("  FAILED: mask %d nrows %d\n", truth.mask, sub.nrows)
			return false
		}
		beta: [REAL_N]f64
		if e := blas.ols_accum_solve(&sub, beta[:k]); e != .None {
			fmt.printf("  FAILED: mask %d solve %v\n", truth.mask, e)
			return false
		}
		for j in 0 ..< k {
			worst = max(worst, abs(beta[j] - truth.beta[j]))
			if abs(beta[j] - truth.beta[j]) > 1e-10 {
				fmt.printf("  FAILED: mask %d beta[%d] %.17e want %.17e\n",
					truth.mask, j, beta[j], truth.beta[j])
				return false
			}
		}
		// RSS drives model ranking, so it must survive the rotation too.
		if abs(blas.ols_accum_rss(&sub) - truth.rss) > 1e-9 * max(1.0, truth.rss) {
			fmt.printf("  FAILED: mask %d rss\n", truth.mask)
			return false
		}
	}
	fmt.printf("    all %d subsets vs numpy, worst beta error %.2e\n", len(SUBSET_TRUTH), worst)

	// Dropping nothing is a copy.
	{
		same_scratch := make([]f64, blas.ols_accum_scratch(REAL_N));defer delete(same_scratch)
		same: blas.Ols_Accum
		blas.ols_accum_init(&same, REAL_N, same_scratch)
		if e := blas.ols_accum_drop_cols(&same, &full, []int{}, dscratch); e != .None {
			fmt.printf("  FAILED: empty drop -> %v\n", e)
			return false
		}
		for i in 0 ..< len(full.tri) {
			if same.tri[i] != full.tri[i] {
				fmt.println("  FAILED: empty drop was not a copy")
				return false
			}
		}
	}

	// Drop order must not matter, and the caller's slice must come back intact.
	{
		a_scratch := make([]f64, blas.ols_accum_scratch(2));defer delete(a_scratch)
		b_scratch := make([]f64, blas.ols_accum_scratch(2));defer delete(b_scratch)
		A, B: blas.Ols_Accum
		blas.ols_accum_init(&A, 2, a_scratch)
		blas.ols_accum_init(&B, 2, b_scratch)
		d1 := [3]int{1, 3, 4}
		d2 := [3]int{4, 1, 3}
		blas.ols_accum_drop_cols(&A, &full, d1[:], dscratch)
		blas.ols_accum_drop_cols(&B, &full, d2[:], dscratch)
		for i in 0 ..< len(A.tri) {
			if A.tri[i] != B.tri[i] {
				fmt.println("  FAILED: drop order changed the result")
				return false
			}
		}
		orig := [3]int{4, 1, 3}
		for i in 0 ..< 3 {
			if d2[i] != orig[i] {
				fmt.println("  FAILED: the caller's drop slice was reordered")
				return false
			}
		}
	}

	// Boundaries.
	{
		s2 := make([]f64, blas.ols_accum_scratch(3));defer delete(s2)
		sub: blas.Ols_Accum
		blas.ols_accum_init(&sub, 3, s2)
		if e := blas.ols_accum_drop_cols(&sub, &full, []int{0}, dscratch);
		   e != .Invalid_Dimension {
			fmt.printf("  FAILED: dst.n mismatch -> %v\n", e)
			return false
		}
		if e := blas.ols_accum_drop_cols(&sub, &full, []int{0, 1, REAL_N}, dscratch);
		   e != .Invalid_Dimension {
			fmt.printf("  FAILED: oob drop -> %v\n", e)
			return false
		}
		if e := blas.ols_accum_drop_cols(&sub, &full, []int{0, 2, 2}, dscratch);
		   e != .Invalid_Dimension {
			fmt.printf("  FAILED: duplicate drop -> %v\n", e)
			return false
		}
		small := make([]f64, blas.ols_accum_drop_scratch(REAL_N) - 1);defer delete(small)
		if e := blas.ols_accum_drop_cols(&sub, &full, []int{0, 1}, small);
		   e != .Scratch_Too_Small {
			fmt.printf("  FAILED: short scratch -> %v\n", e)
			return false
		}
		if e := blas.ols_accum_drop_cols(&full, &full, []int{0}, dscratch);
		   e != .Invalid_Dimension {
			fmt.printf("  FAILED: aliased -> %v\n", e)
			return false
		}
	}

	fmt.println("  PASSED")
	return true
}
