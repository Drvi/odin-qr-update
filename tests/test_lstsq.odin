package tests

// Least-squares tests. Verifies the done-criteria in docs/OLS_PLAN.md section 6.
//
// Ground truth is numpy.linalg.lstsq (LAPACK) on the repository's own real
// data, examples/X_initial.csv + examples/y_initial.csv. Generated, do not
// hand-edit the arrays.

import "core:fmt"
import "core:math"
import "../src/blas"

REAL_M :: 20
REAL_N :: 5
real_x := [REAL_M * REAL_N]f64{
	1.00000000000000000e+00, -4.00079924841446211e+00, 1.42916902884611202e-01, 1.68062838351729704e+00, -3.72597774205915111e-01,
	1.00000000000000000e+00, -1.47808094201990398e-01, 3.09638167004459008e-01, 2.07924501327639710e+00, 1.71548961370036390e+00,
	1.00000000000000000e+00, -4.61559432131522285e-01, 1.07639596616179301e+00, 3.25561907879795220e-01, -6.22003549342410866e-01,
	1.00000000000000000e+00, 5.83551284779582513e-01, -9.79022470960622448e-01, -8.71049005645682245e-02, -1.84157356019706797e+00,
	1.00000000000000000e+00, -1.14717371724153505e+00, -9.16182950306155375e-01, 1.78016082209606297e+00, 8.69726795988284063e-01,
	1.00000000000000000e+00, -8.34612897760916850e-01, -6.97934929397660220e-01, -1.13086393209210101e+00, -9.15905697967967836e-01,
	1.00000000000000000e+00, -1.31296522083111911e+00, 2.90935553106442590e-01, 6.46657923422048531e-01, -4.16676339991469913e-01,
	1.00000000000000000e+00, 9.15137914271532571e-01, 7.24520337394183866e-02, 1.97997973486251505e-01, -2.21600682016919709e-01,
	1.00000000000000000e+00, -1.39958568988279097e+00, 3.38139328126317806e-01, 1.82913555911509096e-01, -1.32353425817789594e+00,
	1.00000000000000000e+00, 2.31443272368586000e-01, 6.45898802068618161e-01, -4.74964814719745984e-02, 1.39157303595394399e+00,
	1.00000000000000000e+00, -8.11616749397245596e-01, -6.89368581601534358e-02, 1.44147726876926008e+00, -2.84284484819326277e-01,
	1.00000000000000000e+00, -6.83425904419321495e-01, 3.13553528418981875e-01, 9.97117504045069991e-02, 1.06796049149547301e+00,
	1.00000000000000000e+00, 1.17120066570115103e-01, 1.21199288075189093e-01, -7.49036347902747979e-01, -1.63730207027157904e+00,
	1.00000000000000000e+00, -1.21538471112442403e+00, 5.03077238991761599e-01, -6.09552474308424963e-01, -7.28530147983039833e-01,
	1.00000000000000000e+00, -6.72866349688102883e-01, -1.05492895104022399e+00, 1.02624886457435394e+00, 1.72629553683881598e+00,
	1.00000000000000000e+00, -3.81756475447123822e-01, 1.54923804002511800e+00, -6.52713435645743356e-01, 5.73855166621199841e-01,
	1.00000000000000000e+00, -2.50113821411634997e-02, 2.48953217857787013e-01, -1.52847411566167102e+00, -1.77370473595421602e-01,
	1.00000000000000000e+00, 6.06892365640902431e-01, -2.34517220807593713e-01, 1.26676371362724205e+00, -7.12388783246402757e-01,
	1.00000000000000000e+00, 4.38454049997726414e-01, -1.33648547770307702e+00, 1.33041280600596890e-01, -8.91586451852176332e-01,
	1.00000000000000000e+00, 4.68819478641890677e-02, -6.96696347658356641e-01, -1.90807701350805187e-01, 1.84553351864431092e+00,
}
real_y := [REAL_M]f64{
	7.38166132922408558e+00,
	4.44427313468801799e+00,
	1.56247373506068810e+00,
	-1.95462184368837799e+00,
	3.95789963789385402e+00,
	1.55676355452626702e+00,
	3.46708321989550683e+00,
	1.15123774041841309e+00,
	2.83466786611645194e+00,
	4.09370034162836305e+00,
	1.53659473663147605e+00,
	4.52170403047698244e+00,
	-1.27278662691589911e-01,
	3.54985983667347504e+00,
	4.48083538157834127e+00,
	4.64608433471740856e+00,
	1.90699560844894389e+00,
	-3.76229817676959177e-01,
	-5.24565338286749783e-01,
	4.60869781852247407e+00,
}
truth_beta := [REAL_N]f64{
	2.04043827528267974e+00,
	-1.48909548321255003e+00,
	5.35156175896667641e-01,
	-2.79793128956932591e-01,
	1.43757594462826233e+00,
}
truth_rss :: 3.01693301371012623e+00

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
