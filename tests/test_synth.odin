package tests

// Tests for the synthetic data generator.
//
// The load-bearing checks are statistical: that the data actually carries the
// correlation, variance and noise level that were asked for. Without those the
// generator would happily produce plausible-looking numbers that silently do
// not match the spec, and every test built on top of it would be meaningless.

import "core:fmt"
import "core:math"
import "core:mem"
import "../src/blas"
import "../src/synth"

// ============================================================================
// Cholesky
// ============================================================================

test_dpotrf :: proc() -> bool {
	fmt.println("Testing dpotrf...")

	// A known SPD matrix; check L*L^T reproduces it.
	N :: 4
	orig := [N * N]f64 {
		4, 2, 1, 0.5,
		2, 5, 1.5, 1,
		1, 1.5, 3, 0.25,
		0.5, 1, 0.25, 2,
	}
	for uplo in ([]blas.Uplo{.Lower, .Upper}) {
		a := orig
		if info := blas.dpotrf(uplo, N, a[:], N); info != 0 {
			fmt.printf("  FAILED: %v reported info %d on an SPD matrix\n", uplo, info)
			return false
		}
		for i in 0 ..< N {
			for j in 0 ..< N {
				s := 0.0
				if uplo == .Lower {
					// (L L^T)[i,j] = sum_p L[i,p] L[j,p], p <= min(i,j)
					for p in 0 ..= min(i, j) {
						s += a[i * N + p] * a[j * N + p]
					}
				} else {
					// (U^T U)[i,j] = sum_p U[p,i] U[p,j]
					for p in 0 ..= min(i, j) {
						s += a[p * N + i] * a[p * N + j]
					}
				}
				if abs(s - orig[i * N + j]) > 1e-12 {
					fmt.printf(
						"  FAILED: %v reconstruct [%d,%d] = %.17e want %.17e\n",
						uplo, i, j, s, orig[i * N + j],
					)
					return false
				}
			}
		}
	}

	// Not positive definite: must report which leading minor failed, not
	// return a plausible factor.
	{
		bad := [4]f64{1, 2, 2, 1} // eigenvalues 3 and -1
		if info := blas.dpotrf(.Lower, 2, bad[:], 2); info != 2 {
			fmt.printf("  FAILED: indefinite matrix reported info %d, want 2\n", info)
			return false
		}
	}
	{
		bad := [4]f64{-1, 0, 0, 1} // fails immediately
		if info := blas.dpotrf(.Lower, 2, bad[:], 2); info != 1 {
			fmt.printf("  FAILED: negative pivot reported info %d, want 1\n", info)
			return false
		}
	}
	// A singular (semi-definite) matrix must also be rejected: a zero pivot
	// would divide by zero.
	{
		bad := [4]f64{1, 1, 1, 1}
		if info := blas.dpotrf(.Lower, 2, bad[:], 2); info != 2 {
			fmt.printf("  FAILED: singular matrix reported info %d, want 2\n", info)
			return false
		}
	}
	fmt.println("  PASSED")
	return true
}

// ============================================================================
// Generator: does the data carry the requested statistics?
// ============================================================================

test_synth_statistics :: proc() -> bool {
	fmt.println("Testing generated data matches the requested statistics...")

	K :: 3
	M :: 200_000
	// Standard error of a sample correlation at this M is ~1/sqrt(M) = 0.0022,
	// and of a variance ~sqrt(2/M) = 0.0032 relative, so these tolerances are
	// several sigma wide and will not flake.
	CORR_TOL :: 0.02
	VAR_REL_TOL :: 0.03

	corr := [K * K]f64 {
		1.0, 0.6, -0.3,
		0.6, 1.0, 0.1,
		-0.3, 0.1, 1.0,
	}
	variance := [K]f64{4.0, 0.25, 9.0}

	// Terms = the three base variables alone, so the design matrix columns ARE
	// the base variables and their sample statistics are directly checkable.
	terms := [K * K]u8 {
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
	}
	coef := [K]f64{1.0, 2.0, 3.0}

	scratch := make([]f64, synth.synth_scratch(K));defer delete(scratch)
	spec: synth.Spec
	if e := synth.synth_init(&spec, K, corr[:], variance[:], terms[:], coef[:], 0.0, scratch);
	   e != .None {
		fmt.printf("  FAILED: init %v\n", e)
		return false
	}

	x := make([]f64, M * K);defer delete(x)
	y := make([]f64, M);defer delete(y)
	SEED :: u32(20260817)
	if e := synth.synth_rows(&spec, SEED, 0, x, K, y, M); e != .None {
		fmt.printf("  FAILED: rows %v\n", e)
		return false
	}

	// Sample means, variances, correlations.
	mean: [K]f64
	for i in 0 ..< M {
		for j in 0 ..< K {mean[j] += x[i * K + j]}
	}
	for j in 0 ..< K {mean[j] /= f64(M)}

	cov: [K * K]f64
	for i in 0 ..< M {
		for a in 0 ..< K {
			da := x[i * K + a] - mean[a]
			for b in 0 ..< K {
				cov[a * K + b] += da * (x[i * K + b] - mean[b])
			}
		}
	}
	for i in 0 ..< K * K {cov[i] /= f64(M - 1)}

	for j in 0 ..< K {
		// Mean should be ~0; the base variables are centred.
		if abs(mean[j]) > 5.0 * math.sqrt_f64(variance[j] / f64(M)) {
			fmt.printf("  FAILED: mean[%d] = %.6f, too far from 0\n", j, mean[j])
			return false
		}
		got := cov[j * K + j]
		if abs(got - variance[j]) > VAR_REL_TOL * variance[j] {
			fmt.printf("  FAILED: variance[%d] = %.6f want %.6f\n", j, got, variance[j])
			return false
		}
	}
	for a in 0 ..< K {
		for b in 0 ..< K {
			r := cov[a * K + b] / math.sqrt_f64(cov[a * K + a] * cov[b * K + b])
			want := corr[a * K + b]
			if abs(r - want) > CORR_TOL {
				fmt.printf("  FAILED: corr[%d,%d] = %.4f want %.4f\n", a, b, r, want)
				return false
			}
		}
	}
	// Third and fourth standardised moments: a Gaussian has skew 0 and excess
	// kurtosis 0. This is what catches a broken normal generator -- a sum of
	// uniforms, say, would pass the variance and correlation checks above while
	// having excess kurtosis well below zero. Standard errors at this M are
	// sqrt(6/M) = 0.0055 and sqrt(24/M) = 0.011, so these bounds are ~5 sigma.
	for j in 0 ..< K {
		sd := math.sqrt_f64(cov[j * K + j])
		m3, m4 := 0.0, 0.0
		for i in 0 ..< M {
			z := (x[i * K + j] - mean[j]) / sd
			z2 := z * z
			m3 += z2 * z
			m4 += z2 * z2
		}
		skew := m3 / f64(M)
		exkurt := m4 / f64(M) - 3.0
		if abs(skew) > 0.03 {
			fmt.printf("  FAILED: base %d skew = %+.4f, not Gaussian\n", j, skew)
			return false
		}
		if abs(exkurt) > 0.06 {
			fmt.printf("  FAILED: base %d excess kurtosis = %+.4f, not Gaussian\n", j, exkurt)
			return false
		}
	}

	fmt.printf("  PASSED (variances, all %d correlations, skew and kurtosis)\n", K * K)
	return true
}

test_synth_noise_level :: proc() -> bool {
	fmt.println("Testing sigma controls the residual scale...")

	K :: 2
	M :: 100_000
	corr := [K * K]f64{1, 0, 0, 1}
	variance := [K]f64{1, 1}
	terms := [3 * K]u8{0, 0, 1, 0, 0, 1} // intercept, b0, b1
	coef := [3]f64{5.0, -2.0, 0.75}

	for sigma in ([]f64{0.0, 0.1, 2.5}) {
		scratch := make([]f64, synth.synth_scratch(K));defer delete(scratch)
		spec: synth.Spec
		if e := synth.synth_init(
			&spec, K, corr[:], variance[:], terms[:], coef[:], sigma, scratch,
		); e != .None {
			fmt.printf("  FAILED: init %v\n", e)
			return false
		}
		x := make([]f64, M * 3);defer delete(x)
		y := make([]f64, M);defer delete(y)
	SEED :: u32(99)
		synth.synth_rows(&spec, SEED, 0, x, 3, y, M)

		// Residual against the TRUE coefficients is exactly the injected noise.
		ss := 0.0
		for i in 0 ..< M {
			pred := 0.0
			for t in 0 ..< 3 {pred += coef[t] * x[i * 3 + t]}
			d := y[i] - pred
			ss += d * d
		}
		rms := math.sqrt_f64(ss / f64(M))

		if sigma == 0.0 {
			// Noiseless must be exact, not merely small.
			if rms > 1e-12 {
				fmt.printf("  FAILED: sigma=0 gave residual rms %.3e\n", rms)
				return false
			}
		} else {
			// Standard error of the rms estimate is sigma/sqrt(2M); 4% is many
			// sigma at M = 100k.
			if abs(rms - sigma) > 0.04 * sigma {
				fmt.printf("  FAILED: sigma=%.3f gave residual rms %.5f\n", sigma, rms)
				return false
			}
		}
	}
	fmt.println("  PASSED")
	return true
}

test_synth_polynomial_terms :: proc() -> bool {
	fmt.println("Testing polynomial and interaction terms...")

	K :: 3
	M :: 64
	corr := [K * K]f64{1, 0.2, 0, 0.2, 1, 0, 0, 0, 1}
	variance := [K]f64{1, 2, 0.5}

	// Hand-written table exercising intercept, linear, powers, interaction.
	NT :: 7
	terms := [NT * K]u8 {
		0, 0, 0, // 1
		1, 0, 0, // b0
		0, 1, 0, // b1
		2, 0, 0, // b0^2
		0, 0, 3, // b2^3
		1, 1, 0, // b0*b1
		1, 0, 1, // b0*b2
	}
	coef := [NT]f64{1, 1, 1, 1, 1, 1, 1}

	scratch := make([]f64, synth.synth_scratch(K));defer delete(scratch)
	spec: synth.Spec
	if e := synth.synth_init(&spec, K, corr[:], variance[:], terms[:], coef[:], 0.0, scratch);
	   e != .None {
		fmt.printf("  FAILED: init %v\n", e)
		return false
	}
	if synth.synth_term_count(&spec) != NT {
		fmt.printf("  FAILED: term count %d want %d\n", synth.synth_term_count(&spec), NT)
		return false
	}

	x := make([]f64, M * NT);defer delete(x)
	y := make([]f64, M);defer delete(y)
	SEED :: u32(4)
	synth.synth_rows(&spec, SEED, 0, x, NT, y, M)

	// The exponent algebra must hold exactly, row by row: columns 3..6 are
	// products of columns 1 and 2 and the (unobserved) b2. b2 is recoverable
	// from column 4 as the cube root only up to sign, so check the relations
	// that do not need it.
	for i in 0 ..< M {
		r := x[i * NT:]
		if r[0] != 1.0 {
			fmt.printf("  FAILED: row %d intercept = %.17e\n", i, r[0])
			return false
		}
		b0, b1 := r[1], r[2]
		if abs(r[3] - b0 * b0) > 1e-12 * max(1.0, abs(r[3])) {
			fmt.printf("  FAILED: row %d b0^2\n", i)
			return false
		}
		if abs(r[5] - b0 * b1) > 1e-12 * max(1.0, abs(r[5])) {
			fmt.printf("  FAILED: row %d b0*b1\n", i)
			return false
		}
		// b2 from the cube term, then check the interaction agrees.
		b2 := math.pow_f64(abs(r[4]), 1.0 / 3.0)
		if r[4] < 0 {b2 = -b2}
		if abs(r[6] - b0 * b2) > 1e-9 * max(1.0, abs(r[6])) {
			fmt.printf("  FAILED: row %d b0*b2: %.17e vs %.17e\n", i, r[6], b0 * b2)
			return false
		}
		// y is the sum of all terms with unit coefficients.
		sum := 0.0
		for t in 0 ..< NT {sum += r[t]}
		if abs(y[i] - sum) > 1e-12 * max(1.0, abs(sum)) {
			fmt.printf("  FAILED: row %d y = %.17e want %.17e\n", i, y[i], sum)
			return false
		}
	}

	// synth_terms_poly must lay out the basis it documents.
	{
		DEG :: 2
		nt := synth.synth_terms_poly_count(K, DEG)
		if nt != 1 + K * DEG {
			fmt.printf("  FAILED: poly count %d\n", nt)
			return false
		}
		buf := make([]u8, nt * K);defer delete(buf)
		if e := synth.synth_terms_poly(buf, K, DEG); e != .None {
			fmt.printf("  FAILED: terms_poly %v\n", e)
			return false
		}
		want := []u8 {
			0, 0, 0,
			1, 0, 0,
			2, 0, 0,
			0, 1, 0,
			0, 2, 0,
			0, 0, 1,
			0, 0, 2,
		}
		for i in 0 ..< nt * K {
			if buf[i] != want[i] {
				fmt.printf("  FAILED: terms_poly[%d] = %d want %d\n", i, buf[i], want[i])
				return false
			}
		}
	}
	fmt.println("  PASSED")
	return true
}

// ============================================================================
// End to end: the generator and the solver must agree
// ============================================================================

test_synth_ols_recovers_coefficients :: proc() -> bool {
	fmt.println("Testing OLS recovers the planted coefficients...")

	K :: 3
	M :: 20_000
	corr := [K * K]f64{1, 0.4, -0.2, 0.4, 1, 0.15, -0.2, 0.15, 1}
	variance := [K]f64{1.0, 2.0, 0.5}

	DEG :: 2
	NT :: 1 + K * DEG
	terms: [NT * K]u8
	synth.synth_terms_poly(terms[:], K, DEG)
	// Intercept, b0, b0^2, b1, b1^2, b2, b2^2
	coef := [NT]f64{2.0, -1.5, 0.25, 0.8, -0.1, 3.0, 0.5}

	sscratch := make([]f64, synth.synth_scratch(K));defer delete(sscratch)
	spec: synth.Spec
	if e := synth.synth_init(&spec, K, corr[:], variance[:], terms[:], coef[:], 0.0, sscratch);
	   e != .None {
		fmt.printf("  FAILED: init %v\n", e)
		return false
	}

	x := make([]f64, M * NT);defer delete(x)
	y := make([]f64, M);defer delete(y)
	SEED :: u32(777)
	synth.synth_rows(&spec, SEED, 0, x, NT, y, M)

	// Noiseless: the fit must reproduce the planted coefficients almost exactly.
	abuf := make([]f64, blas.ols_accum_scratch(NT));defer delete(abuf)
	acc: blas.Ols_Accum
	blas.ols_accum_init(&acc, NT, abuf)
	if _, e := blas.ols_accum_rows(&acc, x, NT, y, M); e != .None {
		fmt.printf("  FAILED: accumulate %v\n", e)
		return false
	}
	beta: [NT]f64
	if e := blas.ols_accum_solve(&acc, beta[:]); e != .None {
		fmt.printf("  FAILED: solve %v\n", e)
		return false
	}
	worst := 0.0
	for t in 0 ..< NT {
		worst = max(worst, abs(beta[t] - coef[t]))
	}
	if worst > 1e-9 {
		fmt.printf("  FAILED: noiseless recovery worst error %.3e\n", worst)
		for t in 0 ..< NT {
			fmt.printf("    term %d: got %.10f want %.10f\n", t, beta[t], coef[t])
		}
		return false
	}
	if blas.ols_accum_rss(&acc) > 1e-14 * f64(M) {
		fmt.printf("  FAILED: noiseless RSS = %.3e\n", blas.ols_accum_rss(&acc))
		return false
	}

	// With noise: coefficients should land within a few standard errors, and
	// the residual scale should recover sigma.
	SIGMA :: 0.5
	spec2: synth.Spec
	s2 := make([]f64, synth.synth_scratch(K));defer delete(s2)
	synth.synth_init(&spec2, K, corr[:], variance[:], terms[:], coef[:], SIGMA, s2)
	SEED2 :: u32(778)
	synth.synth_rows(&spec2, SEED2, 0, x, NT, y, M)

	b2 := make([]f64, blas.ols_accum_scratch(NT));defer delete(b2)
	acc2: blas.Ols_Accum
	blas.ols_accum_init(&acc2, NT, b2)
	blas.ols_accum_rows(&acc2, x, NT, y, M)
	beta2: [NT]f64
	if e := blas.ols_accum_solve(&acc2, beta2[:]); e != .None {
		fmt.printf("  FAILED: noisy solve %v\n", e)
		return false
	}
	// A crude but valid bound: with M = 20k and sigma = 0.5, no coefficient of
	// a well-conditioned design should be off by more than 0.05.
	for t in 0 ..< NT {
		if abs(beta2[t] - coef[t]) > 0.05 {
			fmt.printf("  FAILED: noisy term %d got %.6f want %.6f\n", t, beta2[t], coef[t])
			return false
		}
	}
	rms := math.sqrt_f64(blas.ols_accum_rss(&acc2) / f64(M))
	if abs(rms - SIGMA) > 0.05 * SIGMA {
		fmt.printf("  FAILED: recovered sigma %.5f want %.5f\n", rms, SIGMA)
		return false
	}
	fmt.printf("  PASSED (noiseless worst error %.2e, sigma recovered as %.4f)\n", worst, rms)
	return true
}

// ============================================================================
// Reproducibility, chunking, boundaries, allocation
// ============================================================================

test_synth_reproducible_and_chunkable :: proc() -> bool {
	fmt.println("Testing reproducibility and chunked generation...")

	K :: 2
	NT :: 3
	M :: 500
	corr := [K * K]f64{1, 0.5, 0.5, 1}
	variance := [K]f64{1, 3}
	terms := [NT * K]u8{0, 0, 1, 0, 0, 2}
	coef := [NT]f64{1, 2, 3}

	gen :: proc(seed: u64, chunk: int, x: []f64, y: []f64) {
		K :: 2
		NT :: 3
		M :: 500
		corr := [K * K]f64{1, 0.5, 0.5, 1}
		variance := [K]f64{1, 3}
		terms := [NT * K]u8{0, 0, 1, 0, 0, 2}
		coef := [NT]f64{1, 2, 3}
		sc := make([]f64, synth.synth_scratch(K));defer delete(sc)
		spec: synth.Spec
		synth.synth_init(&spec, K, corr[:], variance[:], terms[:], coef[:], 0.25, sc)
		done := 0
		for done < M {
			take := min(chunk, M - done)
			synth.synth_rows(&spec, u32(seed), done, x[done * NT:], NT, y[done:], take)
			done += take
		}
	}

	a_x := make([]f64, M * NT);defer delete(a_x)
	a_y := make([]f64, M);defer delete(a_y)
	b_x := make([]f64, M * NT);defer delete(b_x)
	b_y := make([]f64, M);defer delete(b_y)

	// Same seed, same everything: identical bytes.
	gen(1234, M, a_x, a_y)
	gen(1234, M, b_x, b_y)
	for i in 0 ..< M * NT {
		if a_x[i] != b_x[i] {
			fmt.printf("  FAILED: same seed differed at x[%d]\n", i)
			return false
		}
	}
	for i in 0 ..< M {
		if a_y[i] != b_y[i] {
			fmt.printf("  FAILED: same seed differed at y[%d]\n", i)
			return false
		}
	}

	// A different seed must give different data, or the seeding is broken.
	gen(1235, M, b_x, b_y)
	same := true
	for i in 0 ..< M * NT {
		if a_x[i] != b_x[i] {same = false;break}
	}
	if same {
		fmt.println("  FAILED: different seeds produced identical data")
		return false
	}

	// Chunked generation must equal one big call: the stream continues across
	// calls, so this is bit-identical rather than merely similar.
	for chunk in ([]int{1, 7, 128}) {
		gen(1234, chunk, b_x, b_y)
		for i in 0 ..< M * NT {
			if a_x[i] != b_x[i] {
				fmt.printf("  FAILED: chunk %d differed at x[%d]\n", chunk, i)
				return false
			}
		}
		for i in 0 ..< M {
			if a_y[i] != b_y[i] {
				fmt.printf("  FAILED: chunk %d differed at y[%d]\n", chunk, i)
				return false
			}
		}
	}

	// A wider destination stride must not disturb the values or the spare
	// columns between them.
	WIDE :: NT + 2
	w_x := make([]f64, M * WIDE);defer delete(w_x)
	for i in 0 ..< M * WIDE {w_x[i] = -1.0}
	w_y := make([]f64, M);defer delete(w_y)
	{
		sc := make([]f64, synth.synth_scratch(K));defer delete(sc)
		spec: synth.Spec
		synth.synth_init(&spec, K, corr[:], variance[:], terms[:], coef[:], 0.25, sc)
	SEED :: u32(1234)
		synth.synth_rows(&spec, SEED, 0, w_x, WIDE, w_y, M)
	}
	for i in 0 ..< M {
		for t in 0 ..< NT {
			if w_x[i * WIDE + t] != a_x[i * NT + t] {
				fmt.printf("  FAILED: wide stride differs at [%d,%d]\n", i, t)
				return false
			}
		}
		for t in NT ..< WIDE {
			if w_x[i * WIDE + t] != -1.0 {
				fmt.printf("  FAILED: wide stride wrote spare column %d\n", t)
				return false
			}
		}
	}
	fmt.println("  PASSED")
	return true
}

test_synth_boundaries :: proc() -> bool {
	fmt.println("Testing generator boundary policies...")

	K :: 2
	good_corr := [K * K]f64{1, 0.5, 0.5, 1}
	good_var := [K]f64{1, 1}
	terms := [2 * K]u8{0, 0, 1, 0}
	coef := [2]f64{1, 1}
	scratch := make([]f64, synth.synth_scratch(K));defer delete(scratch)
	spec: synth.Spec

	// k < 1
	if e := synth.synth_init(&spec, 0, good_corr[:], good_var[:], terms[:], coef[:], 0, scratch);
	   e != .Invalid_Dimension {
		fmt.printf("  FAILED: k=0 -> %v\n", e)
		return false
	}
	// short scratch
	{
		small := make([]f64, synth.synth_scratch(K) - 1);defer delete(small)
		if e := synth.synth_init(
			&spec, K, good_corr[:], good_var[:], terms[:], coef[:], 0, small,
		); e != .Scratch_Too_Small {
			fmt.printf("  FAILED: short scratch -> %v\n", e)
			return false
		}
	}
	// terms length not a multiple of k
	{
		bad := [3]u8{0, 0, 1}
		if e := synth.synth_init(
			&spec, K, good_corr[:], good_var[:], bad[:], coef[:], 0, scratch,
		); e != .Invalid_Dimension {
			fmt.printf("  FAILED: ragged terms -> %v\n", e)
			return false
		}
	}
	// too few coefficients for the terms given
	{
		one := [1]f64{1}
		if e := synth.synth_init(
			&spec, K, good_corr[:], good_var[:], terms[:], one[:], 0, scratch,
		); e != .Invalid_Dimension {
			fmt.printf("  FAILED: short coef -> %v\n", e)
			return false
		}
	}
	// asymmetric correlation
	{
		bad := [K * K]f64{1, 0.5, 0.4, 1}
		if e := synth.synth_init(
			&spec, K, bad[:], good_var[:], terms[:], coef[:], 0, scratch,
		); e != .Invalid_Correlation {
			fmt.printf("  FAILED: asymmetric corr -> %v\n", e)
			return false
		}
	}
	// diagonal not 1
	{
		bad := [K * K]f64{2, 0.5, 0.5, 1}
		if e := synth.synth_init(
			&spec, K, bad[:], good_var[:], terms[:], coef[:], 0, scratch,
		); e != .Invalid_Correlation {
			fmt.printf("  FAILED: bad diagonal -> %v\n", e)
			return false
		}
	}
	// |corr| > 1
	{
		bad := [K * K]f64{1, 1.5, 1.5, 1}
		if e := synth.synth_init(
			&spec, K, bad[:], good_var[:], terms[:], coef[:], 0, scratch,
		); e != .Invalid_Correlation {
			fmt.printf("  FAILED: corr>1 -> %v\n", e)
			return false
		}
	}
	// non-positive-definite but individually legal correlation matrix: the
	// classic three-way inconsistency. This is the one a caller writes by hand.
	{
		K3 :: 3
		bad := [K3 * K3]f64 {
			1.0, 0.9, 0.9,
			0.9, 1.0, -0.9,
			0.9, -0.9, 1.0,
		}
		v3 := [K3]f64{1, 1, 1}
		t3 := [K3]u8{1, 0, 0}
		c3 := [1]f64{1}
		s3 := make([]f64, synth.synth_scratch(K3));defer delete(s3)
		sp3: synth.Spec
		if e := synth.synth_init(&sp3, K3, bad[:], v3[:], t3[:], c3[:], 0, s3);
		   e != .Not_Positive_Definite {
			fmt.printf("  FAILED: inconsistent corr -> %v\n", e)
			return false
		}
	}
	// zero and negative variance
	for v in ([][K]f64{{0, 1}, {-1, 1}}) {
		vv := v
		if e := synth.synth_init(
			&spec, K, good_corr[:], vv[:], terms[:], coef[:], 0, scratch,
		); e != .Invalid_Variance {
			fmt.printf("  FAILED: variance %v -> %v\n", vv, e)
			return false
		}
	}
	// negative sigma
	if e := synth.synth_init(
		&spec, K, good_corr[:], good_var[:], terms[:], coef[:], -1.0, scratch,
	); e != .Invalid_Sigma {
		fmt.printf("  FAILED: negative sigma -> %v\n", e)
		return false
	}

	// Now a valid spec, and check synth_rows' own boundaries.
	if e := synth.synth_init(
		&spec, K, good_corr[:], good_var[:], terms[:], coef[:], 0.1, scratch,
	); e != .None {
		fmt.printf("  FAILED: valid init -> %v\n", e)
		return false
	}
	x := make([]f64, 10 * 2);defer delete(x)
	y := make([]f64, 10);defer delete(y)
	SEED :: u32(1)

	if e := synth.synth_rows(&spec, SEED, 0, x, 1, y, 4); e != .Invalid_Dimension {
		fmt.printf("  FAILED: ldx < nterms -> %v\n", e)
		return false
	}
	if e := synth.synth_rows(&spec, SEED, 0, x, 2, y, -1); e != .Invalid_Dimension {
		fmt.printf("  FAILED: negative count -> %v\n", e)
		return false
	}
	if e := synth.synth_rows(&spec, SEED, 0, x, 2, y, 11); e != .Invalid_Dimension {
		fmt.printf("  FAILED: overrun -> %v\n", e)
		return false
	}
	if e := synth.synth_rows(&spec, SEED, 0, x, 2, y, 0); e != .None {
		fmt.printf("  FAILED: count=0 -> %v\n", e)
		return false
	}
	fmt.println("  PASSED")
	return true
}

test_synth_no_allocation :: proc() -> bool {
	fmt.println("Testing the generator under mem.panic_allocator...")

	saved_a := context.allocator
	saved_t := context.temp_allocator
	defer {
		context.allocator = saved_a
		context.temp_allocator = saved_t
	}
	context.allocator = mem.panic_allocator()
	context.temp_allocator = mem.panic_allocator()

	K :: 3
	DEG :: 2
	NT :: 1 + K * DEG
	M :: 256

	corr := [K * K]f64{1, 0.3, 0, 0.3, 1, 0.1, 0, 0.1, 1}
	variance := [K]f64{1, 2, 0.5}
	terms: [NT * K]u8
	if e := synth.synth_terms_poly(terms[:], K, DEG); e != .None {
		fmt.printf("  FAILED: terms_poly %v\n", e)
		return false
	}
	coef := [NT]f64{1, 2, 0.1, -1, 0.2, 0.5, -0.3}

	scratch: [K * K + 2 * K]f64
	if len(scratch) < synth.synth_scratch(K) {
		fmt.println("  FAILED: stack scratch too small")
		return false
	}
	spec: synth.Spec
	if e := synth.synth_init(
		&spec, K, corr[:], variance[:], terms[:], coef[:], 0.3, scratch[:],
	); e != .None {
		fmt.printf("  FAILED: init %v\n", e)
		return false
	}

	x: [M * NT]f64
	y: [M]f64
	SEED :: u32(5)
	if e := synth.synth_rows(&spec, SEED, 0, x[:], NT, y[:], M); e != .None {
		fmt.printf("  FAILED: rows %v\n", e)
		return false
	}

	// And straight into a fit, still with no heap.
	abuf: [(NT + 1) * (NT + 1) + (NT + 1)]f64
	acc: blas.Ols_Accum
	blas.ols_accum_init(&acc, NT, abuf[:])
	blas.ols_accum_rows(&acc, x[:], NT, y[:], M)
	beta: [NT]f64
	if e := blas.ols_accum_solve(&acc, beta[:]); e != .None {
		fmt.printf("  FAILED: solve %v\n", e)
		return false
	}
	for t in 0 ..< NT {
		if beta[t] != beta[t] {
			fmt.println("  FAILED: NaN in beta")
			return false
		}
	}
	fmt.println("  PASSED (generate and fit with no heap at all)")
	return true
}

// ============================================================================
// Properties that only the counter-based scheme can offer
// ============================================================================

test_synth_sigma_leaves_x_alone :: proc() -> bool {
	fmt.println("Testing sigma changes y but not X...")

	// The point of separate streams. With one shared sequential stream the
	// noise draw shifts every subsequent base draw, so raising sigma silently
	// produces a different design matrix and "the same data with more noise"
	// is not expressible.
	// The spec lives in `gen` below; only the shapes are needed out here.
	NT :: 4
	M :: 500
	SEED :: u32(31337)

	ref_x := make([]f64, M * NT);defer delete(ref_x)
	ref_y := make([]f64, M);defer delete(ref_y)
	got_x := make([]f64, M * NT);defer delete(got_x)
	got_y := make([]f64, M);defer delete(got_y)

	gen :: proc(sigma: f64, seed: u32, x: []f64, y: []f64) -> synth.Synth_Error {
		K :: 3
		NT :: 4
		M :: 500
		corr := [K * K]f64{1, 0.5, 0.2, 0.5, 1, -0.3, 0.2, -0.3, 1}
		variance := [K]f64{1, 2, 0.5}
		terms := [NT * K]u8{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1}
		coef := [NT]f64{1, 2, -1, 0.5}
		sc := make([]f64, synth.synth_scratch(K));defer delete(sc)
		spec: synth.Spec
		if e := synth.synth_init(&spec, K, corr[:], variance[:], terms[:], coef[:], sigma, sc);
		   e != .None {
			return e
		}
		return synth.synth_rows(&spec, seed, 0, x, NT, y, M)
	}

	if e := gen(0.0, SEED, ref_x, ref_y); e != .None {
		fmt.printf("  FAILED: noiseless gen %v\n", e)
		return false
	}
	for sigma in ([]f64{0.01, 0.5, 10.0}) {
		if e := gen(sigma, SEED, got_x, got_y); e != .None {
			fmt.printf("  FAILED: gen sigma=%.2f -> %v\n", sigma, e)
			return false
		}
		// X must be bit-identical across noise levels.
		for i in 0 ..< M * NT {
			if got_x[i] != ref_x[i] {
				fmt.printf("  FAILED: sigma=%.2f changed X at [%d]\n", sigma, i)
				return false
			}
		}
		// y must differ, or sigma is being ignored.
		same := true
		for i in 0 ..< M {
			if got_y[i] != ref_y[i] {same = false;break}
		}
		if same {
			fmt.printf("  FAILED: sigma=%.2f left y unchanged\n", sigma)
			return false
		}
	}
	fmt.println("  PASSED (X bit-identical across sigma 0 / 0.01 / 0.5 / 10)")
	return true
}

test_synth_row_addressable :: proc() -> bool {
	fmt.println("Testing rows are individually addressable and order-free...")

	K :: 2
	NT :: 3
	M :: 300
	corr := [K * K]f64{1, 0.3, 0.3, 1}
	variance := [K]f64{1, 1}
	terms := [NT * K]u8{0, 0, 1, 0, 0, 1}
	coef := [NT]f64{1, 1, 1}
	SEED :: u32(8675309)

	sc := make([]f64, synth.synth_scratch(K));defer delete(sc)
	spec: synth.Spec
	synth.synth_init(&spec, K, corr[:], variance[:], terms[:], coef[:], 0.3, sc)

	all_x := make([]f64, M * NT);defer delete(all_x)
	all_y := make([]f64, M);defer delete(all_y)
	synth.synth_rows(&spec, SEED, 0, all_x, NT, all_y, M)

	// One row at a time, requested out of order and skipping most of them.
	one_x: [NT]f64
	one_y: [1]f64
	for r in ([]int{0, 299, 17, 250, 1, 100}) {
		if e := synth.synth_rows(&spec, SEED, r, one_x[:], NT, one_y[:], 1); e != .None {
			fmt.printf("  FAILED: row %d -> %v\n", r, e)
			return false
		}
		for t in 0 ..< NT {
			if one_x[t] != all_x[r * NT + t] {
				fmt.printf("  FAILED: row %d col %d: %.17e vs %.17e\n",
					r, t, one_x[t], all_x[r * NT + t])
				return false
			}
		}
		if one_y[0] != all_y[r] {
			fmt.printf("  FAILED: row %d y: %.17e vs %.17e\n", r, one_y[0], all_y[r])
			return false
		}
	}

	// A whole range generated backwards, batch by batch, must still match.
	rev_x := make([]f64, M * NT);defer delete(rev_x)
	rev_y := make([]f64, M);defer delete(rev_y)
	CH :: 37
	start := ((M - 1) / CH) * CH
	for start >= 0 {
		take := min(CH, M - start)
		synth.synth_rows(&spec, SEED, start, rev_x[start * NT:], NT, rev_y[start:], take)
		start -= CH
	}
	for i in 0 ..< M * NT {
		if rev_x[i] != all_x[i] {
			fmt.printf("  FAILED: reverse-order generation differs at [%d]\n", i)
			return false
		}
	}
	fmt.println("  PASSED (single rows, out of order, and reverse-order batches all match)")
	return true
}

test_synth_stream_independence :: proc() -> bool {
	fmt.println("Testing hash quality: stream and serial independence...")

	// A counter-based generator lives or dies on its hash. Two failure modes
	// matter here and neither would be caught by the moment tests:
	//
	//   * correlated streams -- the noise would correlate with X, biasing every
	//     fitted coefficient while the marginal distributions stayed perfect
	//   * serial correlation between adjacent indices -- adjacent indices are
	//     exactly what consecutive rows use
	N :: 200_000
	SEED :: u32(0xC0FFEE)
	// 4.5 / sqrt(N) is about 0.010; a correlation this size is ~4.5 sigma.
	TOL :: 0.010

	corr_of :: proc(a: []f64, b: []f64) -> f64 {
		n := len(a)
		ma, mb := 0.0, 0.0
		for i in 0 ..< n {ma += a[i];mb += b[i]}
		ma /= f64(n);mb /= f64(n)
		sab, saa, sbb := 0.0, 0.0, 0.0
		for i in 0 ..< n {
			da := a[i] - ma
			db := b[i] - mb
			sab += da * db
			saa += da * da
			sbb += db * db
		}
		return sab / math.sqrt_f64(saa * sbb)
	}

	base := make([]f64, N);defer delete(base)
	noise := make([]f64, N);defer delete(noise)
	next := make([]f64, N);defer delete(next)
	other_seed := make([]f64, N);defer delete(other_seed)

	for i in 0 ..< N {
		base[i] = synth.synth_normal(SEED, synth.STREAM_BASE, u32(i))
		noise[i] = synth.synth_normal(SEED, synth.STREAM_NOISE, u32(i))
		next[i] = synth.synth_normal(SEED, synth.STREAM_BASE, u32(i + 1))
		other_seed[i] = synth.synth_normal(SEED + 1, synth.STREAM_BASE, u32(i))
	}

	checks := []struct {
		name: string,
		a:    []f64,
		b:    []f64,
	} {
		{"base vs noise stream", base, noise},
		{"adjacent indices", base, next},
		{"adjacent seeds", base, other_seed},
		{"noise vs next base", noise, next},
	}
	for c in checks {
		r := corr_of(c.a, c.b)
		if abs(r) > TOL {
			fmt.printf("  FAILED: %s correlation %+.5f exceeds %.3f\n", c.name, r, TOL)
			return false
		}
		fmt.printf("    %-22s r = %+.5f\n", c.name, r)
	}

	// The uniform feeding Box-Muller must be uniform, not merely unbiased:
	// check occupancy of 64 equal bins. Expected count N/64 with sd
	// sqrt(N/64), so 5 sigma is 5*sqrt(3125) = 279 on 3125.
	BINS :: 64
	counts: [BINS]int
	for i in 0 ..< N {
		// Recover the uniform via the standard normal CDF is awkward; instead
		// bin the raw hash, which is what uniformity is a property of.
		h := synth.synth_hash(SEED, synth.STREAM_BASE, u32(i))
		counts[int(h >> 26)] += 1
	}
	expect := f64(N) / f64(BINS)
	chi2 := 0.0
	for c in counts {
		d := f64(c) - expect
		chi2 += d * d / expect
	}
	// 63 degrees of freedom: the 99.9th percentile is about 112.
	if chi2 > 112.0 {
		fmt.printf("  FAILED: hash bin chi-square %.1f on 63 df, too high\n", chi2)
		return false
	}
	fmt.printf("    hash uniformity chi2 = %.1f on 63 df (99.9%% point 112)\n", chi2)
	fmt.println("  PASSED")
	return true
}
