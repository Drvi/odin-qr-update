package model_iteration_example

// Trying models by hand, against data that keeps arriving.
//
// The person at the keyboard decides which predictors to try. This example
// shows what that costs: after one pass over the data, each hypothesis they
// test is a select from the cached triangle and touches no rows at all.
// Measured at m = 1,000,000 and p = 10, that is ~0.5 us per model tried,
// against ~87 ms to refit from the data. See docs/OLS_RESULTS.md.
//
// There is deliberately no search here, and no scoring rule. The library
// reports RSS for whatever model you ask for; deciding which model is RIGHT is
// yours. A criterion like AIC or BIC is a few lines of arithmetic over
// (rss, nrows, k) if you want one, and that is exactly why the library does
// not pick one for you.
//
// The whole program runs under mem.panic_allocator: every buffer is a fixed
// array, and any hidden allocation in the library would abort it. This is what
// "no allocations the caller could not have supplied" looks like in practice --
// it works the same from an arena, a frame allocator, or no heap at all.

import "core:fmt"
import "core:math"
import "core:mem"
import blas "../../src/blas"

P :: 8 // candidate predictors, decided up front, accumulated once
M0 :: 4000 // first batch of observations
M1 :: 2000 // a second batch, collected later

// The response actually depends on predictors 0, 2 and 5. The point of the
// exercise is that the person trying models does not know that yet.
TRUE_COEF := [P]f64{3.0, 0.0, -1.75, 0.0, 0.0, 0.9, 0.0, 0.0}
NOISE :: 0.35

// All storage is fixed. Nothing here is heap-allocated, so nothing here needs
// an allocator the caller did not choose.
x0: [M0 * P]f64
y0: [M0]f64
x1: [M1 * P]f64
y1: [M1]f64

RNG :: struct {
	state: u64,
}
ru :: proc(r: ^RNG) -> f64 {
	r.state = r.state * 6364136223846793005 + 1442695040888963407
	return f64((r.state >> 11) & 0x1FFFFFFFFFFFFF) / f64(0x1FFFFFFFFFFFFF)
}
rnorm :: proc(r: ^RNG) -> f64 {
	u1 := ru(r)
	for u1 < 1e-12 {u1 = ru(r)}
	return math.sqrt_f64(-2.0 * math.ln_f64(u1)) * math.cos_f64(2.0 * math.PI * ru(r))
}

fill :: proc(r: ^RNG, x: []f64, y: []f64, m: int) {
	for i in 0 ..< m {
		x[i * P] = 1.0
		for j in 1 ..< P {x[i * P + j] = rnorm(r)}
		s := 0.0
		for j in 0 ..< P {s += x[i * P + j] * TRUE_COEF[j]}
		y[i] = s + NOISE * rnorm(r)
	}
}

// A hypothesis the user wants to test: a name and the predictors it uses.
Hypothesis :: struct {
	name: string,
	keep: []int,
}

// try evaluates one hypothesis against the cached fit and prints what came
// back. It does not judge, rank, or recommend.
try :: proc(h: Hypothesis, full: ^blas.Ols_Accum, scratch: []f64) {
	k := len(h.keep)

	sub: blas.Ols_Accum
	if e := blas.ols_accum_init(&sub, k, scratch); e != .None {
		fmt.printf("  %s: init failed, %v\n", h.name, e)
		return
	}
	if e := blas.ols_accum_select(&sub, full, h.keep); e != .None {
		fmt.printf("  %s: select failed, %v\n", h.name, e)
		return
	}

	beta: [P]f64
	if e := blas.ols_accum_solve(&sub, beta[:k]); e != .None {
		// Collinear or underdetermined hypotheses say so rather than
		// returning a plausible-looking vector.
		fmt.printf("  %s: %v\n", h.name, e)
		return
	}

	rss := blas.ols_accum_rss(&sub)
	fmt.printf("  %s\n", h.name)
	fmt.printf("    predictors %v,  k = %d\n", h.keep, k)
	fmt.printf("    RSS %.4f   residual rms %.5f\n", rss, math.sqrt_f64(rss / f64(sub.nrows)))
	fmt.printf("    coefficients")
	for j in 0 ..< k {
		fmt.printf("  b%d=%.4f", h.keep[j], beta[j])
	}
	fmt.println()
}

main :: proc() {
	// Any allocation from here on is a bug, in this file or in the library.
	context.allocator = mem.panic_allocator()
	context.temp_allocator = mem.panic_allocator()

	rng := RNG{31337}
	fill(&rng, x0[:], y0[:], M0)
	fill(&rng, x1[:], y1[:], M1)

	// Scratch, sized by the library's own size procedures. Fixed arrays, so
	// the caller can see and budget every byte.
	full_buf: [(P + 1) * (P + 1) + (P + 1)]f64
	sub_buf: [(P + 1) * (P + 1) + (P + 1)]f64
	batch_buf: [(P + 1) * (P + 1) + (P + 1)]f64

	fmt.println("=== Trying models by hand ===")
	fmt.printf(
		"%d candidate predictors accumulated once; each hypothesis below reads\n",
		P,
	)
	fmt.printf(
		"only the %d-byte triangle, never the %d rows.\n\n",
		len(full_buf) * 8,
		M0,
	)

	full: blas.Ols_Accum
	if e := blas.ols_accum_init(&full, P, full_buf[:]); e != .None {
		fmt.printf("init failed: %v\n", e)
		return
	}

	// One pass over the data. Everything after this is free.
	if _, e := blas.ols_accum_rows(&full, x0[:], P, y0[:], M0); e != .None {
		fmt.printf("accumulate failed: %v\n", e)
		return
	}
	fmt.printf("--- %d rows accumulated ---\n\n", full.nrows)

	// A plausible sequence of hypotheses somebody might work through: start
	// simple, add a term, overshoot, back off. The order and the reasoning are
	// the user's; the library just answers.
	fmt.println("Hypotheses, in the order someone might think of them:")
	fmt.println()
	for h in ([]Hypothesis {
		{"intercept only", {0}},
		{"intercept + p1", {0, 1}},
		{"intercept + p2", {0, 2}},
		{"intercept + p2 + p5", {0, 2, 5}},
		{"everything", {0, 1, 2, 3, 4, 5, 6, 7}},
		{"drop the intercept", {2, 5}},
		{"p2 twice (a mistake)", {2, 2}},
	}) {
		try(h, &full, sub_buf[:])
	}

	// New data arrives, collected separately. Fold it in without going back to
	// the first batch, then re-check whichever model you had settled on.
	fmt.println()
	fmt.println("--- a second batch arrives, collected separately ---")
	batch: blas.Ols_Accum
	blas.ols_accum_init(&batch, P, batch_buf[:])
	if _, e := blas.ols_accum_rows(&batch, x1[:], P, y1[:], M1); e != .None {
		fmt.printf("batch accumulate failed: %v\n", e)
		return
	}
	if e := blas.ols_accum_merge(&full, &batch); e != .None {
		fmt.printf("merge failed: %v\n", e)
		return
	}
	fmt.printf("merged; now %d rows, still %d bytes of state\n\n", full.nrows, len(full_buf) * 8)

	try(Hypothesis{"intercept + p2 + p5, refitted", {0, 2, 5}}, &full, sub_buf[:])

	// A hypothesis the cached triangle CANNOT answer: a derived predictor that
	// was never accumulated. Adding a column needs its inner products against
	// every retained row, and those rows are not in the triangle -- so this one
	// costs a pass over the data where every hypothesis above cost ~0.5 us.
	// ols_accum_rows_gather does the pass, and picks the columns and skips
	// unwanted rows while it is there.
	fmt.println()
	fmt.println("--- a predictor the triangle never saw, and two rows dropped ---")
	fmt.println("  (this one needs a pass over the data; everything above did not)")

	// Materialise the derived column into the spare slot of the table.
	for i in 0 ..< M0 {
		x0[i * P + 7] = x0[i * P + 2] * x0[i * P + 5]
	}

	derived_buf: [(P + 1) * (P + 1) + (P + 1)]f64
	derived: blas.Ols_Accum
	blas.ols_accum_init(&derived, 4, derived_buf[:])
	cols := [4]int{0, 2, 5, 7} // intercept, p2, p5, and p2*p5
	drop := [2]int{10, 200}    // two observations judged bad
	if _, e := blas.ols_accum_rows_gather(
		&derived, x0[:], P, y0[:], 0, M0, cols[:], drop[:],
	); e != .None {
		fmt.printf("  gather failed: %v\n", e)
		return
	}
	dbeta: [4]f64
	if e := blas.ols_accum_solve(&derived, dbeta[:]); e != .None {
		fmt.printf("  solve failed: %v\n", e)
		return
	}
	fmt.printf("    %d of %d rows used, k = 4\n", derived.nrows, M0)
	drss := blas.ols_accum_rss(&derived)
	fmt.printf("    RSS %.4f   residual rms %.5f\n", drss, math.sqrt_f64(drss / f64(derived.nrows)))
	fmt.printf("    coefficients")
	for j in 0 ..< 4 {
		fmt.printf("  c%d=%.4f", cols[j], dbeta[j])
	}
	fmt.println()
	fmt.println("    (the interaction term is noise here, so it buys nothing)")

	fmt.println()
	fmt.println("For reference, the coefficients the data was generated from:")
	fmt.printf("  b0=%.4f  b2=%.4f  b5=%.4f  (all others exactly zero)\n",
		TRUE_COEF[0], TRUE_COEF[2], TRUE_COEF[5])
}
