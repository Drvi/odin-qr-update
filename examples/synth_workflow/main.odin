package synth_workflow_example

// Generate data whose answer you already know, then go and find it.
//
// This is the loop the generator exists for. You plant a model -- correlated
// predictors with chosen variances, a chosen set of polynomial terms, chosen
// coefficients, a chosen noise level -- and then try hypotheses against it. When
// a fit comes back wrong you know it is the fitting that is wrong, because the
// truth was yours to begin with.
//
// The planted model here is deliberately awkward in three ways a real dataset
// would be:
//
//   * b0 and b1 are correlated 0.85, so their coefficients are hard to separate
//   * the response is quadratic in b0, so a linear-only model cannot fit it
//   * b2 is pure noise with a coefficient of exactly zero
//
// Runs under mem.panic_allocator with fixed arrays throughout: generating and
// fitting together touch no heap.

import "core:fmt"
import "core:math"
import "core:mem"
import blas "../../src/blas"
import synth "../../src/synth"

K :: 3 // base variables
M :: 20_000 // observations
SIGMA :: 0.4 // Gaussian noise on the response

// Terms of the planted model, as exponent vectors over (b0, b1, b2).
// Also the columns of the design matrix, in this order.
NT :: 6
TERMS := [NT * K]u8 {
	0, 0, 0, // intercept
	1, 0, 0, // b0
	2, 0, 0, // b0^2      <- the quadratic a linear model will miss
	0, 1, 0, // b1
	0, 0, 1, // b2        <- irrelevant, true coefficient 0
	1, 1, 0, // b0*b1     <- interaction, true coefficient 0
}
TRUE_COEF := [NT]f64{1.5, -2.0, 0.75, 0.5, 0.0, 0.0}

// b0 and b1 strongly correlated; b2 independent.
CORR := [K * K]f64 {
	1.00, 0.85, 0.00,
	0.85, 1.00, 0.00,
	0.00, 0.00, 1.00,
}
VARIANCE := [K]f64{1.0, 4.0, 0.25}

x: [M * NT]f64
y: [M]f64

TERM_NAME := [NT]string{"1", "b0", "b0^2", "b1", "b2", "b0*b1"}

Hypothesis :: struct {
	name: string,
	keep: []int,
}

try :: proc(h: Hypothesis, full: ^blas.Ols_Accum, scratch: []f64) {
	k := len(h.keep)
	sub: blas.Ols_Accum
	if e := blas.ols_accum_init(&sub, k, scratch); e != .None {
		fmt.printf("  %s: %v\n", h.name, e)
		return
	}
	if e := blas.ols_accum_select(&sub, full, h.keep); e != .None {
		fmt.printf("  %s: %v\n", h.name, e)
		return
	}
	beta: [NT]f64
	if e := blas.ols_accum_solve(&sub, beta[:k]); e != .None {
		fmt.printf("  %s: %v\n", h.name, e)
		return
	}

	rss := blas.ols_accum_rss(&sub)
	rms := math.sqrt_f64(rss / f64(sub.nrows))
	fmt.printf("  %s\n", h.name)
	fmt.printf("    residual rms %.4f", rms)
	// Noise was injected at SIGMA, so a model that has captured the structure
	// cannot do better than that, and one that has not will be visibly worse.
	if rms < SIGMA * 1.02 {
		fmt.printf("   (at the noise floor of %.2f)\n", f64(SIGMA))
	} else {
		fmt.printf("   (%.1fx the noise floor -- structure left over)\n", rms / SIGMA)
	}
	fmt.printf("   ")
	for j in 0 ..< k {
		fmt.printf("  %s=%+.4f", TERM_NAME[h.keep[j]], beta[j])
	}
	fmt.println()
	fmt.printf("    true ")
	for j in 0 ..< k {
		fmt.printf("  %s=%+.4f", TERM_NAME[h.keep[j]], TRUE_COEF[h.keep[j]])
	}
	fmt.println()
}

main :: proc() {
	context.allocator = mem.panic_allocator()
	context.temp_allocator = mem.panic_allocator()

	// ---- plant the model ----
	sscratch: [K * K + 2 * K]f64
	spec: synth.Spec
	if e := synth.synth_init(
		&spec, K, CORR[:], VARIANCE[:], TERMS[:], TRUE_COEF[:], SIGMA, sscratch[:],
	); e != .None {
		fmt.printf("synth_init failed: %v\n", e)
		return
	}

	rng: synth.Rng
	synth.rng_seed(&rng, 20260817)

	fmt.println("=== Generate data with a known answer, then look for it ===")
	fmt.println()
	fmt.printf("planted model, %d rows, noise sigma %.2f:\n", M, f64(SIGMA))
	for t in 0 ..< NT {
		if TRUE_COEF[t] != 0 {
			fmt.printf("    %+.4f * %s\n", TRUE_COEF[t], TERM_NAME[t])
		}
	}
	fmt.println("    (b2 and b0*b1 are present in the data but have coefficient 0)")
	fmt.printf("    corr(b0,b1) = %.2f, variances %v\n\n", CORR[1], VARIANCE)

	// ---- generate, in chunks, straight into the accumulator ----
	// Nothing forces the whole table to be resident; this just keeps it around
	// so the same rows can be reused for the gather demonstration below.
	abuf: [(NT + 1) * (NT + 1) + (NT + 1)]f64
	sbuf: [(NT + 1) * (NT + 1) + (NT + 1)]f64
	full: blas.Ols_Accum
	blas.ols_accum_init(&full, NT, abuf[:])

	CHUNK :: 4096
	done := 0
	for done < M {
		take := min(CHUNK, M - done)
		if e := synth.synth_rows(
			&spec, &rng, x[done * NT:], NT, y[done:], take,
		); e != .None {
			fmt.printf("synth_rows failed: %v\n", e)
			return
		}
		if _, e := blas.ols_accum_rows(
			&full, x[done * NT:], NT, y[done:], take,
		); e != .None {
			fmt.printf("accumulate failed: %v\n", e)
			return
		}
		done += take
	}
	fmt.printf("generated and accumulated %d rows in chunks of %d\n", full.nrows, CHUNK)
	fmt.printf("resident state: %d bytes\n\n", len(abuf) * 8)

	// ---- go looking ----
	fmt.println("Hypotheses:")
	fmt.println()
	for h in ([]Hypothesis {
		{"linear in b0 only", {0, 1}},
		{"linear in b0 and b1", {0, 1, 3}},
		{"add the quadratic", {0, 1, 2, 3}},
		{"everything", {0, 1, 2, 3, 4, 5}},
		{"the planted model", {0, 1, 2, 3}},
	}) {
		try(h, &full, sbuf[:])
	}

	fmt.println()
	fmt.println("Note what the correlation did: with corr(b0,b1) = 0.85, the")
	fmt.println("\"linear in b0 only\" fit puts b1's effect into b0's coefficient.")
	fmt.println("The planted values only reappear once b1 is in the model.")
}
