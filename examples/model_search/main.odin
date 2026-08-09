package model_search_example

// Iterating on a model: which predictors, over which data.
//
// The pattern this example exists to show:
//
//   1. Decide the CANDIDATE predictors up front and accumulate all of them
//      once. One pass over the data, O(m*p^2).
//   2. Every sub-model after that is ols_accum_select from the cached
//      triangle: O(p*k^2), independent of m, measured at well under a
//      microsecond. Score it with ols_accum_rss, which is free.
//   3. New data folds in with ols_accum_rows; data collected separately folds
//      in with ols_accum_merge. Neither needs the old rows back.
//
// The thing to avoid is step 2 re-reading the rows. Measured on this machine
// at m = 1,000,000 and p = 10, scoring one candidate model costs about 87 ms
// if you re-read the data and about 0.5 us if you select from the triangle.
// See docs/OLS_RESULTS.md.
//
// Note what is NOT here: no library call for scoring or ranking models. BIC
// below is four lines of caller arithmetic over values the accumulator already
// exposes, which is why the library does not provide it.

import "core:fmt"
import "core:math"
import blas "../../src/blas"

P :: 8 // candidate predictors, including the intercept
M0 :: 4000 // observations in the first batch
M1 :: 2000 // observations collected later

// The response really depends on predictors 0, 2 and 5 only. Everything else
// is noise the search should reject.
TRUE_TERMS :: bit_set[0 ..< P]{0, 2, 5}
TRUE_COEF := [P]f64{3.0, 0.0, -1.75, 0.0, 0.0, 0.9, 0.0, 0.0}
NOISE :: 0.35

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

gen :: proc(r: ^RNG, m: int) -> (x: []f64, y: []f64) {
	x = make([]f64, m * P)
	y = make([]f64, m)
	for i in 0 ..< m {
		x[i * P] = 1.0
		for j in 1 ..< P {x[i * P + j] = rnorm(r)}
		s := 0.0
		for j in 0 ..< P {s += x[i * P + j] * TRUE_COEF[j]}
		y[i] = s + NOISE * rnorm(r)
	}
	return
}

mask_string :: proc(mask: int) -> string {
	buf: [P]byte
	for i in 0 ..< P {
		buf[i] = mask & (1 << uint(i)) != 0 ? '1' : '.'
	}
	return fmt.tprint(string(buf[:]))
}

// Right-pad to a column width.
//
// Numeric width specifiers are avoided throughout this file: in this Odin
// version "%-6d" pads with '0' rather than ' ', so 20 renders as "200000".
// Stringifying first and padding as a string sidesteps that.
pad :: proc(s: string, w: int) -> string {
	if len(s) >= w {return s}
	spaces := "                                        "
	return fmt.tprintf("%s%s", s, spaces[:min(w - len(s), len(spaces))])
}

// Bayesian information criterion. Lower is better. RSS alone always prefers
// more predictors, so a complexity penalty is what makes the search pick the
// true model instead of the full one.
bic :: proc(rss: f64, nrows: int, k: int) -> f64 {
	n := f64(nrows)
	if rss <= 0 {return -math.INF_F64}
	return n * math.ln_f64(rss / n) + f64(k) * math.ln_f64(n)
}

Result :: struct {
	mask: int,
	k:    int,
	rss:  f64,
	bic:  f64,
}

// Score every non-empty subset of the cached candidates. Touches no data.
search :: proc(full: ^blas.Ols_Accum, scratch: []f64) -> (best: Result) {
	best.bic = math.INF_F64
	keep: [P]int

	for mask in 1 ..< (1 << P) {
		k := 0
		for i in 0 ..< P {
			if mask & (1 << uint(i)) != 0 {
				keep[k] = i
				k += 1
			}
		}

		sub: blas.Ols_Accum
		if blas.ols_accum_init(&sub, k, scratch) != .None {continue}
		if blas.ols_accum_select(&sub, full, keep[:k]) != .None {continue}

		// A collinear candidate is reported rather than silently scored.
		beta: [P]f64
		if blas.ols_accum_solve(&sub, beta[:k]) != .None {continue}

		r := Result {
			mask = mask,
			k    = k,
			rss  = blas.ols_accum_rss(&sub),
		}
		r.bic = bic(r.rss, sub.nrows, k)
		if r.bic < best.bic {best = r}
	}
	return
}

report :: proc(label: string, full: ^blas.Ols_Accum, scratch: []f64) -> Result {
	best := search(full, scratch)
	fmt.printf(
		"  %s n=%s best=%s  k=%d  RSS=%s  BIC=%s",
		pad(label, 22),
		pad(fmt.tprintf("%d", full.nrows), 7),
		mask_string(best.mask),
		best.k,
		pad(fmt.tprintf("%.3f", best.rss), 10),
		pad(fmt.tprintf("%.2f", best.bic), 11),
	)

	found: bit_set[0 ..< P]
	for i in 0 ..< P {
		if best.mask & (1 << uint(i)) != 0 {found += {i}}
	}
	fmt.printf("  %s\n", found == TRUE_TERMS ? "<- true model" : "<- MISSED")
	return best
}

main :: proc() {
	fmt.println("=== Model search over cached candidates ===")
	fmt.printf(
		"%d candidate predictors, %d subsets to score, true model is %s\n\n",
		P,
		(1 << P) - 1,
		mask_string(0b00100101),
	)

	rng := RNG{31337}
	x0, y0 := gen(&rng, M0);defer delete(x0);defer delete(y0)
	x1, y1 := gen(&rng, M1);defer delete(x1);defer delete(y1)

	// One block of scratch, reused by every sub-model in the search. The
	// search allocates nothing.
	sub_scratch := make([]f64, blas.ols_accum_scratch(P));defer delete(sub_scratch)

	fs := make([]f64, blas.ols_accum_scratch(P));defer delete(fs)
	full: blas.Ols_Accum
	blas.ols_accum_init(&full, P, fs)

	// -- Step 1: accumulate the candidates once, in growing amounts, and watch
	// the search converge on the true model as data arrives.
	fmt.println("--- Search quality as data accumulates ---")
	for target in ([]int{20, 100, 500, M0}) {
		take := target - full.nrows
		if take <= 0 {continue}
		if _, e := blas.ols_accum_rows(&full, x0[full.nrows * P:], P, y0[full.nrows:], take);
		   e != .None {
			fmt.printf("  accumulate failed: %v\n", e)
			return
		}
		report(fmt.tprintf("after %d rows", full.nrows), &full, sub_scratch)
	}

	// -- Step 2: a second batch of data, collected separately, folded in
	// without revisiting the first batch.
	fmt.println()
	fmt.println("--- Merging a separately collected batch ---")
	bs := make([]f64, blas.ols_accum_scratch(P));defer delete(bs)
	batch: blas.Ols_Accum
	blas.ols_accum_init(&batch, P, bs)
	blas.ols_accum_rows(&batch, x1, P, y1, M1)
	fmt.printf("  second batch accumulated on its own: %d rows\n", batch.nrows)

	if e := blas.ols_accum_merge(&full, &batch); e != .None {
		fmt.printf("  merge failed: %v\n", e)
		return
	}
	best := report("after merge", &full, sub_scratch)

	// -- Step 3: the chosen model's coefficients, from the same cached triangle.
	fmt.println()
	fmt.println("--- Coefficients of the selected model ---")
	keep: [P]int
	k := 0
	for i in 0 ..< P {
		if best.mask & (1 << uint(i)) != 0 {keep[k] = i;k += 1}
	}
	sub: blas.Ols_Accum
	blas.ols_accum_init(&sub, k, sub_scratch)
	blas.ols_accum_select(&sub, &full, keep[:k])
	beta: [P]f64
	if e := blas.ols_accum_solve(&sub, beta[:k]); e != .None {
		fmt.printf("  solve failed: %v\n", e)
		return
	}
	for j in 0 ..< k {
		fmt.printf(
			"  predictor %d: %s (true %s)\n",
			keep[j],
			pad(fmt.tprintf("%.4f", beta[j]), 10),
			pad(fmt.tprintf("%.4f", TRUE_COEF[keep[j]]), 8),
		)
	}

	fmt.println()
	fmt.printf(
		"  resident state for the whole search: %d bytes, independent of the %d rows seen\n",
		(blas.ols_accum_scratch(P) * 2) * 8,
		full.nrows,
	)
}
