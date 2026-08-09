package incremental_example

// Fitting a least-squares model with the work spread across frames.
//
// Two regimes, both shown below:
//
//   1. All the data is already in memory and you just cannot afford one big
//      solve this frame. Absorb a slice of the rows each frame.
//
//   2. Rows arrive over time (samples, telemetry, measurements) and you never
//      want to hold them all. Absorb each row as it shows up and throw it away.
//
// The same Ols_Accum serves both. It holds (n+1)^2 + (n+1) f64 -- 336 bytes at
// n = 5 -- no matter how many rows it has seen, and it is solvable at any
// point, so you can read off a current best estimate mid-stream.
//
// There is no step function or resumable job object here on purpose: per-row
// cost is uniform and the final solve is O(n^2), so your own loop is the state
// machine and progress is just acc.nrows / total. Measured cost of chunking is
// zero above a batch of ~64 rows (docs/OLS_RESULTS.md).

import "core:fmt"
import "core:math"
import "core:time"
import blas "../../src/blas"

N :: 4 // predictors, including the intercept
M :: 50_000 // observations

FRAME_BUDGET_MS :: 2.0 // slice of a frame we are willing to spend

RNG :: struct {
	state: u64,
}

rng_uniform :: proc(r: ^RNG) -> f64 {
	r.state = r.state * 6364136223846793005 + 1442695040888963407
	return f64((r.state >> 11) & 0x1FFFFFFFFFFFFF) / f64(0x1FFFFFFFFFFFFF)
}

rng_normal :: proc(r: ^RNG) -> f64 {
	u1 := rng_uniform(r)
	for u1 < 1e-12 {
		u1 = rng_uniform(r)
	}
	u2 := rng_uniform(r)
	return math.sqrt_f64(-2.0 * math.ln_f64(u1)) * math.cos_f64(2.0 * math.PI * u2)
}

BETA_TRUE := [N]f64{1.5, -0.8, 2.25, 0.4}

// One observation: intercept, then N-1 predictors, and the response.
make_sample :: proc(r: ^RNG, row: []f64) -> f64 {
	row[0] = 1.0
	for j in 1 ..< N {
		row[j] = rng_normal(r)
	}
	y := 0.0
	for j in 0 ..< N {
		y += row[j] * BETA_TRUE[j]
	}
	return y + 0.25 * rng_normal(r)
}

report :: proc(label: string, acc: ^blas.Ols_Accum) {
	beta: [N]f64
	err := blas.ols_accum_solve(acc, beta[:])
	if err != .None {
		fmt.printf("  %-22s %v\n", label, err)
		return
	}
	rss := blas.ols_accum_rss(acc)
	fmt.printf("  %-22s beta = [", label)
	for j in 0 ..< N {
		if j > 0 {fmt.printf(", ")}
		fmt.printf("%7.4f", beta[j])
	}
	// RSS over nrows observations; sqrt(RSS/nrows) is the typical residual.
	fmt.printf("]  rms = %.4f\n", math.sqrt_f64(rss / f64(acc.nrows)))
}

// Rows absorbed between clock reads. Bounds how far past the deadline a
// frame can run: at most this many rows of work. Measured cost of batching is
// zero above ~64 rows (docs/OLS_RESULTS.md), so this is free, and a tick read
// is tens of nanoseconds against tens of microseconds of work per interval.
// Lower it if you are targeting hardware slow enough that the overshoot
// matters; the correctness of the result does not depend on it.
BUDGET_CHECK_ROWS :: 512

// absorb_within folds rows until the time budget is spent or the data runs
// out, whichever comes first. Returns whether every row has been absorbed.
//
// This is the whole of "spread the work across frames" -- there is no step
// function or job object in the library because this loop is the state machine
// and it belongs to the caller, who is the only one who knows the budget.
absorb_within :: proc(
	acc: ^blas.Ols_Accum,
	x: []f64,
	y: []f64,
	m: int,
	budget_ms: f64,
) -> (
	done: bool,
	err: blas.Ols_Error,
) {
	start := time.tick_now()
	for acc.nrows < m {
		take := min(BUDGET_CHECK_ROWS, m - acc.nrows)
		if _, e := blas.ols_accum_rows(acc, x[acc.nrows * N:], N, y[acc.nrows:], take);
		   e != .None {
			return false, e
		}
		if time.duration_milliseconds(time.tick_since(start)) >= budget_ms {
			break
		}
	}
	return acc.nrows >= m, .None
}

// ============================================================================
// Regime 1: resident data, chunked across frames
// ============================================================================

chunked_across_frames :: proc() {
	fmt.println("--- Regime 1: 50,000 resident rows, chunked across frames ---")

	rng := RNG{2024}
	x := make([]f64, M * N);defer delete(x)
	y := make([]f64, M);defer delete(y)
	for i in 0 ..< M {
		y[i] = make_sample(&rng, x[i * N:][:N])
	}

	scratch := make([]f64, blas.ols_accum_scratch(N));defer delete(scratch)
	acc: blas.Ols_Accum
	if e := blas.ols_accum_init(&acc, N, scratch); e != .None {
		fmt.printf("init failed: %v\n", e)
		return
	}

	fmt.printf("  budget %.1f ms/frame, spent against the clock\n", FRAME_BUDGET_MS)

	frame := 0
	for acc.nrows < M {
		// Spend the budget rather than a precomputed row count. A rows-per-frame
		// constant tuned on one machine overshoots the budget on a slower one
		// and wastes it on a faster one; the Steam hardware survey spans both
		// directions. Reading the clock instead needs no calibration and is
		// correct on hardware this was never run on.
		if _, err := absorb_within(&acc, x, y, M, FRAME_BUDGET_MS); err != .None {
			fmt.printf("  bad data at row %d: %v\n", acc.nrows, err)
			return
		}
		frame += 1

		progress := f64(acc.nrows) / f64(M)
		fmt.printf("  frame %2d  %5.1f%%  ", frame, progress * 100)
		bars := int(progress * 20)
		for i in 0 ..< 20 {
			fmt.printf(i < bars ? "#" : ".")
		}
		fmt.printf("  %d/%d rows\n", acc.nrows, M)
	}

	report("final", &acc)
	fmt.printf("  true                   beta = [")
	for j in 0 ..< N {
		if j > 0 {fmt.printf(", ")}
		fmt.printf("%7.4f", BETA_TRUE[j])
	}
	fmt.println("]")
	fmt.println()
}

// ============================================================================
// Regime 2: rows arrive over time and are never stored
// ============================================================================

streaming :: proc() {
	fmt.println("--- Regime 2: streaming, one row at a time, nothing retained ---")

	rng := RNG{7}
	scratch := make([]f64, blas.ols_accum_scratch(N));defer delete(scratch)
	acc: blas.Ols_Accum
	blas.ols_accum_init(&acc, N, scratch)

	fmt.printf(
		"  resident: %d bytes, and it stays that way for any number of rows\n",
		blas.ols_accum_scratch(N) * 8,
	)

	row: [N]f64
	checkpoints := []int{2, 10, 100, 1_000, 20_000}
	next := 0

	for i in 1 ..= 20_000 {
		yv := make_sample(&rng, row[:])

		// One sample in, immediately folded into the triangle. `row` and `yv`
		// can be reused or discarded straight after this call.
		_, err := blas.ols_accum_rows(&acc, row[:], N, []f64{yv}, 1)
		if err != .None {
			fmt.printf("  rejected sample %d: %v\n", i, err)
			continue
		}

		if next < len(checkpoints) && i == checkpoints[next] {
			report(fmt.tprintf("after %d rows", i), &acc)
			next += 1
		}
	}
	fmt.println()
}

// ============================================================================
// Bad data does not poison the fit
// ============================================================================

bad_data :: proc() {
	fmt.println("--- Bad samples are rejected, not absorbed ---")

	rng := RNG{99}
	scratch := make([]f64, blas.ols_accum_scratch(N));defer delete(scratch)
	acc: blas.Ols_Accum
	blas.ols_accum_init(&acc, N, scratch)

	row: [N]f64
	rejected := 0
	for i in 0 ..< 500 {
		yv := make_sample(&rng, row[:])

		// Simulate a sensor glitch.
		if i % 97 == 0 {
			row[2] = math.nan_f64()
		}

		if _, err := blas.ols_accum_rows(&acc, row[:], N, []f64{yv}, 1); err != .None {
			rejected += 1
		}
	}

	fmt.printf("  %d samples offered, %d rejected, %d absorbed\n", 500, rejected, acc.nrows)
	report("still finite", &acc)
	fmt.println()
}

main :: proc() {
	fmt.println("=== Least squares across frames ===")
	fmt.println()
	chunked_across_frames()
	streaming()
	bad_data()
}
