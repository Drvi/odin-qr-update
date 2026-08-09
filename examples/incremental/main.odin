package incremental_example

// Fitting with the work spread across frames, and with data arriving over time.
//
// Two regimes, both shown below:
//
//   1. All the data is already in memory and you cannot afford one big solve
//      this frame. Absorb rows until the frame budget is spent.
//
//   2. Rows arrive over time (samples, telemetry, measurements) and you never
//      want to hold them all. Absorb each row as it shows up and drop it.
//
// The same Ols_Accum serves both. It holds (n+1)^2 + (n+1) f64 -- 240 bytes at
// n = 4 -- no matter how many rows it has seen, and it is solvable at any
// point, so a current estimate is always available mid-stream.
//
// There is no step function or resumable job object here on purpose: per-row
// cost is uniform and the final solve is O(n^2), so your own loop is the state
// machine and progress is just acc.nrows / total. Measured cost of chunking is
// zero above a batch of ~64 rows (docs/OLS_RESULTS.md).
//
// Everything runs under mem.panic_allocator with fixed arrays throughout, so
// any hidden allocation -- here or in the library -- aborts the program.

import "core:fmt"
import "core:math"
import "core:mem"
import "core:time"
import blas "../../src/blas"

N :: 4 // predictors, including the intercept
M :: 50_000 // observations held in memory for regime 1

FRAME_BUDGET_MS :: 2.0 // slice of a frame we are willing to spend

// Rows absorbed between clock reads. Bounds how far past the deadline a frame
// can run: at most this many rows of work. Batching is free above ~64 rows
// (docs/OLS_RESULTS.md), and a tick read is tens of nanoseconds against tens of
// microseconds of work per interval. Lower it when targeting hardware slow
// enough that the overshoot matters; results do not depend on it.
BUDGET_CHECK_ROWS :: 512

BETA_TRUE := [N]f64{1.5, -0.8, 2.25, 0.4}

// Fixed storage. Nothing here needs an allocator the caller did not choose.
data_x: [M * N]f64
data_y: [M]f64

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

// One observation: intercept, then N-1 predictors, and the response.
make_sample :: proc(r: ^RNG, row: []f64) -> f64 {
	row[0] = 1.0
	for j in 1 ..< N {
		row[j] = rnorm(r)
	}
	y := 0.0
	for j in 0 ..< N {
		y += row[j] * BETA_TRUE[j]
	}
	return y + 0.25 * rnorm(r)
}

report :: proc(rows: int, acc: ^blas.Ols_Accum) {
	beta: [N]f64
	if e := blas.ols_accum_solve(acc, beta[:]); e != .None {
		fmt.printf("    after %d rows: %v\n", rows, e)
		return
	}
	rss := blas.ols_accum_rss(acc)
	fmt.printf("    after %d rows: beta = [", rows)
	for j in 0 ..< N {
		if j > 0 {fmt.printf(", ")}
		fmt.printf("%.4f", beta[j])
	}
	fmt.printf("]  rms = %.4f\n", math.sqrt_f64(rss / f64(acc.nrows)))
}

// absorb_within folds rows until the time budget is spent or the data runs
// out. Returns whether everything has been absorbed.
//
// This is the whole of "spread the work across frames". There is no step
// function in the library because this loop is the state machine, and it
// belongs to the caller -- the only one who knows the budget. Spending against
// the clock rather than a precomputed row count is what keeps it correct on
// hardware it was never measured on.
absorb_within :: proc(acc: ^blas.Ols_Accum, m: int, budget_ms: f64) -> blas.Ols_Error {
	start := time.tick_now()
	for acc.nrows < m {
		take := min(BUDGET_CHECK_ROWS, m - acc.nrows)
		if _, e := blas.ols_accum_rows(
			acc,
			data_x[acc.nrows * N:],
			N,
			data_y[acc.nrows:],
			take,
		); e != .None {
			return e
		}
		if time.duration_milliseconds(time.tick_since(start)) >= budget_ms {
			break
		}
	}
	return .None
}

chunked_across_frames :: proc() {
	fmt.println("--- Regime 1: 50,000 resident rows, chunked across frames ---")

	rng := RNG{2024}
	for i in 0 ..< M {
		data_y[i] = make_sample(&rng, data_x[i * N:][:N])
	}

	scratch: [(N + 1) * (N + 1) + (N + 1)]f64
	acc: blas.Ols_Accum
	if e := blas.ols_accum_init(&acc, N, scratch[:]); e != .None {
		fmt.printf("  init failed: %v\n", e)
		return
	}

	fmt.printf("  budget %.1f ms/frame, spent against the clock\n", FRAME_BUDGET_MS)

	frame := 0
	for acc.nrows < M {
		if e := absorb_within(&acc, M, FRAME_BUDGET_MS); e != .None {
			fmt.printf("  bad data at row %d: %v\n", acc.nrows, e)
			return
		}
		frame += 1

		progress := f64(acc.nrows) / f64(M)
		fmt.printf("  frame %d: ", frame)
		bars := int(progress * 20)
		for i in 0 ..< 20 {
			fmt.printf(i < bars ? "#" : ".")
		}
		fmt.printf("  %.1f%%  %d/%d rows\n", progress * 100, acc.nrows, M)
	}

	report(acc.nrows, &acc)
	fmt.printf("    true:              beta = [")
	for j in 0 ..< N {
		if j > 0 {fmt.printf(", ")}
		fmt.printf("%.4f", BETA_TRUE[j])
	}
	fmt.println("]")
	fmt.println()
}

streaming :: proc() {
	fmt.println("--- Regime 2: streaming, one row at a time, nothing retained ---")

	rng := RNG{7}
	scratch: [(N + 1) * (N + 1) + (N + 1)]f64
	acc: blas.Ols_Accum
	blas.ols_accum_init(&acc, N, scratch[:])

	fmt.printf(
		"  resident: %d bytes, and it stays that way for any number of rows\n",
		len(scratch) * 8,
	)

	row: [N]f64
	yv: [1]f64
	checkpoints := [?]int{2, 10, 100, 1_000, 20_000}
	next := 0

	for i in 1 ..= 20_000 {
		yv[0] = make_sample(&rng, row[:])

		// One sample in, folded straight into the triangle. `row` and `yv` are
		// reused on the next iteration; the accumulator kept what it needed.
		if _, e := blas.ols_accum_rows(&acc, row[:], N, yv[:], 1); e != .None {
			fmt.printf("  rejected sample %d: %v\n", i, e)
			continue
		}

		if next < len(checkpoints) && i == checkpoints[next] {
			report(i, &acc)
			next += 1
		}
	}
	fmt.println()
}

bad_data :: proc() {
	fmt.println("--- Bad samples are rejected, not absorbed ---")

	rng := RNG{99}
	scratch: [(N + 1) * (N + 1) + (N + 1)]f64
	acc: blas.Ols_Accum
	blas.ols_accum_init(&acc, N, scratch[:])

	row: [N]f64
	yv: [1]f64
	rejected := 0
	for i in 0 ..< 500 {
		yv[0] = make_sample(&rng, row[:])

		// Simulate a sensor glitch.
		if i % 97 == 0 {
			row[2] = math.nan_f64()
		}

		if _, e := blas.ols_accum_rows(&acc, row[:], N, yv[:], 1); e != .None {
			rejected += 1
		}
	}

	fmt.printf("  500 samples offered, %d rejected, %d absorbed\n", rejected, acc.nrows)
	report(acc.nrows, &acc)
	fmt.println()
}

main :: proc() {
	// Any allocation from here on is a bug, in this file or in the library.
	context.allocator = mem.panic_allocator()
	context.temp_allocator = mem.panic_allocator()

	fmt.println("=== Least squares across frames ===")
	fmt.println()
	chunked_across_frames()
	streaming()
	bad_data()
}
