# OLS Results — Verification and Measurement

Step 7 of `docs/OLS_PLAN.md`. Reports what was measured, what matched, and
what was **not** verified.

Machine: the development container this was built in (Linux x86-64).
Compiler: Odin `dev-2026-08-nightly:902106f`, `-o:speed`.
All figures below are measured on that machine. They are not portable
constants; re-measure on your target.

## Bugs found and fixed before the new work

Found by reading the code and confirmed with a probe program before changing
anything. Each is a silent wrong answer, not a crash. All five now have
regression tests in `tests/test_lstsq.odin`.

| # | Site | Symptom | Cause |
|---|---|---|---|
| 1 | `dnrm2` | returned **0.0** for any vector with norm below ~1e-160 | Two of the four Blue's-algorithm constants had the wrong sign in the exponent: `tsml` computed as `2^510` instead of `2^-511`, `ssml` as `2^-484` instead of `2^537`. Every value took the "tiny" branch and was scaled into underflow. |
| 2 | `dgemv` | **silent no-op** on any row-major matrix with `m > n` | Validated `lda >= max(1,m)`, the Fortran column-major rule. In row-major the bound is `n`. A tall design matrix — the shape OLS always has — was rejected and the routine returned without touching `y`. |
| 3 | `dger` | same silent no-op, same cause | as above |
| 4 | `dlarf` | **did nothing** for tall `C` | It is built on `dgemv` + `dger`, so it inherited 2 and 3. Applying a Householder reflector to a tall matrix was a no-op. |
| 5 | `dgeqrf` | produced `Inf` on data with entries above ~1e154 | Summed squares directly to get the column norm. Also present in `dlarfg`, which computed `sqrt(alpha*alpha + xnorm*xnorm)`. |

Bug 4 was latent rather than active: `dgeqrf` open-coded its own reflector
application instead of calling `dlarf`. Routing `dgeqrf` through
`dlarfg`/`dlarf` fixed bug 5 and removed that duplication, so `dlarf` is now
on the live path and covered by the QR tests.

`dtrsv` carried a comment claiming it assumed column-major storage and needed
Upper/Lower swapped for row-major. That was wrong — the algorithm is written
in terms of `A(i,j)` and is correct as-is. The comment was corrected rather
than the code; acting on it would have introduced a bug.

## Measurement: dense solve, old vs new

`n = 5`. "old" is `examples/regression_example.odin:solve_ls_qr` verbatim,
which builds the explicit `m x m` `Q`. "new" is `ols_solve_dense`.
MiB is bytes the transform touches.

| m | old ms | new ms | speedup | old MiB | new MiB |
|---|---|---|---|---|---|
| 100 | 0.131 | 0.0046 | 28x | 0.1 | 0.005 |
| 500 | 3.515 | 0.0213 | 165x | 2.0 | 0.023 |
| 1 000 | 24.892 | 0.0449 | 555x | 7.7 | 0.046 |
| 2 000 | 1570.714 | 0.0898 | 17 484x | 30.7 | 0.092 |
| 4 000 | 5320.998 | 0.1844 | 28 858x | 122.5 | 0.183 |
| 8 000 | 28814.390 | 0.3698 | 77 916x | 489.1 | 0.366 |

The old column is superlinear past `m = 1000` because 489 MiB of `Q` stops
fitting anywhere useful in the cache hierarchy; those figures vary run to run
(an earlier isolated run gave 16.7 s at `m = 8000` rather than 28.8 s). The
new column is stable and linear in `m`. The speedup ratios are therefore
order-of-magnitude statements, not precise constants — the load-bearing fact
is that one path is quadratic in `m` and the other is linear.

**Criterion 5 (dense at `m = 8000, n = 5` under one 60 Hz frame): met.**
0.3698 ms against a 16.67 ms budget, 45x headroom.

## Measurement: where the frame budget actually breaks

New dense path, `n = 5`:

| m | ms | frames at 60 Hz |
|---|---|---|
| 10 000 | 0.490 | 0.03 |
| 100 000 | 6.791 | 0.41 |
| 1 000 000 | 156.326 | 9.38 |
| 4 000 000 | 680.250 | 40.81 |

A single resident solve fits in one frame up to roughly `m = 250 000`.
Frame-spreading is only needed above that. The timings include copying the
input, because `ols_solve_dense` destroys it.

## Measurement: accumulator

| m | ms | Mrows/s | resident bytes | frames |
|---|---|---|---|---|
| 10 000 | 1.134 | 8.82 | 336 | 0.07 |
| 100 000 | 11.324 | 8.83 | 336 | 0.68 |
| 1 000 000 | 110.700 | 9.03 | 336 | 6.64 |
| 4 000 000 | 472.991 | 8.46 | 336 | 28.37 |

Throughput is flat at ~8.8 M rows/s and resident memory does not move.

**Criterion 6 (accumulator memory independent of `m`): met.** 336 bytes at
every `m` above. Against `n`, holding `m` fixed:

| n | 2 | 5 | 10 | 25 | 50 | 100 |
|---|---|---|---|---|---|---|
| bytes | 96 | 336 | 1 056 | 5 616 | 21 216 | 82 416 |

`O(n^2)`, as designed.

## Measurement: does chunking across frames cost anything?

`m = 1 000 000`, `n = 5`, same total work split into different batch sizes:

| rows per batch | total ms |
|---|---|
| 1 | 114.637 |
| 64 | 110.612 |
| 4 096 | 110.598 |
| 1 000 000 | 110.654 |

Chunking is free above a batch of ~64, and costs 3.6% even in the degenerate
one-row-per-call case. This is what justifies having no step function or phase
machine: the caller can pick any batch size that fits their budget and pay
nothing for the choice.

Sizing a batch from a frame budget, using the measured 8.8 M rows/s: a 2 ms
slice of a 16.67 ms frame absorbs about 17 600 rows. Measure the rate on your
own target rather than reusing that number.

## Correctness verification

Ground truth is `numpy.linalg.lstsq` (LAPACK) on the repository's own real
data, `examples/X_initial.csv` + `y_initial.csv` (`20 x 5`, `cond = 2.3265`).

| Criterion | Result |
|---|---|
| 1. Both transforms match ground truth, `beta` and `RSS`, tol 1e-12 | met |
| 2. Dense and accumulator agree with each other, tol 1e-10 | met (checked on an independent `400 x 7` problem, and both against a residual recomputed from the untouched data) |
| 3. Accumulator independent of batch size, chunks 1 / 3 / 7 / all | met, and **bit-identical** — the same rotations happen in the same order regardless of grouping, so the test asserts exact equality rather than a tolerance |
| 4. Every boundary policy provoked by a test | met — `Scratch_Too_Small`, `Invalid_Dimension` (`n<1`, `ldx<n`), `Not_Enough_Rows` (streaming and `m<n`), `Non_Finite_Input` (NaN mid-batch and Inf in `y`), `Rank_Deficient` (duplicated column, both paths), `count == 0` no-op |
| 5. Dense `m=8000` inside one frame | met, 0.3698 ms |
| 6. Accumulator memory flat in `m` | met, 336 bytes |

The NaN test also checks recovery: after a batch aborts on row 4, the
accumulator still holds exactly 4 rows, absorbing the remaining good rows
succeeds, and the resulting `beta` is finite.

Full suite: 21 tests, 0 failures (13 pre-existing, 8 new).

## What was NOT verified

- **Only one real dataset exists** (`20 x 5`). Everything at larger `m` uses
  synthetic data generated to match its character — intercept column of exact
  `1.0`, remaining columns `~N(0,1)`. If real workloads are differently
  conditioned, the accuracy results may not carry over. This is the weakest
  part of the evidence.
- **Accuracy was not measured on ill-conditioned input.** `cond = 2.3265` is
  benign. The Givens accumulator and Householder QR are both backward stable
  in theory, but the plan's disproof condition — "Transform B materially less
  accurate than Transform A" — was only tested at low condition numbers.
- **No measurement on target hardware.** All figures come from one
  container on one machine, single run per configuration, no variance
  reported. The old-path timings visibly drift between runs.
- **`n > 100` was not measured.** The design assumes small `n`; the
  accumulator's per-row `O(n^2)` cost makes it the wrong machine for large
  `n`, but where the crossover sits was not established.
- **Not multithreaded.** See `issues/004-parallel-accumulation.md`.
- **`delrows` / `delcolsq` were not revisited.** They are outside this
  subsystem and keep their existing test coverage.
