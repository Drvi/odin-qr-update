# OLS Results — Verification and Measurement

Step 7 of `docs/OLS_PLAN.md`. Reports what was measured, what matched, and
what was **not** verified.

Machine: `Intel(R) Xeon(R) @ 2.80GHz`, 4 vCPU, 1 thread/core, **33 MiB L3**,
AVX-512 present (`avx512f/dq/cd/bw/vl/vnni`). Linux x86-64 container.
Compiler: Odin `dev-2026-08-nightly:902106f`, `-o:speed`. Single-threaded.

**This is a server part and it is not a representative target.** Per the Steam
hardware survey, the modal gaming machine is a 6-core (27.52%) or 8-core
(27.85%) consumer CPU; 21.12% of Intel parts report 2.3-2.69 GHz. So the clock
here is ordinary, but 33 MiB of L3 is generous against consumer parts, and most
consumer gaming CPUs have no AVX-512 at all. Numbers measured here on anything
cache-resident are optimistic.

Because of that, the section below establishes **what each path is bound by**
before quoting any absolute figure, and the shipped example takes no measured
constant from this document — it spends a time budget against the clock instead.

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

## Measurement: what each path is bound by

This is the measurement that decides how far any of the others travel. Input
working set swept from inside L1 to far past the 33 MiB L3, `n = 5`:

| input working set | accumulator | dense |
|---|---|---|
| 47 KiB | 8.71 Mrows/s | 24.90 Mrows/s |
| 234 KiB | 8.85 | 23.67 |
| 938 KiB | 8.77 | 22.89 |
| 4.6 MiB | 8.80 | 16.99 |
| 18 MiB | 8.79 | 15.90 |
| 92 MiB | 8.63 | 6.48 |
| 366 MiB | 8.62 | 6.40 |

**The accumulator is flat to within 2.6% across an 8000x span of working set.**
It sustains 0.79 GFLOP/s and 0.39 GiB/s — far below anything this CPU can
saturate — so it is neither bandwidth-bound nor FLOP-bound. It is latency-bound
on the dependent chain of Givens rotations, with 336 bytes of hot state that
lives in L1 on any machine. That makes it the portable one: it should scale
with clock and IPC, and a consumer part with a quarter of this L3 has nothing
to lose here. 0.39 GiB/s is a rounding error against any DDR4 system.

**The dense path falls 3.9x** once the input stops fitting in L3. It streams
`X` and is genuinely memory-sensitive, so its numbers are the ones that will
not survive a move to a smaller-cache machine.

### Where the frame budget breaks, restated

Earlier drafts of this document read a threshold of `m = 250 000` off the
`m = 100 000` timing. That row is 4.6 MiB — comfortably L3-resident here — so
the figure was an in-cache extrapolation and too optimistic. Using the
out-of-cache rate of 6.4 Mrows/s instead, one 16.67 ms frame buys about
**107 000 rows** on this machine, and less on a machine with less cache or a
lower clock.

| m | ms | frames at 60 Hz |
|---|---|---|
| 10 000 | 0.490 | 0.03 |
| 100 000 | 6.791 | 0.41 |
| 1 000 000 | 156.326 | 9.38 |
| 4 000 000 | 680.250 | 40.81 |

Treat `m ~ 100 000` as the order of magnitude at which the resident path stops
fitting a frame here, and calibrate rather than inheriting it. The timings
include copying the input, because `ols_solve_dense` destroys it.

## Measurement: cost against n

`m = 200 000`. The accumulator is `O(n^2)` per row by construction, but the
inner `drot` vectorizes better as it lengthens, so realised cost grows more
slowly than `n^2`:

| n | Mrows/s | GFLOP/s | state bytes | us per 1000 rows |
|---|---|---|---|---|
| 2 | 22.60 | 0.41 | 96 | 44 |
| 4 | 10.72 | 0.64 | 240 | 93 |
| 8 | 5.68 | 1.23 | 720 | 176 |
| 16 | 2.73 | 2.23 | 2 448 | 366 |
| 32 | 1.16 | 3.68 | 8 976 | 862 |
| 64 | 0.43 | 5.33 | 34 320 | 2 340 |

`n = 2 -> 64` is a 32x rise in `n` and would be 693x on flop count alone, but
throughput drops only 52x, because achieved GFLOP/s climbs 13x over the same
range. At `n = 64` the state is 34 KiB and starts to leave L1, which is where
the assumption that `n` is small begins to pay for itself.

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

### Sizing a batch without inheriting this machine

Do not convert a frame budget into a row count with a rows-per-second constant
from this document. Such a constant is wrong in both directions on hardware it
was not measured on — it overruns the budget on a slower part and leaves the
budget unspent on a faster one — and the survey spread is wide enough that both
happen in practice.

`examples/incremental/main.odin` spends the budget against the clock instead:
absorb `BUDGET_CHECK_ROWS` (512) rows, check elapsed time, repeat until the
budget is gone. This needs no calibration, no tuning constant, and no knowledge
of the target. It is correct on hardware this was never run on, which is the
property that actually matters.

Its two costs are both bounded by the measurements above: a tick read every 512
rows against ~58 us of work per interval here (proportionally more on a slower
machine, so the relative overhead only falls), and an overshoot of at most 512
rows past the deadline. Batching at 512 is free per the table above.

An earlier version of that example carried `MEASURED_ROWS_PER_SEC ::
8_800_000.0` taken from this machine. That was the wrong shape of solution and
has been removed.

## Measurement: model iteration

The intended use is experimentation — trying predictors against data, both
changing. The waste to remove is re-reading the rows for every candidate.

Scoring every non-empty subset of `p = 10` candidate predictors (1023 models).
"naive" gathers the subset's columns and re-accumulates all `m` rows each time;
"cached" accumulates the 10-predictor triangle once and then calls
`ols_accum_select` per subset. Both give the same coefficients — the
`select == direct fit` test asserts it:

| m | naive | cached | speedup |
|---|---|---|---|
| 1 000 | 119.9 ms | 0.866 ms | 139x |
| 10 000 | 1 218 ms | 2.815 ms | 433x |
| 100 000 | 8 117 ms | 23.0 ms | 353x |
| 1 000 000 | 88 946 ms | 224.6 ms | 396x |

(naive at `m >= 100 000` extrapolated from a 64-subset sample; the full run
takes minutes.)

The totals understate it, because nearly all of "cached" is the one
accumulation pass. Isolating the per-operation cost:

| p | `ols_accum_select` | `ols_accum_merge` |
|---|---|---|
| 4 | 0.119 us | 0.133 us |
| 8 | 0.346 us | 0.403 us |
| 12 | 0.747 us | 0.636 us |
| 16 | 1.290 us | 1.109 us |
| 24 | 2.758 us | 2.283 us |

So at `m = 1 000 000`, `p = 10`: the first model costs one pass (~220 ms), and
**every model after it costs about 0.5 us instead of 87 ms** — a marginal
improvement of roughly 170 000x. Both operations read only the `(p+1)^2`
triangle, so their cost depends on `p` alone and not on `m`; that is by
construction rather than by measurement, since neither routine touches the
data.

This is what makes exhaustive search practical: 1023 models at `m = 1 000 000`
in 0.5 ms of marginal work, against 89 seconds of refitting.

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

Full suite: **29 tests, 0 failures** (13 pre-existing, 16 new).

Tests added for model iteration and for gaps this document previously listed
as unverified:

| Test | What it pins down |
|---|---|
| `select all 31 subsets` | Every non-empty subset of the real 20x5 data, `beta` and `RSS`, against numpy. Catches any error in the select transform for any subset shape. |
| `select == direct fit` | Selecting from the superset triangle equals gathering those columns from raw data and fitting them, including reordered `keep`. |
| `merge` | Split/merge equals all-at-once; reverse order agrees; 3-way pairwise agrees; an empty accumulator is an identity element. |
| `merge then select` | The combined loop, checked against numpy for all 31 subsets. |
| `select/merge boundaries` | Short `keep`, out-of-range index, negative index, duplicate index, aliasing on both routines, `n` mismatch, widening. Plus: a collinear sub-model reports `.Rank_Deficient` while a good sibling still solves. |
| `apply_qt vs explicit Q` | The core optimization against the `dorgqr` path it replaced: max difference **8.88e-16**, the factored matrix is restored byte-for-byte, and the norm is preserved. |
| `conditioning 1e2..1e8` | Both paths against numpy across four condition numbers (table above). |
| `degenerate shapes` | `n = 1`; `m == n`; exact fits give negligible `RSS`; duplicating every row leaves `beta` unmoved and doubles `RSS`; rows fed in reverse order give the same fit. |

One of these initially failed, and the test was wrong rather than the code: it
asserted `RSS == 0.0` exactly for an exact fit, but the triangle corner carries
~8.9e-16 of rounding residue which squares to 7.9e-31. The assertion is now
relative to `||y||^2`.

## What was NOT verified

- **Only one real dataset exists** (`20 x 5`). Everything at larger `m` uses
  synthetic data generated to match its character — intercept column of exact
  `1.0`, remaining columns `~N(0,1)`. If real workloads are differently
  conditioned, the accuracy results may not carry over. This is the weakest
  part of the evidence.
- ~~Accuracy was not measured on ill-conditioned input.~~ **Now closed.**
  `test_ols_conditioning` fits designs built as `U*diag(s)*V^T` with
  geometrically spaced singular values, giving `cond(X)` of `1e2` to `1e8`, and
  compares both paths against numpy. Worst relative coefficient error:

  | cond(X) | accumulator | dense | apart |
  |---|---|---|---|
  | 1e2 | 2.22e-15 | 1.33e-15 | 3.55e-15 |
  | 1e4 | 2.71e-13 | 6.75e-14 | 2.03e-13 |
  | 1e6 | 3.81e-11 | 2.75e-12 | 3.53e-11 |
  | 1e8 | 2.80e-08 | 1.60e-07 | 1.32e-07 |

  Both track the `cond(X) * eps` bound expected of a backward-stable solver.
  **The plan's disproof condition is refuted:** the Givens accumulator is not
  materially less accurate than Householder QR, and at `cond = 1e8` it is
  actually the better of the two. Still untested above `1e8`, where the
  `rcond` rank check starts rejecting.
- **Nothing was measured on representative target hardware.** Every figure
  comes from one server-class container — 33 MiB L3, AVX-512, 4 vCPU — with a
  single run per configuration and no variance reported. The old-path timings
  visibly drift between runs. What has been established is the *shape*: the
  accumulator is latency-bound with a 336-byte working set and flat across an
  8000x sweep, so it should port predictably; the dense path is memory-bound
  with a measured 3.9x cache cliff, so it should not. Neither claim has been
  checked on a consumer part, and the accumulator's portability argument in
  particular is an inference from the boundedness measurement, not a
  measurement on other hardware.
- **No AVX-512-free measurement.** Most consumer gaming CPUs lack it. The
  inner `drot` is where any vectorisation would land, and the `n`-sweep shows
  achieved GFLOP/s climbing 13x with `n`, which suggests the compiler is
  vectorising it — so a machine without AVX-512 may behave differently,
  especially at larger `n`. Not tested; no `-microarch` sweep was run.
- **`n > 64` was not measured.** The design assumes small `n`. At `n = 64` the
  accumulator state is 34 KiB and is leaving L1, which is where the small-`n`
  assumption starts to earn its keep, but the crossover was not pinned down.
- **Single-threaded only, and core count was not exploited.** The survey's
  modal machine has 6-8 physical cores; this uses one. See
  `issues/004-parallel-accumulation.md`.
- **Not multithreaded.** See `issues/004-parallel-accumulation.md`.
- **`delrows` / `delcolsq` were not revisited.** They are outside this
  subsystem and keep their existing test coverage.
