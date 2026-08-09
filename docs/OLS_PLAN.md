# Plan: Least-Squares (OLS) Solving

Tier 2 (new subsystem). Written before implementation, per
`context/data-oriented-design.md`.

## 1. Frame it

**Problem.** Solve `min ||X*beta - y||` for `beta`, where `X` is `m x n`
real, `m >= n`. Callers may need the work spread across several frames of a
real-time loop, with a progress figure they can display.

**Why.** The repository already computes QR factorizations and QR updates but
has no least-squares entry point. The only worked example
(`examples/regression_example.odin:solve_ls_qr`) open-codes one, and that
open-coded path is quadratic in `m` (see §3).

**Limits.** Real `f64` only. Dense `X`. No rank-revealing pivoting, so
rank-deficient systems are rejected rather than solved in a minimum-norm
sense.

**Fallback.** If the transform below does not hold up, the existing
`dgeqrf` + `dorgqr` + `dtrsv` path still works and stays in the library.

## 2. Get the data

Measured from `examples/X_initial.csv`, `y_initial.csv` — the only real input
in the repository:

- Shape `20 x 5`, dense, row-major, `f64`.
- Column 0 is exactly `1.0` in all 20 rows (intercept).
- Remaining entries approximately `N(0,1)`; observed range `[-4.0008, 2.0792]`.
- `cond(X) = 2.3265` — well conditioned.
- Ground truth (numpy `lstsq`): `beta = [2.04043827528268, -1.48909548321255,
  0.535156175896668, -0.279793128956933, 1.437575944628262]`,
  `RSS = 3.016933013710126`. The repository's current output matches this to
  `7.8e-16`.

**Five data questions.**

1. *Shape, volume, source?* Tall and thin: `m` rows of `n` predictors, `m >> n`.
   The one real sample is `20 x 5`. The request to spread work over frames is
   itself data: it only makes sense if `m` is large enough to blow a frame
   budget, so the design must hold for `m` in the thousands-to-millions with
   `n` small.
2. *Most frequent values / distribution?* Dense `f64`, no exploitable zeros
   except the intercept column. No structure worth specializing on.
3. *Acceptable ranges / out-of-range?* `NaN`/`Inf` in samples is a live risk
   for measured input. Collinear predictors (rank deficiency) is a live risk
   whenever a caller adds a redundant feature. `m < n` is *guaranteed* to
   occur transiently in any streaming use. All three need stated policy.
4. *Stable vs changing?* `n` is fixed for a given fit. `m` grows, possibly
   without bound, possibly arriving over time.
5. *What does the solution touch unnecessarily?* See §3 — the current path
   materializes an `m x m` matrix that the answer does not depend on.

**ASSUMPTION: `n` is small (order 1-100) while `m` may be unbounded** —
affects the choice to keep an `(n+1)^2` accumulator resident and to treat
per-row cost `O(n^2)` as acceptable. If `n` were also large, row-at-a-time
accumulation would be the wrong machine.

**ASSUMPTION: callers want a single coefficient vector per fit, not multiple
right-hand sides** — affects the decision to take one `y` rather than a
block. Filed as `issues/001-multiple-rhs.md`.

**The platform is consumer hardware, not this development machine.** Per the
Steam hardware survey the modal target is a 6-core (27.52%) or 8-core (27.85%)
consumer CPU, commonly clocked 2.3-2.69 GHz, typically without AVX-512 and
with materially less L3 than the 33 MiB server part these measurements were
taken on. This is a real platform constraint, so it is stated here rather than
discovered later: **no rows-per-second constant measured here may be baked into
shipped code.** Budgets are spent against the clock, and §3 establishes what
each transform is bound by before quoting any absolute figure.

## 3. State the cost

Measured on this machine, `n = 5`, `-o:speed`, current
`solve_ls_qr` path (`examples/regression_example.odin`):

| m | bytes touched | full solve | `dgeqrf` alone |
|---|---|---|---|
| 100 | 0.1 MiB | 0.126 ms | 0.003 ms |
| 500 | 2.0 MiB | 3.551 ms | 0.014 ms |
| 1000 | 7.7 MiB | 17.689 ms | 0.032 ms |
| 2000 | 30.7 MiB | 420.126 ms | 0.067 ms |
| 4000 | 122.5 MiB | 2854.562 ms | 0.131 ms |
| 8000 | 489.1 MiB | 16745.719 ms | 0.265 ms |

One frame at 60 Hz is 16.67 ms.

The dominant expense is not the factorization. It is `dorgqr` building an
explicit `m x m` `Q` (489 MiB at `m = 8000`) and then a second `m x m` pass to
form `Q^T*b`. Both are quadratic in `m`; the factorization is linear in `m` and
is 63,000x cheaper at `m = 8000`.

The platform paying this cost is main memory bandwidth and the frame budget.

## 4. Design the transform

Simplification question 1 — *can we skip this entirely?* — applies to `Q`.
`beta` depends on `Q` only through the `n`-vector `Q^T*y`. `Q` is a product of
`n` Householder reflectors already stored in the factored `X`; applying them
to `y` costs `O(m*n)` and needs no extra storage. The `m x m` matrix is pure
waste. Two transforms follow, for the two regimes that actually occur.

### Transform A — resident data

    IN   X : f64[m*ldx]  row-major, ldx >= n, m >= n >= 1, caller-owned
         y : f64[m]      caller-owned
         scratch : f64[ols_dense_scratch(n)]  caller-owned
    OUT  beta : f64[n]   caller-owned, written only on success
         rss  : f64      residual sum of squares, ||X*beta - y||^2
    MUTATES X and y in place (both are destroyed; documented, not incidental)
    LIFETIME all buffers caller-owned for the duration of the call only

    factor  : X -> R (upper n x n) + reflectors + tau     [dgeqrf, 2mn^2 - 2n^3/3]
    applyQt : y -> Q^T*y                                  [dlarf x n, 4mn]
    rank    : diag(R) -> accept / .Rank_Deficient         [n]
    solve   : R*beta = (Q^T*y)[0:n]                       [dtrsv, n^2]
    rss     : sum of squares of (Q^T*y)[n:m]              [m-n]

Peak extra memory: `2n` f64. Independent of `m`.

### Transform B — streaming or frame-chunked data

Keeps the `R` factor of the augmented matrix `[X | y]`, which is
`(n+1) x (n+1)` upper triangular, and folds rows into it with Givens
rotations. After absorbing every row:

    tri[0:n, 0:n] = R of X        tri[0:n, n] = Q^T*y      tri[n, n] = +/-||resid||

so `beta` and `RSS` both fall out of one array with no second pass over the
data.

    IN   tri : f64[(n+1)^2] + row scratch f64[n+1]   one caller-owned block
         X   : f64[count*ldx], y : f64[count]        one batch, read-only
    OUT  tri updated in place; acc.nrows += absorbed
    MUTATES tri only. Input batch is not modified.
    LIFETIME tri lives across frames and is owned by the caller.

    absorb_rows : (tri, batch) -> tri                [3n(n+1) per row]
    solve       : tri -> beta                        [n^2, once at the end]

Peak memory: `(n+1)^2 + (n+1)` f64, **independent of `m`**. The caller never
has to hold all `m` rows.

Progress is `acc.nrows / total`. There is no phase machine, no resumable
state object, and no step function: per-row cost is uniform and the final
solve is `O(n^2)`, so the caller's own loop is the state machine. This is the
main thing the simplification pass removed — see §5.

### Transform C — model iteration (added after the use case was clarified)

The dominant operation is not a single solve. It is *fit, tweak, re-fit*:
experimenting with which predictors and which observations. The waste to remove
is re-reading the `m` rows for every candidate. Two properties of the triangle
make that avoidable, because `T'T = A'A` for `A = [X | y]`:

1. Selecting columns commutes with the Gram product, so the `R` factor of any
   subset of the predictors is derivable from the triangle alone.
2. The `p+1` rows of the triangle span the same row space as the data behind
   them, so triangles over disjoint row sets combine by folding one into the
   other.

<!-- -->

    ols_accum_select : (superset tri, keep[k]) -> sub-model tri   [O(p*k^2)]
    ols_accum_merge  : (tri_a, tri_b)          -> tri_a+b         [O(p^3)]

Both are **independent of m**: they read only the `(p+1)^2` triangle. The
intended pattern is therefore *decide the candidate predictors up front,
accumulate all of them once, then subset freely* — which is also why there is
no "add a predictor" operation. Adding one genuinely needs the data, whereas
selecting from a superset does not, so the superset is accumulated once and the
question never arises.

Merge covers the other axis: accumulate one triangle per data segment, then any
union of segments is `O(p^3)`.

`ols_accum_merge` requires the two row sets to be **disjoint**. Overlap
double-counts silently and cannot be detected from the triangles, so it is the
caller's invariant, stated in the contract.

Removing observations is *not* supported; see
`issues/006-downdating-observations.md`.

### Boundary policy (explicit, at every input)

| Condition | Policy |
|---|---|
| `n < 1`, `ldx < n`, `count < 0`, short slice | reject, `.Invalid_Dimension`, no mutation |
| scratch smaller than the size function says | reject, `.Scratch_Too_Small`, no mutation |
| `NaN`/`Inf` in a row | absorb rows before it, stop, return index + `.Non_Finite_Input` |
| fewer rows absorbed than `n` | reject at solve, `.Not_Enough_Rows` |
| `min\|diag(R)\| <= rcond * max\|diag(R)\|` | reject at solve, `.Rank_Deficient` |

A poisoned row is rejected *before* it enters `tri`, because a single `NaN`
propagates through every subsequent rotation and cannot be removed. Rows are
validated one at a time immediately before absorption, so the row is already
in cache, the accumulator is never left invalid, and the caller learns which
row was bad.

## 5. Simplification pass

| Question | Applied |
|---|---|
| Skip entirely? | **Yes — dropped the explicit `m x m` `Q`.** Removes both quadratic passes and 489 MiB at `m = 8000`. |
| Skip entirely? | **Yes — dropped the phase/step state machine.** Uniform per-row cost plus an `O(n^2)` tail means `rows_done / total` is the whole progress model. No `Ols_Job`, no phase enum, no `step()`. |
| Compute once? | `Q^T*y` folded into the same triangle as `R`, so `RSS` needs no second pass over the data. |
| Reduce frequency? | Rank check runs once per solve, not per row. |
| Different representation? | **Yes — augmented `[X | y]` triangle.** One array carries `R`, `Q^T*y` and the residual norm together. |
| Further constraints? | `n` small (assumption above) is what makes an `(n+1)^2` resident accumulator the right machine. |

Also dropped from the first draft of this work, as speculative generality with
no present requirement: weighted least squares, ridge regularization,
covariance / standard errors / R-squared, accumulator merging for
multithreading, allocator plumbing and `init`/`destroy` pairs. Filed under
`issues/`.

## 6. Define done

1. Both transforms reproduce the numpy ground truth on the real `20 x 5` data
   to `<= 1e-12`, coefficients and `RSS`.
2. Transform A and Transform B agree with each other to `<= 1e-10` on the
   same input.
3. Transform B's result does not depend on how rows are split into batches:
   chunk sizes 1, 3, 7 and all-at-once give bit-comparable results.
4. Every boundary policy in §4 has a test that provokes it.
5. Measured cost of Transform A at `m = 8000, n = 5` is under one 60 Hz frame.
6. Transform B's memory does not grow with `m`.

**Evidence that would disprove the approach:** if Transform B's Givens path
were materially less accurate than Transform A on the real data, or if its
measured per-row cost made the streaming regime slower than just buffering
rows and calling Transform A.

## 7. Verify

Recorded in `docs/OLS_RESULTS.md` after implementation: what was measured,
what matched, and what was not verified.
