# Synthetic Regression Data

`src/synth` generates `(X, y)` whose answer is known in advance. Written to the
rules in `context/data-oriented-design.md`; this is the plan and the results in
one file, since the subsystem is small.

## Frame it

**Problem.** Produce design matrices and responses for exercising the
least-squares code, with the true coefficients chosen by the caller, so that a
wrong fit is unambiguously a fitting bug rather than a modelling mistake.

**Controls asked for.** Gaussian noise level, a correlation matrix over the
inputs, per-input variances, and polynomial inputs.

**Limits.** Real `f64`. Gaussian base variables and Gaussian response noise
only. Homoskedastic. No missing values, categorical predictors or
autocorrelation — see `issues/007-generator-scope.md`.

## The data, and the one representation that covers it

A **term** is a vector of exponents over the base variables; its value in a row
is the product of each base variable raised to its exponent:

| exponents | term | role |
|---|---|---|
| `{0,0,0}` | `1` | intercept |
| `{1,0,0}` | `b0` | linear |
| `{0,2,0}` | `b1^2` | quadratic |
| `{1,0,1}` | `b0*b2` | interaction |
| `{3,0,0}` | `b0^3` | cubic |

One flat `u8` table, `nterms x k`, row-major. That is the whole mechanism —
polynomial regression needs no separate path because it *is* ordinary OLS over
an expanded term set, which is how a caller uses it anyway. `synth_terms_poly`
fills the common univariate basis; anything else is a hand-written table.

The base variables are drawn `N(0, Sigma)` with `Sigma = D*corr*D`,
`D = diag(sqrt(variance))`, using a Cholesky factor computed once by
`synth_init`. Generating a row is then `k` normal draws, a `k^2/2` triangular
multiply, and one pass over the terms.

**Caveat worth stating loudly:** `corr` describes the *base variables*. Terms of
degree > 1 are not Gaussian and their pairwise correlations are not `corr` —
the design matrix holds functions of the base variables, not the base variables
themselves. That is the point (it is what makes polynomial fitting interesting)
but it means you cannot read column correlations of `X` off the spec.

## Contract and boundary policy

`synth_init` borrows the caller's `terms`, `coef` and `scratch`; nothing is
copied and nothing is allocated. `synth_scratch(k) = k*k + 2*k` f64.

Every input condition is checked and reported distinctly, with no partial state:

| condition | error |
|---|---|
| `k < 1`, ragged `terms`, short `coef`, bad stride, short slice | `.Invalid_Dimension` |
| scratch below `synth_scratch(k)` | `.Scratch_Too_Small` |
| `corr` not symmetric to `1e-9`, diagonal not 1, or an entry outside `[-1,1]` | `.Invalid_Correlation` |
| any variance not strictly positive | `.Invalid_Variance` |
| `sigma < 0` | `.Invalid_Sigma` |
| `corr` symmetric and in range but not positive definite | `.Not_Positive_Definite` |

The last one matters most, because it is the mistake a caller actually makes: a
correlation matrix can be entirely legal entry by entry and still describe no
real distribution. Three variables mutually correlated `+0.9, +0.9, -0.9` is the
classic case, and it is rejected rather than silently producing data with some
other correlation. Detection is `dpotrf` failing, which is exact.

Zero variance is rejected rather than allowed: it would mean a constant
predictor, which is what an all-zero term row already provides, and it makes
`Sigma` singular.

## Reproducibility

The `Rng` (xoshiro256\*\* seeded through SplitMix64, with a cached Box-Muller
spare) is held by the caller, so the stream is explicit. Same seed gives
identical bytes, which is what makes a failing case re-runnable. Chunked
generation is bit-identical to one large call, because the stream simply
continues — so generation can be spread over frames or streamed straight into
an `Ols_Accum` without ever holding all rows.

## Verified

`dpotrf` is new and general (row-major, both triangles, `contextless`), so it is
tested on its own: `L*L^T` and `U^T*U` reconstruct a known SPD matrix to
`1e-12`, and indefinite, negative-pivot and *singular* matrices each report the
right failing minor rather than returning a plausible factor.

The generator's statistics are checked two ways.

**In-suite** (`test_synth_statistics`, `k = 3`, 200 000 rows): sample variances
within 3%, all nine correlations within 0.02, means at zero, and — the check
that actually pins down the random source — skew within 0.03 and excess
kurtosis within 0.06 of the Gaussian values. A broken normal generator such as
a sum of uniforms would pass variance and correlation while failing kurtosis.

**Independently, with numpy**, on a separate 100 000-row `k = 4` sample dumped
to disk:

| | requested | measured |
|---|---|---|
| variances | `4, 0.25, 9, 1` | `4.011, 0.252, 9.027, 0.995` (max 0.8% rel) |
| correlations | 6 distinct off-diagonals | max abs error 0.0081 |
| skew / excess kurtosis | 0 / 0 | within `0.011` / `0.025` |
| a `b0^2` column vs `base0**2` | exact | max abs error `0.0` |

Other checks in the suite: `sigma = 0` gives residuals below `1e-12` rather than
merely small; `sigma` of 0.1 and 2.5 are recovered to 4%; exponent algebra holds
row by row including a cube and two interactions; OLS recovers planted
coefficients to `3.6e-14` noiseless and recovers `sigma = 0.5` as `0.5015` from
20 000 noisy rows; same seed reproduces bit-identically and a different seed does
not; chunk sizes 1, 7 and 128 match one call bit-for-bit; a wider destination
stride leaves the spare columns untouched; and the whole
generate-then-fit path runs under `mem.panic_allocator` with fixed arrays.

`examples/synth_workflow` demonstrates the payoff, and doubles as a consistency
check: with `corr(b0,b1) = 0.85` and a planted `0.75*b0^2`, fitting without the
quadratic biases the intercept by `+0.75`, which is exactly
`coef(b0^2) * Var(b0) = 0.75 * 1.0`. The omitted term lands where theory says it
must.

## Not verified

- **Only Gaussian.** No test covers heavy-tailed or skewed error, because none
  is generated. If the fitting code is ever used on such data, this generator
  will not have exercised that.
- **Degree > 3 untested,** and high powers of a correlated base variable make a
  badly conditioned design very quickly. `u8` exponents permit up to 255, which
  is far past anything numerically sensible; the type does not stop you.
- **Statistical tolerances are calibrated for the row counts used.** They are
  several standard errors wide at 100k–200k rows and would flake if the counts
  were cut substantially.
- **No performance measurement.** Generation cost has not been benchmarked; it
  is not on any hot path measured elsewhere. Per row it is `k` normal draws plus
  `k^2/2` plus the total term degree, all with `O(1)` memory.
