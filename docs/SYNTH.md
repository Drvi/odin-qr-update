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

## Why the random source is counter-based (and what that means for GPU)

Every draw is a pure function of `(seed, stream, index)`. Nothing is carried
between calls; there is no state to advance.

That was adopted to fix a real defect, not for elegance. The first version used
a sequential stream (xoshiro256\*\* with a cached Box-Muller spare), and the
noise draw shared that stream with the base variables — so **raising `sigma`
silently produced a different `X`**. "The same data with more noise", the most
natural experiment for a generator whose job is checking a solver, was not
expressible. Separate counter-indexed streams make `X` bit-identical across
noise levels, which `test_synth_sigma_leaves_x_alone` now pins down.

Three further properties come along with it:

- **Row-addressable.** Row *i* depends only on `(seed, i)`, so any row is
  generatable alone, out of order, or in parallel, with no coordination.
  `test_synth_row_addressable` checks single rows, out-of-order requests and
  reverse-order batches all match one sequential call.
- **Order-independent reproducibility.** Chunk boundaries cannot affect output,
  because nothing crosses them.
- **32-bit integer core**, which is what a GPU wants.

### On GPU specifically

The GPU-hostile part of Gaussian generation is not the Gaussian, it is state and
divergence:

| method | GPU verdict |
|---|---|
| Ziggurat | **Measured 3.75x faster than Box-Muller on CPU** (see below). On GPU its rejection loop has a data-dependent trip count, and at the measured 2.73% slow-path rate only 41% of 32-lane warps stay entirely on the fast path — so the advantage is likely eaten there, though that is reasoning and not a measurement. |
| Marsaglia polar | Same rejection problem. |
| **Box-Muller** | Fine, and **slower than ziggurat on CPU by 3.75x**. `log`, `sqrt`, `sincos` all have hardware approximations (SFU, typically quarter rate). Fixed cost, no divergence, vectorises trivially. Used here. |
| **Inverse CDF / probit** | Structurally the most GPU-friendly: one uniform in, one normal out, no pairing and nothing to cache. Costs being an approximation — Acklam is ~1e-9 relative, Wichura AS241 ~1e-16 for more polynomial. Worth switching to if the pairing ever becomes awkward. |
| Sum of uniforms (CLT) | Cheapest and wrong. Passes variance and correlation, fails the tails — `test_synth_statistics`' kurtosis check exists to reject exactly this. |

**What ports as-is:** `synth_hash` is `u32` add/xor/shift/multiply only and
translates verbatim to GLSL/HLSL/WGSL. `synth_normal_pair` is four
transcendentals and no branches. The `L*z` correlation transform is `k^2/2`
multiply-adds per thread, and the Cholesky factor itself is computed once on the
CPU and uploaded as `k*k` floats in a constant buffer — `dpotrf` does not need
to run on the GPU at all. The term expansion is a short fixed loop over a `u8`
table.

**What does not port:** `f64`. Consumer GPUs run double precision at 1/32 to
1/64 rate, so a GPU version should be `f32` throughout. That also collapses the
53-bit magnitude uniform to 32 bits, which moves the Box-Muller tail cut from
about 8.5 sigma to about 6.2 sigma — irrelevant for `f32` data, worth knowing.

**Versus rngeasy's approach.** rngeasy keeps 64 bits of state and advances a
xoroshiro64\*\* stream, with portability handled by `#define RNGSTATE_REF`
(`RngState&` on CPU, `inout RngState` in GLSL). That is a perfectly good GPU
choice — two registers per thread — but it needs per-thread seeding and makes
output depend on how many values each thread consumed. Counter-based needs no
per-thread state and no seeding discipline at all: `normal(seed, stream, index)`
is the whole interface. The tradeoff is recomputation instead of caching, which
is measured below. rngeasy has no Gaussian generator, so nothing here contradicts
it.

**Not verified: none of this has been run on a GPU.** There is no GPU in this
project. The claims above are about the shape of the code — no state, no
rejection, 32-bit integer arithmetic, fixed instruction count per draw — not
measurements. Filed as `issues/008-gpu-port.md`.

### Ziggurat, measured

The table above originally called ziggurat "fastest on CPU, worst on GPU" from
general knowledge. Both halves were then measured, and the first half is
understated while the second is overstated.

4,000,000 draws, min of 7, Marsaglia-Tsang 128-level ziggurat driven by the
*same* `synth_hash` bit source so the comparison isolates the transform:

| | ns/draw | vs ziggurat |
|---|---|---|
| ziggurat | **7.93** | — |
| Box-Muller, both halves consumed (how `synth_rows` calls it) | 29.73 | 3.75x |
| Box-Muller, single draw, half discarded | 55.53 | 7.0x |

The ziggurat output was checked before being timed, because a fast wrong
generator is worthless: mean `-0.00037`, sd `1.00082`, skew `+0.0046`, excess
kurtosis `-0.0004`.

**So ziggurat is decisively faster on CPU.** At `k = 5` the Gaussian transform is
about 73% of generation cost, so adopting it would roughly halve `synth_rows`
(230 ns/row down to about 110 ns/row), which would put generation level with the
OLS accumulate it feeds rather than at twice its cost.

**It is also better in one respect that has nothing to do with speed:** the
ziggurat tail is sampled by exponential rejection and is therefore unbounded,
where Box-Muller from a 53-bit uniform truncates at about 8.5 sigma.

**Three things it costs, which is why it has not simply been adopted:**

1. *Resolution.* The classic ziggurat derives the value from one 32-bit word, so
   a draw carries at most 32 bits of entropy against the 53 currently used. Not
   wrong, but coarser, and this is an `f64` library.
2. *State.* The tables are `128 x (u32 + 2 x f64)` = 2560 bytes, and they need
   `exp`/`log` to build, so they cannot be compile-time constants in Odin. Today
   `src/synth` has no global state at all. The consistent home would be the
   caller's `synth_scratch` block, growing it by 2560 bytes — workable, but it
   is real state where there is currently none.
3. *Vectorisation.* Ziggurat's table access is a per-lane gather and its accept
   test is per-lane, so it does not vectorise; Box-Muller does. This machine has
   AVX-512. **A vectorised Box-Muller was not measured**, and could close a
   meaningful part of the 3.75x — so the gap above is for scalar code only.

**On GPU the claim was too strong.** The measured slow-path rate is 2.73%, so
`P(all 32 lanes fast) = 0.988^32`-style arithmetic gives **0.412** — 58.8% of
warps contain at least one diverging lane, and the whole warp then pays the slow
path (a `log` or an `exp` plus extra draws) on top of the fast one. That is a
real penalty, plausibly enough to erase a 3.75x scalar lead, but "worst on GPU"
overstated it: the slow path is a few extra operations, not a long loop, and
published results are mixed. Calling it *uncertain on GPU* is the defensible
position, and it stays uncertain until somebody measures it on hardware.

**Ziggurat is compatible with the counter-based design.** This was checked rather
than assumed: reserving 8 hash slots per draw (`index*8 + attempt`) keeps
`zig_normal(seed, stream, index)` a pure function of its index despite the
rejection loop, so row-addressability and reproducibility would survive the
switch. Statelessness is not an argument against it.

Recorded in `issues/009-ziggurat.md`. Not adopted, because generation is on no
shipped hot path and the resolution and state costs are real; the measurements
are here so the decision can be made on data rather than reputation.

### Cost of statelessness, measured

`k = 5`, 6 terms, 1,000,000 rows, min of 7, same machine as `docs/OLS_RESULTS.md`:

| | ns/row | Mrows/s |
|---|---|---|
| stateful xoshiro + cached spare (previous version) | 217.1 | 4.61 |
| counter-based, draws indexed per row | 286.9 | 3.49 |
| counter-based, draws indexed **globally** | **230.8** | **4.33** |

The middle row is what naive statelessness costs: at `k = 5` it computes four
Box-Muller pairs per row where caching computed three, because the spare cannot
cross the base/noise boundary or the row boundary. 4/3 is 33%, and 34% was
measured.

Indexing draws globally — base draw *j* of row *r* is index `r*k + j` — lets
pairs straddle those boundaries, and a pair cache local to the batch call
recovers most of it. The cache holds only values already derivable from the
index, so results are unchanged; the addressability and chunk-invariance tests
still pass unmodified. Residual overhead is **6.3%**, which is the extra hash
work (three hashes at two `splitmix32` rounds each per pair, against one
xoshiro advance) plus the cache branch.

Component costs, same conditions: `synth_hash` is **1.23 ns**, and a full
`synth_normal` draw is **58.3 ns**. So the hash — the part that had to change
for statelessness — is under 2% of a draw, and the four transcendentals are
essentially all of it. That is also the part that would get relatively cheaper
on a GPU, where they are hardware instructions.

Global indexing bounds the row count: `(first_row + count) * k` must stay under
`2^32`, or draw indices wrap and rows alias each other. That is checked and
returns `.Invalid_Dimension` rather than silently repeating data. At `k = 100`
the limit is about 43 million rows.

## Reproducibility

`synth_rows` takes a `seed` and an absolute `first_row`; there is no generator
object. Same `(seed, first_row)` always gives the same rows, on any machine, in
any order, so a failing case is re-runnable from two integers. Chunked
generation is bit-identical to one large call because nothing crosses a call
boundary — generation can be spread over frames or streamed straight into an
`Ols_Accum` without ever holding all rows.

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
