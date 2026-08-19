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
| Ziggurat | Measured 3.75x faster than *scalar* Box-Muller, but **slower than vectorised Box-Muller** (7.93 vs 4.86 ns/draw; see below). On GPU its rejection loop has a data-dependent trip count, and at the measured 2.73% slow-path rate only 41% of 32-lane warps stay entirely on the fast path — so the advantage is likely eaten there, though that is reasoning and not a measurement. |
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

### Vectorised Box-Muller, measured

Written after the ziggurat result, to test whether the 3.75x gap was really
about the algorithm or just about scalar code. It was mostly the latter.

Eight lanes (`#simd[8]f64`), same counter-based `synth_hash` so draws stay
addressable, with **hand-written vector `ln` and `sincos`** because there is no
vector libm available. Those are the risk: a polynomial that is slightly wrong
yields numbers that still look Gaussian to a moment check and are not. So the
transform was validated before the distribution, and the distribution was
validated with a negative control.

**Accuracy of the vector transcendentals**, against libm over the domain
Box-Muller actually uses (`u` swept log-uniformly from `1e-300` to `1`):

| | worst error |
|---|---|
| `vln` | **2.0 ULP** (at `x = 2.07e-2`) |
| `vcos(2*pi*u)` | 6.66e-16 absolute |
| `vsin(2*pi*u)` | 7.22e-16 absolute |

**Agreement with the scalar generator**, same seed and same indices, 1,000,000
draws: max absolute difference **3.0e-15**, max relative difference 3.8e-11
(that figure is for deviates near zero, where relative error is meaningless);
60.6% of lanes differ in the last bits, as expected from a different but
equally valid evaluation order.

**Throughput**, 4,000,000 draws, min of 7:

| | ns/draw |
|---|---|
| **vectorised Box-Muller, `-microarch:native` (AVX-512)** | **4.86** |
| vectorised Box-Muller, baseline ISA | 9.40 |
| scalar Box-Muller, both halves consumed | 26.79 |
| scalar ziggurat (from the section above) | 7.93 |

**So vectorising Box-Muller beats scalar ziggurat by 1.63x on this machine**,
and is 5.51x faster than the scalar Box-Muller it replaces. On baseline ISA
without AVX-512 it lands at 9.40 ns, roughly level with ziggurat (1.19x slower).

That answers the question the ziggurat section left open, and it answers it
against ziggurat. Note the comparison is vectorised Box-Muller against *scalar*
ziggurat: **no vectorised ziggurat was attempted**, because its table access is
a per-lane gather and its accept test is per-lane, which is the same property
that makes it awkward on a GPU. That is a reason to expect it to vectorise
poorly, not a measurement.

### The distributional battery, and why it has a negative control

Speed is worthless if the samples are subtly wrong, and "subtly wrong" is
exactly what a hand-written polynomial risks. Three samples of 5,000,000 were
tested side by side:

- **VEC** — the vectorised generator under evaluation
- **SCALAR** — the generator already in the library, as the control for
  *no worse than what we have*
- **BROKEN** — the sum of twelve uniforms minus six. Mean 0 and variance 1
  *exactly*, so it passes anything that stops at the second moment, but excess
  kurtosis is `-0.1` and the support is clipped at `+/-6`. This is the control
  for *the battery has power*. A battery that accepts all three measures
  nothing.

| statistic | VEC | SCALAR | BROKEN |
|---|---|---|---|
| z(mean) | 1.08 | 1.08 | -0.51 |
| z(variance) | 0.11 | 0.11 | 1.31 |
| z(skewness) | -0.07 | -0.07 | -0.52 |
| z(excess kurtosis) | 0.40 | 0.40 | **-46.18** |
| KS blocks rejected (of 50) | 0 | 0 | **4** |
| KS p-value uniformity | 0.668 | 0.668 | **5.1e-19** |
| Cramer-von Mises p-value uniformity | 0.615 | 0.615 | **1.6e-22** |
| Anderson-Darling A^2, median (H0 ~0.78) | 0.754 | 0.754 | **2.865** |
| A^2 blocks over the 1% point | 0 | 0 | **13** |
| tail z at \|z\|>1 | 0.11 | 0.11 | **20.24** |
| tail z at \|z\|>3 | -0.65 | -0.65 | **-28.67** |
| tail z at \|z\|>4 | 1.20 | 1.20 | **-12.80** |
| max \|z\| (expected ~5.20) | 5.714 | 5.714 | 4.727 |
| autocorrelation, lags 1..64 | all < 3e-4 | all < 3e-4 | all < 4e-4 |
| corr(Box-Muller pair halves) | -0.00002 | -0.00002 | -0.00098 |
| KS(radius^2 vs chi2(2)) p | 0.406 | 0.406 | **6.3e-10** |
| chi-square, 256 bins, p | 0.866 | 0.866 | **2.3e-255** |

Four things worth drawing out.

**VEC and SCALAR agree to every digit shown, on every statistic.** With a
maximum absolute difference of 3.0e-15 between them that is what should happen,
and confirming it is the point: the vectorised transform is not merely
*acceptable*, it is indistinguishable from the reference.

**The broken control passes mean, variance and skewness.** Every one of those
z-scores is under 1.4. It is caught only by kurtosis, by the tails, by the
distributional tests and by the radius check. That is the entire argument for
having them, and it is why `test_synth_statistics`' kurtosis bound was worth
adding earlier.

**Single giant tests were deliberately avoided.** A KS test over 5,000,000
points is powerful enough to reject on floating-point discreteness alone, which
says nothing useful. The battery instead runs KS and Cramer-von Mises on 50
independent blocks of 100,000 and then asks whether the resulting p-values are
themselves Uniform(0,1) — which separates "this sample is not normal" from
"this sample is large".

**The radius check tests the joint distribution, not the marginals.** For a
genuine Box-Muller pair `x^2 + y^2` must be `chi2(2)`; a generator with correct
marginals but a mis-distributed angle would fail here and pass everything else.

The battery is `tools/normal_battery.py` (scipy). The subset with demonstrated
power — kurtosis, tail exceedances, pair correlation, radius, and the negative
control — is reproduced in Odin as
`tests/test_synth.odin:test_synth_distribution_battery`, so it runs on every
build rather than only when someone remembers to.

**Not adopted into `src/synth` yet**, and the reason is structural rather than
numerical. `synth_rows` draws `k` normals per row with `k` typically 3-10, so
filling 8 lanes requires vectorising *across* rows: generating a block of rows'
draws into scratch first, then doing the per-row Cholesky and term expansion.
That is a bounded change but it grows `synth_scratch(k)` — a published contract —
and adds ~200 lines of hand-written numerics to a library that currently has
none. Generation is on no shipped hot path. The working implementation is kept
at `tools/vecbm_reference.odin.txt` with its validation harness; see
`issues/010-vectorised-box-muller.md`.

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
