# 010 — Adopt the vectorised Box-Muller

Built, validated and measured; not landed. Implementation and its validation
harness are at tools/vecbm_reference.odin.txt.

WHY IT WINS. 4,000,000 draws, min of 7:

  vectorised Box-Muller, AVX-512    4.86 ns/draw
  vectorised Box-Muller, baseline   9.40 ns/draw
  scalar Box-Muller (current)      26.79 ns/draw
  scalar ziggurat                   7.93 ns/draw

5.51x over what the library does today, and 1.63x faster than ziggurat, which
settles issue 009 against ziggurat. On hardware without AVX-512 it is level
with ziggurat rather than ahead.

WHY IT IS SAFE. Hand-written vector ln and sincos were validated against libm
first: ln worst 2.0 ULP over 1e-300..1, cos and sin worst 7e-16 absolute.
Output agrees with the scalar generator to 3.0e-15 absolute on the same
indices. A 5,000,000-sample goodness-of-fit battery (moments with z-scores,
blocked KS and Cramer-von Mises with a uniformity test on the p-values,
Anderson-Darling, tail exceedances to 6 sigma, autocorrelation, pair-half
correlation, radius^2 vs chi2(2), 256-bin chi-square) cannot distinguish it
from the scalar generator on any statistic, while rejecting a
sum-of-twelve-uniforms control on kurtosis (z = -46), tails (z = -29 at 3
sigma), A^2, the radius test and chi-square.

WHAT LANDING IT COSTS.

- synth_rows draws k normals per row with k typically 3-10, so filling 8 lanes
  means vectorising ACROSS rows: generate a block of rows' draws into scratch,
  then do the per-row Cholesky transform and term expansion. The global draw
  indexing already makes a block of rows a contiguous index range, so the
  restructure is mechanical.
- synth_scratch(k) grows by roughly BLOCK*(k+1) doubles. At BLOCK=16, k=10 that
  is ~1.4 KB; at k=100 it is ~13 KB. synth_scratch is a published contract, so
  this is a visible API change.
- ~200 lines of hand-written numerics enter a library that currently has none.
  They are validated, but they are code someone has to maintain, and the
  validation harness has to come with them.
- The win depends on -microarch:native. Without it the advantage over ziggurat
  disappears, and shipping -march-specific builds is a decision of its own.

NOT DONE because generation is on no hot path measured in docs/OLS_RESULTS.md,
and the costs above are certain while the need is not. The measurements exist
so the call can be made on data.

If it is landed: keep tools/normal_battery.py in the loop, and keep the
negative control. The battery only means something because it demonstrably
rejects a broken generator.
