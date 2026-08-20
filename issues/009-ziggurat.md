# 009 — Ziggurat instead of Box-Muller

Measured, not speculated. 4,000,000 draws, min of 7, Marsaglia-Tsang 128-level
ziggurat driven by the same synth_hash bit source:

  ziggurat                            7.93 ns/draw
  Box-Muller, both halves consumed   29.73 ns/draw   3.75x slower
  Box-Muller, single draw            55.53 ns/draw   7.0x slower

Distribution verified first: mean -0.00037, sd 1.00082, skew +0.0046, excess
kurtosis -0.0004.

Adopting it would roughly halve synth_rows (230 -> ~110 ns/row at k=5), since
the Gaussian transform is about 73% of generation cost there. It would also give
unbounded tails, where Box-Muller from a 53-bit uniform truncates near 8.5 sigma.

Costs:

- Resolution: CORRECTED. This originally said "drops from 53 bits to 32", which
  overstated it. The value is hz*wn[iz] and the per-strip scale differs across
  128 strips, so the reachable value set is nearer 2^39 than 2^32. Counting
  exact ties in 5,000,000 samples gives ZERO for ziggurat, Box-Muller and
  vectorised Box-Muller alike. Real in principle, undetectable at that n.
- 2560 bytes of tables that need exp/log to build, so not compile-time
  constants. src/synth currently has zero global state; the consistent home is
  the caller's synth_scratch block, +2560 bytes.
- No vectorisation: per-lane gather plus per-lane accept test. Box-Muller
  vectorises trivially and this machine has AVX-512. A VECTORISED BOX-MULLER WAS
  NOT MEASURED and could close much of the gap; measure that before switching.

GPU: measured slow-path rate 2.73%, so only 41.2% of 32-lane warps stay entirely
on the fast path and 58.8% pay the slow path warp-wide. Probably enough to erase
the scalar lead, but unmeasured on hardware and published results are mixed.

Compatible with counter-based indexing: verified that reserving 8 hash slots per
draw (index*8 + attempt) keeps zig_normal(seed, stream, index) a pure function
despite the rejection loop, so addressability and reproducibility survive.

RESOLVED AGAINST ZIGGURAT. The vectorised Box-Muller called for above was then
built and measured: 4.86 ns/draw with AVX-512, against ziggurat's 7.93. So the
3.75x gap was mostly scalar code rather than the algorithm, and the faster
option is also the one with 53-bit resolution, no tables and no divergence. See
issues/010-vectorised-box-muller.md.

Ziggurat is not adopted. It would still be the better choice on hardware
without AVX-512, where vectorised Box-Muller lands at 9.40 ns/draw and is
1.19x slower -- worth remembering if the target ever changes.

The benchmark implementation is not in the repository. Reproduce it from
Marsaglia & Tsang (2000) "The Ziggurat Method for Generating Random Variables",
JSS 5(8), driving it from synth_hash as described above.
