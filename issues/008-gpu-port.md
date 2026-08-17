# 008 — GPU port of the generator

src/synth is shaped so a GPU port is mechanical, but no port exists and nothing
has been measured on a GPU. What is already true:

- synth_hash is u32 add/xor/shift/multiply only; translates verbatim to
  GLSL/HLSL/WGSL.
- Every draw is a pure function of (seed, stream, index), so a thread owning
  row i needs no state, no seeding step and no coordination.
- No rejection loops anywhere, so no warp divergence from the RNG.
- The Cholesky factor is computed on the CPU and is k*k floats; the per-thread
  work is only L*z plus the term expansion.

What a port has to decide:

- f64 -> f32. Consumer GPUs run doubles at 1/32-1/64 rate. This also collapses
  the 53-bit magnitude uniform to 32 bits, moving the Box-Muller tail cut from
  ~8.5 to ~6.2 sigma.
- Whether to keep Box-Muller or switch to an inverse-CDF approximation. Box-
  Muller pairs two outputs, which suits a thread generating k base variables and
  is exact. Probit is 1:1 and needs no pairing at all, at the cost of being an
  approximation (Acklam ~1e-9, Wichura AS241 ~1e-16).
- How to verify. The statistical tests in tests/test_synth.odin are the contract:
  variance, correlation, skew, kurtosis, stream independence and hash
  uniformity. A port should reproduce them on GPU-generated data rather than
  being assumed correct because the source looks the same.

Not done because there is no GPU here and no measured requirement. The cost of
NOT doing it is currently zero: generation is not on any hot path measured in
docs/OLS_RESULTS.md.
