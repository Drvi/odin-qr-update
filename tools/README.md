# tools

Validation and evaluation tooling. Nothing here is compiled into the library.

## normal_battery.py

Goodness-of-fit battery for the normal generators, using scipy. Tests three
samples side by side: the candidate, the scalar generator already in the
library (the control for "no worse than what we have"), and a deliberately
broken generator (the control for "the battery has power").

    <odin bench that dumps VBM_VEC.bin / VBM_SCALAR.bin / VBM_BROKEN.bin>
    python tools/normal_battery.py <dir>

Covers moments with z-scores, blocked Kolmogorov-Smirnov and Cramer-von Mises
with a uniformity test on the resulting p-values, Anderson-Darling against a
fully specified N(0,1), tail exceedance counts out to 6 sigma, serial
autocorrelation, Box-Muller pair-half correlation, the radius^2 ~ chi2(2) joint
check, and a 256-bin chi-square.

The subset with demonstrated power is reproduced in Odin in
`tests/test_synth.odin:test_synth_distribution_battery`, negative control
included, so it runs on every build.

## vecbm_reference.odin.txt

A working, validated vectorised Box-Muller (8 lanes, hand-written vector `ln`
and `sincos`). Kept as `.txt` so it is not compiled. Measured at 4.86 ns/draw
with AVX-512 against 26.79 scalar, and statistically indistinguishable from the
scalar generator. See docs/SYNTH.md for the numbers and
issues/010-vectorised-box-muller.md for what adopting it would involve.
