# 004 — Parallel accumulation by merging triangles

Two `Ols_Accum` over disjoint row ranges can be merged by absorbing the `n+1`
rows of one triangle into the other, because the triangle is itself a valid
set of rows spanning the same row space. That makes row-parallel OLS trivial:
partition rows, accumulate independently with no shared state, merge pairwise.
Merge cost is `O(n^3)`, negligible against `O(m*n^2)`.

Not implemented: unmeasured. At the measured per-row cost the single-threaded
accumulator already handles large `m` well inside a frame, so there is no
demonstrated need. Verify with a measurement before building it.
