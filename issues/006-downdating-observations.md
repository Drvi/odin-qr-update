# 006 — Removing observations from a fit

`ols_accum_rows` and `ols_accum_merge` only ever add data. There is no way to
remove observations already folded into a triangle — to drop an outlier, expire
a stale window, or undo a segment merged by mistake.

This matters for "fit the right model to the right DATA": predictor
experimentation is cheap (`ols_accum_select`), but data experimentation is only
cheap in the additive direction.

What exists today:

- Re-accumulate the wanted rows. O(m*n^2), correct, and the only option if the
  removed rows are not separable in advance.
- Keep one accumulator per segment and merge the subset you want. O(n^3) per
  combination and exact, but it requires deciding the segmentation up front.
  This is the recommended pattern and what examples/model_iteration demonstrates.

What is missing: true downdating, removing a row from an existing triangle via
hyperbolic rotations. Deliberately not implemented. Downdating is numerically
unstable in a way ordinary updating is not — the triangle loses positive
definiteness when the removed row carries a large share of the fit, and the
failure is silent, producing a triangle that still looks well-formed. Adding it
would need a breakdown test and a documented policy for what happens when the
downdate fails, neither of which can be designed sensibly without a real case
that demands it.

If this comes up: get the real data first. How many rows are removed, how often,
and can they be segregated into their own accumulator instead?
