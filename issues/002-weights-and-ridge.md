# 002 — Weighted least squares and ridge regularization

Both are cheap to add to the accumulator and were deliberately left out.

- Weights: scale each row by `sqrt(w_i)` before absorbing. ~3 lines.
- Ridge: absorb `n` extra rows holding `sqrt(lambda)*I` with `y = 0`. ~8 lines.
  Would also make rank-deficient systems solvable instead of rejected.

Not implemented: no present requirement, and both change the meaning of
`nrows`, `RSS` and the degrees of freedom in ways that need a caller to
specify before they can be defined correctly (in particular, ridge rows are
not observations and must not count toward `nrows`).
