# 001 — Multiple right-hand sides

`ols_solve_dense` and `ols_accum_solve` take a single `y` and produce a single
`beta`. Solving several right-hand sides against one `X` would share the
factorization and turn the `dtrsv` into a `dtrsm`.

Not implemented: no present caller needs it. The data question it turns on is
whether callers ever fit several responses against one design matrix; nothing
in the repository does today.

Cost if added: `applyQt` gains a column loop (already supported by `dlarf`,
which takes a column count), `beta` becomes `n x nrhs`, and the accumulator's
triangle grows to `(n+nrhs)^2`. The accumulator change is the expensive one.
