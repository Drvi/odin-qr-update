# 003 — Covariance, standard errors, R-squared

**Standard errors: DONE.** `ols_accum_stderr` returns
`se[j] = sigma * ||row j of R^-1||` with `sigma^2 = RSS/(nrows - n)`, at
`n^3/3 + n^2` and `n^2` caller-supplied scratch. Verified against numpy to
3.4e-16 on three models over the real data.

This was deferred here originally, on the grounds that the degrees-of-freedom
convention shifts once weights or ridge exist. That was over-cautious: they do
not exist, so `dof = nrows - n` is unambiguous for everything the library
builds, and without standard errors there was no way to tell a real coefficient
from a spurious one -- which is the subsystem's main use. If weights or ridge
are ever added (issue 002), this formula and the dof count both change, and the
procedure documents that assumption.

Still not provided, and still deliberately:

- **The full covariance matrix.** `ols_accum_stderr` computes `R^-1` internally
  and takes row norms; returning the whole `sigma^2 * R^-1 * R^-T` would cost
  an extra `n^3/2` and `n^2` of output. Worth adding for testing linear
  combinations of coefficients or joint confidence regions; nobody has asked.
- **R-squared, adjusted R-squared, AIC, BIC, p-values.** All are arithmetic over
  `(rss, nrows, n)` and the t-statistics, every one of which is now exposed.
  These are scoring rules, and choosing the model is the user\'s job -- the
  library reports the spread, not a verdict.
