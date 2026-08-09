# 003 — Covariance, standard errors, R-squared

`cov(beta) = sigma^2 * R^-1 * R^-T` with `sigma^2 = RSS / (nrows - n)`.
Requires inverting the `n x n` triangle: `n^2` scratch and `n^3/3` work.

Not implemented: `beta` and `RSS` are what was asked for. Adding these means
picking a degrees-of-freedom convention, which depends on whether weights or
ridge rows (issue 002) are present.
