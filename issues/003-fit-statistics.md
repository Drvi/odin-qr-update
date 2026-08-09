# 003 — Covariance, standard errors, R-squared

`cov(beta) = sigma^2 * R^-1 * R^-T` with `sigma^2 = RSS / (nrows - n)`.
Requires inverting the `n x n` triangle: `n^2` caller-supplied scratch and
`n^3/3` work. No allocation would be needed; it would follow the same
`*_scratch` convention as everything else.

Not implemented. `beta` and `RSS` are what was asked for, and both are already
exposed. Anything built on top of them — standard errors, R-squared, AIC, BIC,
adjusted R-squared — is caller arithmetic over `(rss, nrows, k)`.

That is deliberate rather than lazy: **choosing the model is the user's job**,
so the library does not ship a scoring rule. Providing one would mean picking a
degrees-of-freedom convention on the user's behalf, and that choice changes
again the moment weights or ridge rows appear (issue 002). Covariance proper
has no such ambiguity and could be added if a caller needs it; a built-in
"which model is best" must not be.
