# 012 — The rest of the QR update routines

delcols is now used, via ols_accum_drop_cols: applied to the AUGMENTED triangle
it removes predictors with no Q and no data pass, at O(d*p^2) against select's
O(p*k^2) -- a measured 37x at p=30 for a single drop. See docs/OLS_RESULTS.md.

The others, and where they stand:

addrows. Already what ols_accum_rows does. Folding new rows into a triangle with
Givens rotations IS addrows, specialised to the augmented case. Nothing to
adopt; the general routine stays for callers working on a bare R.

addcols. Inserts predictors, and needs Q^T applied to the new column. The
accumulator cannot supply that -- it discards Q by construction, which is the
whole reason adding a predictor costs a data pass (ols_accum_rows_gather).
BUT the dense path keeps the Householder reflectors, and ols_apply_qt can apply
them to a new column in O(m*p). So for a caller who factored with
ols_solve_dense and kept `a` and `tau`, adding a predictor is O(m*p) + O(p^2)
rather than the O(m*p^2) of a full refit -- a factor of p. That is a real case
and it is NOT implemented: it needs an entry point that carries the factored
matrix and tau alongside the triangle, which is a different state object from
Ols_Accum. Worth doing if "add a term" turns out to be a common move.

delcolsq / addcolsq. The variants that also update an explicit m x m Q. Nothing
here needs Q, and materialising it is the quadratic cost this subsystem was
built to remove. They stay for callers who genuinely want Q.

delrows. Removing observations. Untouched, and the one to be careful with --
see issues/006. The implementation shifts rows and re-triangularises rather than
downdating, and has not been verified against a from-scratch fit the way
delcols now has. Do not assume it is correct because delcols is.
