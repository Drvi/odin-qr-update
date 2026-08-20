# 011 — Faster multivariate normal sampling for structured Sigma

synth_mvn_rows factors a general Sigma with Cholesky and applies it per row at
O(k^2). One structural special case is already exploited: a diagonal Sigma is
detected at init and drops to O(k), worth a measured 1.82x at k=64 and 2.93x at
k=128 (docs/SYNTH.md). Three more are available and not implemented.

EQUICORRELATION. Sigma = sigma^2 * ((1-rho)I + rho*11^T), every pair equally
correlated. Sampling is

    x_i = sigma * ( sqrt(1-rho)*z_i + sqrt(rho)*w )

with one w ~ N(0,1) shared across the row: O(k) instead of O(k^2), same class of
win as the diagonal path. Valid for rho in [-1/(k-1), 1]. Detectable at init by
checking the off-diagonals are all equal, which is O(k^2) once. This is the most
likely to be worth doing: equicorrelated matrices are common in test data
precisely because they are easy to write down.

LOW-RANK PLUS DIAGONAL. Sigma = Lambda*Lambda^T + Psi with Lambda k-by-r and Psi
diagonal. Sampling is x = Lambda*f + sqrt(Psi)*eps with f an r-vector, so O(k*r)
instead of O(k^2). Needs a different input format -- the caller supplies Lambda
and Psi rather than a correlation matrix -- so it is a second entry point, not a
detected special case.

EIGENDECOMPOSITION INSTEAD OF CHOLESKY. Sigma = Q*Lambda*Q^T, x = Q*sqrt(Lambda)*z.
No cheaper (still O(k^2) per row, and the factorisation is dearer), but it
accepts positive SEMI-definite Sigma where Cholesky does not. A caller who wants
two perfectly correlated variables, or a rank-deficient Sigma from a factor
model, currently gets .Not_Positive_Definite. That is a correct report of what
Cholesky found, but it may not be the behaviour they want, and there is no way
to ask for the other one. Needs a symmetric eigensolver, which this repository
does not have.

None implemented: no caller has asked, and each needs either a new input format
or a new factorisation. The diagonal case was done because identity correlation
is what you get by default when you have not asked for correlation at all.
