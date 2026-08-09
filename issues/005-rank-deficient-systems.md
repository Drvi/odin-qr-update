# 005 — Rank-deficient systems are rejected, not solved

`ols_*_solve` returns `.Rank_Deficient` when the `R` diagonal ratio falls below
`rcond`. It does not produce a minimum-norm solution.

Doing so properly needs column-pivoted QR (`dgeqp3`) or an SVD, neither of
which exists in this repository. The diagonal-ratio test is a cheap proxy for
the condition number, not a condition estimate; it can accept a matrix that a
proper estimator would reject.

Explicitly chosen behaviour: fail loudly rather than return a plausible-looking
vector.
