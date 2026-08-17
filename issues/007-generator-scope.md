# 007 — Generator: what it deliberately does not produce

src/synth generates Gaussian base variables with a chosen covariance, arbitrary
polynomial/interaction terms, and homoskedastic Gaussian response noise. Real
measured data is often none of those. Left out, with the reason:

- Heteroskedastic noise (variance depending on the inputs). Cheap to add --
  scale sigma by a function of the row -- but "which function" is a modelling
  choice nobody has stated.
- Non-Gaussian error: heavy tails, skew, outliers. This is the one most likely
  to matter, because it is what would justify robust fitting rather than OLS.
  Needs a stated distribution first.
- Missing values, and NaN injection for exercising the solvers' rejection
  paths. Currently done by hand in the tests, which is adequate for those tests.
- Categorical / one-hot predictors. Representable today as 0/1 columns the
  caller writes, but not generated.
- Autocorrelated or time-series structure. Would need state carried between
  rows, which the current per-row generation deliberately does not have.

None are speculative-generality candidates: each is a real property of real
data. They are absent because no requirement has named one, and each needs a
distribution or a function specified before it can be implemented correctly
rather than arbitrarily.
