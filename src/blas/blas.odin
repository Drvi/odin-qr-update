// Package blas provides Basic Linear Algebra Subprograms (BLAS) and
// LAPACK auxiliary routines implemented in Odin.
//
// This package includes:
// - BLAS Level 1: Vector operations (dcopy, dscal, daxpy, ddot, dnrm2, drot, etc.)
// - BLAS Level 2: Matrix-vector operations (dgemv, dtrsv, dger, dtrmv)
// - LAPACK auxiliaries: Givens rotation generation (dlartg), Householder reflectors (dlarfg)
// - QR update routines: delcols, delcolsq, addcols, addcolsq, addrows, delrows
// - Cholesky factorization (cholesky.odin): dpotrf
// - Least squares (lstsq.odin): ols_solve_dense and Ols_Accum, for solving
//   min ||X*beta - y||. See docs/OLS_PLAN.md and docs/OLS_RESULTS.md.
//
// Every procedure here is declared `proc "contextless"`. The package allocates
// nothing -- all memory is caller-supplied, sized by the *_scratch procedures --
// and `contextless` is what makes that a compile-time property rather than a
// convention: `make` inside one of these is a compile error. It also means the
// package is callable where no Odin context exists, such as a C callback or a
// thread not bootstrapped by the Odin runtime.
//
// All routines follow the conventions of the Reference BLAS/LAPACK implementation.
//
// STORAGE: matrices are row-major (C-style). `lda` is the ROW STRIDE -- the
// distance between the starts of consecutive rows -- so it must be at least the
// column count n, not the row count m. This is the opposite of the Fortran
// leading dimension, and getting it backwards makes routines silently return
// without doing anything on tall matrices. Element (i,j) is a[i*lda + j].

package blas

// Re-export everything from sub-modules
// (All definitions are in the same package, so they're automatically available)
