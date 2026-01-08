// Package blas provides Basic Linear Algebra Subprograms (BLAS) and
// LAPACK auxiliary routines implemented in Odin.
//
// This package includes:
// - BLAS Level 1: Vector operations (dcopy, dscal, daxpy, ddot, dnrm2, drot, etc.)
// - BLAS Level 2: Matrix-vector operations (dgemv, dtrsv, dger, dtrmv)
// - LAPACK auxiliaries: Givens rotation generation (dlartg), Householder reflectors (dlarfg)
// - QR update routines: delcols, delcolsq, addcols, addcolsq, addrows, delrows
//
// All routines follow the conventions of the Reference BLAS/LAPACK implementation.
// Matrices are stored in row-major order (C-style) with leading dimension lda
// specifying the stride between rows.

package blas

// Re-export everything from sub-modules
// (All definitions are in the same package, so they're automatically available)
