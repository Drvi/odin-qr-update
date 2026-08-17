package blas

import "core:math"

// ============================================================================
// Cholesky factorization
// ============================================================================

// dpotrf computes the Cholesky factorization of a real symmetric positive
// definite matrix A, row-major with row stride lda >= n:
//
//    A = L * L**T   (uplo == .Lower)
//    A = U**T * U   (uplo == .Upper)
//
// The factor overwrites the corresponding triangle of A. Following LAPACK, the
// opposite triangle is not read and not written, so a caller storing only half
// a symmetric matrix can leave the other half as garbage.
//
// Returns the 1-based index of the leading minor that is not positive definite,
// or 0 on success. A nonzero return means A was not positive definite and the
// contents of A are undefined -- this is the expected outcome for an
// inconsistent correlation matrix, so it is a value to check, not an
// exceptional case.
//
// Cost: n^3/3 flops. No workspace.
dpotrf :: proc "contextless" (uplo: Uplo, n: int, a: []f64, lda: int) -> int {
	if n < 0 || lda < max(1, n) {
		return 1
	}
	if n == 0 {
		return 0
	}

	if uplo == .Lower {
		for j in 0 ..< n {
			// Diagonal: a[j,j] - sum of squares of the row to its left.
			d := a[j * lda + j]
			for p in 0 ..< j {
				v := a[j * lda + p]
				d -= v * v
			}
			if !(d > 0.0) {
				// Catches zero, negative and NaN.
				return j + 1
			}
			piv := math.sqrt_f64(d)
			a[j * lda + j] = piv

			// Column below the diagonal.
			inv := 1.0 / piv
			for i in j + 1 ..< n {
				s := a[i * lda + j]
				for p in 0 ..< j {
					s -= a[i * lda + p] * a[j * lda + p]
				}
				a[i * lda + j] = s * inv
			}
		}
	} else {
		for j in 0 ..< n {
			d := a[j * lda + j]
			for p in 0 ..< j {
				v := a[p * lda + j]
				d -= v * v
			}
			if !(d > 0.0) {
				return j + 1
			}
			piv := math.sqrt_f64(d)
			a[j * lda + j] = piv

			inv := 1.0 / piv
			for i in j + 1 ..< n {
				s := a[j * lda + i]
				for p in 0 ..< j {
					s -= a[p * lda + j] * a[p * lda + i]
				}
				a[j * lda + i] = s * inv
			}
		}
	}
	return 0
}
