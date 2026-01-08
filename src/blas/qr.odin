package blas

import "core:math"

// ============================================================================
// QR Factorization routine
// ============================================================================

// dgeqrf computes a QR factorization of a real m-by-n matrix A:
//    A = Q * R
// where Q is an m-by-m orthogonal matrix and R is an m-by-n upper trapezoidal matrix.
//
// This implementation uses Householder reflections.
//
// Parameters:
//   m     - Number of rows of A
//   n     - Number of columns of A
//   a     - On entry: m-by-n matrix A
//           On exit: upper triangle contains R; below diagonal contains Householder vectors
//   lda   - Leading dimension of A
//   tau   - Array of length min(m, n) containing scalar factors of Householder reflectors
//   work  - Workspace of size at least n
dgeqrf :: proc(m: int, n: int, a: []f64, lda: int, tau: []f64, work: []f64) {
    k := min(m, n)

    for i in 0 ..< k {
        // Generate elementary reflector H(i) to annihilate A(i+1:m-1, i)
        // First, extract the column from row i onwards
        col_len := m - i

        if col_len > 1 {
            // Get the subcolumn starting at (i, i)
            alpha := a[i * lda + i]

            // Compute norm of subcolumn below diagonal
            xnorm: f64 = 0.0
            for j in i + 1 ..< m {
                xnorm += a[j * lda + i] * a[j * lda + i]
            }
            xnorm = math.sqrt(xnorm)

            if xnorm == 0.0 && alpha >= 0.0 {
                tau[i] = 0.0
            } else {
                // beta = -sign(alpha) * ||[alpha; x]||
                norm_full := math.sqrt(alpha * alpha + xnorm * xnorm)
                beta := -math.copy_sign(norm_full, alpha)

                // tau = (beta - alpha) / beta
                tau[i] = (beta - alpha) / beta

                // v = [alpha - beta; x] / (alpha - beta)
                scale := 1.0 / (alpha - beta)
                for j in i + 1 ..< m {
                    a[j * lda + i] *= scale
                }

                // Store beta on the diagonal
                a[i * lda + i] = beta
            }
        } else {
            tau[i] = 0.0
        }

        // Apply H(i) to A(i:m-1, i+1:n-1) from the left
        if i + 1 < n && tau[i] != 0.0 {
            // Save the diagonal element and set it to 1 for the reflector
            aii := a[i * lda + i]
            a[i * lda + i] = 1.0

            // work = A(i:m-1, i+1:n-1)^T * v
            for j in i + 1 ..< n {
                work[j - (i + 1)] = 0.0
                for jj in i ..< m {
                    work[j - (i + 1)] += a[jj * lda + j] * a[jj * lda + i]
                }
            }

            // A(i:m-1, i+1:n-1) -= tau * v * work^T
            for j in i + 1 ..< n {
                for jj in i ..< m {
                    a[jj * lda + j] -= tau[i] * a[jj * lda + i] * work[j - (i + 1)]
                }
            }

            // Restore the diagonal
            a[i * lda + i] = aii
        }
    }
}

// dorgqr generates an m-by-m orthogonal matrix Q from the output of dgeqrf.
//
// Q is defined as the product of k elementary reflectors:
//    Q = H(0) * H(1) * ... * H(k-1)
// where k = min(m, n).
//
// Parameters:
//   m     - Number of rows of Q
//   n     - Number of columns originally in A (determines k = min(m, n))
//   a     - On entry: output from dgeqrf (Householder vectors below diagonal)
//           Not modified
//   lda   - Leading dimension of A
//   tau   - Scalar factors from dgeqrf
//   q     - On exit: m-by-m orthogonal matrix Q
//   ldq   - Leading dimension of Q
//   work  - Workspace of size at least m
dorgqr :: proc(m: int, n: int, a: []f64, lda: int, tau: []f64, q: []f64, ldq: int, work: []f64) {
    // Initialize Q to identity
    for i in 0 ..< m {
        for j in 0 ..< m {
            q[i * ldq + j] = (i == j) ? 1.0 : 0.0
        }
    }

    k := min(m, n)

    // Apply H(k-1), H(k-2), ..., H(0) to Q
    for ii in 0 ..< k {
        i := k - 1 - ii  // Go backwards

        if tau[i] == 0.0 {
            continue
        }

        // Construct v: v[i] = 1, v[i+1:m] from a
        // Apply H(i) = I - tau * v * v^T to Q from the left
        // Q := H(i) * Q = Q - tau * v * (v^T * Q)

        // For each column j of Q, compute v^T * Q(:, j)
        for j in 0 ..< m {
            // Compute v^T * Q(:, j)
            dot: f64 = q[i * ldq + j]  // v[i] = 1
            for jj in i + 1 ..< m {
                dot += a[jj * lda + i] * q[jj * ldq + j]
            }

            // Q(i:m-1, j) -= tau * v * dot
            q[i * ldq + j] -= tau[i] * dot  // v[i] = 1
            for jj in i + 1 ..< m {
                q[jj * ldq + j] -= tau[i] * a[jj * lda + i] * dot
            }
        }
    }
}

// extract_r extracts the R factor from the output of dgeqrf.
// R is the upper triangular (or trapezoidal if m < n) part of A.
//
// Parameters:
//   m     - Number of rows of A and R
//   n     - Number of columns of A and R
//   a     - Output from dgeqrf
//   lda   - Leading dimension of A
//   r     - On exit: m-by-n upper triangular/trapezoidal matrix R
//   ldr   - Leading dimension of R
extract_r :: proc(m: int, n: int, a: []f64, lda: int, r: []f64, ldr: int) {
    for i in 0 ..< m {
        for j in 0 ..< n {
            if j >= i {
                r[i * ldr + j] = a[i * lda + j]
            } else {
                r[i * ldr + j] = 0.0
            }
        }
    }
}

// ============================================================================
// BLAS Level 3 - Matrix-Matrix operations (basic)
// ============================================================================

// dgemm performs one of the matrix-matrix operations
//    C := alpha*op(A)*op(B) + beta*C
// where op(X) is one of op(X) = X or op(X) = X**T
//
// This is a straightforward O(n^3) implementation for correctness.
// For high performance, this would need blocking and SIMD optimization.
dgemm :: proc(
    transa: Transpose,
    transb: Transpose,
    m: int,
    n: int,
    k: int,
    alpha: f64,
    a: []f64,
    lda: int,
    b: []f64,
    ldb: int,
    beta: f64,
    c: []f64,
    ldc: int,
) {
    ZERO :: 0.0
    ONE :: 1.0

    // Quick return if possible
    if m == 0 || n == 0 || ((alpha == ZERO || k == 0) && beta == ONE) {
        return
    }

    // Scale C by beta
    if beta == ZERO {
        for i in 0 ..< m {
            for j in 0 ..< n {
                c[i * ldc + j] = ZERO
            }
        }
    } else if beta != ONE {
        for i in 0 ..< m {
            for j in 0 ..< n {
                c[i * ldc + j] *= beta
            }
        }
    }

    if alpha == ZERO {
        return
    }

    // Compute C := alpha*op(A)*op(B) + C
    nota := transa == .No_Trans
    notb := transb == .No_Trans

    if nota {
        if notb {
            // C := alpha*A*B + C
            for i in 0 ..< m {
                for l in 0 ..< k {
                    temp := alpha * a[i * lda + l]
                    for j in 0 ..< n {
                        c[i * ldc + j] += temp * b[l * ldb + j]
                    }
                }
            }
        } else {
            // C := alpha*A*B**T + C
            for i in 0 ..< m {
                for j in 0 ..< n {
                    temp: f64 = ZERO
                    for l in 0 ..< k {
                        temp += a[i * lda + l] * b[j * ldb + l]
                    }
                    c[i * ldc + j] += alpha * temp
                }
            }
        }
    } else {
        if notb {
            // C := alpha*A**T*B + C
            for i in 0 ..< m {
                for l in 0 ..< k {
                    temp := alpha * a[l * lda + i]
                    for j in 0 ..< n {
                        c[i * ldc + j] += temp * b[l * ldb + j]
                    }
                }
            }
        } else {
            // C := alpha*A**T*B**T + C
            for i in 0 ..< m {
                for j in 0 ..< n {
                    temp: f64 = ZERO
                    for l in 0 ..< k {
                        temp += a[l * lda + i] * b[j * ldb + l]
                    }
                    c[i * ldc + j] += alpha * temp
                }
            }
        }
    }
}

// dtrsm solves one of the matrix equations
//    op(A)*X = alpha*B, or X*op(A) = alpha*B
// where alpha is a scalar, X and B are m by n matrices, A is a unit, or
// non-unit, upper or lower triangular matrix and op(A) is one of
//    op(A) = A or op(A) = A**T
// The matrix X is overwritten on B.
dtrsm :: proc(
    side: Side,
    uplo: Uplo,
    transa: Transpose,
    diag: Diag,
    m: int,
    n: int,
    alpha: f64,
    a: []f64,
    lda: int,
    b: []f64,
    ldb: int,
) {
    ZERO :: 0.0
    ONE :: 1.0

    // Quick return if possible
    if m == 0 || n == 0 {
        return
    }

    // Scale B by alpha if alpha is zero
    if alpha == ZERO {
        for i in 0 ..< m {
            for j in 0 ..< n {
                b[i * ldb + j] = ZERO
            }
        }
        return
    }

    nounit := diag == .Non_Unit
    lside := side == .Left

    if lside {
        // Form B := alpha*inv(A)*B or B := alpha*inv(A**T)*B
        if transa == .No_Trans {
            // Form B := alpha*inv(A)*B
            if uplo == .Upper {
                for j in 0 ..< n {
                    if alpha != ONE {
                        for i in 0 ..< m {
                            b[i * ldb + j] *= alpha
                        }
                    }
                    for k := m - 1; k >= 0; k -= 1 {
                        if b[k * ldb + j] != ZERO {
                            if nounit {
                                b[k * ldb + j] /= a[k * lda + k]
                            }
                            for i in 0 ..< k {
                                b[i * ldb + j] -= b[k * ldb + j] * a[i * lda + k]
                            }
                        }
                    }
                }
            } else {
                for j in 0 ..< n {
                    if alpha != ONE {
                        for i in 0 ..< m {
                            b[i * ldb + j] *= alpha
                        }
                    }
                    for k in 0 ..< m {
                        if b[k * ldb + j] != ZERO {
                            if nounit {
                                b[k * ldb + j] /= a[k * lda + k]
                            }
                            for i in k + 1 ..< m {
                                b[i * ldb + j] -= b[k * ldb + j] * a[i * lda + k]
                            }
                        }
                    }
                }
            }
        } else {
            // Form B := alpha*inv(A**T)*B
            if uplo == .Upper {
                for j in 0 ..< n {
                    for i in 0 ..< m {
                        temp := alpha * b[i * ldb + j]
                        for k in 0 ..< i {
                            temp -= a[k * lda + i] * b[k * ldb + j]
                        }
                        if nounit {
                            temp /= a[i * lda + i]
                        }
                        b[i * ldb + j] = temp
                    }
                }
            } else {
                for j in 0 ..< n {
                    for i := m - 1; i >= 0; i -= 1 {
                        temp := alpha * b[i * ldb + j]
                        for k in i + 1 ..< m {
                            temp -= a[k * lda + i] * b[k * ldb + j]
                        }
                        if nounit {
                            temp /= a[i * lda + i]
                        }
                        b[i * ldb + j] = temp
                    }
                }
            }
        }
    } else {
        // Form B := alpha*B*inv(A) or B := alpha*B*inv(A**T)
        if transa == .No_Trans {
            // Form B := alpha*B*inv(A)
            if uplo == .Upper {
                for j in 0 ..< n {
                    if alpha != ONE {
                        for i in 0 ..< m {
                            b[i * ldb + j] *= alpha
                        }
                    }
                    for k in 0 ..< j {
                        if a[k * lda + j] != ZERO {
                            for i in 0 ..< m {
                                b[i * ldb + j] -= a[k * lda + j] * b[i * ldb + k]
                            }
                        }
                    }
                    if nounit {
                        temp := ONE / a[j * lda + j]
                        for i in 0 ..< m {
                            b[i * ldb + j] *= temp
                        }
                    }
                }
            } else {
                for j := n - 1; j >= 0; j -= 1 {
                    if alpha != ONE {
                        for i in 0 ..< m {
                            b[i * ldb + j] *= alpha
                        }
                    }
                    for k in j + 1 ..< n {
                        if a[k * lda + j] != ZERO {
                            for i in 0 ..< m {
                                b[i * ldb + j] -= a[k * lda + j] * b[i * ldb + k]
                            }
                        }
                    }
                    if nounit {
                        temp := ONE / a[j * lda + j]
                        for i in 0 ..< m {
                            b[i * ldb + j] *= temp
                        }
                    }
                }
            }
        } else {
            // Form B := alpha*B*inv(A**T)
            if uplo == .Upper {
                for k := n - 1; k >= 0; k -= 1 {
                    if nounit {
                        temp := ONE / a[k * lda + k]
                        for i in 0 ..< m {
                            b[i * ldb + k] *= temp
                        }
                    }
                    for j in 0 ..< k {
                        if a[j * lda + k] != ZERO {
                            temp := a[j * lda + k]
                            for i in 0 ..< m {
                                b[i * ldb + j] -= temp * b[i * ldb + k]
                            }
                        }
                    }
                    if alpha != ONE {
                        for i in 0 ..< m {
                            b[i * ldb + k] *= alpha
                        }
                    }
                }
            } else {
                for k in 0 ..< n {
                    if nounit {
                        temp := ONE / a[k * lda + k]
                        for i in 0 ..< m {
                            b[i * ldb + k] *= temp
                        }
                    }
                    for j in k + 1 ..< n {
                        if a[j * lda + k] != ZERO {
                            temp := a[j * lda + k]
                            for i in 0 ..< m {
                                b[i * ldb + j] -= temp * b[i * ldb + k]
                            }
                        }
                    }
                    if alpha != ONE {
                        for i in 0 ..< m {
                            b[i * ldb + k] *= alpha
                        }
                    }
                }
            }
        }
    }
}
