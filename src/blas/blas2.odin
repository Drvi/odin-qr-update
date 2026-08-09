package blas

// ============================================================================
// BLAS Level 2 routines - Matrix-Vector operations
// Direct translation from Reference BLAS Fortran code
// ============================================================================

// Transpose flags
Transpose :: enum {
    No_Trans,   // 'N' - No transpose
    Trans,      // 'T' - Transpose
    Conj_Trans, // 'C' - Conjugate transpose (same as Trans for real)
}

// Triangular matrix type
Uplo :: enum {
    Upper, // 'U' - Upper triangular
    Lower, // 'L' - Lower triangular
}

// Diagonal type
Diag :: enum {
    Non_Unit, // 'N' - Non-unit diagonal
    Unit,     // 'U' - Unit diagonal
}

// Side for matrix operations
Side :: enum {
    Left,  // 'L'
    Right, // 'R'
}

// dgemv performs one of the matrix-vector operations
//    y := alpha*A*x + beta*y, or y := alpha*A**T*x + beta*y
// where alpha and beta are scalars, x and y are vectors and A is an
// m by n matrix.
// Translated from Reference BLAS dgemv.f
//
// Parameters:
//   trans - specifies the operation to be performed
//   m     - number of rows of matrix A
//   n     - number of columns of matrix A
//   alpha - scalar alpha
//   a     - matrix A stored in row-major order (m x n)
//   lda   - row stride of A: the distance between the starts of consecutive
//           rows. Must be at least n. (This is the row-major analogue of the
//           Fortran leading dimension, which is a column stride of at least m.)
//   x     - input vector
//   incx  - increment for x
//   beta  - scalar beta
//   y     - input/output vector
//   incy  - increment for y
dgemv :: proc(
    trans: Transpose,
    m: int,
    n: int,
    alpha: f64,
    a: []f64,
    lda: int,
    x: []f64,
    incx: int,
    beta: f64,
    y: []f64,
    incy: int,
) {
    ZERO :: 0.0
    ONE :: 1.0

    // Test the input parameters. A is row-major, so the row stride must span a
    // full row of n elements; requiring lda >= m (the Fortran rule) would
    // silently reject every tall matrix, which is the common shape here.
    if m < 0 || n < 0 || lda < max(1, n) || incx == 0 || incy == 0 {
        return  // Would call XERBLA in Fortran
    }

    // Quick return if possible
    if m == 0 || n == 0 || (alpha == ZERO && beta == ONE) {
        return
    }

    // Set lenx and leny, the lengths of the vectors x and y
    lenx, leny: int
    if trans == .No_Trans {
        lenx = n
        leny = m
    } else {
        lenx = m
        leny = n
    }

    // Set up the start points in x and y
    kx := 0
    ky := 0
    if incx < 0 {
        kx = -(lenx - 1) * incx
    }
    if incy < 0 {
        ky = -(leny - 1) * incy
    }

    // Start the operations. In this version the elements of A are
    // accessed sequentially with one pass through A.

    // First form y := beta*y
    if beta != ONE {
        if incy == 1 {
            if beta == ZERO {
                for i in 0 ..< leny {
                    y[i] = ZERO
                }
            } else {
                for i in 0 ..< leny {
                    y[i] = beta * y[i]
                }
            }
        } else {
            iy := ky
            if beta == ZERO {
                for _ in 0 ..< leny {
                    y[iy] = ZERO
                    iy += incy
                }
            } else {
                for _ in 0 ..< leny {
                    y[iy] = beta * y[iy]
                    iy += incy
                }
            }
        }
    }

    if alpha == ZERO {
        return
    }

    if trans == .No_Trans {
        // Form y := alpha*A*x + y
        jx := kx
        if incy == 1 {
            for j in 0 ..< n {
                temp := alpha * x[jx]
                for i in 0 ..< m {
                    y[i] = y[i] + temp * a[i * lda + j]
                }
                jx += incx
            }
        } else {
            for j in 0 ..< n {
                temp := alpha * x[jx]
                iy := ky
                for i in 0 ..< m {
                    y[iy] = y[iy] + temp * a[i * lda + j]
                    iy += incy
                }
                jx += incx
            }
        }
    } else {
        // Form y := alpha*A**T*x + y
        jy := ky
        if incx == 1 {
            for j in 0 ..< n {
                temp: f64 = ZERO
                for i in 0 ..< m {
                    temp = temp + a[i * lda + j] * x[i]
                }
                y[jy] = y[jy] + alpha * temp
                jy += incy
            }
        } else {
            for j in 0 ..< n {
                temp: f64 = ZERO
                ix := kx
                for i in 0 ..< m {
                    temp = temp + a[i * lda + j] * x[ix]
                    ix += incx
                }
                y[jy] = y[jy] + alpha * temp
                jy += incy
            }
        }
    }
}

// dtrsv solves one of the systems of equations
//    A*x = b, or A**T*x = b
// where b and x are n element vectors and A is an n by n unit, or
// non-unit, upper or lower triangular matrix.
// Translated from Reference BLAS dtrsv.f
//
// A is row-major with row stride lda >= n; element (i,j) is a[i*lda + j].
// The Fortran algorithm carries over unchanged because it is written in terms
// of A(i,j) rather than in terms of the storage order -- only the memory
// access pattern differs, not the arithmetic.
dtrsv :: proc(
    uplo: Uplo,
    trans: Transpose,
    diag: Diag,
    n: int,
    a: []f64,
    lda: int,
    x: []f64,
    incx: int,
) {
    ZERO :: 0.0

    // Test the input parameters
    if n < 0 || lda < max(1, n) || incx == 0 {
        return  // Would call XERBLA in Fortran
    }

    // Quick return if possible
    if n == 0 {
        return
    }

    nounit := diag == .Non_Unit

    // Set up the start point in x if the increment is not unity
    kx := 0
    if incx < 0 {
        kx = -(n - 1) * incx
    } else if incx != 1 {
        kx = 0
    }

    // Start the operations
    if trans == .No_Trans {
        // Form x := inv(A)*x
        if uplo == .Upper {
            if incx == 1 {
                for j := n - 1; j >= 0; j -= 1 {
                    if x[j] != ZERO {
                        if nounit {
                            x[j] = x[j] / a[j * lda + j]
                        }
                        temp := x[j]
                        for i := j - 1; i >= 0; i -= 1 {
                            x[i] = x[i] - temp * a[i * lda + j]
                        }
                    }
                }
            } else {
                jx := kx + (n - 1) * incx
                for j := n - 1; j >= 0; j -= 1 {
                    if x[jx] != ZERO {
                        if nounit {
                            x[jx] = x[jx] / a[j * lda + j]
                        }
                        temp := x[jx]
                        ix := jx
                        for i := j - 1; i >= 0; i -= 1 {
                            ix -= incx
                            x[ix] = x[ix] - temp * a[i * lda + j]
                        }
                    }
                    jx -= incx
                }
            }
        } else {
            // Lower triangular
            if incx == 1 {
                for j in 0 ..< n {
                    if x[j] != ZERO {
                        if nounit {
                            x[j] = x[j] / a[j * lda + j]
                        }
                        temp := x[j]
                        for i in j + 1 ..< n {
                            x[i] = x[i] - temp * a[i * lda + j]
                        }
                    }
                }
            } else {
                jx := kx
                for j in 0 ..< n {
                    if x[jx] != ZERO {
                        if nounit {
                            x[jx] = x[jx] / a[j * lda + j]
                        }
                        temp := x[jx]
                        ix := jx
                        for i in j + 1 ..< n {
                            ix += incx
                            x[ix] = x[ix] - temp * a[i * lda + j]
                        }
                    }
                    jx += incx
                }
            }
        }
    } else {
        // Form x := inv(A**T)*x
        if uplo == .Upper {
            if incx == 1 {
                for j in 0 ..< n {
                    temp := x[j]
                    for i in 0 ..< j {
                        temp = temp - a[i * lda + j] * x[i]
                    }
                    if nounit {
                        temp = temp / a[j * lda + j]
                    }
                    x[j] = temp
                }
            } else {
                jx := kx
                for j in 0 ..< n {
                    temp := x[jx]
                    ix := kx
                    for i in 0 ..< j {
                        temp = temp - a[i * lda + j] * x[ix]
                        ix += incx
                    }
                    if nounit {
                        temp = temp / a[j * lda + j]
                    }
                    x[jx] = temp
                    jx += incx
                }
            }
        } else {
            // Lower triangular
            if incx == 1 {
                for j := n - 1; j >= 0; j -= 1 {
                    temp := x[j]
                    for i := n - 1; i > j; i -= 1 {
                        temp = temp - a[i * lda + j] * x[i]
                    }
                    if nounit {
                        temp = temp / a[j * lda + j]
                    }
                    x[j] = temp
                }
            } else {
                kx = kx + (n - 1) * incx
                jx := kx
                for j := n - 1; j >= 0; j -= 1 {
                    temp := x[jx]
                    ix := kx
                    for i := n - 1; i > j; i -= 1 {
                        temp = temp - a[i * lda + j] * x[ix]
                        ix -= incx
                    }
                    if nounit {
                        temp = temp / a[j * lda + j]
                    }
                    x[jx] = temp
                    jx -= incx
                }
            }
        }
    }
}

// dger performs the rank 1 operation A := alpha*x*y**T + A
// where alpha is a scalar, x is an m element vector, y is an n element
// vector and A is an m by n matrix stored row-major with row stride lda >= n.
dger :: proc(
    m: int,
    n: int,
    alpha: f64,
    x: []f64,
    incx: int,
    y: []f64,
    incy: int,
    a: []f64,
    lda: int,
) {
    ZERO :: 0.0

    // Test the input parameters. See dgemv: row-major storage means the
    // stride bound is n, not m.
    if m < 0 || n < 0 || lda < max(1, n) || incx == 0 || incy == 0 {
        return
    }

    // Quick return if possible
    if m == 0 || n == 0 || alpha == ZERO {
        return
    }

    // Set up the start points in x and y
    kx := 0
    ky := 0
    if incx < 0 {
        kx = -(m - 1) * incx
    }
    if incy < 0 {
        ky = -(n - 1) * incy
    }

    // Start the operations
    jy := ky
    if incx == 1 {
        for j in 0 ..< n {
            if y[jy] != ZERO {
                temp := alpha * y[jy]
                for i in 0 ..< m {
                    a[i * lda + j] = a[i * lda + j] + x[i] * temp
                }
            }
            jy += incy
        }
    } else {
        for j in 0 ..< n {
            if y[jy] != ZERO {
                temp := alpha * y[jy]
                ix := kx
                for i in 0 ..< m {
                    a[i * lda + j] = a[i * lda + j] + x[ix] * temp
                    ix += incx
                }
            }
            jy += incy
        }
    }
}

// dtrmv performs one of the matrix-vector operations
//    x := A*x, or x := A**T*x
// where x is an n element vector and A is an n by n unit, or non-unit,
// upper or lower triangular matrix.
dtrmv :: proc(
    uplo: Uplo,
    trans: Transpose,
    diag: Diag,
    n: int,
    a: []f64,
    lda: int,
    x: []f64,
    incx: int,
) {
    ZERO :: 0.0

    // Test the input parameters
    if n < 0 || lda < max(1, n) || incx == 0 {
        return
    }

    // Quick return if possible
    if n == 0 {
        return
    }

    nounit := diag == .Non_Unit

    // Set up the start point in x
    kx := 0
    if incx < 0 {
        kx = -(n - 1) * incx
    }

    // Start the operations
    if trans == .No_Trans {
        // Form x := A*x
        if uplo == .Upper {
            if incx == 1 {
                for j in 0 ..< n {
                    if x[j] != ZERO {
                        temp := x[j]
                        for i in 0 ..< j {
                            x[i] = x[i] + temp * a[i * lda + j]
                        }
                        if nounit {
                            x[j] = x[j] * a[j * lda + j]
                        }
                    }
                }
            } else {
                jx := kx
                for j in 0 ..< n {
                    if x[jx] != ZERO {
                        temp := x[jx]
                        ix := kx
                        for i in 0 ..< j {
                            x[ix] = x[ix] + temp * a[i * lda + j]
                            ix += incx
                        }
                        if nounit {
                            x[jx] = x[jx] * a[j * lda + j]
                        }
                    }
                    jx += incx
                }
            }
        } else {
            // Lower triangular
            if incx == 1 {
                for j := n - 1; j >= 0; j -= 1 {
                    if x[j] != ZERO {
                        temp := x[j]
                        for i := n - 1; i > j; i -= 1 {
                            x[i] = x[i] + temp * a[i * lda + j]
                        }
                        if nounit {
                            x[j] = x[j] * a[j * lda + j]
                        }
                    }
                }
            } else {
                kx = kx + (n - 1) * incx
                jx := kx
                for j := n - 1; j >= 0; j -= 1 {
                    if x[jx] != ZERO {
                        temp := x[jx]
                        ix := kx
                        for i := n - 1; i > j; i -= 1 {
                            x[ix] = x[ix] + temp * a[i * lda + j]
                            ix -= incx
                        }
                        if nounit {
                            x[jx] = x[jx] * a[j * lda + j]
                        }
                    }
                    jx -= incx
                }
            }
        }
    } else {
        // Form x := A**T*x
        if uplo == .Upper {
            if incx == 1 {
                for j := n - 1; j >= 0; j -= 1 {
                    temp := x[j]
                    if nounit {
                        temp = temp * a[j * lda + j]
                    }
                    for i := j - 1; i >= 0; i -= 1 {
                        temp = temp + a[i * lda + j] * x[i]
                    }
                    x[j] = temp
                }
            } else {
                jx := kx + (n - 1) * incx
                for j := n - 1; j >= 0; j -= 1 {
                    temp := x[jx]
                    ix := jx
                    if nounit {
                        temp = temp * a[j * lda + j]
                    }
                    for i := j - 1; i >= 0; i -= 1 {
                        ix -= incx
                        temp = temp + a[i * lda + j] * x[ix]
                    }
                    x[jx] = temp
                    jx -= incx
                }
            }
        } else {
            // Lower triangular
            if incx == 1 {
                for j in 0 ..< n {
                    temp := x[j]
                    if nounit {
                        temp = temp * a[j * lda + j]
                    }
                    for i in j + 1 ..< n {
                        temp = temp + a[i * lda + j] * x[i]
                    }
                    x[j] = temp
                }
            } else {
                jx := kx
                for j in 0 ..< n {
                    temp := x[jx]
                    ix := jx
                    if nounit {
                        temp = temp * a[j * lda + j]
                    }
                    for i in j + 1 ..< n {
                        ix += incx
                        temp = temp + a[i * lda + j] * x[ix]
                    }
                    x[jx] = temp
                    jx += incx
                }
            }
        }
    }
}
