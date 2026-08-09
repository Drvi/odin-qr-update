package blas

import "core:math"

// ============================================================================
// LAPACK Auxiliary routines
// Direct translation from Reference LAPACK Fortran code
// ============================================================================

// Machine constants for f64
// These are computed based on IEEE 754 double precision format
SAFMIN :: 2.2250738585072014e-308  // Smallest normalized positive number
SAFMAX :: 4.4942328371557898e+307  // 1/SAFMIN (approximately)
EPS :: 2.2204460492503131e-16      // Machine epsilon

// dlapy2 returns sqrt(x**2 + y**2) without overflow or destructive underflow
// for intermediate values. Translated from Reference LAPACK dlapy2.f.
//
// Squaring the arguments directly overflows to Inf once |x| exceeds about
// 1.3e154, which is well inside the range of values a well-scaled f64 problem
// can legitimately contain.
dlapy2 :: proc "contextless" (x: f64, y: f64) -> f64 {
    xabs := abs(x)
    yabs := abs(y)

    // NaN propagates rather than being swallowed by the max/min below.
    if xabs != xabs {
        return xabs
    }
    if yabs != yabs {
        return yabs
    }

    w := max(xabs, yabs)
    z := min(xabs, yabs)

    if z == 0.0 {
        return w
    }
    ratio := z / w
    return w * math.sqrt_f64(1.0 + ratio * ratio)
}

// dlartg generates a plane rotation with real cosine and real sine.
//
// Given the Cartesian coordinates (f, g) of a point, this routine computes
// the parameters c, s, and r such that:
//
//    [  c   s ] [ f ]   [ r ]
//    [ -s   c ] [ g ] = [ 0 ]
//
// where r = sign(f) * sqrt(f^2 + g^2), and c, s are:
//    c = f / r
//    s = g / r
//
// The routine handles scaling to prevent overflow/underflow.
// Translated from Reference LAPACK dlartg.f90 (February 2021)
dlartg :: proc "contextless" (f: f64, g: f64) -> (c: f64, s: f64, r: f64) {
    ZERO :: 0.0
    ONE :: 1.0
    HALF :: 0.5

    // Compute safe min/max for scaling
    rtmin := math.sqrt_f64(SAFMIN)
    rtmax := math.sqrt_f64(SAFMAX / 2.0)

    f1 := abs(f)
    g1 := abs(g)

    if g == ZERO {
        c = ONE
        s = ZERO
        r = f
    } else if f == ZERO {
        c = ZERO
        s = ONE if g >= ZERO else -ONE
        r = g1
    } else if f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax {
        // Normal case: values are in safe range
        d := math.sqrt_f64(f * f + g * g)
        c = f1 / d
        r = d if f >= ZERO else -d
        s = g / r
    } else {
        // Need to scale to avoid overflow/underflow
        u := min(SAFMAX, max(SAFMIN, f1, g1))
        fs := f / u
        gs := g / u
        d := math.sqrt_f64(fs * fs + gs * gs)
        c = abs(fs) / d
        r = d if f >= ZERO else -d
        s = gs / r
        r = r * u
    }

    return c, s, r
}

// dlartgp generates a plane rotation so that the diagonal is nonnegative.
// This is like dlartg but ensures r >= 0.
dlartgp :: proc "contextless" (f: f64, g: f64) -> (c: f64, s: f64, r: f64) {
    c, s, r = dlartg(f, g)
    if r < 0.0 {
        c = -c
        s = -s
        r = -r
    }
    return c, s, r
}

// drotg constructs a Givens plane rotation.
// This is the BLAS level 1 routine, simpler than dlartg.
// Given (a, b), computes (c, s, r, z) such that:
//    [  c  s ] [ a ]   [ r ]
//    [ -s  c ] [ b ] = [ 0 ]
drotg :: proc "contextless" (a: f64, b: f64) -> (c: f64, s: f64, r: f64, z: f64) {
    roe := b
    if abs(a) > abs(b) {
        roe = a
    }

    scale := abs(a) + abs(b)
    if scale == 0.0 {
        c = 1.0
        s = 0.0
        r = 0.0
        z = 0.0
    } else {
        r = scale * math.sqrt_f64((a / scale) * (a / scale) + (b / scale) * (b / scale))
        if roe < 0.0 {
            r = -r
        }
        c = a / r
        s = b / r
        z = 1.0
        if abs(a) > abs(b) {
            z = s
        }
        if abs(b) >= abs(a) && c != 0.0 {
            z = 1.0 / c
        }
    }

    return c, s, r, z
}

// dlarfg generates an elementary reflector H of order n, such that
//    H * ( alpha ) = ( beta ), H**T * H = I.
//        (   x   )   (   0  )
//
// H is represented in the form
//    H = I - tau * ( 1 ) * ( 1 v**T ),
//                  ( v )
// where tau is a real scalar and v is a real (n-1)-element vector.
//
// If the elements of x are all zero and alpha is non-negative, tau = 0
// and H is taken to be the unit matrix.
//
// Parameters:
//   n     - Order of the elementary reflector
//   alpha - On entry, the value alpha. On exit, overwritten by beta.
//   x     - Vector of length (n-1). On entry, the vector x. On exit, overwritten by v.
//   incx  - Increment between elements of x
//
// Returns:
//   beta  - The value beta
//   tau   - The value tau
dlarfg :: proc "contextless" (n: int, alpha_in: f64, x: []f64, incx: int) -> (beta: f64, tau: f64) {
    ZERO :: 0.0
    ONE :: 1.0

    alpha := alpha_in  // Make a local mutable copy

    if n <= 1 {
        tau = ZERO
        beta = alpha
        return
    }

    xnorm := dnrm2(n - 1, x, incx)

    if xnorm == ZERO && alpha >= ZERO {
        // H = I
        tau = ZERO
        beta = alpha
        return
    }

    // General case
    beta = -math.copy_sign_f64(dlapy2(alpha, xnorm), alpha)

    // Scale if beta is too small
    safmin := SAFMIN / EPS
    knt := 0

    if abs(beta) < safmin {
        // Need to scale x and recalculate beta
        rsafmn := ONE / safmin
        for abs(beta) < safmin {
            knt += 1
            dscal(n - 1, rsafmn, x, incx)
            beta = beta * rsafmn
            alpha = alpha * rsafmn
        }
        // New beta is at most 1, at least safmin
        xnorm = dnrm2(n - 1, x, incx)
        beta = -math.copy_sign_f64(dlapy2(alpha, xnorm), alpha)
    }

    tau = (beta - alpha) / beta
    dscal(n - 1, ONE / (alpha - beta), x, incx)

    // If alpha is subnormal, it may lose relative accuracy
    for _ in 0 ..< knt {
        beta = beta * safmin
    }

    return beta, tau
}

// dlarf applies an elementary reflector H to a real m by n matrix C,
// from either the left or the right. H is represented in the form
//    H = I - tau * v * v**T
// where tau is a real scalar and v is a real vector.
//
// If tau = 0, then H is taken to be the unit matrix.
//
// Parameters:
//   side  - 'L': form H * C; 'R': form C * H
//   m     - Number of rows of matrix C
//   n     - Number of columns of matrix C
//   v     - Vector of length m (if side = 'L') or n (if side = 'R')
//   incv  - Increment between elements of v
//   tau   - Value tau in the representation of H
//   c     - On entry, the m by n matrix C. On exit, C is overwritten by H * C or C * H.
//   ldc   - Leading dimension of C
//   work  - Workspace array of length n (if side = 'L') or m (if side = 'R')
dlarf :: proc "contextless" (
    side: Side,
    m: int,
    n: int,
    v: []f64,
    incv: int,
    tau: f64,
    c: []f64,
    ldc: int,
    work: []f64,
) {
    ZERO :: 0.0
    ONE :: 1.0

    if tau == ZERO {
        return
    }

    applyleft := side == .Left

    if applyleft {
        // Form H * C
        // w := C**T * v
        dgemv(.Trans, m, n, ONE, c, ldc, v, incv, ZERO, work, 1)
        // C := C - v * w**T
        dger(m, n, -tau, v, incv, work, 1, c, ldc)
    } else {
        // Form C * H
        // w := C * v
        dgemv(.No_Trans, m, n, ONE, c, ldc, v, incv, ZERO, work, 1)
        // C := C - w * v**T
        dger(m, n, -tau, work, 1, v, incv, c, ldc)
    }
}

// dlaset initializes an m-by-n matrix A to beta on the diagonal and alpha
// on the offdiagonals.
dlaset :: proc "contextless" (uplo: Uplo, m: int, n: int, alpha: f64, beta_diag: f64, a: []f64, lda: int) {
    if uplo == .Upper {
        // Set the upper triangle of A to alpha
        for j in 0 ..< n {
            for i in 0 ..< min(j, m) {
                a[i * lda + j] = alpha
            }
            if j < m {
                a[j * lda + j] = beta_diag
            }
        }
    } else if uplo == .Lower {
        // Set the lower triangle of A to alpha
        for j in 0 ..< n {
            if j < m {
                a[j * lda + j] = beta_diag
            }
            for i in j + 1 ..< m {
                a[i * lda + j] = alpha
            }
        }
    } else {
        // Set the full matrix
        for j in 0 ..< n {
            for i in 0 ..< m {
                a[i * lda + j] = alpha
            }
        }
        for i in 0 ..< min(m, n) {
            a[i * lda + i] = beta_diag
        }
    }
}
