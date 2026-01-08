package blas

import "core:math"

// ============================================================================
// BLAS Level 1 routines - Vector operations
// Direct translation from Reference BLAS Fortran code
// ============================================================================

// dcopy copies a vector x to a vector y.
// Translated from Reference BLAS dcopy.f
dcopy :: proc(n: int, x: []f64, incx: int, y: []f64, incy: int) {
    if n <= 0 {
        return
    }

    if incx == 1 && incy == 1 {
        // Code for both increments equal to 1
        // Clean-up loop
        m := n % 7
        for i in 0 ..< m {
            y[i] = x[i]
        }
        if n < 7 {
            return
        }
        // Unrolled loop
        for i := m; i < n; i += 7 {
            y[i] = x[i]
            y[i + 1] = x[i + 1]
            y[i + 2] = x[i + 2]
            y[i + 3] = x[i + 3]
            y[i + 4] = x[i + 4]
            y[i + 5] = x[i + 5]
            y[i + 6] = x[i + 6]
        }
    } else {
        // Code for unequal increments or equal increments not equal to 1
        ix := 0
        iy := 0
        if incx < 0 {
            ix = (-n + 1) * incx
        }
        if incy < 0 {
            iy = (-n + 1) * incy
        }
        for _ in 0 ..< n {
            y[iy] = x[ix]
            ix += incx
            iy += incy
        }
    }
}

// dscal scales a vector by a constant.
// Translated from Reference BLAS dscal.f
dscal :: proc(n: int, da: f64, x: []f64, incx: int) {
    if n <= 0 || incx <= 0 || da == 1.0 {
        return
    }

    if incx == 1 {
        // Code for increment equal to 1
        // Clean-up loop
        m := n % 5
        for i in 0 ..< m {
            x[i] = da * x[i]
        }
        if n < 5 {
            return
        }
        // Unrolled loop
        for i := m; i < n; i += 5 {
            x[i] = da * x[i]
            x[i + 1] = da * x[i + 1]
            x[i + 2] = da * x[i + 2]
            x[i + 3] = da * x[i + 3]
            x[i + 4] = da * x[i + 4]
        }
    } else {
        // Code for increment not equal to 1
        nincx := n * incx
        for i := 0; i < nincx; i += incx {
            x[i] = da * x[i]
        }
    }
}

// daxpy computes y := a*x + y
// Translated from Reference BLAS daxpy.f
daxpy :: proc(n: int, da: f64, x: []f64, incx: int, y: []f64, incy: int) {
    if n <= 0 {
        return
    }
    if da == 0.0 {
        return
    }

    if incx == 1 && incy == 1 {
        // Code for both increments equal to 1
        // Clean-up loop
        m := n % 4
        for i in 0 ..< m {
            y[i] = y[i] + da * x[i]
        }
        if n < 4 {
            return
        }
        // Unrolled loop
        for i := m; i < n; i += 4 {
            y[i] = y[i] + da * x[i]
            y[i + 1] = y[i + 1] + da * x[i + 1]
            y[i + 2] = y[i + 2] + da * x[i + 2]
            y[i + 3] = y[i + 3] + da * x[i + 3]
        }
    } else {
        // Code for unequal increments or equal increments not equal to 1
        ix := 0
        iy := 0
        if incx < 0 {
            ix = (-n + 1) * incx
        }
        if incy < 0 {
            iy = (-n + 1) * incy
        }
        for _ in 0 ..< n {
            y[iy] = y[iy] + da * x[ix]
            ix += incx
            iy += incy
        }
    }
}

// ddot forms the dot product of two vectors.
// Translated from Reference BLAS ddot.f
ddot :: proc(n: int, x: []f64, incx: int, y: []f64, incy: int) -> f64 {
    dtemp: f64 = 0.0

    if n <= 0 {
        return 0.0
    }

    if incx == 1 && incy == 1 {
        // Code for both increments equal to 1
        // Clean-up loop
        m := n % 5
        for i in 0 ..< m {
            dtemp = dtemp + x[i] * y[i]
        }
        if n < 5 {
            return dtemp
        }
        // Unrolled loop
        for i := m; i < n; i += 5 {
            dtemp = dtemp + x[i] * y[i] + x[i + 1] * y[i + 1] +
                    x[i + 2] * y[i + 2] + x[i + 3] * y[i + 3] + x[i + 4] * y[i + 4]
        }
    } else {
        // Code for unequal increments or equal increments not equal to 1
        ix := 0
        iy := 0
        if incx < 0 {
            ix = (-n + 1) * incx
        }
        if incy < 0 {
            iy = (-n + 1) * incy
        }
        for _ in 0 ..< n {
            dtemp = dtemp + x[ix] * y[iy]
            ix += incx
            iy += incy
        }
    }
    return dtemp
}

// dnrm2 computes the Euclidean norm of a vector.
// Translated from Reference BLAS dnrm2.f90
// Uses Blue's scaling algorithm for numerical stability
dnrm2 :: proc(n: int, x: []f64, incx: int) -> f64 {
    ZERO :: 0.0
    ONE :: 1.0

    if n <= 0 {
        return ZERO
    }

    // Blue's scaling constants for f64
    // These prevent overflow and underflow
    RADIX :: 2.0  // Base of floating point

    // For f64: minexponent = -1021, maxexponent = 1024, digits = 53
    MIN_EXP :: -1021
    MAX_EXP :: 1024
    DIGITS :: 53

    // tsml = RADIX ^ ceiling((minexponent - 1) * 0.5)
    // tbig = RADIX ^ floor((maxexponent - digits + 1) * 0.5)
    // ssml = RADIX ^ (-floor((minexponent - digits) * 0.5))
    // sbig = RADIX ^ (-ceiling((maxexponent + digits - 1) * 0.5))

    tsml := math.pow(RADIX, f64((-MIN_EXP - 1 + 1) / 2))  // ~1.49e-154
    tbig := math.pow(RADIX, f64((MAX_EXP - DIGITS + 1) / 2))  // ~1.99e+146
    ssml := math.pow(RADIX, f64(-(-MIN_EXP - DIGITS) / 2))  // ~4.47e+161
    sbig := math.pow(RADIX, f64(-(MAX_EXP + DIGITS - 1 + 1) / 2))  // ~1.12e-162

    // Compute the sum of squares in 3 accumulators:
    // abig -- sums of squares scaled down to avoid overflow
    // asml -- sums of squares scaled up to avoid underflow
    // amed -- sums of squares that do not require scaling

    notbig := true
    asml: f64 = ZERO
    amed: f64 = ZERO
    abig: f64 = ZERO

    ix := 0
    if incx < 0 {
        ix = -(n - 1) * incx
    }

    for _ in 0 ..< n {
        ax := abs(x[ix])
        if ax > tbig {
            abig = abig + (ax * sbig) * (ax * sbig)
            notbig = false
        } else if ax < tsml {
            if notbig {
                asml = asml + (ax * ssml) * (ax * ssml)
            }
        } else {
            amed = amed + ax * ax
        }
        ix += incx
    }

    // Combine accumulators
    scl: f64
    sumsq: f64

    if abig > ZERO {
        // Combine abig and amed if abig > 0
        if amed > ZERO || amed != amed {  // amed != amed checks for NaN
            abig = abig + (amed * sbig) * sbig
        }
        scl = ONE / sbig
        sumsq = abig
    } else if asml > ZERO {
        // Combine amed and asml if asml > 0
        if amed > ZERO || amed != amed {
            amed_sqrt := math.sqrt(amed)
            asml_sqrt := math.sqrt(asml) / ssml
            ymin, ymax: f64
            if asml_sqrt > amed_sqrt {
                ymin = amed_sqrt
                ymax = asml_sqrt
            } else {
                ymin = asml_sqrt
                ymax = amed_sqrt
            }
            scl = ONE
            sumsq = ymax * ymax * (ONE + (ymin / ymax) * (ymin / ymax))
        } else {
            scl = ONE / ssml
            sumsq = asml
        }
    } else {
        // Otherwise all values are mid-range
        scl = ONE
        sumsq = amed
    }

    return scl * math.sqrt(sumsq)
}

// drot applies a plane rotation.
// Translated from Reference BLAS drot.f
drot :: proc(n: int, x: []f64, incx: int, y: []f64, incy: int, c: f64, s: f64) {
    if n <= 0 {
        return
    }

    if incx == 1 && incy == 1 {
        // Code for both increments equal to 1
        for i in 0 ..< n {
            dtemp := c * x[i] + s * y[i]
            y[i] = c * y[i] - s * x[i]
            x[i] = dtemp
        }
    } else {
        // Code for unequal increments or equal increments not equal to 1
        ix := 0
        iy := 0
        if incx < 0 {
            ix = (-n + 1) * incx
        }
        if incy < 0 {
            iy = (-n + 1) * incy
        }
        for _ in 0 ..< n {
            dtemp := c * x[ix] + s * y[iy]
            y[iy] = c * y[iy] - s * x[ix]
            x[ix] = dtemp
            ix += incx
            iy += incy
        }
    }
}

// dasum computes the sum of absolute values.
dasum :: proc(n: int, x: []f64, incx: int) -> f64 {
    if n <= 0 || incx <= 0 {
        return 0.0
    }

    dtemp: f64 = 0.0

    if incx == 1 {
        // Clean-up loop
        m := n % 6
        for i in 0 ..< m {
            dtemp += abs(x[i])
        }
        if n < 6 {
            return dtemp
        }
        // Unrolled loop
        for i := m; i < n; i += 6 {
            dtemp += abs(x[i]) + abs(x[i + 1]) + abs(x[i + 2]) +
                     abs(x[i + 3]) + abs(x[i + 4]) + abs(x[i + 5])
        }
    } else {
        nincx := n * incx
        for i := 0; i < nincx; i += incx {
            dtemp += abs(x[i])
        }
    }
    return dtemp
}

// idamax finds the index of element having maximum absolute value.
// Returns 0-based index (unlike Fortran which returns 1-based).
idamax :: proc(n: int, x: []f64, incx: int) -> int {
    if n < 1 || incx <= 0 {
        return -1
    }
    if n == 1 {
        return 0
    }

    if incx == 1 {
        // Code for increment equal to 1
        dmax := abs(x[0])
        idmax := 0
        for i in 1 ..< n {
            if abs(x[i]) > dmax {
                idmax = i
                dmax = abs(x[i])
            }
        }
        return idmax
    } else {
        // Code for increment not equal to 1
        ix := 0
        dmax := abs(x[0])
        idmax := 0
        ix += incx
        for i in 1 ..< n {
            if abs(x[ix]) > dmax {
                idmax = i
                dmax = abs(x[ix])
            }
            ix += incx
        }
        return idmax
    }
}

// dswap interchanges two vectors.
dswap :: proc(n: int, x: []f64, incx: int, y: []f64, incy: int) {
    if n <= 0 {
        return
    }

    if incx == 1 && incy == 1 {
        // Code for both increments equal to 1
        // Clean-up loop
        m := n % 3
        for i in 0 ..< m {
            dtemp := x[i]
            x[i] = y[i]
            y[i] = dtemp
        }
        if n < 3 {
            return
        }
        // Unrolled loop
        for i := m; i < n; i += 3 {
            dtemp := x[i]
            x[i] = y[i]
            y[i] = dtemp
            dtemp = x[i + 1]
            x[i + 1] = y[i + 1]
            y[i + 1] = dtemp
            dtemp = x[i + 2]
            x[i + 2] = y[i + 2]
            y[i + 2] = dtemp
        }
    } else {
        ix := 0
        iy := 0
        if incx < 0 {
            ix = (-n + 1) * incx
        }
        if incy < 0 {
            iy = (-n + 1) * incy
        }
        for _ in 0 ..< n {
            dtemp := x[ix]
            x[ix] = y[iy]
            y[iy] = dtemp
            ix += incx
            iy += incy
        }
    }
}
