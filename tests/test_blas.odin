package tests

import "core:fmt"
import "core:math"
import "../src/blas"

// Tolerance for floating point comparisons
EPSILON :: 1.0e-10

// Helper to check if two floats are approximately equal
approx_eq :: proc(a, b: f64, tol: f64 = EPSILON) -> bool {
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
}

// Helper to check if a matrix is approximately zero
is_zero :: proc(a: []f64, m, n, lda: int, tol: f64 = EPSILON) -> bool {
    for i in 0 ..< m {
        for j in 0 ..< n {
            if abs(a[i * lda + j]) > tol {
                return false
            }
        }
    }
    return true
}

// Helper to compute Frobenius norm
frobenius_norm :: proc(a: []f64, m, n, lda: int) -> f64 {
    sum: f64 = 0.0
    for i in 0 ..< m {
        for j in 0 ..< n {
            sum += a[i * lda + j] * a[i * lda + j]
        }
    }
    return math.sqrt_f64(sum)
}

// Helper to print a matrix
print_matrix :: proc(name: string, a: []f64, m, n, lda: int) {
    fmt.printf("%s (%d x %d):\n", name, m, n)
    for i in 0 ..< m {
        for j in 0 ..< n {
            fmt.printf("%10.6f ", a[i * lda + j])
        }
        fmt.println()
    }
    fmt.println()
}

// ============================================================================
// BLAS Level 1 Tests
// ============================================================================

test_dcopy :: proc() -> bool {
    fmt.println("Testing dcopy...")

    x := []f64{1.0, 2.0, 3.0, 4.0, 5.0}
    y := []f64{0.0, 0.0, 0.0, 0.0, 0.0}

    blas.dcopy(5, x, 1, y, 1)

    for i in 0 ..< 5 {
        if !approx_eq(x[i], y[i]) {
            fmt.printf("  FAILED: x[%d] = %f, y[%d] = %f\n", i, x[i], i, y[i])
            return false
        }
    }

    fmt.println("  PASSED")
    return true
}

test_dscal :: proc() -> bool {
    fmt.println("Testing dscal...")

    x := []f64{1.0, 2.0, 3.0, 4.0, 5.0}
    blas.dscal(5, 2.0, x, 1)

    expected := []f64{2.0, 4.0, 6.0, 8.0, 10.0}
    for i in 0 ..< 5 {
        if !approx_eq(x[i], expected[i]) {
            fmt.printf("  FAILED: x[%d] = %f, expected %f\n", i, x[i], expected[i])
            return false
        }
    }

    fmt.println("  PASSED")
    return true
}

test_daxpy :: proc() -> bool {
    fmt.println("Testing daxpy...")

    x := []f64{1.0, 2.0, 3.0, 4.0, 5.0}
    y := []f64{5.0, 4.0, 3.0, 2.0, 1.0}

    blas.daxpy(5, 2.0, x, 1, y, 1)

    expected := []f64{7.0, 8.0, 9.0, 10.0, 11.0}
    for i in 0 ..< 5 {
        if !approx_eq(y[i], expected[i]) {
            fmt.printf("  FAILED: y[%d] = %f, expected %f\n", i, y[i], expected[i])
            return false
        }
    }

    fmt.println("  PASSED")
    return true
}

test_ddot :: proc() -> bool {
    fmt.println("Testing ddot...")

    x := []f64{1.0, 2.0, 3.0, 4.0, 5.0}
    y := []f64{5.0, 4.0, 3.0, 2.0, 1.0}

    result := blas.ddot(5, x, 1, y, 1)
    expected: f64 = 1 * 5 + 2 * 4 + 3 * 3 + 4 * 2 + 5 * 1  // = 35

    if !approx_eq(result, expected) {
        fmt.printf("  FAILED: result = %f, expected %f\n", result, expected)
        return false
    }

    fmt.println("  PASSED")
    return true
}

test_dnrm2 :: proc() -> bool {
    fmt.println("Testing dnrm2...")

    x := []f64{3.0, 4.0}
    result := blas.dnrm2(2, x, 1)
    expected: f64 = 5.0  // sqrt(9 + 16) = 5

    if !approx_eq(result, expected) {
        fmt.printf("  FAILED: result = %f, expected %f\n", result, expected)
        return false
    }

    fmt.println("  PASSED")
    return true
}

test_drot :: proc() -> bool {
    fmt.println("Testing drot...")

    // Test with a 45-degree rotation (c = s = 1/sqrt(2))
    c := 1.0 / math.sqrt_f64(2.0)
    s := 1.0 / math.sqrt_f64(2.0)

    x := []f64{1.0, 0.0}
    y := []f64{0.0, 1.0}

    blas.drot(2, x, 1, y, 1, c, s)

    // After rotation: x' = c*x + s*y, y' = c*y - s*x
    // x[0]: c*1 + s*0 = c
    // y[0]: c*0 - s*1 = -s
    // x[1]: c*0 + s*1 = s
    // y[1]: c*1 - s*0 = c

    if !approx_eq(x[0], c) || !approx_eq(y[0], -s) ||
       !approx_eq(x[1], s) || !approx_eq(y[1], c) {
        fmt.printf("  FAILED: rotation result incorrect\n")
        return false
    }

    fmt.println("  PASSED")
    return true
}

// ============================================================================
// LAPACK Auxiliary Tests
// ============================================================================

test_dlartg :: proc() -> bool {
    fmt.println("Testing dlartg...")

    // Test case 1: f = 3, g = 4 -> r = 5, c = 3/5, s = 4/5
    c, s, r := blas.dlartg(3.0, 4.0)

    if !approx_eq(r, 5.0) || !approx_eq(c, 0.6) || !approx_eq(s, 0.8) {
        fmt.printf("  FAILED: c = %f, s = %f, r = %f\n", c, s, r)
        return false
    }

    // Verify: c*f + s*g = r and c*g - s*f = 0
    check1 := c * 3.0 + s * 4.0
    check2 := c * 4.0 - s * 3.0

    if !approx_eq(check1, r) || !approx_eq(check2, 0.0) {
        fmt.printf("  FAILED: rotation verification failed\n")
        return false
    }

    // Test case 2: g = 0
    c2, s2, r2 := blas.dlartg(5.0, 0.0)
    if !approx_eq(c2, 1.0) || !approx_eq(s2, 0.0) || !approx_eq(r2, 5.0) {
        fmt.printf("  FAILED: g=0 case failed\n")
        return false
    }

    // Test case 3: f = 0
    c3, s3, r3 := blas.dlartg(0.0, 5.0)
    if !approx_eq(c3, 0.0) || !approx_eq(abs(s3), 1.0) || !approx_eq(r3, 5.0) {
        fmt.printf("  FAILED: f=0 case failed\n")
        return false
    }

    fmt.println("  PASSED")
    return true
}

// ============================================================================
// BLAS Level 2 Tests
// ============================================================================

test_dgemv :: proc() -> bool {
    fmt.println("Testing dgemv...")

    // Test A * x where A is 2x3, x is 3x1
    // A = [[1, 2, 3], [4, 5, 6]]
    // x = [1, 2, 3]
    // A * x = [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3] = [14, 32]

    a := []f64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
    x := []f64{1.0, 2.0, 3.0}
    y := []f64{0.0, 0.0}

    blas.dgemv(.No_Trans, 2, 3, 1.0, a, 3, x, 1, 0.0, y, 1)

    if !approx_eq(y[0], 14.0) || !approx_eq(y[1], 32.0) {
        fmt.printf("  FAILED: y = [%f, %f], expected [14, 32]\n", y[0], y[1])
        return false
    }

    fmt.println("  PASSED")
    return true
}

test_dtrsv :: proc() -> bool {
    fmt.println("Testing dtrsv...")

    // Test solving U * x = b where U is upper triangular
    // U = [[2, 1], [0, 3]]
    // b = [5, 6]
    // Solution: x[1] = 6/3 = 2, x[0] = (5 - 1*2)/2 = 1.5

    u := []f64{2.0, 1.0, 0.0, 3.0}
    x := []f64{5.0, 6.0}

    blas.dtrsv(.Upper, .No_Trans, .Non_Unit, 2, u, 2, x, 1)

    if !approx_eq(x[0], 1.5) || !approx_eq(x[1], 2.0) {
        fmt.printf("  FAILED: x = [%f, %f], expected [1.5, 2.0]\n", x[0], x[1])
        return false
    }

    fmt.println("  PASSED")
    return true
}

// ============================================================================
// QR Factorization Tests
// ============================================================================

test_dgeqrf :: proc() -> bool {
    fmt.println("Testing dgeqrf...")

    // Test with a simple 3x2 matrix
    // A = [[1, 2], [3, 4], [5, 6]]
    m :: 3
    n :: 2

    a := []f64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
    a_copy := []f64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
    tau := []f64{0.0, 0.0}
    work := []f64{0.0, 0.0, 0.0}

    blas.dgeqrf(m, n, a, n, tau, work)

    // Generate Q
    q := make([]f64, m * m)
    defer delete(q)
    blas.dorgqr(m, n, a, n, tau, q, m, work)

    // Extract R
    r := make([]f64, m * n)
    defer delete(r)
    blas.extract_r(m, n, a, n, r, n)

    // Verify Q is orthogonal: Q^T * Q = I
    qtq := make([]f64, m * m)
    defer delete(qtq)
    blas.dgemm(.Trans, .No_Trans, m, m, m, 1.0, q, m, q, m, 0.0, qtq, m)

    for i in 0 ..< m {
        for j in 0 ..< m {
            expected: f64 = (i == j) ? 1.0 : 0.0
            if !approx_eq(qtq[i * m + j], expected, 1.0e-9) {
                fmt.printf("  FAILED: Q^T*Q is not identity\n")
                return false
            }
        }
    }

    // Verify Q * R = A
    qr := make([]f64, m * n)
    defer delete(qr)
    blas.dgemm(.No_Trans, .No_Trans, m, n, m, 1.0, q, m, r, n, 0.0, qr, n)

    for i in 0 ..< m {
        for j in 0 ..< n {
            if !approx_eq(qr[i * n + j], a_copy[i * n + j], 1.0e-9) {
                fmt.printf("  FAILED: Q*R != A at (%d, %d)\n", i, j)
                return false
            }
        }
    }

    fmt.println("  PASSED")
    return true
}

// ============================================================================
// QR Update Tests
// ============================================================================

test_delcols :: proc() -> bool {
    fmt.println("Testing delcols...")

    // Start with a 4x4 upper triangular matrix R
    // Delete column 1 (0-indexed), should result in 4x3 upper triangular
    m :: 4
    n :: 4

    // R = [[1, 2, 3, 4], [0, 5, 6, 7], [0, 0, 8, 9], [0, 0, 0, 10]]
    r := []f64{
        1.0, 2.0, 3.0, 4.0,
        0.0, 5.0, 6.0, 7.0,
        0.0, 0.0, 8.0, 9.0,
        0.0, 0.0, 0.0, 10.0,
    }

    work := make([]f64, 10)
    defer delete(work)

    blas.delcols(m, n, 1, 1, r, n, work)

    // After deleting column 1, columns 2,3 shift left
    // Result should be upper triangular in the first 3 columns
    // Check that it's upper triangular
    new_n := n - 1

    for i in 0 ..< m {
        for j in 0 ..< i {
            if j < new_n && abs(r[i * n + j]) > 1.0e-10 {
                fmt.printf("  FAILED: not upper triangular at (%d, %d) = %f\n", i, j, r[i * n + j])
                return false
            }
        }
    }

    fmt.println("  PASSED")
    return true
}

test_addcols :: proc() -> bool {
    fmt.println("Testing addcols...")

    // Start with a 3x2 upper triangular matrix R (with space for 3 columns)
    m :: 3
    n :: 2
    ldr :: 4  // Leave room for new column

    // R = [[1, 2, 0, 0], [0, 3, 0, 0], [0, 0, 0, 0]]
    r := []f64{
        1.0, 2.0, 0.0, 0.0,
        0.0, 3.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    }

    // New column to add (already in Q^T basis for this simple test)
    u := []f64{4.0, 5.0, 6.0}

    work := make([]f64, 10)
    defer delete(work)

    // Add column at position 1
    blas.addcols(m, n, 1, 1, r, ldr, u, 1, work)

    // Result should be 3x3 upper triangular
    new_n := n + 1

    for i in 0 ..< m {
        for j in 0 ..< i {
            if j < new_n && abs(r[i * ldr + j]) > 1.0e-10 {
                fmt.printf("  FAILED: not upper triangular at (%d, %d) = %f\n", i, j, r[i * ldr + j])
                return false
            }
        }
    }

    fmt.println("  PASSED")
    return true
}

test_qr_update_consistency :: proc() -> bool {
    fmt.println("Testing QR update consistency...")

    // Create a 4x4 matrix A, compute QR, delete a column, verify result
    m :: 4
    n :: 4

    // A = random-ish well-conditioned matrix
    a := []f64{
        4.0, 2.0, 1.0, 3.0,
        2.0, 5.0, 2.0, 1.0,
        1.0, 2.0, 4.0, 2.0,
        3.0, 1.0, 2.0, 5.0,
    }
    a_copy := make([]f64, m * n)
    defer delete(a_copy)
    for i in 0 ..< m * n {
        a_copy[i] = a[i]
    }

    tau := make([]f64, min(m, n))
    defer delete(tau)
    work := make([]f64, n)
    defer delete(work)

    // Compute QR factorization
    blas.dgeqrf(m, n, a, n, tau, work)

    // Generate Q
    q := make([]f64, m * m)
    defer delete(q)
    blas.dorgqr(m, n, a, n, tau, q, m, work)

    // Extract R
    r := make([]f64, m * n)
    defer delete(r)
    blas.extract_r(m, n, a, n, r, n)

    // Delete column 1
    work2 := make([]f64, 20)
    defer delete(work2)
    blas.delcolsq(m, n, 1, 1, q, m, r, n, work2)

    // Verify Q is still orthogonal (first 3 columns)
    qtq := make([]f64, m * m)
    defer delete(qtq)
    blas.dgemm(.Trans, .No_Trans, m, m, m, 1.0, q, m, q, m, 0.0, qtq, m)

    for i in 0 ..< m {
        for j in 0 ..< m {
            expected: f64 = (i == j) ? 1.0 : 0.0
            if !approx_eq(qtq[i * m + j], expected, 1.0e-8) {
                fmt.printf("  FAILED: Q not orthogonal after delcols at (%d, %d)\n", i, j)
                return false
            }
        }
    }

    // Verify Q * R equals A with column 1 deleted
    // A with column 1 deleted: columns 0, 2, 3
    a_reduced := make([]f64, m * (n - 1))
    defer delete(a_reduced)
    a_reduced[0] = a_copy[0]
    a_reduced[1] = a_copy[2]
    a_reduced[2] = a_copy[3]
    a_reduced[3] = a_copy[4]
    a_reduced[4] = a_copy[6]
    a_reduced[5] = a_copy[7]
    a_reduced[6] = a_copy[8]
    a_reduced[7] = a_copy[10]
    a_reduced[8] = a_copy[11]
    a_reduced[9] = a_copy[12]
    a_reduced[10] = a_copy[14]
    a_reduced[11] = a_copy[15]

    qr := make([]f64, m * (n - 1))
    defer delete(qr)
    blas.dgemm(.No_Trans, .No_Trans, m, n - 1, m, 1.0, q, m, r, n, 0.0, qr, n - 1)

    max_err: f64 = 0.0
    for i in 0 ..< m {
        for j in 0 ..< n - 1 {
            err := abs(qr[i * (n - 1) + j] - a_reduced[i * (n - 1) + j])
            max_err = max(max_err, err)
        }
    }

    if max_err > 1.0e-8 {
        fmt.printf("  FAILED: Q*R != A_reduced, max error = %e\n", max_err)
        return false
    }

    fmt.println("  PASSED")
    return true
}

// ============================================================================
// Main test runner
// ============================================================================

Test :: struct {
    name: string,
    fn:   proc() -> bool,
}

main :: proc() {
    fmt.println("=== BLAS Library Tests ===\n")

    passed := 0
    failed := 0

    // BLAS Level 1 tests
    fmt.println("--- BLAS Level 1 ---")
    if test_dcopy() {
        passed += 1
    } else {
        failed += 1
    }
    if test_dscal() {
        passed += 1
    } else {
        failed += 1
    }
    if test_daxpy() {
        passed += 1
    } else {
        failed += 1
    }
    if test_ddot() {
        passed += 1
    } else {
        failed += 1
    }
    if test_dnrm2() {
        passed += 1
    } else {
        failed += 1
    }
    if test_drot() {
        passed += 1
    } else {
        failed += 1
    }

    // LAPACK Auxiliary tests
    fmt.println("\n--- LAPACK Auxiliaries ---")
    if test_dlartg() {
        passed += 1
    } else {
        failed += 1
    }

    // BLAS Level 2 tests
    fmt.println("\n--- BLAS Level 2 ---")
    if test_dgemv() {
        passed += 1
    } else {
        failed += 1
    }
    if test_dtrsv() {
        passed += 1
    } else {
        failed += 1
    }

    // QR Factorization tests
    fmt.println("\n--- QR Factorization ---")
    if test_dgeqrf() {
        passed += 1
    } else {
        failed += 1
    }

    // QR Update tests
    fmt.println("\n--- QR Updates ---")
    if test_delcols() {
        passed += 1
    } else {
        failed += 1
    }
    if test_addcols() {
        passed += 1
    } else {
        failed += 1
    }
    if test_qr_update_consistency() {
        passed += 1
    } else {
        failed += 1
    }

    // Numerical regression tests (see docs/OLS_RESULTS.md)
    fmt.println("\n--- Numerical robustness ---")
    for t in ([]Test {
        {"dnrm2 scaling", test_dnrm2_scaling},
        {"row-major lda checks", test_row_major_lda_checks},
        {"dgeqrf extreme scale", test_dgeqrf_extreme_scale},
    }) {
        if t.fn() {passed += 1} else {failed += 1}
    }

    // Least squares (docs/OLS_PLAN.md section 6)
    fmt.println("\n--- Least squares ---")
    for t in ([]Test {
        {"dense ground truth", test_ols_dense_ground_truth},
        {"accumulator ground truth", test_ols_accum_ground_truth},
        {"chunk invariance", test_ols_accum_chunk_invariance},
        {"transforms agree", test_ols_transforms_agree},
        {"boundary policies", test_ols_boundaries},
    }) {
        if t.fn() {passed += 1} else {failed += 1}
    }

    fmt.println("\n=== Summary ===")
    fmt.printf("Passed: %d, Failed: %d\n", passed, failed)

    if failed > 0 {
        fmt.println("SOME TESTS FAILED!")
    } else {
        fmt.println("ALL TESTS PASSED!")
    }
}
