package regression_example

import "core:fmt"
import "core:os"
import "core:strings"
import "../src/blas"
import synth "../src/synth"

// Random numbers come from src/synth, which is the generator the test suite
// validates against a goodness-of-fit battery. Draws are indexed rather than
// streamed, so each use below names its own stream and cannot disturb another.
SEED :: u32(42)
STREAM_X :: u32(10) // design matrix entries
STREAM_NOISE :: u32(11) // response noise
STREAM_NEWCOL :: u32(12) // the added column
STREAM_NEWROW :: u32(13) // the added rows
STREAM_NEWY :: u32(14) // responses for the added rows

rnorm :: proc(stream: u32, index: int) -> f64 {
    return synth.synth_normal(SEED, stream, u32(index))
}

// Solve least squares problem using QR factorization: min ||Ax - b||
// Returns coefficients x.
//
// This wraps blas.ols_solve_dense, which applies the Householder reflectors
// straight to b instead of materializing Q. The previous version of this
// procedure built the full m x m Q via dorgqr and then did a second m x m pass
// to form Q^T*b; that was 489 MiB and 28 s at m = 8000 against 0.37 ms here.
// See docs/OLS_RESULTS.md.
// All working memory is passed in. An earlier version allocated the copies and
// the scratch internally, which hides the cost from the caller and forces the
// context allocator on them; the library itself allocates nothing, and helpers
// built on it should not either. Size the buffers with blas.ols_dense_scratch.
solve_ls_qr :: proc(
    m: int, n: int,
    a: []f64, lda: int,
    b: []f64,
    x: []f64,
    a_copy: []f64,   // >= m*lda, destroyed
    b_copy: []f64,   // >= m, destroyed
    scratch: []f64,  // >= blas.ols_dense_scratch(n)
) {
    // ols_solve_dense destroys both inputs, so hand it copies.
    for i in 0 ..< m * lda {
        a_copy[i] = a[i]
    }
    for i in 0 ..< m {
        b_copy[i] = b[i]
    }

    _, err := blas.ols_solve_dense(m, n, a_copy[:m * lda], lda, b_copy[:m], x, scratch)
    if err != .None {
        fmt.printf("solve_ls_qr: %v\n", err)
        for i in 0 ..< n {
            x[i] = 0.0
        }
    }
}

// Compute residual norm ||Ax - b||
residual_norm :: proc(m: int, n: int, a: []f64, lda: int, x: []f64, b: []f64) -> f64 {
    return blas.ols_residual_norm(m, n, a, lda, x, b)
}

// Compare two coefficient vectors
coefficients_match :: proc(x1: []f64, x2: []f64, n: int, tol: f64) -> bool {
    for i in 0 ..< n {
        if abs(x1[i] - x2[i]) > tol {
            return false
        }
    }
    return true
}

// Write matrix to file for Julia verification
write_matrix :: proc(filename: string, a: []f64, m: int, n: int, lda: int) {
    sb := strings.builder_make()
    defer strings.builder_destroy(&sb)

    for i in 0 ..< m {
        for j in 0 ..< n {
            if j > 0 {
                strings.write_string(&sb, ",")
            }
            strings.write_string(&sb, fmt.tprintf("%.15e", a[i * lda + j]))
        }
        strings.write_string(&sb, "\n")
    }

    if err := os.write_entire_file(filename, transmute([]u8)strings.to_string(sb)); err != nil {
        fmt.printf("failed to write %s: %v\n", filename, err)
    }
}

// Write vector to file
write_vector :: proc(filename: string, v: []f64, n: int) {
    sb := strings.builder_make()
    defer strings.builder_destroy(&sb)

    for i in 0 ..< n {
        strings.write_string(&sb, fmt.tprintf("%.15e\n", v[i]))
    }

    if err := os.write_entire_file(filename, transmute([]u8)strings.to_string(sb)); err != nil {
        fmt.printf("failed to write %s: %v\n", filename, err)
    }
}

main :: proc() {
    fmt.println("=== Multiple Regression with QR Updates ===\n")

    // Problem dimensions
    M :: 20      // Number of observations
    N :: 5       // Number of predictors (including intercept)

    // Generate random design matrix X (M x N)
    // First column is all 1s (intercept)
    lda :: N + 3  // Extra space for adding columns
    x_full := make([]f64, (M + 5) * lda)  // Extra space for adding rows
    defer delete(x_full)

    for i in 0 ..< M {
        x_full[i * lda + 0] = 1.0  // Intercept
        for j in 1 ..< N {
            x_full[i * lda + j] = rnorm(STREAM_X, i * N + j)
        }
    }

    // Generate true coefficients
    beta_true := []f64{2.0, -1.5, 0.8, -0.3, 1.2}

    // Generate response y = X * beta + noise
    y_full := make([]f64, M + 5)
    defer delete(y_full)

    for i in 0 ..< M {
        y_full[i] = 0.0
        for j in 0 ..< N {
            y_full[i] += x_full[i * lda + j] * beta_true[j]
        }
        y_full[i] += 0.5 * rnorm(STREAM_NOISE, i)  // Add noise
    }

    // Work arrays. Sized once here for the largest case any call below uses,
    // then reused -- the solver never allocates, so the caller owns every byte.
    work := make([]f64, 100)
    defer delete(work)

    MAX_M :: M + 5
    MAX_LDA :: lda
    ls_a := make([]f64, MAX_M * MAX_LDA)
    defer delete(ls_a)
    ls_b := make([]f64, MAX_M)
    defer delete(ls_b)
    ls_scratch := make([]f64, blas.ols_dense_scratch(N + 3))
    defer delete(ls_scratch)

    // =========================================================================
    // 1. Fit initial model
    // =========================================================================
    fmt.println("--- Initial Model Fit ---")

    beta_hat := make([]f64, N)
    defer delete(beta_hat)

    solve_ls_qr(M, N, x_full, lda, y_full, beta_hat, ls_a, ls_b, ls_scratch)

    fmt.println("True coefficients:     ", beta_true[:])
    fmt.printf("Estimated coefficients: [")
    for i in 0 ..< N {
        if i > 0 {
            fmt.printf(", ")
        }
        fmt.printf("%.4f", beta_hat[i])
    }
    fmt.println("]")

    resid := residual_norm(M, N, x_full, lda, beta_hat, y_full)
    fmt.printf("Residual norm: %.6f\n\n", resid)

    // Save initial data for Julia
    write_matrix("examples/X_initial.csv", x_full, M, N, lda)
    write_vector("examples/y_initial.csv", y_full, M)
    write_vector("examples/beta_initial.csv", beta_hat, N)

    // =========================================================================
    // 2. Delete a column (remove predictor 2) using QR update
    // =========================================================================
    fmt.println("--- Delete Column 2 (QR Update vs Full Recompute) ---")

    // Create matrices for QR factorization
    a_qr := make([]f64, M * lda)
    defer delete(a_qr)
    for i in 0 ..< M * lda {
        a_qr[i] = x_full[i]
    }

    tau := make([]f64, min(M, N))
    defer delete(tau)

    // Compute initial QR
    blas.dgeqrf(M, N, a_qr, lda, tau, work)

    // Generate Q
    q := make([]f64, M * M)
    defer delete(q)
    blas.dorgqr(M, N, a_qr, lda, tau, q, M, work)

    // Extract R (allocate extra space for addrows test later)
    r := make([]f64, (M + 5) * lda)
    defer delete(r)
    blas.extract_r(M, N, a_qr, lda, r, lda)

    // Delete column 2 using QR update
    blas.delcolsq(M, N, 2, 1, q, M, r, lda, work)

    // Solve using updated QR: compute Q^T * y, then back-substitute
    qtb := make([]f64, M)
    defer delete(qtb)
    for i in 0 ..< M {
        sum: f64 = 0.0
        for j in 0 ..< M {
            sum += q[j * M + i] * y_full[j]
        }
        qtb[i] = sum
    }

    beta_del_update := make([]f64, N - 1)
    defer delete(beta_del_update)
    for i in 0 ..< N - 1 {
        beta_del_update[i] = qtb[i]
    }
    blas.dtrsv(.Upper, .No_Trans, .Non_Unit, N - 1, r, lda, beta_del_update, 1)

    // Now recompute from scratch for comparison
    // Create X with column 2 removed: columns 0,1,3,4
    x_del := make([]f64, M * (N - 1))
    defer delete(x_del)
    for i in 0 ..< M {
        x_del[i * (N - 1) + 0] = x_full[i * lda + 0]
        x_del[i * (N - 1) + 1] = x_full[i * lda + 1]
        x_del[i * (N - 1) + 2] = x_full[i * lda + 3]
        x_del[i * (N - 1) + 3] = x_full[i * lda + 4]
    }

    beta_del_recompute := make([]f64, N - 1)
    defer delete(beta_del_recompute)
    solve_ls_qr(M, N - 1, x_del, N - 1, y_full, beta_del_recompute, ls_a, ls_b, ls_scratch)

    fmt.printf("Coefficients (QR update):   [")
    for i in 0 ..< N - 1 {
        if i > 0 {
            fmt.printf(", ")
        }
        fmt.printf("%.6f", beta_del_update[i])
    }
    fmt.println("]")

    fmt.printf("Coefficients (recomputed):  [")
    for i in 0 ..< N - 1 {
        if i > 0 {
            fmt.printf(", ")
        }
        fmt.printf("%.6f", beta_del_recompute[i])
    }
    fmt.println("]")

    if coefficients_match(beta_del_update, beta_del_recompute, N - 1, 1e-8) {
        fmt.println("Result: MATCH!")
    } else {
        fmt.println("Result: MISMATCH!")
    }
    fmt.println()

    // Save for Julia
    write_matrix("examples/X_delcol.csv", x_del, M, N - 1, N - 1)
    write_vector("examples/beta_delcol_update.csv", beta_del_update, N - 1)
    write_vector("examples/beta_delcol_recompute.csv", beta_del_recompute, N - 1)

    // =========================================================================
    // 3. Add a new column using QR update
    // =========================================================================
    fmt.println("--- Add New Column (QR Update vs Full Recompute) ---")

    // Reset QR factorization for original data
    for i in 0 ..< M * lda {
        a_qr[i] = x_full[i]
    }
    blas.dgeqrf(M, N, a_qr, lda, tau, work)
    blas.dorgqr(M, N, a_qr, lda, tau, q, M, work)
    blas.extract_r(M, N, a_qr, lda, r, lda)

    // Generate a new column
    new_col := make([]f64, M)
    defer delete(new_col)
    for i in 0 ..< M {
        new_col[i] = rnorm(STREAM_NEWCOL, i)
    }

    // Add column at position 3 using QR update
    blas.addcolsq(M, N, 3, 1, q, M, r, lda, new_col, 1, work)

    // Solve using updated QR
    for i in 0 ..< M {
        sum: f64 = 0.0
        for j in 0 ..< M {
            sum += q[j * M + i] * y_full[j]
        }
        qtb[i] = sum
    }

    beta_add_update := make([]f64, N + 1)
    defer delete(beta_add_update)
    for i in 0 ..< N + 1 {
        beta_add_update[i] = qtb[i]
    }
    blas.dtrsv(.Upper, .No_Trans, .Non_Unit, N + 1, r, lda, beta_add_update, 1)

    // Recompute from scratch
    x_add := make([]f64, M * (N + 1))
    defer delete(x_add)
    for i in 0 ..< M {
        // Columns 0,1,2, then new_col, then 3,4
        x_add[i * (N + 1) + 0] = x_full[i * lda + 0]
        x_add[i * (N + 1) + 1] = x_full[i * lda + 1]
        x_add[i * (N + 1) + 2] = x_full[i * lda + 2]
        x_add[i * (N + 1) + 3] = new_col[i]
        x_add[i * (N + 1) + 4] = x_full[i * lda + 3]
        x_add[i * (N + 1) + 5] = x_full[i * lda + 4]
    }

    beta_add_recompute := make([]f64, N + 1)
    defer delete(beta_add_recompute)
    solve_ls_qr(M, N + 1, x_add, N + 1, y_full, beta_add_recompute, ls_a, ls_b, ls_scratch)

    fmt.printf("Coefficients (QR update):   [")
    for i in 0 ..< N + 1 {
        if i > 0 {
            fmt.printf(", ")
        }
        fmt.printf("%.6f", beta_add_update[i])
    }
    fmt.println("]")

    fmt.printf("Coefficients (recomputed):  [")
    for i in 0 ..< N + 1 {
        if i > 0 {
            fmt.printf(", ")
        }
        fmt.printf("%.6f", beta_add_recompute[i])
    }
    fmt.println("]")

    if coefficients_match(beta_add_update, beta_add_recompute, N + 1, 1e-8) {
        fmt.println("Result: MATCH!")
    } else {
        fmt.println("Result: MISMATCH!")
    }
    fmt.println()

    // Save for Julia
    write_matrix("examples/X_addcol.csv", x_add, M, N + 1, N + 1)
    write_vector("examples/new_column.csv", new_col, M)
    write_vector("examples/beta_addcol_update.csv", beta_add_update, N + 1)
    write_vector("examples/beta_addcol_recompute.csv", beta_add_recompute, N + 1)

    // =========================================================================
    // 4. Add new rows using QR update
    // =========================================================================
    fmt.println("--- Add New Rows (QR Update vs Full Recompute) ---")

    // Reset QR factorization
    for i in 0 ..< M * lda {
        a_qr[i] = x_full[i]
    }
    blas.dgeqrf(M, N, a_qr, lda, tau, work)
    blas.extract_r(M, N, a_qr, lda, r, lda)

    // Generate 3 new rows
    NUM_NEW_ROWS :: 3
    new_rows := make([]f64, NUM_NEW_ROWS * N)
    defer delete(new_rows)
    new_y := make([]f64, NUM_NEW_ROWS)
    defer delete(new_y)

    for i in 0 ..< NUM_NEW_ROWS {
        new_rows[i * N + 0] = 1.0  // Intercept
        for j in 1 ..< N {
            new_rows[i * N + j] = rnorm(STREAM_NEWROW, i * N + j)
        }
        // Generate corresponding y value
        new_y[i] = 0.0
        for j in 0 ..< N {
            new_y[i] += new_rows[i * N + j] * beta_true[j]
        }
        new_y[i] += 0.5 * rnorm(STREAM_NEWY, i)
    }

    // Add rows using QR update
    blas.addrows(M, N, NUM_NEW_ROWS, r, lda, new_rows, N, work)

    // For addrows, we need to solve the new system
    // The updated R is (M+NUM_NEW_ROWS) x N upper triangular
    // We need Q^T * y_extended, but addrows doesn't track Q

    // For this example, let's just verify R is correct by recomputing
    x_extended := make([]f64, (M + NUM_NEW_ROWS) * N)
    defer delete(x_extended)
    y_extended := make([]f64, M + NUM_NEW_ROWS)
    defer delete(y_extended)

    for i in 0 ..< M {
        for j in 0 ..< N {
            x_extended[i * N + j] = x_full[i * lda + j]
        }
        y_extended[i] = y_full[i]
    }
    for i in 0 ..< NUM_NEW_ROWS {
        for j in 0 ..< N {
            x_extended[(M + i) * N + j] = new_rows[i * N + j]
        }
        y_extended[M + i] = new_y[i]
    }

    beta_addrow_recompute := make([]f64, N)
    defer delete(beta_addrow_recompute)
    solve_ls_qr(M + NUM_NEW_ROWS, N, x_extended, N, y_extended, beta_addrow_recompute, ls_a, ls_b, ls_scratch)

    // Verify R from update matches R from full recompute
    a_qr_full := make([]f64, (M + NUM_NEW_ROWS) * N)
    defer delete(a_qr_full)
    for i in 0 ..< (M + NUM_NEW_ROWS) * N {
        a_qr_full[i] = x_extended[i]
    }
    tau_full := make([]f64, min(M + NUM_NEW_ROWS, N))
    defer delete(tau_full)
    blas.dgeqrf(M + NUM_NEW_ROWS, N, a_qr_full, N, tau_full, work)

    r_full := make([]f64, (M + NUM_NEW_ROWS) * N)
    defer delete(r_full)
    blas.extract_r(M + NUM_NEW_ROWS, N, a_qr_full, N, r_full, N)

    // Compare R matrices (just the upper N x N part)
    r_match := true
    max_diff: f64 = 0.0
    for i in 0 ..< N {
        for j in i ..< N {
            diff := abs(abs(r[i * lda + j]) - abs(r_full[i * N + j]))
            max_diff = max(max_diff, diff)
            if diff > 1e-8 {
                r_match = false
            }
        }
    }

    fmt.printf("R matrix max absolute difference: %.2e\n", max_diff)
    fmt.printf("Coefficients after adding rows: [")
    for i in 0 ..< N {
        if i > 0 {
            fmt.printf(", ")
        }
        fmt.printf("%.6f", beta_addrow_recompute[i])
    }
    fmt.println("]")

    if r_match {
        fmt.println("R matrices: MATCH (up to sign)!")
    } else {
        fmt.println("R matrices: MISMATCH!")
    }
    fmt.println()

    // Save for Julia
    write_matrix("examples/X_addrows.csv", x_extended, M + NUM_NEW_ROWS, N, N)
    write_vector("examples/y_addrows.csv", y_extended, M + NUM_NEW_ROWS)
    write_matrix("examples/new_rows.csv", new_rows, NUM_NEW_ROWS, N, N)
    write_vector("examples/new_y.csv", new_y, NUM_NEW_ROWS)
    write_vector("examples/beta_addrows.csv", beta_addrow_recompute, N)

    // =========================================================================
    // Summary
    // =========================================================================
    fmt.println("=== Summary ===")
    fmt.println("Data files written to examples/ directory for Julia verification.")
    fmt.println("Run: julia examples/verify_regression.jl")
}
