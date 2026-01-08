package blas

import "core:math"

// ============================================================================
// QR Factorization Updating Routines
// Based on "Updating the QR factorization and the least squares problem"
// by Sven Hammarling
// ============================================================================

// delcols deletes columns k through k+p-1 from the QR factorization.
//
// Given the QR factorization A = Q * R of an m by n matrix A, this routine
// produces the QR factorization A' = Q' * R' of the m by (n-p) matrix A'
// formed by deleting columns k through k+p-1 from A.
//
// The routine uses Givens rotations to zero out the elements that appear
// below the diagonal after the column deletion.
//
// Parameters:
//   m     - Number of rows of R (and A)
//   n     - Number of columns of R before deletion
//   k     - First column to delete (0-indexed)
//   p     - Number of columns to delete
//   r     - On entry: m by n upper trapezoidal matrix R
//           On exit: m by (n-p) upper trapezoidal matrix R'
//           Elements in columns k through n-p-1 contain the updated R
//   ldr   - Leading dimension of R
//   work  - Workspace of size at least 2*min(m-1, n-k-p)
//           Used to store cosines and sines of rotations
//
// Note: This routine does not update Q. Use delcolsq if Q update is needed.
delcols :: proc(m: int, n: int, k: int, p: int, r: []f64, ldr: int, work: []f64) {
    // Quick return if nothing to do
    if p <= 0 || k >= n || m <= 0 {
        return
    }

    // Ensure we don't delete more columns than exist
    nump := min(p, n - k)
    new_n := n - nump

    // After deleting columns k through k+p-1, we need to shift columns
    // k+p through n-1 left to positions k through n-p-1
    // This creates subdiagonal elements in the (k+1) to min(m, n-p) rows
    // that need to be zeroed out using Givens rotations.

    // First, shift the columns (copy columns k+p:n-1 to k:n-p-1)
    for j := k; j < new_n; j += 1 {
        src_col := j + nump
        // Copy column src_col to column j
        for i in 0 ..< m {
            r[i * ldr + j] = r[i * ldr + src_col]
        }
    }

    // Zero out the vacated columns (optional, for cleanliness)
    for j := new_n; j < n; j += 1 {
        for i in 0 ..< m {
            r[i * ldr + j] = 0.0
        }
    }

    // Now we need to eliminate the subdiagonal elements that appeared
    // after the shift. The "bump" extends from row k+1 to min(m-1, new_n-1)
    // and affects columns k to new_n-1.
    //
    // We use Givens rotations applied from the right to zero out elements.
    // For each row i from k+1 to min(m-1, new_n-1), we zero out element (i, i-1)
    // by rotating columns i-1 and i.

    // Actually, the standard approach is to work column by column:
    // After deletion, the matrix has a bulge starting at position (k+1, k).
    // We chase the bulge down and to the right using Givens rotations.

    // Apply Givens rotations to restore upper triangular form
    // The approach: for each column j from k to new_n-2, eliminate the
    // subdiagonal element at position (j+1, j) using a rotation that
    // mixes rows j and j+1.

    for j in k ..< min(new_n - 1, m - 1) {
        // Zero out element (j+1, j) using a rotation of rows j and j+1
        // from columns j to new_n-1

        if r[(j + 1) * ldr + j] == 0.0 {
            continue
        }

        // Generate rotation to zero out r[j+1, j]
        c, s, rr := dlartg(r[j * ldr + j], r[(j + 1) * ldr + j])
        r[j * ldr + j] = rr
        r[(j + 1) * ldr + j] = 0.0

        // Apply rotation to remaining columns j+1 to new_n-1 of rows j and j+1
        if j + 1 < new_n {
            // Get slices for rows j and j+1, starting from column j+1
            row_j_start := j * ldr + (j + 1)
            row_j1_start := (j + 1) * ldr + (j + 1)
            num_cols := new_n - (j + 1)

            drot(num_cols, r[row_j_start:], 1, r[row_j1_start:], 1, c, s)
        }
    }
}

// delcolsq deletes columns k through k+p-1 from the QR factorization,
// and also updates the orthogonal matrix Q.
//
// Given the QR factorization A = Q * R of an m by n matrix A, this routine
// produces the QR factorization A' = Q' * R' of the m by (n-p) matrix A'
// formed by deleting columns k through k+p-1 from A.
//
// Parameters:
//   m     - Number of rows of Q and R
//   n     - Number of columns of R before deletion
//   k     - First column to delete (0-indexed)
//   p     - Number of columns to delete
//   q     - On entry: m by m orthogonal matrix Q
//           On exit: updated orthogonal matrix Q'
//   ldq   - Leading dimension of Q
//   r     - On entry: m by n upper trapezoidal matrix R
//           On exit: m by (n-p) upper trapezoidal matrix R'
//   ldr   - Leading dimension of R
//   work  - Workspace of size at least 2*min(m-1, n-k-p)
delcolsq :: proc(m: int, n: int, k: int, p: int, q: []f64, ldq: int, r: []f64, ldr: int, work: []f64) {
    // Quick return if nothing to do
    if p <= 0 || k >= n || m <= 0 {
        return
    }

    // Ensure we don't delete more columns than exist
    nump := min(p, n - k)
    new_n := n - nump

    // First, shift columns in R (copy columns k+p:n-1 to k:n-p-1)
    for j := k; j < new_n; j += 1 {
        src_col := j + nump
        for i in 0 ..< m {
            r[i * ldr + j] = r[i * ldr + src_col]
        }
    }

    // Zero out the vacated columns
    for j := new_n; j < n; j += 1 {
        for i in 0 ..< m {
            r[i * ldr + j] = 0.0
        }
    }

    // Apply Givens rotations to restore upper triangular form
    // Also accumulate the rotations in Q

    for j in k ..< min(new_n - 1, m - 1) {
        // Zero out element (j+1, j) using a rotation of rows j and j+1

        if r[(j + 1) * ldr + j] == 0.0 {
            continue
        }

        // Generate rotation to zero out r[j+1, j]
        c, s, rr := dlartg(r[j * ldr + j], r[(j + 1) * ldr + j])
        r[j * ldr + j] = rr
        r[(j + 1) * ldr + j] = 0.0

        // Apply rotation to remaining columns of R (rows j and j+1)
        if j + 1 < new_n {
            row_j_start := j * ldr + (j + 1)
            row_j1_start := (j + 1) * ldr + (j + 1)
            num_cols := new_n - (j + 1)
            drot(num_cols, r[row_j_start:], 1, r[row_j1_start:], 1, c, s)
        }

        // Apply rotation to Q (columns j and j+1 of Q)
        // Q' = Q * G^T means we rotate columns j and j+1 of Q
        col_j_start := j
        col_j1_start := j + 1
        // Q is stored in row-major, so columns are at stride ldq
        // For column j: elements are at q[0*ldq+j], q[1*ldq+j], ...
        // We need to apply rotation to columns j and j+1
        for i in 0 ..< m {
            temp := c * q[i * ldq + j] + s * q[i * ldq + j + 1]
            q[i * ldq + j + 1] = c * q[i * ldq + j + 1] - s * q[i * ldq + j]
            q[i * ldq + j] = temp
        }
    }
}

// addcols adds p new columns at position k in the QR factorization.
//
// Given the QR factorization A = Q * R of an m by n matrix A, this routine
// produces the QR factorization A' = Q' * R' of the m by (n+p) matrix A'
// formed by inserting p new columns at position k.
//
// The new columns are given as an m by p matrix U.
//
// Parameters:
//   m     - Number of rows of R (and A)
//   n     - Number of columns of R before addition
//   k     - Position to insert new columns (0-indexed)
//   p     - Number of columns to add
//   r     - On entry: m by n upper trapezoidal matrix R (in m by (n+p) storage)
//           On exit: m by (n+p) upper trapezoidal matrix R'
//   ldr   - Leading dimension of R (must be at least n+p)
//   u     - m by p matrix containing the new columns
//   ldu   - Leading dimension of U
//   work  - Workspace of size at least m
//
// Note: This routine does not update Q. Use addcolsq if Q update is needed.
// The input u is assumed to be premultiplied by Q^T (i.e., u contains Q^T * A_new
// where A_new are the actual new columns to insert).
addcols :: proc(m: int, n: int, k: int, p: int, r: []f64, ldr: int, u: []f64, ldu: int, work: []f64) {
    // Quick return if nothing to do
    if p <= 0 || m <= 0 {
        return
    }

    new_n := n + p

    // First, shift existing columns k through n-1 to positions k+p through n+p-1
    // Work backwards to avoid overwriting
    for j := n - 1; j >= k; j -= 1 {
        dst_col := j + p
        for i in 0 ..< m {
            r[i * ldr + dst_col] = r[i * ldr + j]
        }
    }

    // Copy the new columns from u to positions k through k+p-1 in r
    // Note: u is assumed to contain Q^T * new_columns
    for j in 0 ..< p {
        for i in 0 ..< m {
            r[i * ldr + (k + j)] = u[i * ldu + j]
        }
    }

    // Now we need to restore upper triangular form.
    // The new columns at positions k to k+p-1 have elements below the diagonal.
    // We use Givens rotations to zero out these elements.

    // For each new column j in k to k+p-1:
    // - Zero out elements (j+1, j), (j+2, j), ..., (m-1, j) using Givens rotations
    // - Each rotation mixes rows i and i+1, applied to columns j to new_n-1

    for j in k ..< k + p {
        // Zero out elements below diagonal in column j
        for i := m - 1; i > j; i -= 1 {
            if r[i * ldr + j] == 0.0 {
                continue
            }

            // Generate rotation to zero out r[i, j] using r[i-1, j]
            c, s, rr := dlartg(r[(i - 1) * ldr + j], r[i * ldr + j])
            r[(i - 1) * ldr + j] = rr
            r[i * ldr + j] = 0.0

            // Apply rotation to remaining columns j+1 to new_n-1
            if j + 1 < new_n {
                row_im1_start := (i - 1) * ldr + (j + 1)
                row_i_start := i * ldr + (j + 1)
                num_cols := new_n - (j + 1)
                drot(num_cols, r[row_im1_start:], 1, r[row_i_start:], 1, c, s)
            }
        }
    }

    // After introducing the new columns, we may also need to eliminate any
    // remaining subdiagonal elements in columns k+p to new_n-1
    // This handles the "bulge chasing" for elements that were shifted
    for j in k + p ..< min(new_n - 1, m - 1) {
        if j < m && r[(j + 1) * ldr + j] != 0.0 {
            // Generate rotation to zero out r[j+1, j]
            c, s, rr := dlartg(r[j * ldr + j], r[(j + 1) * ldr + j])
            r[j * ldr + j] = rr
            r[(j + 1) * ldr + j] = 0.0

            // Apply rotation to remaining columns
            if j + 1 < new_n {
                row_j_start := j * ldr + (j + 1)
                row_j1_start := (j + 1) * ldr + (j + 1)
                num_cols := new_n - (j + 1)
                drot(num_cols, r[row_j_start:], 1, r[row_j1_start:], 1, c, s)
            }
        }
    }
}

// addcolsq adds p new columns at position k in the QR factorization,
// and also updates the orthogonal matrix Q.
//
// Given the QR factorization A = Q * R of an m by n matrix A, this routine
// produces the QR factorization A' = Q' * R' of the m by (n+p) matrix A'
// formed by inserting p new columns at position k.
//
// Parameters:
//   m     - Number of rows of Q and R
//   n     - Number of columns of R before addition
//   k     - Position to insert new columns (0-indexed)
//   p     - Number of columns to add
//   q     - On entry: m by m orthogonal matrix Q
//           On exit: updated orthogonal matrix Q'
//   ldq   - Leading dimension of Q
//   r     - On entry: m by n upper trapezoidal matrix R (in m by (n+p) storage)
//           On exit: m by (n+p) upper trapezoidal matrix R'
//   ldr   - Leading dimension of R
//   u     - m by p matrix containing the new columns (the actual columns, not Q^T * columns)
//   ldu   - Leading dimension of U
//   work  - Workspace of size at least m
addcolsq :: proc(
    m: int,
    n: int,
    k: int,
    p: int,
    q: []f64,
    ldq: int,
    r: []f64,
    ldr: int,
    u: []f64,
    ldu: int,
    work: []f64,
) {
    // Quick return if nothing to do
    if p <= 0 || m <= 0 {
        return
    }

    new_n := n + p

    // First, shift existing columns k through n-1 to positions k+p through n+p-1
    for j := n - 1; j >= k; j -= 1 {
        dst_col := j + p
        for i in 0 ..< m {
            r[i * ldr + dst_col] = r[i * ldr + j]
        }
    }

    // Compute Q^T * u and store in columns k to k+p-1 of R
    // This transforms the new columns into the basis of Q
    for j in 0 ..< p {
        // Compute column j of Q^T * u: sum over rows of Q^T times u[:,j]
        for i in 0 ..< m {
            sum: f64 = 0.0
            for ii in 0 ..< m {
                // Q^T[i, ii] = Q[ii, i] for row-major storage
                // Actually Q^T * u[:, j] = sum_ii Q[ii, i] * u[ii, j]
                // But for row-major Q, Q[ii, i] is at q[ii * ldq + i]
                sum += q[ii * ldq + i] * u[ii * ldu + j]
            }
            r[i * ldr + (k + j)] = sum
        }
    }

    // Now restore upper triangular form using Givens rotations
    // Also accumulate the rotations in Q

    for j in k ..< k + p {
        // Zero out elements below diagonal in column j
        for i := m - 1; i > j; i -= 1 {
            if r[i * ldr + j] == 0.0 {
                continue
            }

            // Generate rotation to zero out r[i, j] using r[i-1, j]
            c, s, rr := dlartg(r[(i - 1) * ldr + j], r[i * ldr + j])
            r[(i - 1) * ldr + j] = rr
            r[i * ldr + j] = 0.0

            // Apply rotation to remaining columns j+1 to new_n-1
            if j + 1 < new_n {
                row_im1_start := (i - 1) * ldr + (j + 1)
                row_i_start := i * ldr + (j + 1)
                num_cols := new_n - (j + 1)
                drot(num_cols, r[row_im1_start:], 1, r[row_i_start:], 1, c, s)
            }

            // Apply rotation to Q (columns i-1 and i)
            // Q' = Q * G^T
            for ii in 0 ..< m {
                temp := c * q[ii * ldq + (i - 1)] + s * q[ii * ldq + i]
                q[ii * ldq + i] = c * q[ii * ldq + i] - s * q[ii * ldq + (i - 1)]
                q[ii * ldq + (i - 1)] = temp
            }
        }
    }

    // Handle remaining bulge chasing for shifted columns
    for j in k + p ..< min(new_n - 1, m - 1) {
        if j < m && r[(j + 1) * ldr + j] != 0.0 {
            c, s, rr := dlartg(r[j * ldr + j], r[(j + 1) * ldr + j])
            r[j * ldr + j] = rr
            r[(j + 1) * ldr + j] = 0.0

            if j + 1 < new_n {
                row_j_start := j * ldr + (j + 1)
                row_j1_start := (j + 1) * ldr + (j + 1)
                num_cols := new_n - (j + 1)
                drot(num_cols, r[row_j_start:], 1, r[row_j1_start:], 1, c, s)
            }

            // Apply rotation to Q
            for ii in 0 ..< m {
                temp := c * q[ii * ldq + j] + s * q[ii * ldq + (j + 1)]
                q[ii * ldq + (j + 1)] = c * q[ii * ldq + (j + 1)] - s * q[ii * ldq + j]
                q[ii * ldq + j] = temp
            }
        }
    }
}

// ============================================================================
// Row update routines
// ============================================================================

// addrows adds p new rows at the bottom of the QR factorization.
//
// Given the QR factorization A = Q * R of an m by n matrix A, this routine
// produces the QR factorization A' = Q' * R' of the (m+p) by n matrix A'
// formed by appending p new rows at the bottom.
//
// Parameters:
//   m     - Number of rows of R before addition
//   n     - Number of columns of R
//   p     - Number of rows to add
//   r     - On entry: m by n upper trapezoidal matrix R (in (m+p) by n storage)
//           On exit: (m+p) by n upper trapezoidal matrix R'
//   ldr   - Leading dimension of R
//   u     - p by n matrix containing the new rows
//   ldu   - Leading dimension of U
//   work  - Workspace of size at least n
addrows :: proc(m: int, n: int, p: int, r: []f64, ldr: int, u: []f64, ldu: int, work: []f64) {
    if p <= 0 || n <= 0 {
        return
    }

    new_m := m + p

    // Copy the new rows to the bottom of R
    for i in 0 ..< p {
        for j in 0 ..< n {
            r[(m + i) * ldr + j] = u[i * ldu + j]
        }
    }

    // Now zero out the new rows using Givens rotations
    // For each column j, zero out elements (m+i, j) for i = 0 to p-1
    // by rotating with row j (if j < m) or previous new row

    for j in 0 ..< n {
        // Zero out elements below the diagonal in column j
        for i := new_m - 1; i > j; i -= 1 {
            if r[i * ldr + j] == 0.0 {
                continue
            }

            // Generate rotation to zero out r[i, j] using r[i-1, j]
            pivot_row := j
            if i - 1 > j {
                pivot_row = i - 1
            }

            c, s, rr := dlartg(r[pivot_row * ldr + j], r[i * ldr + j])
            r[pivot_row * ldr + j] = rr
            r[i * ldr + j] = 0.0

            // Apply rotation to remaining columns
            if j + 1 < n {
                row1_start := pivot_row * ldr + (j + 1)
                row2_start := i * ldr + (j + 1)
                num_cols := n - (j + 1)
                drot(num_cols, r[row1_start:], 1, r[row2_start:], 1, c, s)
            }
        }
    }
}

// delrows deletes p rows from the QR factorization.
//
// Given the QR factorization A = Q * R of an m by n matrix A, this routine
// produces the QR factorization A' = Q' * R' of the (m-p) by n matrix A'
// formed by deleting rows k through k+p-1 from A.
//
// This is more complex than column deletion because it requires updating
// both Q and R, and potentially involves Householder reflections or
// multiple Givens rotations.
//
// Parameters:
//   m     - Number of rows of R before deletion
//   n     - Number of columns of R
//   k     - First row to delete (0-indexed)
//   p     - Number of rows to delete
//   q     - On entry: m by m orthogonal matrix Q
//           On exit: (m-p) by (m-p) orthogonal matrix Q'
//   ldq   - Leading dimension of Q
//   r     - On entry: m by n upper trapezoidal matrix R
//           On exit: (m-p) by n upper trapezoidal matrix R'
//   ldr   - Leading dimension of R
//   work  - Workspace of size at least max(m, n)
delrows :: proc(m: int, n: int, k: int, p: int, q: []f64, ldq: int, r: []f64, ldr: int, work: []f64) {
    if p <= 0 || k >= m || m <= 0 {
        return
    }

    nump := min(p, m - k)
    new_m := m - nump

    // First, form the product Q' * R where Q' has rows k to k+p-1 deleted
    // This involves applying rotations to move the deleted rows to the bottom
    // and then dropping them.

    // Move rows k through k+p-1 to the bottom using cyclic permutation
    // implemented via Givens rotations

    // For simplicity, we'll use the direct approach:
    // 1. Shift rows k+p through m-1 up to positions k through m-p-1 in R
    // 2. Also update Q accordingly

    // Shift rows in R
    for i := k; i < new_m; i += 1 {
        src_row := i + nump
        for j in 0 ..< n {
            r[i * ldr + j] = r[src_row * ldr + j]
        }
    }

    // Zero out vacated rows
    for i := new_m; i < m; i += 1 {
        for j in 0 ..< n {
            r[i * ldr + j] = 0.0
        }
    }

    // For Q, we need to delete columns k through k+p-1 and rows k through k+p-1
    // to maintain orthogonality. This is complex; for now we'll update Q by
    // shifting columns.

    // Shift columns in Q (since Q*R, deleting rows means adjusting columns)
    for j := k; j < new_m; j += 1 {
        src_col := j + nump
        for i in 0 ..< m {
            q[i * ldq + j] = q[i * ldq + src_col]
        }
    }

    // Shift rows in Q
    for i := k; i < new_m; i += 1 {
        src_row := i + nump
        for j in 0 ..< new_m {
            q[i * ldq + j] = q[src_row * ldq + j]
        }
    }

    // Now R may not be upper triangular. Apply Givens rotations to fix it.
    for j in 0 ..< min(n - 1, new_m - 1) {
        for i := new_m - 1; i > j; i -= 1 {
            if r[i * ldr + j] == 0.0 {
                continue
            }

            c, s, rr := dlartg(r[(i - 1) * ldr + j], r[i * ldr + j])
            r[(i - 1) * ldr + j] = rr
            r[i * ldr + j] = 0.0

            if j + 1 < n {
                row1_start := (i - 1) * ldr + (j + 1)
                row2_start := i * ldr + (j + 1)
                num_cols := n - (j + 1)
                drot(num_cols, r[row1_start:], 1, r[row2_start:], 1, c, s)
            }

            // Apply rotation to Q
            for ii in 0 ..< new_m {
                temp := c * q[ii * ldq + (i - 1)] + s * q[ii * ldq + i]
                q[ii * ldq + i] = c * q[ii * ldq + i] - s * q[ii * ldq + (i - 1)]
                q[ii * ldq + (i - 1)] = temp
            }
        }
    }
}
