> **Status: superseded for anything actually being built.**
>
> This document was written as a speculative API exploration, before the
> repository adopted the operating rules in `context/data-oriented-design.md`.
> Much of what it proposes is exactly what those rules reject: abstraction
> layers and type parameters added for hypothetical future needs (layout-generic
> matrix types, `f32`/complex support, expression templates, builder patterns,
> GPU hooks), and performance claims with no measurement behind them.
>
> Work on the least-squares subsystem instead follows `docs/OLS_PLAN.md`
> (plan, real data, costs, boundary policy) and `docs/OLS_RESULTS.md`
> (measurements and what was not verified). Where the two disagree, those win.
>
> Kept for the record, and because a few of the observations about the existing
> API still hold. Do not implement from it without re-deriving the need from
> real data first.

# Odin BLAS Library Design Document

## Overview

This document outlines the design for an idiomatic Odin BLAS library that improves upon the current Fortran-style API while maintaining numerical correctness and performance.

## Current State

The current implementation is a direct translation from Reference BLAS Fortran code:
- Matrices represented as flat `[]f64` slices
- Leading dimension (`lda`) parameters everywhere
- Increment parameters (`incx`, `incy`) for strided access
- Character-based flags translated to enums (`Transpose`, `Uplo`, `Diag`)
- No compile-time safety for matrix dimensions
- Manual memory management

## Design Goals

1. **Type Safety**: Catch dimension mismatches at compile time where possible
2. **Ergonomics**: Reduce boilerplate and parameter counts
3. **Zero-Cost Abstractions**: Views and slices without copying
4. **Polymorphism**: Support f32, f64, and potentially complex types
5. **Clarity**: Clear naming with BLAS aliases for experts
6. **Performance**: SIMD and multithreading support
7. **Interoperability**: Easy conversion to/from raw pointers for C FFI

---

## 1. Matrix Types

### 1.1 Core Matrix Structure

```odin
// Generic matrix with compile-time element type
Matrix :: struct($T: typeid) where intrinsics.type_is_numeric(T) {
    data:   []T,           // Underlying storage (owned or borrowed)
    rows:   int,           // Number of rows
    cols:   int,           // Number of columns
    stride: int,           // Elements between consecutive rows (row-major)
                           // or columns (column-major)
    layout: Layout,        // Row-major or column-major
    owns_data: bool,       // Whether this matrix owns its data
}

Layout :: enum {
    Row_Major,    // C-style: elements in a row are contiguous
    Column_Major, // Fortran-style: elements in a column are contiguous
}
```

### 1.2 Specialized Matrix Types

```odin
// Type aliases for common cases
Mat64 :: Matrix(f64)
Mat32 :: Matrix(f32)

// Triangular matrix view (uses same storage, different interpretation)
Triangular_Matrix :: struct($T: typeid) {
    base:   Matrix(T),
    uplo:   Uplo,      // Upper or Lower
    diag:   Diag,      // Unit or Non-unit diagonal
}

// Symmetric matrix view
Symmetric_Matrix :: struct($T: typeid) {
    base:   Matrix(T),
    uplo:   Uplo,      // Which triangle is stored
}

// Band matrix (for future extension)
Band_Matrix :: struct($T: typeid) {
    base:   Matrix(T),
    kl:     int,       // Number of subdiagonals
    ku:     int,       // Number of superdiagonals
}
```

### 1.3 Vector Type

```odin
// Vectors are 1D views, can be row or column
Vector :: struct($T: typeid) {
    data:   []T,
    len:    int,
    stride: int,       // For non-contiguous views
}

Vec64 :: Vector(f64)
Vec32 :: Vector(f32)
```

### 1.4 Compile-Time Sized Matrices (Optional)

For small, fixed-size matrices where dimensions are known at compile time:

```odin
// Stack-allocated fixed-size matrix
Fixed_Matrix :: struct($T: typeid, $M: int, $N: int) {
    data: [M * N]T,
    layout: Layout,
}

// Common fixed sizes
Mat2x2 :: Fixed_Matrix(f64, 2, 2)
Mat3x3 :: Fixed_Matrix(f64, 3, 3)
Mat4x4 :: Fixed_Matrix(f64, 4, 4)
```

---

## 2. Matrix Construction and Memory

### 2.1 Constructors

```odin
// Allocate new matrix
matrix_make :: proc($T: typeid, rows, cols: int, layout := Layout.Row_Major,
                    allocator := context.allocator) -> Matrix(T)

// Create view from existing slice (no allocation)
matrix_from_slice :: proc(data: []$T, rows, cols: int,
                          layout := Layout.Row_Major) -> Matrix(T)

// Create view from raw pointer (for C interop)
matrix_from_ptr :: proc(ptr: [^]$T, rows, cols, stride: int,
                        layout := Layout.Row_Major) -> Matrix(T)

// Create identity matrix
matrix_identity :: proc($T: typeid, n: int, allocator := context.allocator) -> Matrix(T)

// Create zero matrix
matrix_zeros :: proc($T: typeid, rows, cols: int, allocator := context.allocator) -> Matrix(T)
```

### 2.2 Destruction

```odin
// Free matrix data if owned
matrix_destroy :: proc(m: ^Matrix($T)) {
    if m.owns_data && m.data != nil {
        delete(m.data)
    }
    m^ = {}
}
```

### 2.3 Cloning

```odin
// Deep copy
matrix_clone :: proc(m: Matrix($T), allocator := context.allocator) -> Matrix(T)

// Copy into existing matrix
matrix_copy_into :: proc(dst: ^Matrix($T), src: Matrix(T)) -> Error
```

---

## 3. Zero-Copy Views and Submatrices

### 3.1 Submatrix Views

```odin
// Extract submatrix view (no copy)
matrix_submatrix :: proc(m: Matrix($T),
                         row_start, row_end: int,
                         col_start, col_end: int) -> (Matrix(T), Error)

// Example usage:
// A_sub := matrix_submatrix(A, 1, 4, 2, 5)  // rows 1-3, cols 2-4

// Slice notation support via operator overloading (if Odin adds it)
// For now, use explicit calls
```

### 3.2 Row and Column Views

```odin
// Get row as vector view
matrix_row :: proc(m: Matrix($T), i: int) -> (Vector(T), Error)

// Get column as vector view
matrix_col :: proc(m: Matrix($T), j: int) -> (Vector(T), Error)

// Get diagonal as vector view
matrix_diag :: proc(m: Matrix($T), offset := 0) -> (Vector(T), Error)
```

### 3.3 Transpose View

```odin
// Create transposed view (no copy, just swaps interpretation)
matrix_transpose :: proc(m: Matrix($T)) -> Matrix(T) {
    return Matrix(T){
        data   = m.data,
        rows   = m.cols,
        cols   = m.rows,
        stride = m.stride,
        layout = .Column_Major if m.layout == .Row_Major else .Row_Major,
        owns_data = false,  // View never owns data
    }
}
```

### 3.4 Triangular Views

```odin
// Interpret matrix as triangular (no copy)
matrix_as_triangular :: proc(m: Matrix($T), uplo: Uplo,
                              diag := Diag.Non_Unit) -> Triangular_Matrix(T)
```

---

## 4. Element Access

### 4.1 Indexing

```odin
// Safe element access with bounds checking
matrix_get :: proc(m: Matrix($T), i, j: int) -> (T, Error)
matrix_set :: proc(m: ^Matrix($T), i, j: int, value: T) -> Error

// Unsafe fast access (no bounds checking)
matrix_get_unchecked :: #force_inline proc(m: Matrix($T), i, j: int) -> T {
    when m.layout == .Row_Major {
        return m.data[i * m.stride + j]
    } else {
        return m.data[j * m.stride + i]
    }
}

// Operator-style access via procedure overloading
// m[i, j] syntax would require language changes
```

### 4.2 Raw Data Access

```odin
// Get raw pointer for C interop
matrix_raw_data :: proc(m: Matrix($T)) -> [^]T

// Get leading dimension (for BLAS interop)
matrix_leading_dim :: proc(m: Matrix($T)) -> int {
    return m.stride
}
```

---

## 5. Polymorphism Strategy

### 5.1 Type-Parameterized Procedures

```odin
// BLAS operations are generic over numeric types
gemv :: proc(alpha: $T, A: Matrix(T), x: Vector(T),
             beta: T, y: ^Vector(T)) -> Error
    where intrinsics.type_is_float(T)

gemm :: proc(alpha: $T, A, B: Matrix(T),
             beta: T, C: ^Matrix(T)) -> Error
    where intrinsics.type_is_float(T)
```

### 5.2 Specialization via #force_inline

For performance-critical inner loops, provide specialized versions:

```odin
// Generic implementation
dot_generic :: proc(x, y: Vector($T)) -> T { ... }

// Specialized for f64 with SIMD
dot_f64_simd :: proc(x, y: Vector(f64)) -> f64 { ... }

// Dispatch based on type
dot :: proc(x, y: Vector($T)) -> T {
    when T == f64 && ODIN_ARCH == .amd64 {
        return dot_f64_simd(x, y)
    } else {
        return dot_generic(x, y)
    }
}
```

### 5.3 Complex Number Support

```odin
// Complex types
Complex64 :: complex64
Complex128 :: complex128

// Complex-aware operations
dot :: proc(x, y: Vector($T)) -> T {
    when intrinsics.type_is_complex(T) {
        // Use conjugate for complex dot product
        return dotc(x, y)
    } else {
        return dotu(x, y)
    }
}
```

---

## 6. Naming Conventions

### 6.1 Dual Naming Strategy

Provide both descriptive names and traditional BLAS names:

```odin
// Descriptive names (primary API)
matrix_vector_multiply :: proc(...) -> Error
matrix_multiply :: proc(...) -> Error
triangular_solve :: proc(...) -> Error
vector_dot :: proc(...) -> $T

// BLAS aliases (for experts and compatibility)
gemv :: matrix_vector_multiply
gemm :: matrix_multiply
trsv :: triangular_solve
dot  :: vector_dot

// Full BLAS names available in sub-namespace
blas.dgemv :: proc(...)  // Double precision
blas.sgemv :: proc(...)  // Single precision
blas.zgemv :: proc(...)  // Double complex
blas.cgemv :: proc(...)  // Single complex
```

### 6.2 Naming Rules

| Category | Pattern | Example |
|----------|---------|---------|
| Level 1 (vector) | `vector_<operation>` | `vector_scale`, `vector_add` |
| Level 2 (matrix-vector) | `matrix_vector_<op>` | `matrix_vector_multiply` |
| Level 3 (matrix-matrix) | `matrix_<operation>` | `matrix_multiply` |
| Factorizations | `<name>_factorize` | `qr_factorize`, `lu_factorize` |
| Solvers | `<type>_solve` | `triangular_solve`, `least_squares_solve` |
| Updates | `qr_<operation>` | `qr_add_columns`, `qr_delete_rows` |

---

## 7. Error Handling

### 7.1 Error Type

```odin
Error :: enum {
    None,

    // Dimension errors
    Dimension_Mismatch,
    Invalid_Dimensions,
    Not_Square,

    // Index errors
    Out_Of_Bounds,
    Invalid_Stride,

    // Numerical errors
    Singular_Matrix,
    Not_Positive_Definite,
    Convergence_Failed,

    // Memory errors
    Allocation_Failed,
    Buffer_Too_Small,

    // Argument errors
    Invalid_Argument,
    Null_Pointer,
}

// Detailed error with context
Error_Info :: struct {
    code:     Error,
    message:  string,
    location: Source_Code_Location,
}
```

### 7.2 Error Handling Patterns

```odin
// Pattern 1: Return error as second value
result, err := matrix_multiply(A, B, C)
if err != .None {
    log.errorf("Matrix multiply failed: {}", err)
    return err
}

// Pattern 2: or_return for chaining
matrix_multiply(A, B, C) or_return
matrix_add(C, D, E) or_return

// Pattern 3: Assert for internal errors (debug only)
#assert(m.rows > 0, "Invalid matrix dimensions")

// Pattern 4: Panic variants for convenience (use sparingly)
matrix_multiply_must :: proc(A, B, C: Matrix($T)) {
    err := matrix_multiply(A, B, C)
    if err != .None {
        panic(fmt.tprintf("matrix_multiply failed: {}", err))
    }
}
```

### 7.3 Dimension Checking

```odin
// Compile-time checks where possible (fixed-size matrices)
matrix_multiply :: proc(A: Fixed_Matrix($T, $M, $K),
                        B: Fixed_Matrix(T, K, $N)) -> Fixed_Matrix(T, M, N)
// K must match at compile time!

// Runtime checks with helpful messages
@(private)
check_gemm_dims :: proc(A, B, C: Matrix($T)) -> Error {
    if A.cols != B.rows {
        return .Dimension_Mismatch
    }
    if C.rows != A.rows || C.cols != B.cols {
        return .Invalid_Dimensions
    }
    return .None
}
```

---

## 8. SIMD Support

### 8.1 SIMD Vector Types

```odin
import "core:simd"

// SIMD width detection
SIMD_WIDTH_F64 :: #config(SIMD_WIDTH_F64, 4)  // AVX: 4 doubles
SIMD_WIDTH_F32 :: #config(SIMD_WIDTH_F32, 8)  // AVX: 8 floats

F64x4 :: simd.f64x4
F32x8 :: simd.f32x8
```

### 8.2 SIMD-Optimized Kernels

```odin
// SIMD dot product
@(enable_target_feature = "avx2")
dot_f64_avx :: proc(x, y: []f64) -> f64 {
    n := len(x)
    sum := simd.f64x4{}

    i := 0
    for ; i + 4 <= n; i += 4 {
        vx := simd.load(x[i:])
        vy := simd.load(y[i:])
        sum = simd.fma(vx, vy, sum)
    }

    result := simd.reduce_add(sum)

    // Handle remainder
    for ; i < n; i += 1 {
        result += x[i] * y[i]
    }

    return result
}
```

### 8.3 Micro-Kernels for GEMM

```odin
// Register-blocked micro-kernel (e.g., 6x16 for AVX)
GEMM_MR :: 6   // Rows in micro-panel of A
GEMM_NR :: 16  // Columns in micro-panel of B

@(enable_target_feature = "avx2,fma")
gemm_microkernel_6x16 :: proc(
    k: int,
    alpha: f64,
    a: [^]f64,  // MR x k, column-major
    b: [^]f64,  // k x NR, row-major
    beta: f64,
    c: [^]f64,  // MR x NR
    ldc: int,
) { ... }
```

---

## 9. Multithreading

### 9.1 Thread Pool Integration

```odin
import "core:thread"

// Global thread pool (lazily initialized)
@(private)
g_thread_pool: ^thread.Pool

// Configuration
Parallel_Config :: struct {
    num_threads:     int,      // 0 = auto-detect
    min_task_size:   int,      // Minimum elements per task
    grain_size:      int,      // Target elements per task
}

DEFAULT_CONFIG :: Parallel_Config{
    num_threads   = 0,
    min_task_size = 1024,
    grain_size    = 4096,
}
```

### 9.2 Parallel BLAS Operations

```odin
// Parallel GEMM
gemm_parallel :: proc(
    alpha: $T,
    A, B: Matrix(T),
    beta: T,
    C: ^Matrix(T),
    config := DEFAULT_CONFIG,
) -> Error {
    // Determine if parallelization is worthwhile
    work_size := A.rows * B.cols * A.cols
    if work_size < config.min_task_size {
        return gemm(alpha, A, B, beta, C)
    }

    // Partition work
    num_threads := config.num_threads
    if num_threads == 0 {
        num_threads = thread.hardware_parallelism()
    }

    // Use outer-product parallelization
    // Each thread handles a block of rows of C
    rows_per_thread := (C.rows + num_threads - 1) / num_threads

    // Submit tasks
    tasks := make([]thread.Task, num_threads)
    for t in 0 ..< num_threads {
        row_start := t * rows_per_thread
        row_end := min((t + 1) * rows_per_thread, C.rows)

        tasks[t] = thread.pool_add_task(g_thread_pool,
            gemm_task, {alpha, A, B, beta, C, row_start, row_end})
    }

    // Wait for completion
    for task in tasks {
        thread.pool_wait(g_thread_pool, task)
    }

    return .None
}
```

### 9.3 Task-Based API for Composition

```odin
// Future-based API for complex workflows
Computation :: struct($T: typeid) {
    result: ^Matrix(T),
    done:   ^sync.Atomic_Bool,
    error:  Error,
}

// Asynchronous operations
gemm_async :: proc(alpha: $T, A, B: Matrix(T),
                   beta: T, C: ^Matrix(T)) -> Computation(T)

// Wait for completion
computation_wait :: proc(c: Computation($T)) -> Error

// Example usage:
// comp1 := gemm_async(1.0, A, B, 0.0, &C1)
// comp2 := gemm_async(1.0, D, E, 0.0, &C2)
// computation_wait(comp1)
// computation_wait(comp2)
// // Now both C1 and C2 are ready
```

---

## 10. QR Update API Redesign

### 10.1 QR Factorization Object

```odin
// Encapsulate QR factorization state
QR_Factorization :: struct($T: typeid) {
    q:        Matrix(T),    // Orthogonal factor (optional, lazily computed)
    r:        Matrix(T),    // Upper triangular factor
    tau:      []T,          // Householder scalars (compact representation)
    m:        int,          // Original rows
    n:        int,          // Original columns
    has_q:    bool,         // Whether Q is explicitly formed
}

// Create QR factorization
qr_factorize :: proc(A: Matrix($T), compute_q := false) -> (QR_Factorization(T), Error)

// Lazily compute Q if needed
qr_get_q :: proc(qr: ^QR_Factorization($T)) -> (Matrix(T), Error)

// Get R (always available)
qr_get_r :: proc(qr: QR_Factorization($T)) -> Matrix(T)
```

### 10.2 QR Update Operations

```odin
// Add columns at position k
qr_add_columns :: proc(
    qr: ^QR_Factorization($T),
    k: int,                    // Position (0-indexed)
    new_cols: Matrix(T),       // Columns to add
) -> Error

// Delete columns k through k+p-1
qr_delete_columns :: proc(
    qr: ^QR_Factorization($T),
    k: int,                    // First column to delete
    count: int,                // Number of columns
) -> Error

// Add rows at bottom
qr_add_rows :: proc(
    qr: ^QR_Factorization($T),
    new_rows: Matrix(T),
) -> Error

// Delete rows
qr_delete_rows :: proc(
    qr: ^QR_Factorization($T),
    k: int,
    count: int,
) -> Error

// Rank-1 update: A + u*v^T
qr_rank1_update :: proc(
    qr: ^QR_Factorization($T),
    u, v: Vector(T),
) -> Error
```

### 10.3 Least Squares with QR

```odin
// Solve min ||Ax - b||_2 using existing QR
qr_solve_least_squares :: proc(
    qr: QR_Factorization($T),
    b: Vector(T),
) -> (x: Vector(T), residual_norm: T, err: Error)

// Update solution after QR update
qr_update_solution :: proc(
    qr: QR_Factorization($T),
    b: Vector(T),           // Updated RHS
    x_prev: Vector(T),      // Previous solution (for warm start)
) -> (x: Vector(T), err: Error)
```

---

## 11. Builder Pattern for Complex Operations

For operations with many optional parameters:

```odin
// GEMM builder
GEMM_Builder :: struct($T: typeid) {
    alpha:    T,
    beta:     T,
    trans_a:  Transpose,
    trans_b:  Transpose,
    parallel: bool,
    config:   Parallel_Config,
}

gemm_builder :: proc($T: typeid) -> GEMM_Builder(T) {
    return GEMM_Builder(T){
        alpha    = 1,
        beta     = 0,
        trans_a  = .No_Trans,
        trans_b  = .No_Trans,
        parallel = false,
    }
}

// Chainable setters
gemm_alpha :: proc(b: GEMM_Builder($T), alpha: T) -> GEMM_Builder(T)
gemm_beta :: proc(b: GEMM_Builder($T), beta: T) -> GEMM_Builder(T)
gemm_transpose_a :: proc(b: GEMM_Builder($T)) -> GEMM_Builder(T)
gemm_parallel :: proc(b: GEMM_Builder($T), config := DEFAULT_CONFIG) -> GEMM_Builder(T)

// Execute
gemm_execute :: proc(b: GEMM_Builder($T), A, B: Matrix(T), C: ^Matrix(T)) -> Error

// Usage:
// gemm_builder(f64)
//     ->gemm_alpha(2.0)
//     ->gemm_beta(1.0)
//     ->gemm_transpose_a()
//     ->gemm_parallel()
//     ->gemm_execute(A, B, &C)
```

---

## 12. Memory Layout Optimization

### 12.1 Automatic Layout Selection

```odin
// Hint for optimal layout based on usage pattern
Layout_Hint :: enum {
    General,           // No specific pattern
    Row_Access,        // Frequent row access
    Column_Access,     // Frequent column access
    Both,              // Both row and column access
    Multiply_Left,     // Used as left operand in multiply
    Multiply_Right,    // Used as right operand in multiply
}

matrix_make_optimized :: proc($T: typeid, rows, cols: int,
                               hint: Layout_Hint) -> Matrix(T) {
    layout: Layout
    switch hint {
    case .Row_Access, .Multiply_Left:
        layout = .Row_Major
    case .Column_Access, .Multiply_Right:
        layout = .Column_Major
    case:
        layout = .Row_Major  // Default
    }
    return matrix_make(T, rows, cols, layout)
}
```

### 12.2 Packed Formats

```odin
// Packed triangular storage (saves ~50% memory)
Packed_Triangular :: struct($T: typeid) {
    data:  []T,        // n*(n+1)/2 elements
    n:     int,
    uplo:  Uplo,
}

// Convert between packed and full
packed_to_full :: proc(p: Packed_Triangular($T)) -> Matrix(T)
full_to_packed :: proc(m: Matrix($T), uplo: Uplo) -> Packed_Triangular(T)
```

---

## 13. Debugging and Diagnostics

### 13.1 Debug Printing

```odin
// Pretty print matrix
matrix_print :: proc(m: Matrix($T), name := "", precision := 4)

// Print with full precision for debugging
matrix_print_full :: proc(m: Matrix($T), name := "")

// Print structure (dimensions, stride, layout)
matrix_describe :: proc(m: Matrix($T))
```

### 13.2 Validation

```odin
// Check matrix properties
matrix_is_finite :: proc(m: Matrix($T)) -> bool
matrix_is_symmetric :: proc(m: Matrix($T), tol: T = 1e-10) -> bool
matrix_is_orthogonal :: proc(m: Matrix($T), tol: T = 1e-10) -> bool
matrix_is_upper_triangular :: proc(m: Matrix($T), tol: T = 1e-10) -> bool

// Compute condition number (expensive)
matrix_condition_number :: proc(m: Matrix($T)) -> T
```

---

## 14. C Interoperability

### 14.1 C-Compatible Types

```odin
// C-compatible matrix descriptor
C_Matrix :: struct {
    data:   rawptr,
    rows:   c.int,
    cols:   c.int,
    stride: c.int,
    layout: c.int,  // 0 = row-major, 1 = column-major
}

// Convert Odin matrix to C descriptor
matrix_to_c :: proc(m: Matrix($T)) -> C_Matrix

// Wrap C matrix in Odin type (no copy)
matrix_from_c :: proc(cm: C_Matrix, $T: typeid) -> Matrix(T)
```

### 14.2 CBLAS Compatibility Layer

```odin
// CBLAS-compatible function signatures
@(export, link_name="cblas_dgemm")
cblas_dgemm :: proc "c" (
    layout: c.int,
    transA: c.int,
    transB: c.int,
    M, N, K: c.int,
    alpha: f64,
    A: [^]f64, lda: c.int,
    B: [^]f64, ldb: c.int,
    beta: f64,
    C: [^]f64, ldc: c.int,
) { ... }
```

---

## 15. Implementation Phases

### Phase 1: Core Types and Basic Operations
- [ ] Define Matrix, Vector, Error types
- [ ] Implement constructors and destructors
- [ ] Implement view operations (submatrix, transpose)
- [ ] Implement element access
- [ ] Port existing BLAS Level 1 to new API

### Phase 2: Full BLAS Coverage
- [ ] BLAS Level 2 with new types
- [ ] BLAS Level 3 with new types
- [ ] Triangular and symmetric specializations
- [ ] Add descriptive name aliases

### Phase 3: QR Updates Redesign
- [ ] QR_Factorization type
- [ ] Refactor qr_add_columns, qr_delete_columns
- [ ] Refactor qr_add_rows, qr_delete_rows
- [ ] Least squares solver

### Phase 4: Performance Optimization
- [ ] SIMD kernels for critical paths
- [ ] Micro-kernels for GEMM
- [ ] Cache-aware blocking

### Phase 5: Parallelization
- [ ] Thread pool integration
- [ ] Parallel GEMM
- [ ] Parallel QR factorization
- [ ] Task-based async API

### Phase 6: Polish
- [ ] Comprehensive error messages
- [ ] Debug utilities
- [ ] C interop layer
- [ ] Documentation and examples

---

## 16. Example Usage

```odin
import "blas"

main :: proc() {
    // Create matrices
    A := blas.matrix_make(f64, 100, 50)
    defer blas.matrix_destroy(&A)

    B := blas.matrix_make(f64, 50, 30)
    defer blas.matrix_destroy(&B)

    C := blas.matrix_make(f64, 100, 30)
    defer blas.matrix_destroy(&C)

    // Fill with data...

    // Matrix multiply: C = A * B
    blas.matrix_multiply(A, B, &C) or_return

    // Or with explicit parameters:
    blas.gemm(1.0, A, B, 0.0, &C) or_return

    // Submatrix view (no copy)
    A_sub, _ := blas.matrix_submatrix(A, 0, 50, 0, 25)

    // QR factorization
    qr, _ := blas.qr_factorize(A, compute_q = true)
    defer blas.qr_destroy(&qr)

    // Solve least squares
    b := blas.vector_make(f64, 100)
    defer blas.vector_destroy(&b)

    x, residual, _ := blas.qr_solve_least_squares(qr, b)
    defer blas.vector_destroy(&x)

    // Add a column to the factorization
    new_col := blas.vector_make(f64, 100)
    defer blas.vector_destroy(&new_col)

    blas.qr_add_columns(&qr, 25, blas.vector_as_matrix(new_col)) or_return

    // Updated solution
    x_new, _ := blas.qr_solve_least_squares(qr, b)
}
```

---

## 17. Open Questions

1. **Allocator Strategy**: Should matrices carry their allocator, or rely on context?

2. **Copy vs Move Semantics**: How to handle ownership transfer efficiently?

3. **Expression Templates**: Worth implementing lazy evaluation for expressions like `A*B + C*D`?

4. **GPU Support**: Design hooks for future CUDA/OpenCL integration?

5. **Sparse Matrices**: Include in this design or separate package?

6. **Fixed-Size Matrices**: Worth the complexity for small matrix optimizations?
