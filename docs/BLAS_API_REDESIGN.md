# Odin BLAS API Redesign

## Executive Summary

This document proposes a modernized API for the Odin BLAS library that improves ergonomics, type safety, and performance while maintaining compatibility with the low-level Reference BLAS semantics when needed. The design emphasizes:

- **Zero-cost abstractions**: Higher-level types that compile down to the same code
- **Compile-time safety**: Catching dimension mismatches and layout errors at compile time
- **Explicit memory layout**: Row-major vs column-major as part of the type
- **Cheap views**: Submatrices without copying via stride-aware slicing
- **Polymorphic operations**: Generic functions that work across matrix types
- **Clear error handling**: Result types with actionable error information
- **Ergonomic naming**: Readable aliases alongside traditional BLAS names

---

## 1. Matrix Type Design

### 1.1 Core Matrix Type

The fundamental insight is that a matrix is a **strided view over contiguous memory**. We define:

```odin
Matrix :: struct($T: typeid, $Layout: Layout_Kind) {
    data:   []T,      // Underlying storage (borrowed, not owned)
    rows:   int,      // Number of rows (m)
    cols:   int,      // Number of columns (n)
    stride: int,      // Distance between row starts (row-major) or column starts (col-major)
}

Layout_Kind :: enum {
    Row_Major,    // C-style: elements in a row are contiguous
    Col_Major,    // Fortran-style: elements in a column are contiguous
}
```

**Key properties:**
- `$T` is the element type (`f32`, `f64`, `complex64`, `complex128`)
- `$Layout` is compile-time known, enabling layout-specific optimizations
- `data` is a **borrowed slice** - the matrix does not own memory
- `stride` generalizes the traditional `lda` (leading dimension) parameter

### 1.2 Convenience Type Aliases

```odin
// Double precision, row-major (most common in C/Odin code)
Mat64   :: Matrix(f64, .Row_Major)
Mat64c  :: Matrix(f64, .Col_Major)

// Single precision
Mat32   :: Matrix(f32, .Row_Major)
Mat32c  :: Matrix(f32, .Col_Major)

// Complex types
CMat64  :: Matrix(complex128, .Row_Major)
CMat128 :: Matrix(complex128, .Col_Major)

// Vectors (1D case - layout irrelevant, use stride for increment)
Vec64   :: distinct []f64
Vec32   :: distinct []f32
```

### 1.3 Owned Matrix Type

For cases where the matrix owns its storage:

```odin
Owned_Matrix :: struct($T: typeid, $Layout: Layout_Kind) {
    mat:       Matrix(T, Layout),
    allocator: mem.Allocator,
}
```

**Factory functions:**

```odin
// Create owned matrix with default allocator
make_matrix :: proc($T: typeid, $Layout: Layout_Kind, rows, cols: int,
                    allocator := context.allocator) -> Owned_Matrix(T, Layout) {
    data := make([]T, rows * cols, allocator)
    return Owned_Matrix(T, Layout){
        mat = Matrix(T, Layout){
            data   = data,
            rows   = rows,
            cols   = cols,
            stride = cols when Layout == .Row_Major else rows,
        },
        allocator = allocator,
    }
}

// Free owned matrix
delete_matrix :: proc(m: ^Owned_Matrix($T, $L)) {
    delete(m.mat.data, m.allocator)
    m.mat = {}
}
```

### 1.4 Views and Borrowing

The borrowed `Matrix` type enables zero-copy submatrix extraction:

```odin
// Borrow a matrix view from owned storage
borrow :: proc(m: ^Owned_Matrix($T, $L)) -> Matrix(T, L) {
    return m.mat
}

// Create matrix view from raw slice (for interop with existing code)
from_slice :: proc($T: typeid, $Layout: Layout_Kind,
                   data: []T, rows, cols, stride: int) -> Matrix(T, Layout) {
    return Matrix(T, Layout){data = data, rows = rows, cols = cols, stride = stride}
}

// Create matrix from slice with default stride
from_slice_packed :: proc($T: typeid, $Layout: Layout_Kind,
                          data: []T, rows, cols: int) -> Matrix(T, Layout) {
    stride := cols when Layout == .Row_Major else rows
    return from_slice(T, Layout, data, rows, cols, stride)
}
```

---

## 2. Submatrix Views (Zero-Copy Slicing)

### 2.1 Row and Column Extraction

```odin
// Extract row as vector view (zero-copy for row-major)
row :: proc(m: Matrix($T, $L), i: int) -> []T
    where L == .Row_Major {
    start := i * m.stride
    return m.data[start:start + m.cols]
}

// Extract row for column-major (strided view)
row_strided :: proc(m: Matrix($T, $L), i: int) -> Strided_Slice(T)
    where L == .Col_Major {
    return Strided_Slice(T){
        data   = m.data[i:],
        len    = m.cols,
        stride = m.stride,
    }
}

// Extract column as vector view (zero-copy for column-major)
col :: proc(m: Matrix($T, $L), j: int) -> []T
    where L == .Col_Major {
    start := j * m.stride
    return m.data[start:start + m.rows]
}

// Extract column for row-major (strided view)
col_strided :: proc(m: Matrix($T, $L), j: int) -> Strided_Slice(T)
    where L == .Row_Major {
    return Strided_Slice(T){
        data   = m.data[j:],
        len    = m.rows,
        stride = m.stride,
    }
}
```

### 2.2 Strided Slice Type

For non-contiguous views:

```odin
Strided_Slice :: struct($T: typeid) {
    data:   []T,    // Starting position
    len:    int,    // Number of elements
    stride: int,    // Step between elements
}

// Iterator over strided slice
strided_iter :: proc(s: Strided_Slice($T)) -> iter.Iterator(T) {
    // Implementation yields s.data[i * s.stride] for i in 0..<s.len
}
```

### 2.3 Submatrix Extraction

```odin
// Extract submatrix view - zero copy, just adjusts pointers and dimensions
submatrix :: proc(m: Matrix($T, $L),
                  row_start, col_start: int,
                  num_rows, num_cols: int) -> Matrix(T, L) {
    offset := when L == .Row_Major {
        row_start * m.stride + col_start
    } else {
        col_start * m.stride + row_start
    }

    return Matrix(T, L){
        data   = m.data[offset:],
        rows   = num_rows,
        cols   = num_cols,
        stride = m.stride,  // Stride preserved from parent
    }
}

// Shorthand: extract trailing submatrix (common in QR updates)
tail :: proc(m: Matrix($T, $L), row_start, col_start: int) -> Matrix(T, L) {
    return submatrix(m, row_start, col_start, m.rows - row_start, m.cols - col_start)
}

// Extract leading submatrix
head :: proc(m: Matrix($T, $L), num_rows, num_cols: int) -> Matrix(T, L) {
    return submatrix(m, 0, 0, num_rows, num_cols)
}
```

### 2.4 Transposed Views

Transposition is a **layout flip** rather than data movement:

```odin
// Return transposed view (compile-time layout change)
transpose :: proc(m: Matrix($T, $L)) -> Matrix(T, flip_layout(L)) {
    return Matrix(T, flip_layout(L)){
        data   = m.data,
        rows   = m.cols,  // Swap dimensions
        cols   = m.rows,
        stride = m.stride,
    }
}

flip_layout :: proc($L: Layout_Kind) -> Layout_Kind {
    return .Col_Major when L == .Row_Major else .Row_Major
}
```

This means `transpose(transpose(m))` returns a matrix with the original layout - correct by construction.

---

## 3. Polymorphic BLAS Functions

### 3.1 Generic Procedure Groups

Use Odin's procedure overloading to create unified interfaces:

```odin
// Unified interface - dispatches based on argument types
gemm :: proc{gemm_mat, gemm_raw}

// Matrix-based interface (preferred)
gemm_mat :: proc(alpha: $T, a: Matrix(T, $La), b: Matrix(T, $Lb),
                 beta: T, c: ^Matrix(T, $Lc)) -> Blas_Error
    where T == f64 || T == f32 {
    // Validate dimensions at runtime (compile-time would require dependent types)
    if a.cols != b.rows || a.rows != c.rows || b.cols != c.cols {
        return .Dimension_Mismatch
    }

    // Select kernel based on layouts
    when La == .Row_Major && Lb == .Row_Major && Lc == .Row_Major {
        gemm_rrr(alpha, a, b, beta, c)
    } else when La == .Col_Major && Lb == .Col_Major && Lc == .Col_Major {
        gemm_ccc(alpha, a, b, beta, c)
    } else {
        gemm_generic(alpha, a, b, beta, c)
    }

    return nil
}

// Raw BLAS interface (for interop)
gemm_raw :: proc(trans_a, trans_b: Transpose,
                 m, n, k: int,
                 alpha: f64,
                 a: []f64, lda: int,
                 b: []f64, ldb: int,
                 beta: f64,
                 c: []f64, ldc: int) {
    // Original dgemm implementation
}
```

### 3.2 Element Type Polymorphism

Use Odin's parametric polymorphism for type-generic operations:

```odin
// Generic axpy: y = alpha * x + y
axpy :: proc(alpha: $T, x: []T, y: []T)
    where T == f32 || T == f64 || T == complex64 || T == complex128 {
    when T == f32 {
        saxpy_impl(alpha, x, y)
    } else when T == f64 {
        daxpy_impl(alpha, x, y)
    } else when T == complex64 {
        caxpy_impl(alpha, x, y)
    } else {
        zaxpy_impl(alpha, x, y)
    }
}

// Or use intrinsics for simple operations
axpy_simple :: proc(alpha: $T, x, y: []T) #no_bounds_check {
    assert(len(x) == len(y), "axpy: dimension mismatch")
    for i in 0..<len(x) {
        y[i] += alpha * x[i]
    }
}
```

### 3.3 Strided Vector Operations

For BLAS Level 1 with non-unit increments:

```odin
// Strided axpy
axpy_strided :: proc(n: int, alpha: $T,
                     x: []T, incx: int,
                     y: []T, incy: int) #no_bounds_check {
    ix, iy := 0, 0
    if incx < 0 { ix = (-n + 1) * incx }
    if incy < 0 { iy = (-n + 1) * incy }

    for _ in 0..<n {
        y[iy] += alpha * x[ix]
        ix += incx
        iy += incy
    }
}

// Or use Strided_Slice type
axpy_s :: proc(alpha: $T, x, y: Strided_Slice(T)) {
    axpy_strided(x.len, alpha, x.data, x.stride, y.data, y.stride)
}
```

---

## 4. Error Handling

### 4.1 Error Type Definition

```odin
Blas_Error :: enum {
    None,                    // Success
    Dimension_Mismatch,      // Matrix/vector dimensions incompatible
    Invalid_Stride,          // Zero or incompatible stride
    Invalid_Dimension,       // Negative dimension
    Leading_Dim_Too_Small,   // lda < max(1, m)
    Buffer_Too_Small,        // Provided buffer insufficient
    Singular_Matrix,         // Matrix is singular (for triangular solve)
    Not_Positive_Definite,   // Matrix not positive definite (Cholesky)
    Workspace_Too_Small,     // Work array insufficient
    Invalid_Argument,        // Generic invalid parameter
}

// Error with location info (for debugging)
Blas_Error_Info :: struct {
    error:    Blas_Error,
    param:    int,          // Which parameter (1-indexed, BLAS convention)
    message:  string,       // Human-readable description
    location: Source_Code_Location,
}
```

### 4.2 Result Type Pattern

```odin
Blas_Result :: union($T: typeid) {
    T,
    Blas_Error,
}

// Usage example
dot :: proc(x, y: []$T) -> Blas_Result(T) {
    if len(x) != len(y) {
        return .Dimension_Mismatch
    }

    sum: T = 0
    for i in 0..<len(x) {
        sum += x[i] * y[i]
    }
    return sum
}

// Call site
result := dot(x, y)
switch v in result {
case f64:
    fmt.println("Dot product:", v)
case Blas_Error:
    fmt.println("Error:", v)
}

// Or with helper
if value, ok := dot(x, y).?; ok {
    // Use value
}
```

### 4.3 Checked vs Unchecked Variants

Provide both safe and unsafe versions:

```odin
// Safe version - validates inputs, returns Result
gemv :: proc(alpha: $T, a: Matrix(T, $L), x: []T, beta: T, y: []T) -> Blas_Error {
    if a.cols != len(x) || a.rows != len(y) {
        return .Dimension_Mismatch
    }
    gemv_unchecked(alpha, a, x, beta, y)
    return nil
}

// Unchecked version - no validation, maximum performance
gemv_unchecked :: proc(alpha: $T, a: Matrix(T, $L), x: []T, beta: T, y: []T) #no_bounds_check {
    // Implementation assumes all dimensions are valid
}
```

### 4.4 Debug vs Release Behavior

```odin
BLAS_DEBUG :: #config(BLAS_DEBUG, ODIN_DEBUG)

validate :: proc(cond: bool, error: Blas_Error, loc := #caller_location) -> Maybe(Blas_Error) {
    when BLAS_DEBUG {
        if !cond {
            return error
        }
    }
    return nil
}
```

---

## 5. Naming Conventions

### 5.1 Dual Naming Strategy

Keep traditional BLAS names for experts, add readable aliases:

```odin
// Traditional BLAS names (short, familiar to domain experts)
dgemm :: gemm_raw   // Double General Matrix Multiply
dgemv :: gemv_raw   // Double General Matrix-Vector
dtrsm :: trsm_raw   // Double Triangular Solve Multiple
dgeqrf :: qr_raw    // Double General QR Factorization

// Readable aliases (self-documenting)
matrix_multiply      :: gemm
matrix_vector        :: gemv
triangular_solve     :: trsm
qr_factorize         :: qr

// Action-oriented names for updates
qr_delete_columns    :: delcols
qr_add_columns       :: addcols
qr_delete_rows       :: delrows
qr_add_rows          :: addrows
```

### 5.2 Naming Rules

1. **Matrix operations**: `matrix_<operation>` or `mat_<operation>`
2. **Vector operations**: `vec_<operation>` or bare verb
3. **In-place operations**: suffix with `_inplace` or use `^` (pointer) receiver
4. **Out-of-place**: return new matrix
5. **Destructive updates**: clearly documented, prefer `_mut` suffix

```odin
// Examples
scale       :: dscal      // y = alpha * x (in-place)
copy        :: dcopy      // y = x
add_scaled  :: daxpy      // y += alpha * x
norm2       :: dnrm2      // ||x||_2
dot         :: ddot       // x . y
```

### 5.3 Name Length Guidelines

- **1-4 chars**: Reserved for traditional BLAS (`gemm`, `axpy`, `dot`)
- **5-12 chars**: Primary readable names (`mat_mul`, `qr_factor`)
- **13+ chars**: Full descriptive names (`matrix_multiply`, `qr_factorize`)

Expose all three levels; let users import what they prefer:

```odin
// Package organization
blas      // Traditional: dgemm, daxpy, dtrsm
linalg    // Readable: matrix_multiply, qr_factor
la        // Short readable: mat_mul, qr
```

---

## 6. Allocation Patterns

### 6.1 Workspace Management

```odin
Workspace :: struct($T: typeid) {
    data:      []T,
    allocator: mem.Allocator,
    owned:     bool,
}

// Query workspace requirements
qr_workspace_size :: proc(m, n: int) -> int {
    return max(1, n)  // Minimum for Householder reflectors
}

// Create workspace
make_workspace :: proc($T: typeid, size: int,
                       allocator := context.allocator) -> Workspace(T) {
    return Workspace(T){
        data      = make([]T, size, allocator),
        allocator = allocator,
        owned     = true,
    }
}

// Use provided buffer as workspace (no allocation)
use_as_workspace :: proc(buffer: []$T) -> Workspace(T) {
    return Workspace(T){
        data  = buffer,
        owned = false,
    }
}

// Free workspace
delete_workspace :: proc(w: ^Workspace($T)) {
    if w.owned {
        delete(w.data, w.allocator)
    }
    w^ = {}
}
```

### 6.2 Scratch Allocator Integration

```odin
// Use temp allocator for short-lived workspace
with_temp_workspace :: proc($T: typeid, size: int,
                            body: proc(ws: []T)) {
    context.allocator = context.temp_allocator
    ws := make([]T, size)
    defer free_all(context.temp_allocator)
    body(ws)
}

// Usage
with_temp_workspace(f64, qr_workspace_size(m, n), proc(ws: []f64) {
    qr_factorize(a, tau, ws)
})
```

### 6.3 Arena Allocator Pattern

```odin
Blas_Context :: struct {
    arena:     mem.Arena,
    allocator: mem.Allocator,
}

init_blas_context :: proc(backing: []byte) -> Blas_Context {
    ctx: Blas_Context
    mem.arena_init(&ctx.arena, backing)
    ctx.allocator = mem.arena_allocator(&ctx.arena)
    return ctx
}

// Allocate from arena
alloc_matrix :: proc(ctx: ^Blas_Context, $T: typeid, $L: Layout_Kind,
                     rows, cols: int) -> Owned_Matrix(T, L) {
    return make_matrix(T, L, rows, cols, ctx.allocator)
}

// Reset arena (free all at once)
reset_blas_context :: proc(ctx: ^Blas_Context) {
    mem.arena_free_all(&ctx.arena)
}
```

### 6.4 Stack Allocation for Small Matrices

```odin
SMALL_MATRIX_THRESHOLD :: 64  // Elements

// Stack-allocated small matrix
Small_Matrix :: struct($T: typeid, $Rows, $Cols: int) {
    data: [Rows * Cols]T,
}

// Convert to view
to_matrix :: proc(sm: ^Small_Matrix($T, $R, $C)) -> Matrix(T, .Row_Major) {
    return Matrix(T, .Row_Major){
        data   = sm.data[:],
        rows   = R,
        cols   = C,
        stride = C,
    }
}

// Usage: 3x3 rotation matrix on stack
rot: Small_Matrix(f64, 3, 3)
m := to_matrix(&rot)
```

---

## 7. Calling Conventions and Context

### 7.1 Context-Free Core

BLAS core operations should not depend on Odin's implicit context:

```odin
// Core computational kernel - no context dependency
@(no_context)
gemm_kernel :: proc "c" (m, n, k: int,
                          alpha: f64,
                          a: [^]f64, lda: int,
                          b: [^]f64, ldb: int,
                          beta: f64,
                          c: [^]f64, ldc: int) {
    // Pure computation, no allocations, no context
}
```

### 7.2 Context-Aware Wrapper

High-level functions use context for allocation/error handling:

```odin
// Wrapper that uses context
qr_factorize :: proc(a: ^Matrix($T, $L), allocator := context.allocator) -> (tau: []T, err: Blas_Error) {
    tau = make([]T, min(a.rows, a.cols), allocator)
    work := make([]T, qr_workspace_size(a.rows, a.cols), context.temp_allocator)
    defer delete(work, context.temp_allocator)

    err = qr_factorize_with_workspace(a, tau, work)
    return
}

// No-context version
qr_factorize_with_workspace :: proc(a: ^Matrix($T, $L), tau: []T, work: []T) -> Blas_Error #no_context {
    // Implementation
}
```

### 7.3 Explicit vs Implicit Allocator

```odin
// Option A: Explicit allocator parameter (recommended)
make_matrix_explicit :: proc($T: typeid, rows, cols: int,
                             alloc: mem.Allocator) -> Owned_Matrix(T, .Row_Major)

// Option B: Use context allocator (convenient)
make_matrix_context :: proc($T: typeid, rows, cols: int) -> Owned_Matrix(T, .Row_Major) {
    return make_matrix_explicit(T, rows, cols, context.allocator)
}

// Procedure group for flexibility
make_matrix :: proc{make_matrix_explicit, make_matrix_context}
```

---

## 8. Bounds Checking Strategy

### 8.1 Layered Validation

```odin
// Layer 1: API boundary (always check)
gemm :: proc(a, b: Matrix($T, $L), c: ^Matrix(T, L)) -> Blas_Error {
    if a.cols != b.rows {
        return .Dimension_Mismatch
    }
    if a.rows != c.rows || b.cols != c.cols {
        return .Dimension_Mismatch
    }

    gemm_validated(a, b, c)
    return nil
}

// Layer 2: Validated entry point (optional checks based on config)
gemm_validated :: proc(a, b: Matrix($T, $L), c: ^Matrix(T, L)) {
    when BLAS_DEBUG {
        assert(a.cols == b.rows)
        // Additional invariant checks
    }

    gemm_unchecked(a, b, c)
}

// Layer 3: Unchecked kernel (no validation)
@(no_bounds_check)
gemm_unchecked :: proc(a, b: Matrix($T, $L), c: ^Matrix(T, L)) {
    // Tight computational loop
}
```

### 8.2 Compile-Time Dimension Checking

For fixed-size matrices, use Odin's type system:

```odin
// Fixed-size matrix type
Fixed_Matrix :: struct($T: typeid, $Rows, $Cols: int) {
    data: [Rows * Cols]T,
}

// Compile-time checked multiplication
mul_fixed :: proc(a: Fixed_Matrix($T, $M, $K),
                  b: Fixed_Matrix(T, K, $N)) -> Fixed_Matrix(T, M, N) {
    // K must match - this is enforced by the type system
    result: Fixed_Matrix(T, M, N)
    // Implementation
    return result
}

// Usage - dimension mismatch is compile error
a: Fixed_Matrix(f64, 3, 4)
b: Fixed_Matrix(f64, 4, 5)
c := mul_fixed(a, b)  // OK: (3x4) * (4x5) = (3x5)

// This would not compile:
// d: Fixed_Matrix(f64, 3, 3)
// e := mul_fixed(a, d)  // Error: K=4 vs K=3 mismatch
```

### 8.3 Debug Assertions

```odin
// Debug-only bounds verification
@(disabled = !BLAS_DEBUG)
verify_bounds :: proc(m: Matrix($T, $L), i, j: int, loc := #caller_location) {
    if i < 0 || i >= m.rows || j < 0 || j >= m.cols {
        panic(fmt.tprintf("Index (%d, %d) out of bounds for %dx%d matrix",
                          i, j, m.rows, m.cols), loc)
    }
}

// Matrix element access with optional bounds check
at :: proc(m: Matrix($T, $L), i, j: int) -> T {
    when BLAS_DEBUG {
        verify_bounds(m, i, j)
    }

    offset := when L == .Row_Major {
        i * m.stride + j
    } else {
        j * m.stride + i
    }

    return m.data[offset]
}
```

---

## 9. Multithreading and Parallelism

### 9.1 Thread Pool Integration

```odin
Blas_Thread_Config :: struct {
    num_threads:     int,       // 0 = auto-detect
    min_parallel_n:  int,       // Minimum size for parallelization
    chunk_size:      int,       // Work unit size
}

// Global configuration (can be overridden per-call)
@(thread_local)
blas_thread_config: Blas_Thread_Config = {
    num_threads    = 0,
    min_parallel_n = 1024,
    chunk_size     = 64,
}

set_num_threads :: proc(n: int) {
    blas_thread_config.num_threads = n
}
```

### 9.2 Parallel Matrix Multiply

```odin
gemm_parallel :: proc(alpha: $T, a, b: Matrix(T, $L), beta: T, c: ^Matrix(T, L)) {
    m, n, k := a.rows, b.cols, a.cols

    // Fall back to sequential for small matrices
    if m * n < blas_thread_config.min_parallel_n {
        gemm_unchecked(alpha, a, b, beta, c)
        return
    }

    // Partition work by output rows
    num_threads := effective_num_threads()
    rows_per_thread := (m + num_threads - 1) / num_threads

    sync.wait_group_add(&wg, num_threads)

    for t in 0..<num_threads {
        row_start := t * rows_per_thread
        row_end := min(row_start + rows_per_thread, m)

        if row_start >= m { break }

        thread.run(proc(row_start, row_end: int) {
            defer sync.wait_group_done(&wg)

            a_sub := submatrix(a, row_start, 0, row_end - row_start, k)
            c_sub := submatrix(c^, row_start, 0, row_end - row_start, n)

            gemm_unchecked(alpha, a_sub, b, beta, &c_sub)
        })
    }

    sync.wait_group_wait(&wg)
}
```

### 9.3 Job System Interface

```odin
Blas_Job :: struct {
    proc_ptr:  rawptr,           // Function to execute
    args:      rawptr,           // Arguments
    deps:      []^Blas_Job,      // Dependencies (must complete first)
    completed: sync.Atomic(bool),
}

// Create job for matrix operation
make_gemm_job :: proc(alpha: $T, a, b: Matrix(T, $L), beta: T,
                      c: ^Matrix(T, L), deps: ..^Blas_Job) -> ^Blas_Job {
    // Package arguments and create job
}

// Submit to job system
submit_blas_job :: proc(job: ^Blas_Job) {
    // Add to thread pool queue
}

// Wait for completion
wait_blas_job :: proc(job: ^Blas_Job) {
    for !job.completed.load(.Acquire) {
        thread.yield()
    }
}
```

### 9.4 SIMD Hints

```odin
// Inner kernel with SIMD hints
@(no_bounds_check, optimization_mode = .Speed)
gemm_kernel_4x4 :: proc(a: [^]f64, b: [^]f64, c: [^]f64,
                        k, lda, ldb, ldc: int) {
    // 4x4 tile computation
    // Compiler can vectorize with #simd
    c00, c01, c02, c03: f64 = 0, 0, 0, 0
    c10, c11, c12, c13: f64 = 0, 0, 0, 0
    c20, c21, c22, c23: f64 = 0, 0, 0, 0
    c30, c31, c32, c33: f64 = 0, 0, 0, 0

    for l in 0..<k {
        a0, a1, a2, a3 := a[0*lda + l], a[1*lda + l], a[2*lda + l], a[3*lda + l]
        b0, b1, b2, b3 := b[l*ldb + 0], b[l*ldb + 1], b[l*ldb + 2], b[l*ldb + 3]

        c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3
        c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3
        c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3
        c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3
    }

    c[0*ldc + 0] += c00; c[0*ldc + 1] += c01; c[0*ldc + 2] += c02; c[0*ldc + 3] += c03
    c[1*ldc + 0] += c10; c[1*ldc + 1] += c11; c[1*ldc + 2] += c12; c[1*ldc + 3] += c13
    c[2*ldc + 0] += c20; c[2*ldc + 1] += c21; c[2*ldc + 2] += c22; c[2*ldc + 3] += c23
    c[3*ldc + 0] += c30; c[3*ldc + 1] += c31; c[3*ldc + 2] += c32; c[3*ldc + 3] += c33
}
```

---

## 10. Additional API Improvements

### 10.1 Builder Pattern for Complex Operations

```odin
Gemm_Builder :: struct($T: typeid) {
    alpha:      T,
    beta:       T,
    a:          Matrix(T, .Row_Major),
    b:          Matrix(T, .Row_Major),
    c:          ^Matrix(T, .Row_Major),
    trans_a:    Transpose,
    trans_b:    Transpose,
    parallel:   bool,
}

gemm_builder :: proc($T: typeid) -> Gemm_Builder(T) {
    return Gemm_Builder(T){alpha = 1, beta = 0}
}

with_alpha :: proc(b: ^Gemm_Builder($T), alpha: T) -> ^Gemm_Builder(T) {
    b.alpha = alpha
    return b
}

with_a :: proc(b: ^Gemm_Builder($T), a: Matrix(T, .Row_Major),
               trans: Transpose = .No_Trans) -> ^Gemm_Builder(T) {
    b.a = a
    b.trans_a = trans
    return b
}

// ... similar for other parameters ...

execute :: proc(b: ^Gemm_Builder($T)) -> Blas_Error {
    // Validate and execute
}

// Usage:
gemm_builder(f64)
    ->with_alpha(2.0)
    ->with_a(a_mat)
    ->with_b(b_mat, .Trans)
    ->with_c(&c_mat)
    ->parallel(true)
    ->execute()
```

### 10.2 Expression Templates (Lazy Evaluation)

```odin
// Deferred computation representation
Matrix_Expr :: union($T: typeid) {
    Matrix(T, .Row_Major),
    Matrix_Add(T),
    Matrix_Mul(T),
    Matrix_Scale(T),
}

Matrix_Mul :: struct($T: typeid) {
    left:  ^Matrix_Expr(T),
    right: ^Matrix_Expr(T),
}

// Operator overloading (conceptual - Odin doesn't support operator overloading)
// Instead, provide fluent API:
mul :: proc(a, b: Matrix_Expr($T)) -> Matrix_Expr(T) {
    // Return deferred multiplication
}

// Materialize result
eval :: proc(expr: Matrix_Expr($T), result: ^Matrix(T, .Row_Major)) -> Blas_Error {
    // Evaluate expression tree, potentially fusing operations
}
```

### 10.3 In-Place Operation Markers

```odin
// Marker type for in-place operations
In_Place :: struct($T: typeid) {
    mat: ^Matrix(T, .Row_Major),
}

in_place :: proc(m: ^Matrix($T, $L)) -> In_Place(T) {
    return In_Place(T){mat = m}
}

// Overloaded operation that detects in-place
add :: proc{add_out_of_place, add_in_place}

add_out_of_place :: proc(a, b: Matrix($T, $L)) -> Owned_Matrix(T, L) {
    // Allocate new matrix
}

add_in_place :: proc(a: Matrix($T, $L), b: In_Place(T)) {
    // Modify b.mat directly
}

// Usage:
add(a, b)              // Returns new matrix
add(a, in_place(&b))   // Modifies b
```

### 10.4 Dimension Query Helpers

```odin
// Common dimension queries
is_square :: proc(m: Matrix($T, $L)) -> bool {
    return m.rows == m.cols
}

is_vector :: proc(m: Matrix($T, $L)) -> bool {
    return m.rows == 1 || m.cols == 1
}

is_row_vector :: proc(m: Matrix($T, $L)) -> bool {
    return m.rows == 1
}

is_col_vector :: proc(m: Matrix($T, $L)) -> bool {
    return m.cols == 1
}

is_contiguous :: proc(m: Matrix($T, $L)) -> bool {
    when L == .Row_Major {
        return m.stride == m.cols
    } else {
        return m.stride == m.rows
    }
}

numel :: proc(m: Matrix($T, $L)) -> int {
    return m.rows * m.cols
}
```

### 10.5 Pretty Printing

```odin
// Debug printing
print_matrix :: proc(m: Matrix($T, $L), name := "", precision := 4) {
    if name != "" {
        fmt.printf("%s (%dx%d, %s):\n", name, m.rows, m.cols,
                   "row-major" when L == .Row_Major else "col-major")
    }

    for i in 0..<m.rows {
        fmt.print("[")
        for j in 0..<m.cols {
            fmt.printf(" %.*f", precision, at(m, i, j))
        }
        fmt.println(" ]")
    }
}
```

---

## 11. Migration Path

### 11.1 Compatibility Layer

```odin
// Wrap old API to use new types internally
dgemm_compat :: proc(trans_a, trans_b: Transpose,
                     m, n, k: int,
                     alpha: f64,
                     a: []f64, lda: int,
                     b: []f64, ldb: int,
                     beta: f64,
                     c: []f64, ldc: int) {
    // Convert to new types
    a_mat := from_slice(f64, .Row_Major, a, m, k, lda)
    b_mat := from_slice(f64, .Row_Major, b, k, n, ldb)
    c_mat := from_slice(f64, .Row_Major, c, m, n, ldc)

    // Use new implementation
    gemm(alpha, a_mat, trans_a, b_mat, trans_b, beta, &c_mat)
}
```

### 11.2 Deprecation Strategy

```odin
// Mark old API as deprecated (Odin supports this attribute)
@(deprecated = "Use gemm with Matrix types instead")
dgemm_old :: proc(...) { ... }
```

### 11.3 Package Organization

```
blas/
├── blas.odin           # Public API, type aliases, procedure groups
├── matrix.odin         # Matrix type definitions
├── level1.odin         # Vector operations (dot, axpy, nrm2, etc.)
├── level2.odin         # Matrix-vector operations (gemv, trsv, etc.)
├── level3.odin         # Matrix-matrix operations (gemm, trsm, etc.)
├── lapack/
│   ├── qr.odin         # QR factorization
│   ├── qr_update.odin  # QR update/downdate operations
│   └── householder.odin # Householder transformations
├── internal/
│   ├── kernels.odin    # Low-level computational kernels
│   └── parallel.odin   # Threading infrastructure
├── error.odin          # Error types and handling
└── compat.odin         # Compatibility with Reference BLAS API
```

---

## 12. Summary of Key Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Matrix type** | Struct with borrowed slice + dimensions + stride | Zero-copy views, explicit ownership |
| **Layout** | Compile-time enum parameter | Optimization + prevents accidental mixing |
| **Submatrices** | Zero-copy via stride preservation | Performance, matches BLAS semantics |
| **Polymorphism** | Procedure groups + parametric types | Odin-idiomatic, compile-time dispatch |
| **Error handling** | Result union type + unchecked variants | Safety when needed, speed when trusted |
| **Naming** | Traditional BLAS + readable aliases | Familiar to experts, accessible to newcomers |
| **Allocation** | Caller-managed + workspace helpers | Explicit control, no hidden allocations |
| **Context** | Optional, only for allocation | Core math context-free for performance |
| **Bounds checking** | Layered: API/validated/unchecked | Correctness at boundary, speed in kernel |
| **Threading** | Configurable, opt-in | Single-threaded by default, parallel when requested |

---

## 13. Open Questions

1. **Complex number support**: Implement `complex64`/`complex128` variants now or later?
2. **Integer matrices**: Support `i32`/`i64` for specialized applications?
3. **Sparse matrices**: Separate package or integrate with dense?
4. **GPU acceleration**: Design hooks for future GPU backends?
5. **Automatic differentiation**: Consider AD-friendly API patterns?

---

## Appendix A: Type Hierarchy Diagram

```
                    Owned_Matrix(T, L)
                          │
                          │ borrow()
                          ▼
    ┌──────────────► Matrix(T, L) ◄──────────────┐
    │                     │                       │
    │ from_slice()        │ submatrix()          │ transpose()
    │                     │                       │
    │                     ▼                       │
    │              Matrix(T, L)                   │
    │              (view)                         │
    │                     │                       │
    │                     │ row(), col()          │
    │                     ▼                       │
    │           ┌─────────┴─────────┐             │
    │           ▼                   ▼             │
    │        []T              Strided_Slice(T)    │
    │     (contiguous)         (non-contiguous)   │
    │                                             │
    └─────────────────────────────────────────────┘
```

## Appendix B: Example Usage

```odin
package example

import "blas"
import "core:fmt"

main :: proc() {
    // Create owned matrices
    a := blas.make_matrix(f64, .Row_Major, 3, 4)
    b := blas.make_matrix(f64, .Row_Major, 4, 2)
    c := blas.make_matrix(f64, .Row_Major, 3, 2)
    defer blas.delete_matrix(&a)
    defer blas.delete_matrix(&b)
    defer blas.delete_matrix(&c)

    // Initialize (example: identity-like pattern)
    for i in 0..<3 {
        for j in 0..<4 {
            blas.set(&a.mat, i, j, f64(i + j))
        }
    }

    // Matrix multiply: C = A * B
    if err := blas.gemm(1.0, a.mat, b.mat, 0.0, &c.mat); err != nil {
        fmt.println("Error:", err)
        return
    }

    // Extract submatrix view (zero-copy)
    sub := blas.submatrix(c.mat, 0, 0, 2, 2)

    // Print result
    blas.print_matrix(sub, "C[0:2, 0:2]")

    // QR factorization with automatic workspace
    q, r, tau := blas.qr_factor(&a.mat)
    defer blas.delete_matrix(&q)
    defer blas.delete_matrix(&r)
    defer delete(tau)

    fmt.println("QR factorization complete")
}
```
