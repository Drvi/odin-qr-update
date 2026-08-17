package synth

// ============================================================================
// Synthetic regression data with known ground truth.
//
// Generates (X, y) where you chose the answer in advance, so a wrong fit is
// unambiguously a bug in the fitting and not a modelling mistake.
//
// What you control:
//
//   correlation matrix of the base variables   corr, k x k
//   variance of each base variable             variance, k
//   the model terms                            terms, as exponent vectors
//   the true coefficients                      coef, one per term
//   Gaussian noise on the response             sigma
//
// A TERM is a vector of exponents over the base variables, and its value in a
// row is the product of each base variable raised to its exponent. That one
// representation covers everything:
//
//   {0,0,0}   1              intercept
//   {1,0,0}   b0             linear
//   {0,2,0}   b1^2           quadratic
//   {1,0,1}   b0*b2          interaction
//   {3,0,0}   b0^3           cubic
//
// so polynomial regression needs no separate mechanism -- it is ordinary OLS
// over an expanded set of terms, which is exactly how a caller would use it.
//
// The base variables are drawn N(0, Sigma) with Sigma = D*corr*D, D =
// diag(sqrt(variance)), via a Cholesky factor computed once in synth_init.
// Note that terms of degree > 1 are NOT Gaussian and their pairwise
// correlations are NOT `corr` -- corr describes the base variables, and the
// design matrix holds functions of them.
//
// Allocates nothing: all memory is caller-owned and sized by synth_scratch.
// Every procedure is contextless.
// ============================================================================

import "core:math"
import blas "../blas"

Synth_Error :: enum {
	None = 0,
	Invalid_Dimension, // k < 1, nterms < 1, bad stride, or a short slice
	Scratch_Too_Small, // caller buffer smaller than synth_scratch(k)
	Invalid_Correlation, // not symmetric, diagonal not 1, or |corr| > 1
	Invalid_Variance, // variance not strictly positive
	Not_Positive_Definite, // the correlation matrix is not a consistent one
	Invalid_Sigma, // negative noise level
}

// Tolerance for the symmetry and unit-diagonal checks on `corr`. Loose enough
// that hand-written matrices pass, tight enough to catch a transposed entry.
CORR_TOL :: 1.0e-9

// ============================================================================
// Random source
// ============================================================================

// Rng is xoshiro256** with a cached spare normal deviate. Held by the caller so
// the stream is explicit and reproducible: the same seed always gives the same
// data, which is what makes a failing test re-runnable.
Rng :: struct {
	s:          [4]u64,
	spare:      f64,
	has_spare:  bool,
}

// rng_seed initialises the stream. Any seed is valid, including 0.
rng_seed :: proc "contextless" (r: ^Rng, seed: u64) {
	// SplitMix64 to spread one word over the whole state; xoshiro behaves badly
	// from a nearly-zero state.
	z := seed
	for i in 0 ..< 4 {
		z += 0x9E3779B97F4A7C15
		v := z
		v = (v ~ (v >> 30)) * 0xBF58476D1CE4E5B9
		v = (v ~ (v >> 27)) * 0x94D049BB133111EB
		r.s[i] = v ~ (v >> 31)
	}
	r.has_spare = false
	r.spare = 0
}

@(private = "file")
rotl :: #force_inline proc "contextless" (x: u64, k: uint) -> u64 {
	return (x << k) | (x >> (64 - k))
}

// rng_u64 returns the next raw word.
rng_u64 :: proc "contextless" (r: ^Rng) -> u64 {
	result := rotl(r.s[1] * 5, 7) * 9
	t := r.s[1] << 17
	r.s[2] ~= r.s[0]
	r.s[3] ~= r.s[1]
	r.s[1] ~= r.s[2]
	r.s[0] ~= r.s[3]
	r.s[2] ~= t
	r.s[3] = rotl(r.s[3], 45)
	return result
}

// rng_open01 returns a uniform in (0, 1): never 0, so log() is always safe.
rng_open01 :: proc "contextless" (r: ^Rng) -> f64 {
	// 53 significand bits, shifted into [1, 2) then offset, giving (0,1).
	u := rng_u64(r) >> 11
	return (f64(u) + 0.5) * (1.0 / 9007199254740992.0)
}

// rng_normal returns a standard normal deviate by Box-Muller, generating two at
// a time and keeping the spare.
rng_normal :: proc "contextless" (r: ^Rng) -> f64 {
	if r.has_spare {
		r.has_spare = false
		return r.spare
	}
	u1 := rng_open01(r)
	u2 := rng_open01(r)
	mag := math.sqrt_f64(-2.0 * math.ln_f64(u1))
	ang := 2.0 * math.PI * u2
	r.spare = mag * math.sin_f64(ang)
	r.has_spare = true
	return mag * math.cos_f64(ang)
}

// ============================================================================
// Spec
// ============================================================================

// Spec is the precomputed generator. Flat struct; the slices are views into
// the caller's scratch block and into the caller's own term and coefficient
// arrays, none of which are copied.
//
// `chol` holds the lower Cholesky factor of D*corr*D and is computed once by
// synth_init, so generating rows costs no factorization.
Spec :: struct {
	k:      int, // base variables
	nterms: int, // design columns produced
	sigma:  f64, // Gaussian noise standard deviation on y
	chol:   []f64, // k*k, lower triangular, row stride k
	terms:  []u8, // nterms*k exponents, row-major (borrowed)
	coef:   []f64, // nterms true coefficients (borrowed)
	z:      []f64, // k, scratch for one draw
	base:   []f64, // k, scratch for one row's base variables
}

// Scratch required by synth_init, in f64 elements.
synth_scratch :: proc "contextless" (k: int) -> int {
	return k * k + 2 * k
}

// synth_init validates the spec and precomputes the Cholesky factor.
//
// CONTRACT
//   spec     out  filled in; borrows `terms`, `coef` and `scratch`, all of
//                 which must outlive it and must not be modified afterwards.
//   k        in   number of base variables, >= 1.
//   corr     in   f64[>= k*k], row-major k x k correlation matrix. Must be
//                 symmetric to CORR_TOL, have unit diagonal, entries in
//                 [-1, 1], and be positive definite. Pass the identity for
//                 independent variables. NOT modified.
//   variance in   f64[>= k], each strictly positive. Variance of each base
//                 variable; combined with corr to give the covariance.
//   terms    in   u8[>= nterms*k] exponents, row-major. Row t column j is the
//                 exponent of base variable j in term t. An all-zero row is
//                 the intercept.
//   coef     in   f64[>= nterms], the true coefficient of each term.
//   sigma    in   standard deviation of the Gaussian noise added to y, >= 0.
//                 Pass 0 for an exact noiseless fit.
//   scratch  in   f64[>= synth_scratch(k)], caller-owned, overwritten.
//
// Boundary policy: every condition above is checked and reported as a distinct
// error, with no partial state left behind. An inconsistent correlation matrix
// -- easy to write by hand, for instance three variables mutually correlated
// +0.9, +0.9, -0.9 -- fails as .Not_Positive_Definite rather than silently
// producing data with some other correlation.
synth_init :: proc "contextless" (
	spec: ^Spec,
	k: int,
	corr: []f64,
	variance: []f64,
	terms: []u8,
	coef: []f64,
	sigma: f64,
	scratch: []f64,
) -> Synth_Error {
	if k < 1 {
		return .Invalid_Dimension
	}
	if len(corr) < k * k || len(variance) < k {
		return .Invalid_Dimension
	}
	if len(terms) < k || len(terms) % k != 0 {
		return .Invalid_Dimension
	}
	nterms := len(terms) / k
	if nterms < 1 || len(coef) < nterms {
		return .Invalid_Dimension
	}
	if len(scratch) < synth_scratch(k) {
		return .Scratch_Too_Small
	}
	if !(sigma >= 0.0) {
		return .Invalid_Sigma
	}

	for j in 0 ..< k {
		v := variance[j]
		if !(v > 0.0) {
			// Zero variance means a constant column; that is what an
			// all-zero term row is for, and it would make Sigma singular.
			return .Invalid_Variance
		}
	}
	for i in 0 ..< k {
		if abs(corr[i * k + i] - 1.0) > CORR_TOL {
			return .Invalid_Correlation
		}
		for j in 0 ..< k {
			c := corr[i * k + j]
			if !(c >= -1.0 && c <= 1.0) {
				return .Invalid_Correlation
			}
			if abs(c - corr[j * k + i]) > CORR_TOL {
				return .Invalid_Correlation
			}
		}
	}

	chol := scratch[0:k * k]
	spec.z = scratch[k * k:k * k + k]
	spec.base = scratch[k * k + k:k * k + 2 * k]

	// Sigma = D * corr * D with D = diag(sqrt(variance)).
	for i in 0 ..< k {
		si := math.sqrt_f64(variance[i])
		for j in 0 ..< k {
			chol[i * k + j] = si * corr[i * k + j] * math.sqrt_f64(variance[j])
		}
	}
	if info := blas.dpotrf(.Lower, k, chol, k); info != 0 {
		return .Not_Positive_Definite
	}

	spec.k = k
	spec.nterms = nterms
	spec.sigma = sigma
	spec.chol = chol
	spec.terms = terms
	spec.coef = coef
	return .None
}

// synth_term_count returns how many design columns a spec produces.
synth_term_count :: proc "contextless" (spec: ^Spec) -> int {
	return spec.nterms
}

// ============================================================================
// Generation
// ============================================================================

// synth_rows generates a batch of observations.
//
// BATCH CONTRACT
//   spec  in     initialised by synth_init. Its scratch is reused per row, so
//                one spec cannot be driven from two threads at once; give each
//                thread its own spec and its own Rng.
//   rng   in/out the random stream; advanced. Seed it with rng_seed.
//   x     out    f64[>= (count-1)*ldx + nterms], row-major, row stride
//                ldx >= nterms. Row i column t is the value of term t.
//   ldx   in     row stride, so a caller can generate into a wider table and
//                leave spare columns for terms added later.
//   y     out    f64[>= count], the response.
//   count in     rows to generate, >= 0. 0 is a no-op.
//
// Chunkable: call repeatedly with advancing slices to spread generation over
// frames, or to stream rows straight into an Ols_Accum without ever holding
// them all. The stream continues from wherever the Rng left off, so the
// concatenation of chunked calls is identical to one big call.
//
// Cost per row: k normal draws, k^2/2 for the correlation transform, and one
// pass over the terms whose cost is the total degree. All O(1) memory.
synth_rows :: proc "contextless" (
	spec: ^Spec,
	rng: ^Rng,
	x: []f64,
	ldx: int,
	y: []f64,
	count: int,
) -> Synth_Error {
	k, nterms := spec.k, spec.nterms
	if k < 1 || nterms < 1 {
		return .Invalid_Dimension
	}
	if count < 0 || ldx < nterms {
		return .Invalid_Dimension
	}
	if count == 0 {
		return .None
	}
	if len(x) < (count - 1) * ldx + nterms || len(y) < count {
		return .Invalid_Dimension
	}

	for i in 0 ..< count {
		// Independent standard normals.
		for j in 0 ..< k {
			spec.z[j] = rng_normal(rng)
		}
		// base = L * z, giving covariance L*L' = D*corr*D.
		for a in 0 ..< k {
			s := 0.0
			for b in 0 ..= a {
				s += spec.chol[a * k + b] * spec.z[b]
			}
			spec.base[a] = s
		}

		// Expand the terms. An all-zero exponent row leaves the product at 1,
		// which is the intercept, so no special case is needed.
		row := i * ldx
		resp := 0.0
		for t in 0 ..< nterms {
			te := t * k
			v := 1.0
			for j in 0 ..< k {
				e := spec.terms[te + j]
				for _ in 0 ..< e {
					v *= spec.base[j]
				}
			}
			x[row + t] = v
			resp += spec.coef[t] * v
		}
		if spec.sigma != 0.0 {
			resp += spec.sigma * rng_normal(rng)
		}
		y[i] = resp
	}
	return .None
}

// ============================================================================
// Term-list helpers
// ============================================================================

// synth_terms_poly_count returns how many rows synth_terms_poly will write:
// the intercept plus every power from 1 to degree of every base variable.
// Interactions are not included -- write those rows yourself, since which
// interactions matter is a modelling choice.
synth_terms_poly_count :: proc "contextless" (k: int, degree: int) -> int {
	if k < 1 || degree < 0 {
		return 0
	}
	return 1 + k * degree
}

// synth_terms_poly fills a term list for a full univariate polynomial basis:
// the intercept, then b0, b0^2, ... b0^degree, then b1, b1^2, and so on.
//
//   out  f64-free u8[>= synth_terms_poly_count(k, degree) * k], overwritten.
//
// This is a convenience for the common "is it linear or quadratic in each
// variable" question. Anything else -- selected interactions, mixed degrees --
// is a hand-written exponent table, which is the general form.
synth_terms_poly :: proc "contextless" (out: []u8, k: int, degree: int) -> Synth_Error {
	if k < 1 || degree < 0 {
		return .Invalid_Dimension
	}
	nt := synth_terms_poly_count(k, degree)
	if len(out) < nt * k {
		return .Invalid_Dimension
	}
	for i in 0 ..< nt * k {
		out[i] = 0
	}
	// Row 0 is all zeros: the intercept.
	r := 1
	for j in 0 ..< k {
		for d in 1 ..= degree {
			out[r * k + j] = u8(d)
			r += 1
		}
	}
	return .None
}
