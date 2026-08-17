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
// Random source: counter-based, stateless
// ============================================================================
//
// Every draw is a pure function of (seed, stream, index). There is no state
// carried between calls and nothing to advance, which buys three things:
//
//   1. Row i is generatable without generating rows 0..i-1, so generation
//      parallelises with no coordination and resumes at any offset.
//   2. The base variables and the response noise draw from SEPARATE streams,
//      so changing sigma cannot perturb X. Fitting the same design matrix at
//      several noise levels is then a real experiment. A sequential stream
//      cannot do this: the noise draw shifts everything after it.
//   3. The arithmetic is 32-bit integer only, which is what a GPU wants --
//      see docs/SYNTH.md for what a GLSL/HLSL port needs.
//
// The cost of statelessness is recomputation: a Box-Muller pair is derived
// from its pair index rather than cached, so nothing is remembered but nothing
// is wasted either, because both halves of each pair are consumed.

// Stream identifiers. Distinct streams are statistically independent, which
// test_synth_stream_independence checks rather than assumes.
STREAM_BASE :: u32(0) // the base variables
STREAM_NOISE :: u32(1) // the response noise

// splitmix32 finalizer, as used by rngeasy for seeding. Bijective, so it
// cannot map distinct inputs onto the same output.
@(private = "file")
splitmix32 :: #force_inline proc "contextless" (b: u32) -> u32 {
	x := b
	x += 0x9E3779B9
	x ~= x >> 15
	x *= 0x85EBCA6B
	x ~= x >> 13
	x *= 0xC2B2AE3D
	x ~= x >> 16
	return x
}

// synth_hash mixes (seed, stream, index) into one word.
//
// Two splitmix32 rounds over a Weyl-mixed input. Two rounds rather than one
// because a single round leaves visible structure between adjacent indices,
// and adjacent indices are exactly what this is used with.
synth_hash :: proc "contextless" (seed: u32, stream: u32, index: u32) -> u32 {
	h := seed
	h += index * 0x9E3779B9
	h ~= stream * 0x85EBCA6B
	return splitmix32(splitmix32(h))
}

// synth_uniform53 returns a uniform in (0,1) with 53 significant bits, built
// from two hashes.
//
// The extra hash buys tail depth. A 32-bit uniform bottoms out at 1.2e-10,
// which truncates a Box-Muller normal at about 6.2 sigma; 53 bits reaches
// about 8.5 sigma, past where f64 sampling has any practical meaning.
@(private = "file")
uniform53 :: #force_inline proc "contextless" (seed, stream, index: u32) -> f64 {
	hi := u64(synth_hash(seed, stream, index * 2))
	lo := u64(synth_hash(seed, stream, index * 2 + 1))
	// 53 bits: 32 high, 21 low.
	bits := (hi << 21) | (lo >> 11)
	return (f64(bits) + 0.5) * (1.0 / 9007199254740992.0)
}

// synth_uniform32 returns a uniform in (0,1) with 32 significant bits. Used
// for the Box-Muller angle, where resolution is irrelevant.
@(private = "file")
uniform32 :: #force_inline proc "contextless" (seed, stream, index: u32) -> f64 {
	return (f64(synth_hash(seed, stream, index)) + 0.5) * (1.0 / 4294967296.0)
}

// synth_normal_pair returns two independent standard normal deviates for the
// given pair index. Box-Muller, exact rather than an approximation to the
// inverse CDF, and both halves are returned so nothing is discarded.
//
// Three hashes per pair: two for the magnitude uniform (53 bits, for tail
// depth) and one for the angle.
synth_normal_pair :: proc "contextless" (
	seed: u32,
	stream: u32,
	pair_index: u32,
) -> (
	f64,
	f64,
) {
	// Offset the two sub-streams so the angle hash cannot collide with either
	// magnitude hash.
	u1 := uniform53(seed, stream, pair_index * 2)
	u2 := uniform32(seed, stream ~ 0x5BF03635, pair_index)
	mag := math.sqrt_f64(-2.0 * math.ln_f64(u1))
	ang := 2.0 * math.PI * u2
	return mag * math.cos_f64(ang), mag * math.sin_f64(ang)
}

// synth_normal returns one standard normal deviate at an absolute draw index.
//
// Index-addressable: draw i is derived from pair i/2 and selected by parity, so
// any single draw is reachable without touching the others. Consecutive
// even/odd pairs share a pair computation, so a loop over indices in order
// pays for each Box-Muller pair once if the compiler keeps it, and twice if
// not -- correctness does not depend on which.
synth_normal :: proc "contextless" (seed: u32, stream: u32, index: u32) -> f64 {
	a, b := synth_normal_pair(seed, stream, index / 2)
	return index % 2 == 0 ? a : b
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
//   spec      in   initialised by synth_init. Its per-row scratch is reused, so
//                  one Spec cannot be driven from two threads at once; give each
//                  thread its own Spec (they can share the same seed).
//   seed      in   any u32. Together with first_row it fully determines the
//                  output, so the same (seed, first_row) always gives the same
//                  rows, on any machine, in any order.
//   first_row in   ABSOLUTE row index of the first row of this batch, >= 0.
//                  Row i depends only on (seed, i), so batches may be generated
//                  in any order, in parallel, or not at all.
//   x         out  f64[>= (count-1)*ldx + nterms], row-major, row stride
//                  ldx >= nterms. Row i column t is the value of term t.
//   y         out  f64[>= count], the response.
//   count     in   rows to generate, >= 0. 0 is a no-op.
//
// Because the base variables and the noise use separate streams, changing
// sigma changes only y and leaves X untouched. Fitting one design matrix at
// several noise levels is therefore a controlled experiment, which a shared
// sequential stream cannot offer.
//
// Chunkable and reorderable: the concatenation of any partition of [0, m) is
// identical to one call over the whole range, because nothing is carried
// between calls.
//
// Cost per row: k normal draws (3 hashes per pair of draws), k^2/2 for the
// correlation transform, and one pass over the terms costing their total
// degree. O(1) memory.
synth_rows :: proc "contextless" (
	spec: ^Spec,
	seed: u32,
	first_row: int,
	x: []f64,
	ldx: int,
	y: []f64,
	count: int,
) -> Synth_Error {
	k, nterms := spec.k, spec.nterms
	if k < 1 || nterms < 1 {
		return .Invalid_Dimension
	}
	if count < 0 || first_row < 0 || ldx < nterms {
		return .Invalid_Dimension
	}
	if count == 0 {
		return .None
	}
	if len(x) < (count - 1) * ldx + nterms || len(y) < count {
		return .Invalid_Dimension
	}

	// Draw indices are GLOBAL: base draw j of row r is index r*k + j, and the
	// noise draw for row r is index r. Both are pure functions of the row, so
	// addressability is unaffected -- but because consecutive rows land in
	// consecutive pairs, a pair computed for one draw usually serves the next
	// one too. The two caches below are local to this call and hold only what
	// was already derivable, so they change cost and not results.
	if u64(first_row + count) * u64(k) > 0xFFFFFFFF {
		// Beyond this the global draw index wraps and rows would alias each
		// other's draws. Rejected rather than silently repeating data.
		return .Invalid_Dimension
	}

	NO_PAIR :: ~u64(0)
	bp_idx := NO_PAIR // cached base pair index
	bp_a, bp_b := 0.0, 0.0
	np_idx := NO_PAIR // cached noise pair index
	np_a, np_b := 0.0, 0.0

	for i in 0 ..< count {
		abs_row := u64(first_row + i)

		// Base variables, drawn from a globally indexed stream so that pairs
		// straddle row boundaries and nothing is wasted on odd k.
		for j in 0 ..< k {
			gi := abs_row * u64(k) + u64(j)
			pi := gi >> 1
			if pi != bp_idx {
				bp_a, bp_b = synth_normal_pair(seed, STREAM_BASE, u32(pi))
				bp_idx = pi
			}
			spec.z[j] = gi & 1 == 0 ? bp_a : bp_b
		}

		// base = L * z, giving covariance L*L' = D*corr*D.
		for a in 0 ..< k {
			sum := 0.0
			for b in 0 ..= a {
				sum += spec.chol[a * k + b] * spec.z[b]
			}
			spec.base[a] = sum
		}

		// Expand the terms. An all-zero exponent row leaves the product at 1,
		// which is the intercept, so no special case is needed.
		row := i * ldx
		resp := 0.0
		for t in 0 ..< nterms {
			te := t * k
			v := 1.0
			for jj in 0 ..< k {
				e := spec.terms[te + jj]
				for _ in 0 ..< e {
					v *= spec.base[jj]
				}
			}
			x[row + t] = v
			resp += spec.coef[t] * v
		}
		if spec.sigma != 0.0 {
			// A separate stream, indexed by row: this is what keeps X
			// independent of sigma. Paired across consecutive rows, same as above.
			pi := abs_row >> 1
			if pi != np_idx {
				np_a, np_b = synth_normal_pair(seed, STREAM_NOISE, u32(pi))
				np_idx = pi
			}
			resp += spec.sigma * (abs_row & 1 == 0 ? np_a : np_b)
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
