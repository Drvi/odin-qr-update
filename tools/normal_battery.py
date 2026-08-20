#!/usr/bin/env python
"""Goodness-of-fit battery for the normal generators.

Three samples are tested, and the third is the point:

  VEC     the vectorised Box-Muller under evaluation
  SCALAR  the scalar Box-Muller already in the library -- the control for
          "no worse than what we have"
  BROKEN  sum of 12 uniforms minus 6.  Mean 0 and variance 1 exactly, so it
          sails through any moment check that stops at the second moment, but
          its excess kurtosis is -0.1 and its support is clipped at +/-6.
          This is the control for "the battery has power".  A battery that
          accepts all three is not measuring anything.

Every statistic is reported for all three side by side, so a reader can see
which tests separate them and which do not.
"""
import sys
import numpy as np
from scipy import stats

N_BLOCK = 100_000          # block size for the p-value-uniformity tests
ALPHA = 0.001              # per-test alarm level, deliberately strict


def load(path):
    return np.fromfile(path, dtype=np.float64)


def banner(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def row(label, vals, fmt="%12.5f"):
    print(("  %-34s" + fmt * len(vals)) % ((label,) + tuple(vals)))


def moments(samples):
    """Moments with z-scores against their sampling distributions under H0."""
    banner("1. MOMENTS  (z = deviation in standard errors; |z| > 3 is a flag)")
    n = len(next(iter(samples.values())))
    print(f"  n = {n:,} per sample")
    print()
    print(("  %-34s" + "%12s" * len(samples)) % ("", *samples.keys()))

    stat_rows = []
    for name, f, se, target in [
        ("mean",             lambda x: x.mean(),                 1/np.sqrt(N := n), 0.0),
        ("variance",         lambda x: x.var(ddof=1),            np.sqrt(2/n),      1.0),
        ("skewness",         lambda x: stats.skew(x),            np.sqrt(6/n),      0.0),
        ("excess kurtosis",  lambda x: stats.kurtosis(x),        np.sqrt(24/n),     0.0),
    ]:
        vals = {k: f(v) for k, v in samples.items()}
        row(name, list(vals.values()), "%12.6f")
        stat_rows.append((name, vals, se, target))

    print()
    print("  z-scores:")
    flags = {k: 0 for k in samples}
    for name, vals, se, target in stat_rows:
        zs = {k: (v - target) / se for k, v in vals.items()}
        row("z(" + name + ")", list(zs.values()), "%12.2f")
        for k, z in zs.items():
            if abs(z) > 3:
                flags[k] += 1
    return flags


def gof_blocked(samples):
    """KS, Cramer-von Mises and Anderson-Darling, run per block.

    One test over 5,000,000 points is so powerful that it rejects on
    representation-level discreteness alone, which tells you nothing useful.
    Running the test on independent blocks and then asking whether the p-values
    are themselves Uniform(0,1) is the honest version: it separates "this
    sample is not normal" from "this sample is enormous".
    """
    banner("2. GOODNESS OF FIT, blocked  (p-values should be Uniform(0,1))")
    print(f"  block size {N_BLOCK:,}; testing against the FULLY SPECIFIED N(0,1)")
    print()
    out = {}
    for name, x in samples.items():
        nb = len(x) // N_BLOCK
        ks_p, cvm_p, ad_stat = [], [], []
        for b in range(nb):
            blk = x[b * N_BLOCK:(b + 1) * N_BLOCK]
            ks_p.append(stats.kstest(blk, "norm").pvalue)
            cvm_p.append(stats.cramervonmises(blk, "norm").pvalue)
            # Anderson-Darling A^2 against fully specified N(0,1)
            z = np.sort(blk)
            u = stats.norm.cdf(z)
            u = np.clip(u, 1e-300, 1 - 1e-16)
            i = np.arange(1, len(u) + 1)
            a2 = -len(u) - np.mean((2 * i - 1) * (np.log(u) + np.log(1 - u[::-1])))
            ad_stat.append(a2)
        ks_p = np.array(ks_p); cvm_p = np.array(cvm_p); ad_stat = np.array(ad_stat)
        # Are the p-values uniform?
        u_ks = stats.kstest(ks_p, "uniform").pvalue
        u_cvm = stats.kstest(cvm_p, "uniform").pvalue
        out[name] = dict(
            nb=nb,
            ks_rej=int((ks_p < ALPHA).sum()), ks_unif=u_ks,
            cvm_rej=int((cvm_p < ALPHA).sum()), cvm_unif=u_cvm,
            ad_med=float(np.median(ad_stat)), ad_max=float(ad_stat.max()),
            ad_rej=int((ad_stat > 3.878).sum()),   # 1% point for fully specified A^2
        )
    print(("  %-34s" + "%12s" * len(samples)) % ("", *samples.keys()))
    row("blocks tested",            [out[k]["nb"] for k in samples], "%12d")
    row(f"KS  blocks rejected @{ALPHA}",  [out[k]["ks_rej"] for k in samples], "%12d")
    row("KS  p-value uniformity p",  [out[k]["ks_unif"] for k in samples], "%12.3e")
    row(f"CvM blocks rejected @{ALPHA}", [out[k]["cvm_rej"] for k in samples], "%12d")
    row("CvM p-value uniformity p", [out[k]["cvm_unif"] for k in samples], "%12.3e")
    row("A^2 median (H0 median~0.78)", [out[k]["ad_med"] for k in samples], "%12.4f")
    row("A^2 max",                  [out[k]["ad_max"] for k in samples], "%12.4f")
    row("A^2 blocks over 1% point", [out[k]["ad_rej"] for k in samples], "%12d")
    return out


def tails(samples):
    """Exceedance counts against exact normal probabilities.

    This is where an approximate transform or a truncated uniform shows up, and
    where KS is weakest -- KS is driven by the middle of the distribution.
    """
    banner("3. TAILS  (observed vs expected exceedances, z from the binomial)")
    n = len(next(iter(samples.values())))
    print(("  %-34s" + "%12s" * len(samples)) % ("threshold", *samples.keys()))
    for t in [1, 2, 3, 4, 5, 5.5, 6]:
        p = 2 * stats.norm.sf(t)
        exp = n * p
        se = np.sqrt(n * p * (1 - p))
        zs = []
        for k, x in samples.items():
            obs = int((np.abs(x) > t).sum())
            zs.append((obs - exp) / se if se > 0 else 0.0)
        row(f"|z| > {t}   (expect {exp:,.0f})", zs, "%12.2f")
    print()
    print("  extreme order statistic:")
    row("max |z| observed", [float(np.abs(x).max()) for x in samples.values()], "%12.4f")
    print("    (for n = %d the expected max |z| is about %.2f)"
          % (n, stats.norm.isf(0.5 / n)))


def independence(samples):
    """Serial structure, and the Box-Muller pair correlation specifically.

    The two halves of a Box-Muller pair are cos and sin of the same angle. If
    the angle were badly distributed, or the two hash sub-streams collided, the
    halves would correlate -- and every downstream fit would inherit it while
    the marginals stayed perfect.
    """
    banner("4. INDEPENDENCE")
    print(("  %-34s" + "%12s" * len(samples)) % ("", *samples.keys()))
    for lag in [1, 2, 3, 7, 16, 64]:
        vals = []
        for x in samples.values():
            a, b = x[:-lag], x[lag:]
            vals.append(float(np.corrcoef(a, b)[0, 1]))
        row(f"autocorr lag {lag}", vals, "%12.5f")
    n = len(next(iter(samples.values())))
    print(f"    (standard error of each correlation is {1/np.sqrt(n):.5f})")
    print()
    vals = []
    for x in samples.values():
        even, odd = x[0::2], x[1::2]
        vals.append(float(np.corrcoef(even, odd)[0, 1]))
    row("corr(pair half 0, half 1)", vals, "%12.5f")
    print()
    # Bivariate check: (even, odd) should be a 2-D standard normal, so the
    # radius^2 should be Exponential(1/2) i.e. chi-square with 2 df.
    print("  radius test: for a true B-M pair, x^2+y^2 ~ chi2(2)")
    vals = []
    for x in samples.values():
        r2 = x[0::2] ** 2 + x[1::2] ** 2
        # KS against chi2(2) on a subsample, since this is very powerful
        vals.append(stats.kstest(r2[:200_000], "chi2", args=(2,)).pvalue)
    row("KS(radius^2, chi2(2)) p", vals, "%12.3e")


def chisq(samples):
    banner("5. CHI-SQUARE on 256 equiprobable bins")
    n = len(next(iter(samples.values())))
    edges = stats.norm.ppf(np.linspace(0, 1, 257))
    edges[0], edges[-1] = -np.inf, np.inf
    exp = n / 256
    print(("  %-34s" + "%12s" * len(samples)) % ("", *samples.keys()))
    stat, pv = [], []
    for x in samples.values():
        obs, _ = np.histogram(x, bins=edges)
        c2 = ((obs - exp) ** 2 / exp).sum()
        stat.append(c2)
        pv.append(stats.chi2.sf(c2, 255))
    row("chi-square (255 df)", stat, "%12.2f")
    row("p-value", pv, "%12.3e")


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "."
    import os
    want = [("VEC", "VBM_VEC.bin"), ("SCALAR", "VBM_SCALAR.bin"),
            ("ZIG", "VBM_ZIG.bin"), ("BROKEN", "VBM_BROKEN.bin")]
    samples = {k: load(f"{base}/{f}") for k, f in want
               if os.path.exists(f"{base}/{f}")}
    for k, v in samples.items():
        assert v.size > 0, k
    print("Normal-generator goodness-of-fit battery")
    print("VEC and SCALAR should be indistinguishable from N(0,1).")
    print("BROKEN is a negative control and MUST be rejected, or the battery")
    print("is not testing anything.")

    moments(samples)
    gof_blocked(samples)
    tails(samples)
    independence(samples)
    chisq(samples)
    print()


if __name__ == "__main__":
    main()
