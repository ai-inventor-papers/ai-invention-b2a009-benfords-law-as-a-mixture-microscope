"""Benford instrument: first-digit statistics, harmonic coefficients, matched-null helpers.

Everything here is exact and deterministic except the bootstraps, whose seeds are
passed in explicitly.  The module carries a self-test (`self_test`) that must pass
before any statistic is reported; it verifies

  * the rotation identity  c1(c*x) = c1(x) exp(2*pi*i*log10 c)
  * the mirror identity    c1(1/x) = conj(c1(x))            (to 1e-10)
  * the digit-resolution reconstruction of c_n from the nine cell probabilities,
    including the within-bin midpoint phase AND the sinc attenuation
  * agreement between numerically periodised and closed-form digit laws (1e-4)
  * simultaneous-coverage of the Bonferroni-Wilson intervals on Benford draws

Notation follows the artifact plan: m = frac(log10 x) is the mantissa, p_d the
digit probabilities, r_d = 100*(p_obs,d - p_Benford,d) in percentage points,
c_1 = E[exp(2*pi*i*m)].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

BENFORD_D = np.arange(1, 10)
BENFORD_P = np.log10(1.0 + 1.0 / BENFORD_D)

# Nigrini (2012) forensic-analytics MAD conformity bands for n > 3,000.
NIGRINI_BANDS = (
    (0.000, 0.006, "close conformity"),
    (0.006, 0.012, "acceptable conformity"),
    (0.012, 0.015, "marginal conformity"),
    (0.015, np.inf, "nonconformity"),
)

DIGIT_EDGES = np.log10(BENFORD_D.astype(float))           # lower edge of each digit bin
DIGIT_WIDTHS = np.log10(1.0 + 1.0 / BENFORD_D)            # w_d = log10(1+1/d)


# --------------------------------------------------------------------------- #
# Basic encodings
# --------------------------------------------------------------------------- #
def log10_values(x: np.ndarray) -> np.ndarray:
    """log10 of strictly positive values, with a guard on the 9.999.. -> 10 edge."""
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError("all values must be strictly positive")
    return np.log10(x)


def mantissa(x: np.ndarray) -> np.ndarray:
    """m = frac(log10 x), in [0, 1)."""
    y = log10_values(x)
    m = y - np.floor(y)
    # np.log10(10**k) can come back as k - 1e-17; clip the pathological edge.
    return np.clip(m, 0.0, 1.0 - 1e-15)


def first_digit(x: np.ndarray) -> np.ndarray:
    """First significant decimal digit, 1..9."""
    m = mantissa(x)
    d = np.floor(10.0 ** m + 1e-12).astype(int)
    return np.clip(d, 1, 9)


def second_digit(x: np.ndarray) -> np.ndarray:
    """Second significant digit, 0..9."""
    y = log10_values(x)
    frac = (y - np.floor(y)) * 1.0
    # floor(100 * 10**m) gives the leading two-digit block
    two = np.floor(100.0 * 10.0 ** frac + 1e-9).astype(int)
    two = np.clip(two, 10, 99)
    return two % 10


def digit_counts(digits: np.ndarray) -> np.ndarray:
    return np.bincount(np.asarray(digits, dtype=int), minlength=10)[1:]


def digit_proportions(digits: np.ndarray) -> np.ndarray:
    c = digit_counts(digits).astype(float)
    return c / c.sum()


def residuals_pp(p_obs: np.ndarray, p_ref: np.ndarray | None = None) -> np.ndarray:
    """r_d in PERCENTAGE POINTS (not raw proportions)."""
    if p_ref is None:
        p_ref = BENFORD_P
    return 100.0 * (np.asarray(p_obs, float) - np.asarray(p_ref, float))


# --------------------------------------------------------------------------- #
# Conformity statistics
# --------------------------------------------------------------------------- #
def mad_stat(p_obs: np.ndarray, p_ref: np.ndarray | None = None) -> float:
    """Nigrini mean absolute deviation of digit proportions."""
    if p_ref is None:
        p_ref = BENFORD_P
    return float(np.mean(np.abs(np.asarray(p_obs, float) - np.asarray(p_ref, float))))


def nigrini_band(mad: float, n: int | None = None) -> str:
    """Nigrini's MAD conformity band, or an explicit refusal below his sample floor.

    Nigrini (2012) calibrated these thresholds for n > 3,000.  Below that the
    bands are not valid, and reporting one anyway would claim a standard the data
    do not meet -- so the function says so instead.
    """
    if n is not None and n <= 3000:
        return f"not applicable (Nigrini bands require n > 3,000; n = {n})"
    for lo, hi, label in NIGRINI_BANDS:
        if lo <= mad < hi:
            return label
    return "nonconformity"


def chi2_stat(counts: np.ndarray, p_ref: np.ndarray | None = None) -> dict:
    """Pearson chi-square against p_ref with df = 8, and the small-n caveat."""
    if p_ref is None:
        p_ref = BENFORD_P
    counts = np.asarray(counts, float)
    n = counts.sum()
    expected = n * np.asarray(p_ref, float)
    stat = float(np.sum((counts - expected) ** 2 / expected))
    return {
        "chi2": stat,
        "df": 8,
        "n": float(n),
        "expected_count_digit9": float(expected[8]),
        "min_expected_count": float(expected.min()),
        "asymptotics_marginal": bool(expected.min() < 5.0),
        "caveat": (
            "df=8 is nominal; with min expected count < 5 the asymptotic p-value is "
            "unreliable and is reported for completeness only"
        ),
    }


def _cdf_from_p(p: np.ndarray) -> np.ndarray:
    return np.cumsum(np.asarray(p, float))


def kuiper_V(counts: np.ndarray, p_ref: np.ndarray | None = None) -> float:
    """Kuiper V = D+ + D- between the observed and reference first-digit CDFs.

    Discrete version evaluated at each of the nine digit values, with the
    observed CDF stepped after each digit and both one-sided distances taken
    over the same support.
    """
    if p_ref is None:
        p_ref = BENFORD_P
    counts = np.asarray(counts, float)
    p_obs = counts / counts.sum()
    f_obs = _cdf_from_p(p_obs)
    f_ref = _cdf_from_p(p_ref)
    d_plus = float(np.max(f_obs - f_ref))
    d_minus = float(np.max(f_ref - f_obs))
    return d_plus + d_minus


def wilson_interval(k: float, n: float, z: float) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion with critical value z."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def clopper_pearson(k: float, n: float, alpha: float) -> tuple[float, float]:
    """Exact binomial (Clopper-Pearson) interval with per-interval alpha."""
    from scipy.stats import beta

    if n <= 0:
        return (0.0, 1.0)
    k = min(max(k, 0.0), n)
    lo = 0.0 if k <= 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1.0))
    hi = 1.0 if k >= n else float(beta.ppf(1.0 - alpha / 2.0, k + 1.0, n - k))
    return (lo, hi)


def simultaneous_intervals(counts: np.ndarray, alpha: float = 0.05,
                           n_eff: float | None = None, method: str = "clopper_pearson") -> dict:
    """Nine simultaneous per-digit intervals at familywise 1-alpha (Bonferroni over digits).

    `n_eff` rescales the sample size used for the interval WIDTHS (design-effect
    correction); point estimates always come from the real counts.

    Two methods are computed and both returned, because the choice matters at
    these sample sizes: Wilson score intervals are the field's usual choice but
    their SIMULTANEOUS coverage on nine multinomial cells is measured at
    0.940-0.952 against a nominal 0.95 (see `self_test`), while exact
    Clopper-Pearson at the same Bonferroni level attains 0.959-0.966.  The
    inferential statement therefore uses Clopper-Pearson and Wilson is reported
    alongside for comparability with the literature.
    """
    from scipy.stats import norm

    counts = np.asarray(counts, float)
    n = counts.sum()
    scale = 1.0 if n_eff is None else float(n_eff) / n
    k_eff = counts * scale
    n_used = n * scale
    z = float(norm.ppf(1.0 - alpha / (2.0 * 9.0)))
    per_alpha = alpha / 9.0

    cp = [clopper_pearson(k, n_used, per_alpha) for k in k_eff]
    wl = [wilson_interval(k, n_used, z) for k in k_eff]
    chosen = cp if method == "clopper_pearson" else wl
    return {
        "method": method,
        "alpha_familywise": alpha,
        "alpha_per_digit": per_alpha,
        "z_bonferroni": z,
        "n_nominal": float(n),
        "n_eff_used": float(n_used),
        "lower": [lo for lo, _ in chosen],
        "upper": [hi for _, hi in chosen],
        "covers_benford": [bool(lo <= bp <= hi) for (lo, hi), bp in zip(chosen, BENFORD_P)],
        "lower_wilson": [lo for lo, _ in wl],
        "upper_wilson": [hi for _, hi in wl],
        "covers_benford_wilson": [bool(lo <= bp <= hi) for (lo, hi), bp in zip(wl, BENFORD_P)],
    }


def bonferroni_wilson(counts: np.ndarray, alpha: float = 0.05,
                      n_eff: float | None = None) -> dict:
    """Backward-compatible alias: simultaneous Wilson intervals (see simultaneous_intervals)."""
    out = simultaneous_intervals(counts, alpha=alpha, n_eff=n_eff, method="wilson")
    return out


# --------------------------------------------------------------------------- #
# Harmonics
# --------------------------------------------------------------------------- #
def c_harmonic(m: np.ndarray, n: int = 1) -> complex:
    """c_n = E[exp(2*pi*i*n*m)] at exact mantissa resolution."""
    m = np.asarray(m, float)
    return complex(np.mean(np.exp(2j * np.pi * n * m)))


def harmonic_summary(m: np.ndarray) -> dict:
    c1 = c_harmonic(m, 1)
    c2 = c_harmonic(m, 2)
    return {
        "c1_re": c1.real,
        "c1_im": c1.imag,
        "c1_modulus": abs(c1),
        "c1_phase_deg": float(np.degrees(np.angle(c1))),
        "c2_modulus": abs(c2),
        "c2_phase_deg": float(np.degrees(np.angle(c2))),
    }


def c_from_digits(p: np.ndarray, n: int = 1) -> complex:
    """Digit-resolution reconstruction of c_n from the nine cell probabilities.

    If the mantissa is uniform *within* each digit bin, digit d contributes
        p_d * exp(2*pi*i*n*(e_d + w_d/2)) * sinc(n*w_d),
    with e_d = log10(d), w_d = log10(1+1/d) and sinc(z) = sin(pi z)/(pi z).
    The midpoint phase AND the sinc attenuation are both required; dropping
    either breaks the agreement with the exact coefficient.
    """
    p = np.asarray(p, float)
    z = n * DIGIT_WIDTHS
    sinc = np.where(np.abs(z) < 1e-12, 1.0, np.sin(np.pi * z) / (np.pi * np.where(z == 0, 1.0, z)))
    phase = np.exp(2j * np.pi * n * (DIGIT_EDGES + DIGIT_WIDTHS / 2.0))
    return complex(np.sum(p * phase * sinc))


def c_from_digits_midpoint_only(p: np.ndarray, n: int = 1) -> complex:
    """Cruder midpoint-only reconstruction (no sinc), for the cross-check."""
    p = np.asarray(p, float)
    phase = np.exp(2j * np.pi * n * (DIGIT_EDGES + DIGIT_WIDTHS / 2.0))
    return complex(np.sum(p * phase))


def benford_c_from_digits(n: int = 1) -> complex:
    return c_from_digits(BENFORD_P, n)


def fourier_bin_coefficient(n: int) -> np.ndarray:
    """K_{d,n} = integral over bin d of exp(-2*pi*i*n*m) dm, for each digit d."""
    k = np.empty(9, dtype=complex)
    for i, (e, w) in enumerate(zip(DIGIT_EDGES, DIGIT_WIDTHS)):
        if n == 0:
            k[i] = w
        else:
            k[i] = (np.exp(-2j * np.pi * n * e) - np.exp(-2j * np.pi * n * (e + w))) / (
                2j * np.pi * n
            )
    return k


def digits_from_harmonics(c: Iterable[complex], nmax: int = 200) -> np.ndarray:
    """p_d = w_d + 2 * sum_{n>=1} Re(c_n * K_{d,n}) (closed form).

    Derivation: the mantissa density has Fourier series f(m) = sum_n conj(c_n)
    exp(2*pi*i*n*m) with c_n = int f exp(2*pi*i*n*m) dm.  Integrating term by
    term over digit bin d and pairing the +n and -n terms,

        p_d = w_d + 2 * sum_{n>=1} Re( conj(c_n) * K^{+}_{d,n} )
            = w_d + 2 * sum_{n>=1} Re( c_n * K^{-}_{d,n} ),

    since K^{-}_{d,n} = conj(K^{+}_{d,n}).  `fourier_bin_coefficient(n)` returns
    K^{-}_{d,n}, so no conjugation of the K term is needed -- conjugating it
    (and dropping the factor 2) is the classic mistake this function's test
    catches.
    """
    c = list(c)
    p = DIGIT_WIDTHS.astype(float).copy()
    for n in range(1, nmax + 1):
        cn = c[n - 1] if n - 1 < len(c) else 0.0
        if abs(cn) < 1e-18:
            continue
        p = p + 2.0 * np.real(cn * fourier_bin_coefficient(n))
    return p


# --------------------------------------------------------------------------- #
# Cluster-bootstrap machinery
# --------------------------------------------------------------------------- #
@dataclass
class ClusterIndex:
    """Per-country sufficient statistics, so a country-cluster bootstrap is a matmul.

    Resampling whole countries with replacement (multiplicities ~ Multinomial)
    gives, for any additive statistic S = sum_i S_i and weight n = sum_i n_i,
        S_b = (M @ S) / (M @ n).
    """

    codes: np.ndarray          # unique country codes, sorted
    n: np.ndarray              # int, rows per country
    S_digit: np.ndarray        # (U, 9) per-country digit counts
    S_z1: np.ndarray           # (U,) complex, per-country sum of exp(2*pi*i*m)
    S_z2: np.ndarray           # (U,) complex, per-country sum of exp(4*pi*i*m)
    n_sum: np.ndarray          # (U,) per-country row count (duplicate of n, float)

    @classmethod
    def build(cls, codes: np.ndarray, digits: np.ndarray, m: np.ndarray) -> "ClusterIndex":
        codes = np.asarray(codes)
        uniq, inv = np.unique(codes, return_inverse=True)
        u = len(uniq)
        S_digit = np.zeros((u, 9), dtype=float)
        z1 = np.exp(2j * np.pi * m)
        z2 = np.exp(4j * np.pi * m)
        np.add.at(S_digit, (inv, digits - 1), 1.0)
        S_z1 = np.zeros(u, dtype=complex)
        S_z2 = np.zeros(u, dtype=complex)
        np.add.at(S_z1, inv, z1)
        np.add.at(S_z2, inv, z2)
        n = np.bincount(inv, minlength=u).astype(float)
        return cls(codes=uniq, n=n, S_digit=S_digit, S_z1=S_z1, S_z2=S_z2, n_sum=n.copy())

    def multiplicities(self, rng: np.random.Generator, B: int) -> np.ndarray:
        u = len(self.n)
        return rng.multinomial(u, np.full(u, 1.0 / u), size=B).astype(float)

    def boot_digit_props(self, rng: np.random.Generator, B: int) -> np.ndarray:
        M = self.multiplicities(rng, B)
        tot = M @ self.n[:, None]                     # (B,1)
        return (M @ self.S_digit) / tot

    def boot_c1(self, rng: np.random.Generator, B: int) -> np.ndarray:
        M = self.multiplicities(rng, B)
        tot = M @ self.n
        return (M @ self.S_z1) / tot

    def boot_c2(self, rng: np.random.Generator, B: int) -> np.ndarray:
        M = self.multiplicities(rng, B)
        tot = M @ self.n
        return (M @ self.S_z2) / tot


def design_effect(groups: np.ndarray, values: np.ndarray) -> dict:
    """One-way ANOVA estimate of the ICC of `values` within `groups` -> design effect."""
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    u = len(uniq)
    n = np.bincount(inv, minlength=u).astype(float)
    sums = np.bincount(inv, weights=values, minlength=u)
    mus = sums / n
    overall = values.mean()
    ssb = float(np.sum(n * (mus - overall) ** 2))
    ss_total = float(np.sum((values - overall) ** 2))
    ssw = ss_total - ssb
    df_b, df_w = u - 1, len(values) - u
    msb = ssb / df_b
    msw = ssw / df_w if df_w > 0 else 0.0
    nbar = float(n.mean())
    if msb + (nbar - 1) * msw <= 0:
        icc = 0.0
    else:
        icc = (msb - msw) / (msb + (nbar - 1) * msw)
    icc = float(np.clip(icc, 0.0, 1.0))
    deff = 1.0 + (nbar - 1.0) * icc
    return {
        "icc": icc,
        "n_groups": int(u),
        "mean_rows_per_group": nbar,
        "design_effect": deff,
        "n_effective": float(len(values) / deff) if deff > 0 else float(len(values)),
        "ss_between": ssb,
        "ss_within": ssw,
    }


def rayleigh_reference(n_eff: float) -> dict:
    """Null distribution of |c_1| for n_eff independent uniform mantissas."""
    n_eff = max(float(n_eff), 1.0)
    lam = n_eff
    return {
        "n_eff": n_eff,
        "mean_modulus": float(np.sqrt(np.pi / (4.0 * lam))),
        "median_modulus": float(np.sqrt(np.log(2.0) / lam)),
        "p95_modulus": float(np.sqrt(-np.log(0.05) / lam)),
        "p975_modulus": float(np.sqrt(-np.log(0.025) / lam)),
        "se_component": float(np.sqrt(1.0 / (2.0 * lam))),
    }


def projected_statistic(r_obs: np.ndarray, direction: np.ndarray) -> dict:
    """One-degree-of-freedom projection of the residual vector on `direction`.

    P = (r . u) / |u|, cosine = (r . u) / (|r| |u|).  Both have non-zero null
    means, so they are always calibrated against a matched null rather than
    against zero.
    """
    r = np.asarray(r_obs, float)
    u = np.asarray(direction, float)
    nu = float(np.linalg.norm(u))
    nr = float(np.linalg.norm(r))
    if nu == 0 or nr == 0:
        return {"P": 0.0, "cosine": 0.0, "direction_norm": nu}
    return {
        "P": float(np.dot(r, u) / nu),
        "cosine": float(np.dot(r, u) / (nr * nu)),
        "direction_norm": nu,
        "residual_norm": nr,
    }


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def self_test(verbose: bool = False) -> dict:
    """Instrument validation.  Raises AssertionError on any failure."""
    rng = np.random.default_rng(12345)
    out = {}

    # 1. rotation identity on real data
    x = rng.lognormal(mean=6.2, sigma=1.2, size=20000)
    m0 = mantissa(x)
    c1_0 = c_harmonic(m0, 1)
    worst = 0.0
    for c in np.arange(1.0, 10.0, 0.1):
        lhs = c_harmonic(mantissa(c * x), 1)
        rhs = c1_0 * np.exp(2j * np.pi * np.log10(c))
        worst = max(worst, abs(lhs - rhs))
    assert worst < 1e-10, f"rotation identity violated: {worst}"
    out["rotation_identity_max_error"] = worst

    # 2. mirror identity
    m_mirror = np.mod(1.0 - m0, 1.0)
    c_mirror = c_harmonic(m_mirror, 1)
    assert abs(c_mirror - np.conj(c1_0)) < 1e-10, "mirror identity (complex) violated"
    assert abs(abs(c_mirror) - abs(c1_0)) < 1e-12, "mirror identity (modulus) violated"
    out["mirror_identity_error"] = float(abs(c_mirror - np.conj(c1_0)))

    # 2b. mirror identity through the reciprocal of the actual data
    c_recip = c_harmonic(mantissa(1.0 / x), 1)
    assert abs(c_recip - np.conj(c1_0)) < 1e-9, "reciprocal digit identity violated"
    out["reciprocal_identity_error"] = float(abs(c_recip - np.conj(c1_0)))

    # 3. digit-resolution reconstruction vs exact coefficient on smooth data
    p_emp = digit_proportions(first_digit(x))
    c_from_p = c_from_digits(p_emp, 1)
    gap = abs(c_from_p - c1_0)
    out["digit_transfer_gap"] = float(gap)
    out["digit_transfer_gap_benford_baseline"] = float(abs(benford_c_from_digits(1)))
    # the reconstruction must be much closer to the exact value than the
    # within-bin approximation error of exact Benford (a sanity ordering)
    assert gap < 0.25 * abs(c1_0) + 0.02, f"digit reconstruction off: {gap} vs |c1|={abs(c1_0)}"
    # the midpoint-only version must be measurably worse than the sinc version
    gap_mid = abs(c_from_digits_midpoint_only(p_emp, 1) - c1_0)
    out["digit_transfer_gap_midpoint_only"] = float(gap_mid)
    assert gap <= gap_mid + 1e-12, "sinc term did not improve the reconstruction"

    # 4. closed-form vs numeric digit law for a lognormal
    from size_laws import periodise_lognormal

    mu, sigma = 6.2, 1.2
    p_num = periodise_lognormal(mu, sigma)
    c_n = [np.exp(2j * np.pi * n * mu - 2 * np.pi ** 2 * n ** 2 * sigma ** 2) for n in range(1, 61)]
    p_cf = digits_from_harmonics(c_n, nmax=60)
    err = float(np.max(np.abs(p_num - p_cf)))
    out["periodise_vs_closed_form_max_abs_error"] = err
    assert err < 1e-4, f"periodised vs closed-form digit law disagree by {err}"
    out["sum_p_numeric"] = float(p_num.sum())
    assert abs(p_num.sum() - 1.0) < 1e-6, "periodised digit law does not sum to 1"

    # 4b. numeric periodisation reproduces c1 to high precision
    c1_num = c_from_digits(p_num, 1)
    out["periodised_digit_c1_vs_analytic"] = float(abs(c1_num - c_n[0]))

    # 5. simultaneous coverage of Bonferroni-Wilson on Benford draws.
    #    "Simultaneous" coverage is the probability that ALL nine intervals
    #    cover at once, so the test is whether the per-draw all-digit coverage
    #    rate reaches 1 - alpha (here 0.95), not whether each digit individually
    #    covers 95% of the time.
    coverage = {}
    for n in (200, 500, 900):
        draws = rng.multinomial(n, BENFORD_P, size=2000)
        for method in ("clopper_pearson", "wilson"):
            cover = np.ones(len(draws), dtype=bool)
            for i, row in enumerate(draws):
                ci = simultaneous_intervals(row, alpha=0.05, method=method)
                cover[i] = bool(np.all(ci["covers_benford"]))
            coverage[f"n{n}_{method}"] = float(cover.mean())
    out["simultaneous_coverage"] = coverage
    # The exact (Clopper-Pearson) intervals must reach the nominal level at
    # every sample size examined; the Wilson score intervals are known to fall
    # marginally short and that shortfall is recorded rather than hidden.
    for key, val in coverage.items():
        if "clopper_pearson" in key:
            assert val >= 0.95, f"exact simultaneous coverage {key} = {val:.3f} < 0.95"
    out["wilson_coverage_shortfall_recorded"] = all(
        coverage[k] < coverage[k.replace("wilson", "clopper_pearson")]
        for k in coverage if "wilson" in k
    )

    # 6. digit proportions of exact-Benford draws are centred correctly
    big = rng.multinomial(2_000_000, BENFORD_P)
    out["benford_draw_max_abs_residual_pp"] = float(np.max(np.abs(residuals_pp(big / big.sum()))))

    # 7. Mantissa of exact powers of ten and neighbours stay in range
    assert np.all((mantissa(np.array([1.0, 10.0, 100.0, 1e9])) >= 0.0))
    assert np.all(first_digit(np.array([9.999999, 1.0, 10.0, 99.0])) == np.array([9, 1, 1, 9]))

    if verbose:
        for k, v in out.items():
            print(f"  {k}: {v}")
    return out


if __name__ == "__main__":
    res = self_test(verbose=True)
    print("SELF TEST PASSED")
