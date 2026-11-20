"""The measurement instrument: complex Fourier coefficients of the mantissa
distribution, the digit-resolution reconstruction, and the companion battery
of conventional Benford statistics.

Theory (Berger & Hill 2011; 2019 primer).  Write a positive number as
x = 10**(k + m) with k = floor(log10 x) and m = 10**0 in [0, 1) the
*mantissa* (sometimes called the significand).  Exact Benford behaviour is
equivalent to m ~ Uniform[0, 1), which by Fourier uniqueness is equivalent to

    c_n := E[exp(2*pi*1j*n*m)] = 0   for every non-zero integer n.

The leading instrument of this study is c_1.  Its modulus |c_1| is the
amplitude of the first harmonic of the mantissa density, and its argument is
the phase.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

BENFORD = np.array([math.log10(1.0 + 1.0 / d) for d in range(1, 10)], dtype=float)
#: left edge of the mantissa bin of each leading digit (e_d = log10 d)
DIGIT_EDGES = np.array([math.log10(d) for d in range(1, 11)], dtype=float)
#: width of the mantissa bin of each leading digit (w_d = log10(1 + 1/d))
DIGIT_WIDTHS = np.array([math.log10(1.0 + 1.0 / d) for d in range(1, 10)], dtype=float)


# ---------------------------------------------------------------------------
# mantissa machinery
# ---------------------------------------------------------------------------
def mantissa(x: np.ndarray) -> np.ndarray:
    """Mantissa in [0, 1) of a positive array, computed stably.

    Uses ``log10`` once and removes the integer part; the floating-point
    residual of ``log10`` at exact powers of ten is clipped to [0, 1).
    """
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError("mantissa() requires strictly positive values")
    y = np.log10(x)
    m = y - np.floor(y)
    # guard the 9.999... -> 10.0 rounding edge in log10
    m = np.where(m >= 1.0, m - 1.0, m)
    return np.clip(m, 0.0, 1.0 - 1e-15)


def first_digit(x: np.ndarray) -> np.ndarray:
    """Leading decimal digit (1..9) of a positive array."""
    m = mantissa(x)
    d = np.floor(10.0 ** m + 1e-12).astype(int)
    return np.clip(d, 1, 9)


def first_two_digits(x: np.ndarray) -> np.ndarray:
    """Leading two decimal digits (10..99) of a positive array."""
    y = np.log10(np.asarray(x, dtype=float))
    m2 = y - np.floor(y) + 1.0          # in [1, 2)
    v = np.floor(100.0 * 10.0 ** (m2 - 1.0) + 1e-9).astype(int)
    return np.clip(v, 10, 99)


# ---------------------------------------------------------------------------
# harmonic coefficients
# ---------------------------------------------------------------------------
def c_n(m: np.ndarray, n: int) -> complex:
    """Empirical complex Fourier coefficient c_n of a mantissa sample."""
    m = np.asarray(m, dtype=float)
    if m.size == 0:
        raise ValueError("c_n() on an empty sample")
    return complex(np.mean(np.exp(2j * np.pi * n * m)))


def c_n_vector(m: np.ndarray, nmax: int = 8) -> dict[int, complex]:
    """c_1 .. c_nmax as a dict."""
    return {n: c_n(m, n) for n in range(1, nmax + 1)}


def c_n_digit_vector(p: np.ndarray, n: int) -> complex:
    """Digit-resolution reconstruction of c_n from the nine-cell vector p.

    If the mantissa is uniform *within* each leading-digit bin, the bin's
    contribution is

        p_d * (1/w_d) * int_{e_d}^{e_d+w_d} exp(2j pi n m) dm
          = p_d * exp(2j pi n (e_d + w_d/2)) * sinc(n * w_d),

    with sinc(z) = sin(pi z)/(pi z).  Both the midpoint phase and the sinc
    attenuation are needed; dropping either makes the reconstruction wrong.
    """
    p = np.asarray(p, dtype=float)
    if p.shape != (9,):
        raise ValueError(f"digit vector must have shape (9,), got {p.shape}")
    mid = DIGIT_EDGES[:9] + DIGIT_WIDTHS / 2.0
    z = n * DIGIT_WIDTHS
    sinc = np.where(np.abs(z) < 1e-15, 1.0, np.sin(np.pi * z) / (np.pi * np.where(np.abs(z) < 1e-15, 1.0, z)))
    return complex(np.sum(p * np.exp(2j * np.pi * n * mid) * sinc))


def c_n_digit_midpoint_only(p: np.ndarray, n: int) -> complex:
    """Cruder reconstruction without the sinc attenuation (cross-check only)."""
    p = np.asarray(p, dtype=float)
    mid = DIGIT_EDGES[:9] + DIGIT_WIDTHS / 2.0
    return complex(np.sum(p * np.exp(2j * np.pi * n * mid)))


def digit_probabilities_from_mantissa(m: np.ndarray) -> np.ndarray:
    """Nine-cell leading-digit frequency vector from a mantissa sample."""
    d = first_digit_from_mantissa(m)
    return np.bincount(d, minlength=10)[1:10] / float(np.asarray(m).size)


def first_digit_from_mantissa(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    d = np.floor(10.0 ** m + 1e-12).astype(int)
    return np.clip(d, 1, 9)


# ---------------------------------------------------------------------------
# identities that must hold to machine precision
# ---------------------------------------------------------------------------
def check_rotation_identity(m: np.ndarray, tol: float = 1e-10) -> dict:
    """c_1(c*x) = c_1(x) * exp(2j pi log10 c) exactly, for every scale c."""
    base = c_n(m, 1)
    worst = 0.0
    worst_c = 1.0
    for c in np.arange(1.0, 10.0, 0.1):
        shifted = c_n(np.mod(m + math.log10(c), 1.0), 1)   # mantissa of c*x
        predicted = base * np.exp(2j * np.pi * math.log10(c))
        err = abs(shifted - predicted)
        if err > worst:
            worst, worst_c = err, float(c)
    return {"max_abs_error": worst, "worst_scale": worst_c, "tolerance": tol,
            "passed": bool(worst < tol)}


def check_mirror_identity(m: np.ndarray, tol: float = 1e-10) -> dict:
    """The reciprocal map m -> 1-m conjugates c_1 and preserves |c_1|."""
    m = np.asarray(m, dtype=float)
    a = c_n(m, 1)
    b = c_n(np.mod(1.0 - m, 1.0), 1)
    err_conj = abs(b - np.conj(a))
    err_amp = abs(abs(b) - abs(a))
    return {"abs_error_conjugation": err_conj, "abs_error_amplitude": err_amp,
            "tolerance": tol, "passed": bool(max(err_conj, err_amp) < tol)}


# ---------------------------------------------------------------------------
# conventional companion battery
# ---------------------------------------------------------------------------
def wilson_interval(p_hat: float, n: float, z: float) -> tuple[float, float]:
    """Wilson score interval at critical value z."""
    if n <= 0:
        return (0.0, 1.0)
    denom = 1.0 + z * z / n
    centre = (p_hat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(max(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n), 0.0)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bonferroni_wilson(p_hat: np.ndarray, n: float, alpha: float,
                      widening: float = 1.0) -> list[dict]:
    """Nine simultaneous Wilson intervals at familywise level 1-alpha.

    Bonferroni: each interval at alpha/9.  ``widening`` > 1 inflates the
    critical value, which is the documented remedy if the coverage simulation
    comes in under nominal.
    """
    z = float(stats.norm.ppf(1.0 - alpha / (2.0 * 9))) * widening
    out = []
    for i, p in enumerate(np.asarray(p_hat, dtype=float)):
        lo, hi = wilson_interval(p, n, z)
        out.append({"digit": i + 1, "p_hat": float(p), "lo": lo, "hi": hi,
                    "covers_benford": bool(lo <= BENFORD[i] <= hi)})
    return out


def goodman_intervals(p_hat: np.ndarray, n: float, alpha: float) -> list[dict]:
    """Goodman (1965) simultaneous multinomial intervals (secondary)."""
    a = float(stats.chi2.ppf(1.0 - alpha, 1))
    out = []
    for i, p in enumerate(np.asarray(p_hat, dtype=float)):
        n_a = n + a
        centre = (2.0 * n * p + a) / (2.0 * n_a)
        half = math.sqrt(a) * math.sqrt(max(4.0 * n * p * (1.0 - p) + a, 0.0)) / (2.0 * n_a)
        lo, hi = max(0.0, centre - half), min(1.0, centre + half)
        out.append({"digit": i + 1, "p_hat": float(p), "lo": lo, "hi": hi,
                    "covers_benford": bool(lo <= BENFORD[i] <= hi)})
    return out


def simultaneous_coverage(p: np.ndarray, n: float, alpha: float,
                          widening: float = 1.0) -> float:
    """Familywise coverage of the Bonferroni-Wilson intervals, one draw."""
    iv = bonferroni_wilson(p, n, alpha, widening=widening)
    return 1.0 if all(d["covers_benford"] for d in iv) else 0.0


def kuiper_v(m: np.ndarray) -> dict:
    """Kuiper V of the mantissa sample against Uniform[0, 1)."""
    m = np.sort(np.asarray(m, dtype=float))
    n = m.size
    if n < 2:
        return {"V": float("nan"), "D_plus": float("nan"), "D_minus": float("nan"), "n": int(n)}
    i = np.arange(1, n + 1)
    d_plus = float(np.max(i / n - m))
    d_minus = float(np.max(m - (i - 1) / n))
    return {"V": d_plus + d_minus, "D_plus": d_plus, "D_minus": d_minus, "n": int(n)}


def first_two_digit_mad(x: np.ndarray) -> dict:
    """MAD and chi-square over the 90 first-two-digit cells."""
    d2 = first_two_digits(x)
    obs = np.bincount(d2, minlength=100)[10:100]
    n = obs.sum()
    expected = np.array([math.log10(1.0 + 1.0 / k) for k in range(10, 100)])
    p_hat = obs / n
    return {
        "mad": float(np.mean(np.abs(p_hat - expected))),
        "chi2": float(np.sum((obs - n * expected) ** 2 / (n * expected))),
        "df": 89,
        "n": int(n),
    }


def companion_battery_from_values(x: np.ndarray, alpha: float = 0.05,
                                  n_eff: float | None = None) -> dict:
    """The conventional battery on raw positive values (mantissas derived)."""
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError("companion_battery_from_values requires strictly positive values; "
                         "pass mantissas to companion_battery_from_mantissa instead")
    bat = companion_battery_from_mantissa(mantissa(x), alpha=alpha, n_eff=n_eff)
    bat["first_two_digit"] = first_two_digit_mad(x)
    return bat


def companion_battery_from_mantissa(m: np.ndarray, alpha: float = 0.05,
                                    n_eff: float | None = None) -> dict:
    """The full conventional battery at the same n as the harmonic instrument.

    ``m`` MUST be a mantissa array in [0, 1).  Raw populations are not
    mantissas: feeding them here silently produces a digit vector that is
    100% digit 9, which is exactly the failure mode this split API exists to
    make impossible.
    """
    m = np.asarray(m, dtype=float)
    if m.size == 0:
        raise ValueError("empty mantissa sample")
    if m.min() < 0.0 or m.max() > 1.0:
        raise ValueError(f"values outside [0,1] were passed as mantissas: "
                         f"min={m.min():.4g}, max={m.max():.4g}")
    n_obs = int(m.size)
    p_hat = digit_probabilities_from_mantissa(m)
    resid_pp = 100.0 * (p_hat - BENFORD)
    chi2 = float(n_obs * np.sum((p_hat - BENFORD) ** 2 / BENFORD))
    kp = kuiper_v(m)
    c1 = c_n(m, 1)
    c2 = c_n(m, 2)
    out = {
        "n": n_obs,
        "p_hat": [float(v) for v in p_hat],
        "benford": [float(v) for v in BENFORD],
        "residuals_pp": [float(v) for v in resid_pp],
        "mad": float(np.mean(np.abs(resid_pp)) / 100.0),
        "mad_nigrini_band": nigrini_band(float(np.mean(np.abs(resid_pp)) / 100.0)),
        "chi2": chi2,
        "chi2_df": 8,
        "chi2_p_naive_iid": float(stats.chi2.sf(chi2, 8)),
        "kuiper_V": kp["V"],
        "kuiper_D_plus": kp["D_plus"],
        "kuiper_D_minus": kp["D_minus"],
        "mantissa_mean": float(np.mean(m)),
        "mantissa_mad_from_half": float(np.mean(np.abs(m - 0.5))),
        "c1": {"re": float(c1.real), "im": float(c1.imag)},
        "c2": {"re": float(c2.real), "im": float(c2.imag)},
        "c1_amp": float(abs(c1)),
        "c1_phase": float(np.angle(c1)),
        "per_digit_intervals_naive": bonferroni_wilson(p_hat, n_obs, alpha),
        "goodman_intervals_naive": goodman_intervals(p_hat, n_obs, alpha),
    }
    if n_eff is not None:
        out["per_digit_intervals_neff"] = bonferroni_wilson(p_hat, float(n_eff), alpha)
        out["n_eff_used_for_intervals"] = float(n_eff)
    return out


def nigrini_band(mad: float) -> str:
    """Nigrini (2012) first-digit MAD conformity bands, reported but never used
    as a verdict here (they ignore n and the generating law)."""
    if mad <= 0.006:
        return "close conformity"
    if mad <= 0.012:
        return "acceptable conformity"
    if mad <= 0.015:
        return "marginal conformity"
    return "nonconformity"
