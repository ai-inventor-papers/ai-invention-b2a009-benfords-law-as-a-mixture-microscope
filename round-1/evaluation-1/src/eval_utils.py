"""Shared numeric helpers for the evaluation stages (JSON-safe conversions, bootstraps)."""

from __future__ import annotations

import numpy as np

import benford_core as bc


def jfloat(x) -> float | None:
    if x is None:
        return None
    v = float(x)
    return None if not np.isfinite(v) else v


def jlist(a) -> list:
    return [jfloat(v) for v in np.asarray(a, float).ravel()]


def jcomplex(z: complex) -> dict:
    return {"re": jfloat(np.real(z)), "im": jfloat(np.imag(z)),
            "modulus": jfloat(abs(z)), "phase_deg": jfloat(np.degrees(np.angle(z)))}


def summarise(vals) -> dict:
    v = np.asarray(vals, float)
    if v.size == 0:
        return {}
    return {
        "n": int(v.size),
        "mean": jfloat(v.mean()),
        "sd": jfloat(v.std(ddof=1)) if v.size > 1 else 0.0,
        "p2_5": jfloat(np.percentile(v, 2.5)),
        "p5": jfloat(np.percentile(v, 5.0)),
        "p50": jfloat(np.percentile(v, 50.0)),
        "p95": jfloat(np.percentile(v, 95.0)),
        "p97_5": jfloat(np.percentile(v, 97.5)),
        "min": jfloat(v.min()),
        "max": jfloat(v.max()),
    }


def percentile_of(value: float, null) -> float:
    null = np.asarray(null, float)
    return jfloat(100.0 * np.mean(null < value))


def two_sided_p(value: float, null) -> float:
    null = np.asarray(null, float)
    return jfloat(min(1.0, 2.0 * min(np.mean(null <= value), np.mean(null >= value))))


def bh_adjust(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    return out


def digit_vector(sub, column: str = "d1") -> np.ndarray:
    """Nine cell proportions from a frame column."""
    counts = np.bincount(sub[column].values.astype(int), minlength=10)[1:]
    return counts / counts.sum()


def pooled_harmonic(sub, n: int = 1) -> complex:
    return bc.c_harmonic(sub.mantissa.values, n)


def cluster_bootstrap_ci(cluster: bc.ClusterIndex, rng: np.random.Generator, B: int,
                         statistic: str = "c1") -> dict:
    """Percentile CI of |c_1| (or |c_2|) under a country resample."""
    vals = cluster.boot_c1(rng, B) if statistic == "c1" else cluster.boot_c2(rng, B)
    mod = np.abs(vals)
    return {
        "modulus_ci_low": jfloat(np.percentile(mod, 2.5)),
        "modulus_ci_high": jfloat(np.percentile(mod, 97.5)),
        "modulus_p50": jfloat(np.percentile(mod, 50.0)),
        "phase_ci_low_deg": jfloat(np.percentile(np.degrees(np.angle(vals)), 2.5)),
        "phase_ci_high_deg": jfloat(np.percentile(np.degrees(np.angle(vals)), 97.5)),
        "n_replicates": int(B),
    }


def cluster_bootstrap_residuals(cluster: bc.ClusterIndex, rng: np.random.Generator,
                                B: int, p_ref: np.ndarray) -> dict:
    """Percentile CI for each of the nine residuals in percentage points."""
    props = cluster.boot_digit_props(rng, B)
    r = 100.0 * (props - p_ref)
    return {
        "residuals_pp_ci_low": jlist(np.percentile(r, 2.5, axis=0)),
        "residuals_pp_ci_high": jlist(np.percentile(r, 97.5, axis=0)),
        "M_ci_low": jfloat(np.percentile(np.abs(r).max(axis=1), 2.5) / 100.0),
        "M_ci_high": jfloat(np.percentile(np.abs(r).max(axis=1), 97.5) / 100.0),
        "mads_ci_low": jfloat(np.percentile(np.abs(r).mean(axis=1), 2.5) / 100.0),
        "mads_ci_high": jfloat(np.percentile(np.abs(r).mean(axis=1), 97.5) / 100.0),
        "n_replicates": int(B),
    }


def moving_block_bootstrap_ci(x: np.ndarray, y: np.ndarray, block: int, B: int,
                              rng: np.random.Generator, stat: str = "pearson") -> dict:
    """Moving-block bootstrap of a correlation between two year-indexed series."""
    from scipy import stats as sstats

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < block + 3:
        return {"stat": jfloat(sstats.pearsonr(x, y)[0]), "ci_low": None,
                "ci_high": None, "n": n}

    def f(a, b, kind):
        if kind == "pearson":
            return float(sstats.pearsonr(a, b)[0])
        return float(sstats.spearmanr(a, b)[0])

    obs = f(x, y, stat)
    n_blocks = int(np.ceil(n / block))
    vals = []
    for _ in range(B):
        starts = rng.integers(0, max(1, n - block + 1), n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        idx = np.clip(idx, 0, n - 1)
        try:
            vals.append(f(x[idx], y[idx], stat))
        except ValueError:
            continue
    vals = np.array(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 10:
        return {"stat": jfloat(obs), "ci_low": None, "ci_high": None, "n": n}
    return {
        "stat": jfloat(obs),
        "ci_low": jfloat(np.percentile(vals, 2.5)),
        "ci_high": jfloat(np.percentile(vals, 97.5)),
        "boot_sd": jfloat(vals.std(ddof=1)),
        "n": n,
        "block_length": block,
        "n_replicates": int(vals.size),
    }
