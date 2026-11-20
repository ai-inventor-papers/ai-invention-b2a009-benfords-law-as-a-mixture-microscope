"""Matched nulls for the digit statistics.

The point of a matched null is to replace "is this Benford?" with "is this what the
fitted size law says it should be?".  Exact Benford is a point null with no
mechanism behind it; the fitted law is the mechanism, so departures from IT are
what a calibrated screen has to explain.

Five nulls, each answering a different question:

  N_benford_cluster   exact Benford digits, but resampled by COUNTRY (bootstrap) --
                      the answer to "is the observed pattern what a Benford world
                      would give with this country structure?"
  N_law_iid           i.i.d. draws from the fitted law at the observed n --
                      the classical parametric bootstrap (over-states information,
                      because it drops the country repeat structure).
  N_law_cluster       [PRIMARY] one draw per country from the fitted law, then that
                      country's own even years as a mean-reverting random walk of
                      the observed within-country amplitude.  Reproduces the
                      dependence structure that makes nominal n a fiction.
  N_entity_resample   resample countries with replacement from the observed panel,
                      keeping all their years.  Non-parametric, so it applies where
                      no size law is fitted.
  N_shiftyear         circular year shifts of a predicted series -- an exact null
                      that preserves autocorrelation while destroying year-specific
                      association.

Every null returns ONLY per-replicate scalars or nine-vectors, so storage stays
bounded (B x 11 floats).
"""

from __future__ import annotations

import numpy as np

import benford_core as bc


def _digit_counts_from_mantissas(ms: np.ndarray) -> np.ndarray:
    d = np.clip(np.floor(10.0 ** ms + 1e-12), 1, 9).astype(int)
    return np.bincount(d, minlength=10)[1:].astype(float)


def null_benford_cluster(cluster: bc.ClusterIndex, rng: np.random.Generator,
                         B: int) -> dict:
    """Exact Benford mantissas at the observed country structure."""
    props = cluster.boot_digit_props(rng, B)               # (B, 9)
    r = 100.0 * (props - bc.BENFORD_P)
    c1 = cluster.boot_c1(rng, B)
    c2 = cluster.boot_c2(rng, B)
    return {
        "residuals_pp": r,
        "M": np.abs(r).max(axis=1) / 100.0,
        "abs_c1": np.abs(c1),
        "abs_c2": np.abs(c2),
        "phase_c1_deg": np.degrees(np.angle(c1)),
        "digit_props": props,
    }


def null_law_iid(law, rng: np.random.Generator, B: int, n: int) -> dict:
    """I.i.d. draws from the fitted law at the observed row count.

    Vectorised in chunks; the per-replicate Python loop it replaces was O(B)
    iterations of a full `law.sample` call.
    """
    p_model = law.digit_proportions()
    r = np.empty((B, 9))
    c1 = np.empty(B, dtype=complex)
    c2 = np.empty(B, dtype=complex)
    done = 0
    chunk = 200
    while done < B:
        b = min(chunk, B - done)
        ys = law.sample(rng, n * b)
        ms = (ys % 1.0).reshape(b, n)
        cnt = np.stack([np.bincount(
            np.clip(np.floor(10.0 ** row + 1e-12), 1, 9).astype(int), minlength=10)[1:]
            for row in ms]).astype(float)
        r[done:done + b] = 100.0 * (cnt / n - p_model)
        c1[done:done + b] = np.exp(2j * np.pi * ms).mean(axis=1)
        c2[done:done + b] = np.exp(4j * np.pi * ms).mean(axis=1)
        done += b
    return {
        "residuals_pp": r,
        "M": np.abs(r).max(axis=1) / 100.0,
        "abs_c1": np.abs(c1),
        "abs_c2": np.abs(c2),
        "phase_c1_deg": np.degrees(np.angle(c1)),
        "digit_props": r / 100.0 + p_model,
    }


def null_law_cluster(law, rng: np.random.Generator, B: int,
                     n_per_country: np.ndarray, within_sd: float,
                     p_model: np.ndarray, direction: np.ndarray,
                     chunk: int = 200) -> dict:
    """[PRIMARY] one mantissa per country from the law + a within-country walk.

    `within_sd` is the measured detrended within-country sd of log10 population
    (decades).  Each country gets one draw from the law and then its own sequence
    of deviations, so the digit signals of a country's years are near-replicates --
    which is exactly what the real panel shows.

    Fully vectorised: the per-country loop this used to carry was O(B * U) Python
    iterations and dominated the whole evaluation's runtime.
    """
    n_per = np.asarray(n_per_country, int)
    u = len(n_per)
    pad = int(n_per.max())
    total = int(n_per.sum())

    keep = np.zeros((u, pad), dtype=bool)
    for i, ni in enumerate(n_per):
        keep[i, :ni] = True
    keep_flat = keep.ravel()

    w = np.asarray(law.params["weights"], float)
    mus = np.asarray(law.params["mus"], float)
    sgs = np.asarray(law.params["sigmas"], float)
    p_model = np.asarray(p_model, float)

    r = np.empty((B, 9))
    c1 = np.empty(B, dtype=complex)
    c2 = np.empty(B, dtype=complex)
    proj = np.empty(B)
    direct = np.asarray(direction, float)

    done = 0
    while done < B:
        b = min(chunk, B - done)
        comp = rng.choice(len(w), size=(b, u), p=w / w.sum())
        base = rng.normal(mus[comp], sgs[comp])
        steps = rng.normal(0.0, within_sd, size=(b, u, pad))
        ms = (base[:, :, None] + steps) % 1.0
        flat = ms.reshape(b, -1)[:, keep_flat]
        dg = np.clip(np.floor(10.0 ** flat + 1e-12), 1, 9).astype(np.int64)
        cnt = np.stack([np.bincount(row, minlength=10)[1:] for row in dg]).astype(float)
        p = cnt / total
        rr = 100.0 * (p - p_model)
        r[done:done + b] = rr
        c1[done:done + b] = np.exp(2j * np.pi * flat).mean(axis=1)
        c2[done:done + b] = np.exp(4j * np.pi * flat).mean(axis=1)
        nu = np.linalg.norm(direct)
        proj[done:done + b] = (rr @ direct) / nu if nu > 0 else 0.0
        done += b

    return {
        "residuals_pp": r,
        "M": np.abs(r).max(axis=1) / 100.0,
        "abs_c1": np.abs(c1),
        "abs_c2": np.abs(c2),
        "phase_c1_deg": np.degrees(np.angle(c1)),
        "projection_P": proj,
    }


def null_entity_resample(cluster: bc.ClusterIndex, codes: np.ndarray, digits: np.ndarray,
                         rng: np.random.Generator, B: int) -> dict:
    """Non-parametric country resample: the answer to "how much would this move?"."""
    props = cluster.boot_digit_props(rng, B)
    r = 100.0 * (props - bc.BENFORD_P)
    c1 = cluster.boot_c1(rng, B)
    return {
        "residuals_pp": r,
        "M": np.abs(r).max(axis=1) / 100.0,
        "abs_c1": np.abs(c1),
        "phase_c1_deg": np.degrees(np.angle(c1)),
        "digit_props": props,
    }


def null_shiftyear(observed: np.ndarray, predicted: np.ndarray) -> dict:
    """Every circular year shift of the predicted series: an exact shift null."""
    t = len(observed)
    vals = np.array([np.corrcoef(observed, np.roll(predicted, s))[0, 1]
                     for s in range(1, t)])
    return {"corr": vals, "n_shifts": int(t - 1)}


def matched_envelope_summary(null: dict, observed: dict) -> dict:
    """Percentiles of every observed statistic inside a null sample."""
    out = {}
    for key, obs_key in [("M", "M"), ("abs_c1", "abs_c1"), ("abs_c2", "abs_c2"),
                         ("projection_P", "projection_P")]:
        if key not in null or obs_key not in observed:
            continue
        vals = np.asarray(null[key], float)
        obs = float(observed[obs_key])
        out[key] = {
            "observed": obs,
            "percentile": float(100.0 * np.mean(vals < obs)),
            "two_sided_p": float(min(1.0, 2.0 * min(np.mean(vals <= obs),
                                                    np.mean(vals >= obs)))),
            "null_mean": float(vals.mean()),
            "null_p2_5": float(np.percentile(vals, 2.5)),
            "null_p97_5": float(np.percentile(vals, 97.5)),
        }
    if "residuals_pp" in null and "residuals_pp" in observed:
        r_null = np.asarray(null["residuals_pp"], float)
        r_obs = np.asarray(observed["residuals_pp"], float)
        # null mean cosine of the residual vector with the observed one: the null
        # mean is NOT zero and must be estimated, not assumed.
        dots = r_null @ r_obs
        norms = np.linalg.norm(r_null, axis=1) * np.linalg.norm(r_obs)
        cos_null = np.where(norms > 0, dots / np.where(norms > 0, norms, 1.0), 0.0)
        cos_obs = float(dots.max() * 0 + np.dot(r_obs, r_obs) /
                        max(np.linalg.norm(r_obs) ** 2, 1e-30))
        out["cosine_with_observed"] = {
            "observed": 1.0,
            "null_mean": float(cos_null.mean()),
            "null_p95": float(np.percentile(cos_null, 95.0)),
            "null_p97_5": float(np.percentile(cos_null, 97.5)),
            "self_consistency": cos_obs,
        }
        # per-digit sign probabilities under the null
        out["sign_prob_positive"] = (np.mean(r_null > 0, axis=0)).tolist()
    return out
