"""Cluster-calibrated conformity of the digit law: the headline instrument.

The literature this artifact follows (Mebane 2011; Cho & Gaines 2004;
Bustillos & Leiva 2026) is unambiguous on one point: a significant chi-square
against exact Benford is not evidence of anything unless the sampling
distribution of the statistic is the one the data actually have.  Country
populations are a PANEL -- 236 countries observed over 74 years -- so the
nominal N = 17,424 overstates independent information by roughly the number of
years per country (design effect ~9 by the ICC estimate).  A nominal-n
chi-square therefore rejects Benford almost surely for any real panel, and its
p-value is uninterpretable.

This module builds the null the data need, three ways:

  exact_benford_cluster
      One exact-Benford mantissa per COUNTRY (n = number of countries), then
      that country's own years as a small mean-reverting walk of the measured
      detrended within-country amplitude.  This is the "what would a Benford
      world look like with this country structure?" null.  It is the correct
      null for the question "how closely does Benford fit?".

  nominal_iid
      i.i.d. exact-Benford draws at the nominal row count.  Reported ONLY to
      show how badly the naive null is miscalibrated (this is what a
      spreadsheet chi-square effectively assumes).

  matched_size_law
      draws from the fitted size law at the same country structure.  This is
      the modern matched null for the confirmation question ("is the panel what
      its own generating law says it should be?"), which the confirmation stage
      uses.

Every statistic (MAD, sup-distance M, |c1|, chi-square, Kuiper V) is calibrated
against the SAME simulated replicates, so the five readouts cannot drift apart
from each other and Kuiper V finally gets the bootstrap calibration the plan
requires (it is otherwise reported with no reference distribution at all).
"""

from __future__ import annotations

import numpy as np

import benford_core as bc


# --------------------------------------------------------------------------- #
# Sufficient statistics that can be accumulated over a mantissa sample
# --------------------------------------------------------------------------- #
def statistics_from_mantissas(m: np.ndarray, p_ref: np.ndarray,
                             n_nominal: float | None = None) -> dict:
    """All five conformity readouts from one mantissa sample.

    `n_nominal` is the row count to use in the chi-square denominator; it is
    passed separately because the SIMULATED null has the same row count as the
    data (so it is the data's n), while an entity-level observed sample has a
    different one.
    """
    m = np.asarray(m, float)
    n = int(len(m))
    d = np.clip(np.floor(10.0 ** m + 1e-12), 1, 9).astype(int)
    counts = np.bincount(d, minlength=10)[1:].astype(float)
    p = counts / counts.sum()
    n_chi = float(n_nominal) if n_nominal is not None else float(n)
    expected = n_chi * np.asarray(p_ref, float)
    obs_expected = n_chi * p
    chi2 = float(np.sum((obs_expected - expected) ** 2 / expected))
    f_obs = np.cumsum(p)
    f_ref = np.cumsum(np.asarray(p_ref, float))
    c1 = complex(np.mean(np.exp(2j * np.pi * m)))
    return {
        "n": n,
        "counts": counts,
        "p": p,
        "residuals_pp": 100.0 * (p - np.asarray(p_ref, float)),
        "MAD": float(np.mean(np.abs(p - np.asarray(p_ref, float)))),
        "M": float(np.max(np.abs(p - np.asarray(p_ref, float)))),
        "abs_c1": float(abs(c1)),
        "phase_deg": float(np.degrees(np.angle(c1))),
        "chi2": chi2,
        "kuiper_V": float(np.max(f_obs - f_ref) + np.max(f_ref - f_obs)),
        "c1_re": float(c1.real),
        "c1_im": float(c1.imag),
    }


STAT_KEYS = ("MAD", "M", "abs_c1", "chi2", "kuiper_V")


# --------------------------------------------------------------------------- #
# The clustered exact-Benford null
# --------------------------------------------------------------------------- #
def exact_benford_cluster_null(n_per_country: np.ndarray, within_sd: float,
                               p_ref: np.ndarray, B: int,
                               rng: np.random.Generator,
                               chunk: int = 250) -> dict:
    """Simulate `B` panels from exact Benford under the observed country structure.

    Each replicate: one exact-Benford mantissa per country, then `n_i` yearly
    deviations drawn from N(0, within_sd^2) in decades, wrapped into [0, 1).
    The result reproduces both the digit law and the panel's dependence
    structure, which is what makes the resulting p-values honest.

    Returns an (B, 5) matrix of statistics in STAT_KEYS order plus the
    per-digit residual matrix so sign patterns can be calibrated too.
    """
    n_per = np.asarray(n_per_country, int)
    u = len(n_per)
    total = int(n_per.sum())
    out = np.empty((B, len(STAT_KEYS)))
    resid = np.empty((B, 9))
    pad = int(n_per.max())
    # index that maps the padded (u, pad) block back to the ragged total
    keep = np.zeros((u, pad), dtype=bool)
    for i, ni in enumerate(n_per):
        keep[i, :ni] = True
    keep_flat = keep.ravel()

    done = 0
    while done < B:
        b_here = min(chunk, B - done)
        base = rng.random((b_here, u))                                  # exact Benford
        steps = rng.normal(0.0, within_sd, size=(b_here, u, pad))
        ms = (base[:, :, None] + steps) % 1.0
        flat = ms.reshape(b_here, -1)[:, keep_flat]
        d = np.clip(np.floor(10.0 ** flat + 1e-12), 1, 9).astype(np.int64)
        # digit counts per replicate
        for j in range(b_here):
            counts = np.bincount(d[j], minlength=10)[1:].astype(float)
            p = counts / total
            r = p - np.asarray(p_ref, float)
            resid[done + j] = 100.0 * r
            expected = total * np.asarray(p_ref, float)
            f_obs = np.cumsum(p)
            f_ref = np.cumsum(np.asarray(p_ref, float))
            z = np.mean(np.exp(2j * np.pi * flat[j]))
            out[done + j] = (
                np.mean(np.abs(r)),
                np.max(np.abs(r)),
                abs(z),
                np.sum((total * p - expected) ** 2 / expected),
                np.max(f_obs - f_ref) + np.max(f_ref - f_obs),
            )
        done += b_here
    return {"stats": out, "keys": STAT_KEYS, "residuals_pp": resid,
            "n_replicates": B, "n_countries": u, "n_rows": total}


def nominal_iid_null(n: int, p_ref: np.ndarray, B: int,
                     rng: np.random.Generator, chunk: int = 200) -> dict:
    """The naive null: i.i.d. draws at the NOMINAL row count.

    Reported to expose the miscalibration of the usual test, never to decide
    anything.  Valid only for p_ref = exact Benford, which is the null the
    literature's chi-square implicitly assumes.

    Mantissas are sampled at FULL resolution, one per observation, so every
    statistic here is the same estimator the observation uses and the two are
    commensurable.

    Exact Benford is the UNIFORM mantissa law: if m ~ U(0, 1) then
    P(first digit = d) = log10(d+1) - log10(d) = log10(1 + 1/d), which is Benford
    exactly, with no inverse-CDF transformation needed.  It is worth being
    explicit about this because the tempting inverse-CDF for a UNIFORM-DIGIT law,
    m = log10(1 + 9u), is a different distribution: it makes each digit equally
    likely (p_d = 1/9), not Benford.
    """
    p_ref = np.asarray(p_ref, float)
    expected = n * p_ref
    out = np.empty((B, len(STAT_KEYS)))
    done = 0
    while done < B:
        b_here = min(chunk, B - done)
        # exact Benford: the mantissa itself is U(0, 1)
        m = rng.random((b_here, n))
        d = np.clip(np.floor(10.0 ** m + 1e-12), 1, 9).astype(np.int64)
        counts = np.stack([np.bincount(row, minlength=10)[1:] for row in d]).astype(float)
        p = counts / n
        r = p - p_ref
        c1 = np.exp(2j * np.pi * m).mean(axis=1)
        f_obs = np.cumsum(p, axis=1)
        f_ref = np.cumsum(p_ref)
        out[done:done + b_here, 0] = np.mean(np.abs(r), axis=1)
        out[done:done + b_here, 1] = np.max(np.abs(r), axis=1)
        out[done:done + b_here, 2] = np.abs(c1)
        out[done:done + b_here, 3] = np.sum((counts - expected) ** 2 / expected, axis=1)
        out[done:done + b_here, 4] = (np.max(f_obs - f_ref, axis=1) +
                                      np.max(f_ref - f_obs, axis=1))
        done += b_here
    return {"stats": out, "keys": STAT_KEYS, "n_replicates": B}


def sampler_self_test(rng: np.random.Generator, n: int = 200_000) -> dict:
    """Verify that the null sampler reproduces Benford and that the tempting
    alternative does not.

    A uniform mantissa IS exact Benford, while `log10(1 + 9u)` makes digits
    equally likely.  Getting this backwards silently swaps the null for a
    uniform-digit law and would make every calibrated p-value wrong, so it is
    checked numerically at run time rather than trusted.
    """
    p_ok = np.log10(1.0 + 1.0 / np.arange(1, 10))
    m_uniform = rng.random(n)
    d_uniform = np.clip(np.floor(10.0 ** m_uniform + 1e-12), 1, 9).astype(int)
    p_uniform = np.bincount(d_uniform, minlength=10)[1:] / n
    m_wrong = np.log10(1.0 + 9.0 * rng.random(n))
    d_wrong = np.clip(np.floor(10.0 ** m_wrong + 1e-12), 1, 9).astype(int)
    p_wrong = np.bincount(d_wrong, minlength=10)[1:] / n
    mad_uniform = float(np.mean(np.abs(p_uniform - p_ok)))
    mad_wrong = float(np.mean(np.abs(p_wrong - p_ok)))
    ok = mad_uniform < 0.002 and mad_wrong > 0.02
    if not ok:
        raise AssertionError(
            "null sampler does not reproduce Benford: uniform-mantissa MAD "
            f"{mad_uniform:.4f} (want < 0.002), log10(1+9u) MAD {mad_wrong:.4f} "
            "(want > 0.02)")
    return {"uniform_mantissa_mad_vs_benford": mad_uniform,
            "uniform_digit_law_mad_vs_benford": mad_wrong,
            "n": n, "passes": True}


def calibrated_report(observed: dict, null: dict) -> dict:
    """Observed statistic vs its own simulated reference distribution."""
    stats = null["stats"]
    keys = null["keys"]
    block = {}
    for j, key in enumerate(keys):
        vals = stats[:, j]
        obs = float(observed[key])
        lo = float(np.mean(vals <= obs))
        hi = float(np.mean(vals >= obs))
        block[key] = {
            "observed": obs,
            "percentile": float(100.0 * np.mean(vals < obs)),
            "two_sided_p": float(min(1.0, 2.0 * min(lo, hi))),
            "null_mean": float(vals.mean()),
            "null_p2_5": float(np.percentile(vals, 2.5)),
            "null_p50": float(np.percentile(vals, 50.0)),
            "null_p97_5": float(np.percentile(vals, 97.5)),
            "null_min": float(vals.min()),
            "null_max": float(vals.max()),
            "observed_inside_95_envelope": bool(
                np.percentile(vals, 2.5) <= obs <= np.percentile(vals, 97.5)),
        }
    return block


# --------------------------------------------------------------------------- #
# Per-year model-fitted prediction
# --------------------------------------------------------------------------- #
def year_specific_prediction(panel, years, law_family: str, k: int,
                             seed: int, n_restarts: int = 6) -> dict:
    """Fit the winning family to EACH year's own cross-section and periodise.

    The plan asks for "predicted |c1(y)| and predicted phase from each year's
    own fitted law".  A single pooled fit cannot do this: its harmonic is
    attenuated to exp(-2*pi^2*sigma^2) and is essentially zero, so a pooled
    'prediction' has no year-to-year variation and correlating it with the
    observed series measures the time trend of the mean, not the size law.
    Fitting per year puts the prediction in the same units as the observation.
    """
    import size_laws as sl

    rng = np.random.default_rng(seed)
    rows = []
    for yr in years:
        sub = panel[panel.year == yr]
        y = sub.log10_pop.values
        if law_family == "lognormal_mixture":
            law = sl.fit_mixture(y, k, rng, n_restarts=n_restarts)
        elif law_family == "lognormal":
            law = sl.fit_lognormal(y)
        elif law_family == "dpln":
            law = sl.fit_dpln(y, rng, n_restarts=4)
        else:
            law = sl.fit_truncated_power_law(y)
        if law is None:
            rows.append({"year": int(yr), "status": "fit_failed"})
            continue
        p = law.digit_proportions()
        c = bc.c_from_digits(p, 1)
        obs = bc.c_harmonic(sub.mantissa.values, 1)
        rows.append({
            "year": int(yr),
            "status": "run",
            "n": int(len(sub)),
            "abs_c1_observed": float(abs(obs)),
            "phase_observed_deg": float(np.degrees(np.angle(obs))),
            "abs_c1_predicted": float(abs(c)),
            "phase_predicted_deg": float(np.degrees(np.angle(c))),
            "mad_observed_vs_benford": float(bc.mad_stat(
                bc.digit_proportions(sub.d1.values))),
            "mad_observed_vs_model": float(np.mean(np.abs(
                bc.digit_proportions(sub.d1.values) - p))),
            "sigma_geometric": float(np.exp(np.average(
                np.log(np.asarray(law.params["sigmas"], float)),
                weights=np.asarray(law.params["weights"], float))))
            if law.family == "lognormal_mixture" else float(law.params.get("sigma", np.nan)),
            "sd_log10_pop": float(np.std(y)),
        })
    return {"rows": rows}
