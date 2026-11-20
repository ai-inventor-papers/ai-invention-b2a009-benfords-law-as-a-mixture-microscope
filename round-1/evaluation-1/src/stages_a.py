"""Stages 0-6 of the evaluation: inventory, instrument, fits, power gate, nulls,
reserved-fold confirmation.

Split into a module so eval.py stays readable; eval.py owns the orchestration.
"""

from __future__ import annotations

import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

import benford_core as bc
import null_models as nm
import size_laws as sl
from eval_utils import (
    jcomplex, jfloat, jlist, summarise, percentile_of, two_sided_p,
    cluster_bootstrap_ci, cluster_bootstrap_residuals, digit_vector,
)

MARGIN = 0.05
ATTENUATION_LIMIT = 1.25
MICROSTATE_THRESHOLD = 1_000_000


# --------------------------------------------------------------------------- #
def fold_statistics(sub: pd.DataFrame, p_ref: np.ndarray | None = None,
                    n_eff: float | None = None) -> dict:
    """The full instrument on one set of rows: residuals, MAD, chi2, Kuiper, harmonics."""
    d1 = sub.d1.values.astype(int)
    counts = bc.digit_counts(d1)
    n = int(counts.sum())
    p_obs = counts / n
    ref = bc.BENFORD_P if p_ref is None else np.asarray(p_ref, float)
    r = 100.0 * (p_obs - ref)
    # chi-square and Kuiper must be measured against the SAME reference as the
    # residuals, not always against Benford: when p_ref is the fitted law the
    # whole point is that the null moved.
    chi = bc.chi2_stat(counts, p_ref=ref)
    kv = bc.kuiper_V(counts, p_ref=ref)
    ci = bc.simultaneous_intervals(counts, alpha=0.05, n_eff=n_eff,
                                   method="clopper_pearson")
    harm = bc.harmonic_summary(sub.mantissa.values)
    return {
        "n": n,
        "n_entities": int(sub.code.nunique()),
        "digit_counts": counts.tolist(),
        "digit_proportions": jlist(p_obs),
        "reference": "benford" if p_ref is None else "fitted_law",
        "residuals_pp": jlist(r),
        "MAD": jfloat(bc.mad_stat(p_obs, ref)),
        "MAD_nigrini_band": bc.nigrini_band(bc.mad_stat(p_obs, ref), n=n),
        "M": jfloat(np.max(np.abs(p_obs - ref))),
        "chi2": chi,
        "kuiper_V": jfloat(kv),
        "harmonics": {k: jfloat(v) for k, v in harm.items()},
        "simultaneous_intervals": ci,
        "mantissa_mean": jfloat(np.mean(sub.mantissa.values)),
        "mantissa_mad_from_half": jfloat(np.mean(np.abs(sub.mantissa.values - 0.5))),
    }


def instrument_even_odd(panel: pd.DataFrame) -> dict:
    """Split by year parity and prove the folds are disjoint country-year sets."""
    even = panel[panel.fold == "even"].reset_index(drop=True)
    odd = panel[panel.fold == "odd"].reset_index(drop=True)
    ek = set(map(tuple, even[["code", "year"]].values))
    ok = set(map(tuple, odd[["code", "year"]].values))
    shared = ek & ok
    if shared:
        raise RuntimeError(f"folds share {len(shared)} country-years; aborting")
    logger.info("fold split verified: {} even, {} odd, 0 shared", len(even), len(odd))
    return {
        "definition": "fold = 'even' if year % 2 == 0 else 'odd'; even = screen side, "
                      "odd = reserved confirmation side",
        "n_even": int(len(even)),
        "n_odd": int(len(odd)),
        "n_countries_even": int(even.code.nunique()),
        "n_countries_odd": int(odd.code.nunique()),
        "shared_country_years": 0,
        "years_even": [int(even.year.min()), int(even.year.max())],
        "years_odd": [int(odd.year.min()), int(odd.year.max())],
        "data": {"even": even, "odd": odd},
    }


def scale_rotation_check(panel: pd.DataFrame, factors=(1.5, 3.0, 7.0)) -> dict:
    """Multiply every published value by c; the phase must rotate by 360*log10(c).

    This is a validity check on the phase estimates: a real harmonic signal
    rotates exactly and keeps its modulus, whereas a numerical artefact does not.
    """
    rows = []
    base = bc.c_harmonic(panel.mantissa.values, 1)
    for c in factors:
        x = panel.population.values * c
        m = bc.mantissa(x)
        c1 = bc.c_harmonic(m, 1)
        predicted = (np.degrees(np.angle(base)) + 360.0 * np.log10(c)) % 360.0
        observed = np.degrees(np.angle(c1)) % 360.0
        err = abs((observed - predicted + 180) % 360 - 180)
        rows.append({
            "factor": float(c),
            "expected_rotation_deg": float(360.0 * np.log10(c)),
            "phase_before_deg": jfloat(np.degrees(np.angle(base))),
            "phase_after_deg": jfloat(observed),
            "phase_error_deg": jfloat(err),
            "modulus_before": jfloat(abs(base)),
            "modulus_after": jfloat(abs(c1)),
            "modulus_abs_change": jfloat(abs(abs(c1) - abs(base))),
        })
    return {"rows": rows,
            "identity_holds": bool(all(r["phase_error_deg"] < 1e-6 and
                                       r["modulus_abs_change"] < 1e-9 for r in rows))}
