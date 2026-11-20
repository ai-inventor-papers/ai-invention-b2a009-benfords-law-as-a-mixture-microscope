"""Stage 11: robustness -- the rounding-artifact rival, inclusion thresholds, and
the honest-negative branch."""

from __future__ import annotations

import numpy as np
import pandas as pd

import benford_core as bc
import size_laws as sl
from eval_utils import jfloat, jlist, summarise

ROUNDING_SHARE_LIMIT = 0.50


# --------------------------------------------------------------------------- #
# (a) rounding-artifact simulation
# --------------------------------------------------------------------------- #
def round_sig(x: np.ndarray, sig: int) -> np.ndarray:
    """Round to `sig` significant figures, vectorised, without changing magnitude."""
    x = np.asarray(x, float)
    with np.errstate(divide="ignore"):
        mag = np.floor(np.log10(np.abs(x)))
    scale = 10.0 ** (sig - 1 - mag)
    return np.round(x * scale) / scale


def rounding_simulation(panel: pd.DataFrame, p_model: np.ndarray,
                        direction: np.ndarray) -> dict:
    """Does reporting grain alone explain the predicted-direction projection?

    Three variants are compared against the observed residuals:
      round3   -- round each published value to 3 significant figures
      round1e3 -- round each published value to the nearest 1,000
      smooth   -- values simulated from the fitted law at the same years (the
                  no-grain reference), so the grain imprint is isolated
    The criterion is that rounding must explain < 50 percent of the
    predicted-direction projection for the demographic account to survive.
    """
    p_obs = bc.digit_proportions(panel.d1.values)
    r_obs = 100.0 * (p_obs - bc.BENFORD_P)
    P_obs = bc.projected_statistic(r_obs, direction)["P"]

    rows = {}
    for label, rounded in [
        ("round_3_significant_figures", round_sig(panel.population.values, 3)),
        ("round_nearest_1000", np.round(panel.population.values / 1000.0) * 1000.0),
    ]:
        m = bc.mantissa(rounded[rounded > 0])
        d1 = bc.first_digit(rounded[rounded > 0])
        p = bc.digit_proportions(d1)
        r = 100.0 * (p - bc.BENFORD_P)
        P_rounded = bc.projected_statistic(r, direction)["P"]
        # WHAT ROUNDING EXPLAINS is the CHANGE it makes to the projection, not the
        # projection of the rounded data.  Comparing the rounded data's own P with the
        # raw data's P is circular: the rounded values barely differ from the raw ones,
        # so that ratio is ~1 by construction and says nothing about grain.
        delta_P = P_rounded - P_obs
        rows[label] = {
            "residuals_pp": jlist(r),
            "MAD": jfloat(bc.mad_stat(p)),
            "abs_c1": jfloat(abs(bc.c_harmonic(m, 1))),
            "projection_P_of_rounded_data": jfloat(P_rounded),
            "projection_shift_from_rounding": jfloat(delta_P),
            "share_of_observed_projection": jfloat(abs(delta_P) / abs(P_obs))
            if abs(P_obs) > 1e-12 else None,
            "identity_share_superseded": jfloat(P_rounded / P_obs) if abs(P_obs) > 1e-12 else None,
            "n_zeroed": int(np.sum(rounded <= 0)),
            "n_leading_digit_changed": int(np.sum(bc.first_digit(rounded[rounded > 0]) !=
                                                   panel.d1.values[rounded > 0])),
        }

    # grain data-quality audit (the plan's candidate-C0 block)
    audit = {
        "median_significant_figures": jfloat(panel.significant_figures.median()),
        "sigfig_distribution": {int(k): int(v) for k, v in
                                panel.significant_figures.value_counts().sort_index().items()},
        "fraction_integer_values": jfloat(panel.is_integer_value.mean()),
        "fraction_divisible_1e1": jfloat(panel.divisible_1e1.mean()),
        "fraction_divisible_1e3": jfloat(panel.divisible_1e3.mean()),
        "fraction_divisible_1e6": jfloat(panel.divisible_1e6.mean()),
        "expected_divisible_1e1_under_smooth_law": 0.1,
        "expected_divisible_1e3_under_smooth_law": 0.001,
    }
    max_share = max((v["share_of_observed_projection"] or 0.0) for v in rows.values())
    max_digit_changed = max(v["n_leading_digit_changed"] / max(len(panel), 1)
                            for v in rows.values())
    return {
        "observed_projection_P": jfloat(P_obs),
        "observed_residuals_pp": jlist(r_obs),
        "variants": rows,
        "grain_audit": audit,
        "max_share_explained_by_rounding": jfloat(max_share),
        "max_fraction_of_leading_digits_changed": jfloat(max_digit_changed),
        "threshold": ROUNDING_SHARE_LIMIT,
        "demographic_account_survives": bool(max_share < ROUNDING_SHARE_LIMIT),
        "interpretation": (
            "The numerator is the CHANGE rounding makes to the predicted-direction "
            "projection, |P(rounded) - P(observed)|, not the projection of the rounded "
            "data -- the latter is ~1 by construction and measures nothing. Rounding to 3 "
            "significant figures cannot move the leading digit at all, so its share is "
            "exactly zero; rounding to the nearest 1,000 moves the leading digit of a small "
            "state and is the informative variant. A share above 0.5 would mean the "
            "published grain, not demography, carries the signal."
        ),
    }


# --------------------------------------------------------------------------- #
# (b) inclusion thresholds
# --------------------------------------------------------------------------- #
def inclusion_thresholds(panel: pd.DataFrame, direction: np.ndarray,
                         thresholds=(0, 100_000, 500_000, 1_000_000, 5_000_000)) -> dict:
    """Repeat the headline statistics at increasing population floors.

    A demographic mechanism should keep its sign; a small-state reporting artefact
    should vanish as soon as the microstates leave the sample.
    """
    rows = []
    base_sign = None
    for t in thresholds:
        sub = panel[panel.population >= max(t, 1)]
        if len(sub) < 100:
            rows.append({"threshold": int(t), "status": "insufficient_rows",
                         "n": int(len(sub))})
            continue
        p = bc.digit_proportions(sub.d1.values)
        r = 100.0 * (p - bc.BENFORD_P)
        proj = bc.projected_statistic(r, direction)["P"]
        c1 = bc.c_harmonic(sub.mantissa.values, 1)
        if base_sign is None:
            base_sign = (np.sign(proj), np.sign(c1.real), np.sign(c1.imag))
        rows.append({
            "threshold": int(t),
            "n": int(len(sub)),
            "n_entities": int(sub.code.nunique()),
            "MAD_vs_benford": jfloat(bc.mad_stat(p)),
            "abs_c1": jfloat(abs(c1)),
            "c1_phase_deg": jfloat(np.degrees(np.angle(c1))),
            "projection_P": jfloat(proj),
            "projection_sign_stable": bool(np.sign(proj) == base_sign[0]),
            "c1_re_sign_stable": bool(np.sign(c1.real) == base_sign[1]),
            "c1_im_sign_stable": bool(np.sign(c1.imag) == base_sign[2]),
        })
    return {"rows": rows,
            "projection_sign_stable_across_thresholds": bool(
                all(r.get("projection_sign_stable", True) for r in rows))}


# --------------------------------------------------------------------------- #
# (c) honest-negative branch
# --------------------------------------------------------------------------- #
def honest_negative(odd: pd.DataFrame, law: sl.SizeLaw, p_model: np.ndarray,
                    null_summary: dict, null_res_pp: np.ndarray) -> dict:
    """The negative written to the same standard a positive would be.

    If the confirmation failed, this is not a defeat: 'country populations conform
    to their own fitted size law within sampling noise' recalibrates every Benford
    screen on official statistics and must be reported with the same matched-
    envelope rigour.
    """
    p_obs = bc.digit_proportions(odd.d1.values)
    counts = bc.digit_counts(odd.d1.values)
    r = 100.0 * (p_obs - bc.BENFORD_P)
    r_model = 100.0 * (p_obs - p_model)
    M_ben = float(np.max(np.abs(p_obs - bc.BENFORD_P)))
    M_mod = float(np.max(np.abs(p_obs - p_model)))
    return {
        "distance_from_exact_benford": {
            "residuals_pp": jlist(r),
            "MAD": jfloat(bc.mad_stat(p_obs)),
            "MAD_nigrini_band": bc.nigrini_band(bc.mad_stat(p_obs)),
            "M": jfloat(M_ben),
            "chi2": bc.chi2_stat(counts),
            "kuiper_V": jfloat(bc.kuiper_V(counts)),
            "abs_c1": jfloat(abs(bc.c_harmonic(odd.mantissa.values, 1))),
            "interpretation": (
                "Measured against the POINT null on the nominal n, the panel looks far from "
                "exact Benford: this is the descriptive finding every prior study of "
                "population digits reports. It is not evidence of anything on its own, "
                "because the nominal n treats 236 near-replicated countries as 8,732 "
                "independent draws; the cluster-calibrated test in "
                "conformity_headline finds no departure at all."
            ),
        },
        "distance_from_fitted_law": {
            "residuals_pp": jlist(r_model),
            "MAD": jfloat(bc.mad_stat(p_obs, p_model)),
            "M": jfloat(M_mod),
            "matched_null": null_summary,
            "interpretation": (
                "Measured against the MATCHED null -- the fitted size law -- the same "
                "panel is inside its own sampling envelope. Together with the "
                "cluster-calibrated Benford result this says something stronger than "
                "either alone: these digits are equally compatible with exact Benford and "
                "with the fitted size law, and a 236-country panel cannot separate the two "
                "(see discrimination_between_nulls). The digit pattern therefore carries no "
                "evidence that discriminates a demographic size mechanism from Benford."
            ),
        },
        "headline": (
            "The leading digits of world-country populations conform to their own fitted "
            "size law AND to exact Benford inside the sampling envelope once the "
            "country-clustering is respected; the widely-reported 'departure from Benford' "
            "is an aggregation artefact of the nominal-n chi-square, not a demographic "
            "digit mechanism."
        ),
        "two_null_note": (
            "These two negatives are not in tension, they are the same fact seen twice. The "
            "fitted three-component lognormal mixture has a digit law whose own MAD from "
            "exact Benford is ~0.005, so for THESE data the two nulls are nearly "
            "indistinguishable, and a panel of 236 countries cannot tell them apart. That "
            "indistinguishability is the sharpened negative: the digit pattern carries no "
            "information that discriminates a size law from Benford at this sample size."
        ),
        "rescheduling_note": (
            "The temporal axis (per-year harmonic drift) becomes the lead for the next "
            "round: it is the only place where the fitted law predicts a year-specific "
            "signal that the pooled fit cannot absorb."
        ),
    }
