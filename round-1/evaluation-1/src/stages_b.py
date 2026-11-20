"""Stages 6-11: reserved-fold confirmation, projections, ablations, dose-response,
per-year series, source invariance, robustness, honest-negative branch."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats as sstats

import benford_core as bc
import null_models as nm
import size_laws as sl
from eval_utils import (
    jfloat, jlist, summarise, percentile_of, two_sided_p, bh_adjust,
    cluster_bootstrap_ci, cluster_bootstrap_residuals, digit_vector,
    moving_block_bootstrap_ci,
)
from stages_a import fold_statistics

MARGIN = 0.05
ATTENUATION_LIMIT = 1.25


# --------------------------------------------------------------------------- #
# Stage 6 -- reserved-fold confirmation
# --------------------------------------------------------------------------- #
def confirmation(odd: pd.DataFrame, even: pd.DataFrame, p_model_carried: np.ndarray,
                 m_carried: np.ndarray, ev: dict, od: dict) -> dict:
    """The primary decision: does the even-fold law carry over to the odd years?

    M = max_d |p_obs,reserved(d) - p_model,carried(d)| in PROBABILITY units (the
    plan states the margin as 0.05 in probability units, not percentage points).

    Pre-registered rule: SURVIVOR CONFIRMED iff
        M < 0.05 AND M's matched-null percentile in [2.5, 97.5]
        AND M_reserved / M_screen <= 1.25.
    Every component is reported separately whether or not the conjunction holds,
    and the per-digit breakdown names which digits drive M.
    """
    M_odd = float(np.max(np.abs(od["digit_proportions"] - p_model_carried)))
    M_even = float(np.max(np.abs(ev["digit_proportions"] - p_model_carried)))
    per_digit = np.abs(np.asarray(od["digit_proportions"]) - p_model_carried)
    driver = int(np.argmax(per_digit)) + 1
    ratio = jfloat(M_odd / M_even) if M_even > 0 else None

    pct = percentile_of(M_odd, m_carried)
    in_band = bool(2.5 <= pct <= 97.5) if pct is not None else False
    pass_margin = bool(M_odd < MARGIN)
    pass_ratio = bool(ratio is not None and ratio <= ATTENUATION_LIMIT)
    confirmed = bool(pass_margin and in_band and pass_ratio)

    return {
        "M_reserved": jfloat(M_odd),
        "M_screen": jfloat(M_even),
        "replication_attenuation_ratio": ratio,
        "attenuation_limit": ATTENUATION_LIMIT,
        "M_matched_null_percentile": pct,
        "M_matched_null_in_band_2_5_97_5": in_band,
        "margin_threshold_probability_units": MARGIN,
        "passes_margin": pass_margin,
        "passes_attenuation": pass_ratio,
        "per_digit_abs_gap": jlist(per_digit),
        "largest_gap_digit": driver,
        "largest_gap_value": jfloat(per_digit[driver - 1]),
        "n_reserved": od["n"],
        "n_screen": ev["n"],
        "decision": "SURVIVOR CONFIRMED" if confirmed else "NOT CONFIRMED",
        "rule_invoked": (
            "M_reserved < 0.05 in probability units AND M_reserved's percentile in the "
            "matched-null envelope within [2.5, 97.5] AND M_reserved / M_screen <= 1.25"
        ),
        "parameters_refit_by_evaluation": True,
        "provenance_note": (
            "The screen JSON was not staged, so the even-fold parameters were refitted "
            "here under the identical recipe. This is a refit-transfer, not a frozen "
            "carry-over: the even-fold M and the reserved-fold M are computed against the "
            "same parameter vector, but that vector is estimated from the even fold only "
            "and never sees an odd year."
        ),
    }


# --------------------------------------------------------------------------- #
# Stage 7 -- projected-residual and dose-response
# --------------------------------------------------------------------------- #
def projected_residuals(r_obs: np.ndarray, direction: np.ndarray, null_res: np.ndarray) -> dict:
    """One-df projection of the residual vector on the model-predicted direction.

    Both P and the cosine have non-zero null means (the nine residuals sum to zero),
    so their null means are ESTIMATED from the matched null rather than assumed.
    """
    obs = bc.projected_statistic(r_obs, direction)
    P_null = np.array([bc.projected_statistic(r, direction)["P"] for r in null_res])
    cos_null = np.array([bc.projected_statistic(r, direction)["cosine"] for r in null_res])
    return {
        "direction": jlist(direction / np.linalg.norm(direction)),
        "direction_norm": jfloat(np.linalg.norm(direction)),
        "P": jfloat(obs["P"]),
        "P_percentile": percentile_of(obs["P"], P_null),
        "P_two_sided_p": two_sided_p(obs["P"], P_null),
        "P_null": summarise(P_null),
        "cosine": jfloat(obs["cosine"]),
        "cosine_percentile": percentile_of(obs["cosine"], cos_null),
        "cosine_null_mean": jfloat(cos_null.mean()),
        "cosine_null_p97_5": jfloat(np.percentile(cos_null, 97.5)),
        "cosine_null": summarise(cos_null),
        "note": (
            "The null means of both statistics are estimated from the matched envelope; "
            "assuming 0 (for P) or 4.5/9 (for the cosine) would misstate the test."
        ),
    }


def width_dose_response(odd: pd.DataFrame, law: sl.SizeLaw,
                        p_model: np.ndarray, n_strata: int = 5) -> dict:
    """Grade the residual signal by the width of the country's size-law component.

    Each country is assigned to the mixture component it is most likely to come
    from under the even-fold fit (never refitting on the odd fold), and the strata
    are formed by that component's sigma (its width).  The claim being examined is
    that NARROW components carry the digit signal; a real mechanism produces a
    monotone relation between the stratum's narrow mass and its projected residual.
    """
    if law.family != "lognormal_mixture":
        return {"status": "not_run",
                "reason": f"winner is {law.family}, which has no narrow-component structure; "
                          "the dose-response needs a mixture"}
    w = np.asarray(law.params["weights"], float)
    mus = np.asarray(law.params["mus"], float)
    sgs = np.asarray(law.params["sigmas"], float)
    cols = [(np.log(max(wj, 1e-300)) + sstats.norm.logpdf(odd.log10_pop.values, mj, sj))
            for wj, mj, sj in zip(w, mus, sgs)]
    logp = np.column_stack(cols)
    resp = np.exp(logp - logp.max(axis=1, keepdims=True))
    resp = resp / resp.sum(axis=1, keepdims=True)
    # narrow mass: posterior weight on components with sigma below the median sigma
    narrow = sgs < np.median(sgs)
    narrow_mass = resp[:, narrow].sum(axis=1) if narrow.any() else np.zeros(len(odd))

    direction = p_model - bc.BENFORD_P
    frame = odd.copy()
    frame["narrow_mass"] = narrow_mass
    frame["stratum"] = pd.qcut(frame.narrow_mass, n_strata, labels=False, duplicates="drop")

    rows = []
    for s, sub in frame.groupby("stratum"):
        p = digit_vector(sub)
        r = 100.0 * (p - bc.BENFORD_P)
        proj = bc.projected_statistic(r, direction)
        rows.append({
            "stratum": int(s),
            "n": int(len(sub)),
            "n_countries": int(sub.code.nunique()),
            "mean_narrow_mass": jfloat(sub.narrow_mass.mean()),
            "mean_log10_pop": jfloat(sub.log10_pop.mean()),
            "residuals_pp": jlist(r),
            "projection_P": jfloat(proj["P"]),
            "cosine": jfloat(proj["cosine"]),
            "abs_c1": jfloat(abs(bc.c_harmonic(sub.mantissa.values, 1))),
            "mad_vs_model": jfloat(np.mean(np.abs(p - p_model))),
        })
    rows.sort(key=lambda d: d["mean_narrow_mass"])
    x = np.array([r["mean_narrow_mass"] for r in rows])
    y = np.array([r["projection_P"] for r in rows])
    rho, pval = sstats.spearmanr(x, y) if len(rows) > 2 else (np.nan, np.nan)
    return {
        "status": "run",
        "component_sigmas": jlist(sgs),
        "component_weights": jlist(w),
        "narrow_components": [int(i) for i in np.where(narrow)[0]],
        "n_strata": len(rows),
        "strata": rows,
        "spearman_rho_narrow_mass_vs_projection": jfloat(rho),
        "spearman_p": jfloat(pval),
        "interpretation": (
            "A positive rho means strata dominated by narrow mixture components show a "
            "larger projected residual in the predicted direction, which is the graded "
            "signature the selectivity principle requires. Five strata over the same data "
            "are graded evidence, not five independent tests."
        ),
    }


# --------------------------------------------------------------------------- #
# Stage 8 -- per-year harmonic series
# --------------------------------------------------------------------------- #
def per_year_series(panel: pd.DataFrame, odd: pd.DataFrame, even: pd.DataFrame,
                    law: sl.SizeLaw, rng: np.random.Generator, B: int = 2000,
                    year_predictions: list | None = None) -> dict:
    """Observed vs year-specific predicted |c_1(y)| across reserved years.

    Replaces 38 independent pass/fail decisions (which BH would largely null) with
    one correlation over 37 reserved years.

    The prediction MUST be year-specific or the correlation is meaningless: a
    single pooled fit has its first harmonic attenuated to
    exp(-2*pi^2*sigma^2) ~ 3e-5, so a pooled "prediction" is a constant to six
    decimals and correlating it with the observed series measures the time trend
    of the mean level, not the size law.  `year_predictions` therefore supplies
    the law fitted to each year's OWN cross-section (calibrated.
    year_specific_prediction); the constant is what it replaces and the
    difference is reported so the inflation is visible.
    """
    rows = []
    for yr in sorted(odd.year.unique()):
        sub = odd[odd.year == yr]
        c1 = bc.c_harmonic(sub.mantissa.values, 1)
        idx = bc.ClusterIndex.build(sub.code.values, sub.d1.values, sub.mantissa.values)
        boot = idx.boot_c1(rng, 400)
        rows.append({
            "year": int(yr),
            "n": int(len(sub)),
            "abs_c1_observed": jfloat(abs(c1)),
            "phase_observed_deg": jfloat(np.degrees(np.angle(c1))),
            "abs_c1_ci_low": jfloat(np.percentile(np.abs(boot), 2.5)),
            "abs_c1_ci_high": jfloat(np.percentile(np.abs(boot), 97.5)),
            "mean_log10_pop": jfloat(sub.log10_pop.mean()),
            "mad_observed_vs_benford": jfloat(bc.mad_stat(bc.digit_proportions(sub.d1.values))),
        })
    frame = pd.DataFrame(rows)

    # attach the year-specific prediction (its own cross-section's fitted law)
    pooled_prediction = abs(bc.c_from_digits(law.digit_proportions(), 1))
    if year_predictions:
        pred_map = {r["year"]: r for r in year_predictions if r.get("status") == "run"}
        frame["abs_c1_predicted"] = [pred_map.get(y, {}).get("abs_c1_predicted", np.nan)
                                     for y in frame.year]
        frame["phase_predicted_deg"] = [pred_map.get(y, {}).get("phase_predicted_deg", np.nan)
                                        for y in frame.year]
        frame["mad_predicted_vs_benford"] = [pred_map.get(y, {}).get("mad_observed_vs_benford", np.nan)
                                             for y in frame.year]
    else:
        frame["abs_c1_predicted"] = pooled_prediction
        frame["phase_predicted_deg"] = float(np.degrees(np.angle(bc.c_from_digits(
            law.digit_proportions(), 1))))
        frame["mad_predicted_vs_benford"] = np.nan

    # growth dispersion over the full panel, so the 2-year step is well defined
    full = panel.pivot_table(index="year", columns="code", values="log10_pop")
    growth_sd = full.diff().std(axis=1)
    frame = frame.merge(growth_sd.rename("growth_sd"), left_on="year", right_index=True,
                        how="left")
    for c in ("abs_c1_predicted", "phase_predicted_deg", "mad_predicted_vs_benford",
              "growth_sd"):
        frame[c] = pd.to_numeric(frame[c], errors="coerce")

    def corr_block(x, y, label):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < BLOCK + 3:
            return {"status": "insufficient_years", "n": int(ok.sum()), "label": label}
        if np.std(y[ok]) < 1e-15 or np.std(x[ok]) < 1e-15:
            return {"status": "undefined_constant_series", "label": label, "n": int(ok.sum()),
                    "note": ("this prediction has no year-to-year variation, so a "
                             "correlation with it is undefined -- which is itself the "
                             "finding: a pooled-fit prediction cannot be compared to a "
                             "per-year observed series")}
        pear = moving_block_bootstrap_ci(x[ok], y[ok], BLOCK, B, rng, "pearson")
        spear = moving_block_bootstrap_ci(x[ok], y[ok], BLOCK, B, rng, "spearman")
        return {"pearson": pear, "spearman": spear, "label": label, "n": int(ok.sum())}

    obs = frame.abs_c1_observed.values
    mod_block = corr_block(obs, frame.abs_c1_predicted.values,
                           "observed vs year-specific predicted |c1(y)|")
    # the prediction the year-specific fit REPLACES, for comparison
    constant_block = corr_block(obs, np.full(len(obs), pooled_prediction),
                                "observed vs a pooled constant prediction (what the "
                                "single-fold fit gives; differs only through the level trend)")

    growth_block = corr_block(obs, frame.growth_sd.values,
                              "observed |c1(y)| vs cross-country sd of log growth")
    # how the observed amplitude actually moves over the window
    year_block = corr_block(obs, frame.year.values, "observed |c1(y)| vs calendar year")

    def circ_lin(alpha_deg, beta_deg):
        a = np.radians(np.asarray(alpha_deg, float))
        b = np.radians(np.asarray(beta_deg, float))
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        if len(a) < 5:
            return None
        rxc = np.mean(np.cos(a) * np.cos(b))
        rxs = np.mean(np.cos(a) * np.sin(b))
        ryc = np.mean(np.sin(a) * np.cos(b))
        rys = np.mean(np.sin(a) * np.sin(b))
        rcc = np.mean(np.cos(a) ** 2) - np.mean(np.cos(a)) ** 2
        rss = np.mean(np.sin(a) ** 2) - np.mean(np.sin(a)) ** 2
        rcs = np.mean(np.cos(a) * np.sin(a)) - np.mean(np.cos(a)) * np.mean(np.sin(a))
        num = rxc * rxc + rxs * rxs + ryc * ryc + rys * rys
        den = rcc * rss - rcs * rcs
        return float(np.sqrt(max(num, 0) / max(den, 1e-30)))

    rho_cl = circ_lin(frame.phase_observed_deg.values, frame.phase_predicted_deg.values)
    ang_err = np.abs((frame.phase_observed_deg.values - frame.phase_predicted_deg.values
                      + 180) % 360 - 180)
    ang_err = ang_err[np.isfinite(ang_err)]

    pear = mod_block.get("pearson", {})
    r_val = pear.get("stat") if isinstance(pear, dict) else None
    achieved = bool(r_val is not None and r_val > 0 and
                    (pear.get("ci_low") or -1) > 0)
    return {
        "per_year_table": frame.replace({np.nan: None}).to_dict(orient="records"),
        "pooled_prediction_abs_c1_replaced": jfloat(pooled_prediction),
        "modulus_correlation": mod_block,
        "constant_prediction_correlation": constant_block,
        "growth_mechanism_check": growth_block,
        "year_trend_check": year_block,
        "circular_linear_phase_correlation": jfloat(rho_cl),
        "mean_absolute_phase_error_deg": jfloat(np.mean(ang_err)) if len(ang_err) else None,
        "pre_registered_criterion": {
            "stated_as": "r > 0 with one-sided p < 0.05 AND r >= r_screen - 0.2",
            "r_observed": jfloat(r_val),
            "one_sided_95_lower_bound_above_zero": achieved,
            "note": (
                "With 37 reserved years a correlation must exceed roughly 0.27 to reach "
                "one-sided p < 0.05; the criterion is reported as trend evidence with its "
                "block-bootstrap interval, never upgraded to a confirmation. The prediction "
                "is each year's OWN fitted size law, not the pooled fit: the pooled "
                "prediction is constant to six decimals and correlating with it would "
                "measure the level trend."
            ),
        },
        "status": "trend evidence, wide intervals",
    }


BLOCK = 5
