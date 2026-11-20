"""Stages 9-11: the ablation suite, source invariance with a provenance audit,
and the rounding / inclusion-threshold robustness checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats as sstats

import benford_core as bc
import size_laws as sl
from eval_utils import jfloat, jlist, summarise, percentile_of

MARGIN = 0.05


# --------------------------------------------------------------------------- #
# Stage 9 -- ablation suite
# --------------------------------------------------------------------------- #
def ablation_predictions(law: sl.SizeLaw, odd: pd.DataFrame, name: str,
                         mask: np.ndarray) -> dict:
    """Two predictions of the residual-vector SHIFT for deleting one sub-ensemble.

    Delta_plugin -- delete the deleted rows' mass from the mixture and renormalise
        the remaining components.  This is the MECHANISM statement: what the fitted
        generative account says the deletion should do to the digit law.
    Delta_refit  -- refit the same family on the ablated data and periodise.  This
        is the OBSERVABLE net effect arrived at the conservative way; refitting
        absorbs part of the deletion, so Delta_refit should be smaller.

    Both are computed on the EVEN-fold law (never refitting on odd years) except
    Delta_refit, which by construction refits on the ablated even rows.

    Because the odd-fold fit is not available, the plug-in deletion works on the
    mixture components: the deleted rows' posterior responsibility is removed and
    the remaining weights renormalised.  For a non-mixture winner the plug-in
    prediction is unavailable and only the refit rule is reported.
    """
    p_full = law.digit_proportions()
    out = {"ablation": name, "n_deleted_rows": int(mask.sum()),
           "fraction_deleted": jfloat(mask.mean()), "rules": {}}

    # --- plug-in (mixture only)
    if law.family == "lognormal_mixture":
        w = np.asarray(law.params["weights"], float)
        mus = np.asarray(law.params["mus"], float)
        sgs = np.asarray(law.params["sigmas"], float)
        # responsibility of the DELETED rows, evaluated under the even-fold fit
        # using their own log10 populations
        deleted_y = odd.log10_pop.values[mask]
        if len(deleted_y):
            cols = [np.log(max(wj, 1e-300)) + sstats.norm.logpdf(deleted_y, mj, sj)
                    for wj, mj, sj in zip(w, mus, sgs)]
            logp = np.column_stack(cols)
            resp = np.exp(logp - logp.max(axis=1, keepdims=True))
            resp = resp / resp.sum(axis=1, keepdims=True)
            deleted_share = resp.mean(axis=0)          # share of deleted mass per component
        else:
            deleted_share = np.zeros(len(w))
        w_abl = w - w * deleted_share
        if w_abl.sum() > 1e-9:
            w_abl = w_abl / w_abl.sum()
            p_plugin = sl.periodise_mixture(w_abl, mus, sgs)
            out["rules"]["plugin"] = {
                "available": True,
                "component_deletion_share": jlist(deleted_share),
                "weights_ablated": jlist(w_abl),
                "digit_proportions_ablated": jlist(p_plugin),
                "delta_pp": jlist(100.0 * (p_plugin - p_full)),
            }
        else:
            out["rules"]["plugin"] = {"available": False,
                                      "reason": "ablated weights vanish; plug-in undefined"}
    else:
        out["rules"]["plugin"] = {"available": False,
                                  "reason": f"winner is {law.family}; no mixture components to delete"}

    out["full_digit_proportions"] = jlist(p_full)
    return out


def delta_refit(law_family: str, y_abl: np.ndarray, rng, n_restarts: int = 12) -> dict:
    """Refit the same family on the ablated even-fold data and periodise."""
    try:
        if law_family == "lognormal":
            law2 = sl.fit_lognormal(y_abl)
        elif law_family == "lognormal_mixture":
            law2 = sl.fit_mixture(y_abl, 3, rng, n_restarts=n_restarts)
        elif law_family == "dpln":
            law2 = sl.fit_dpln(y_abl, rng, n_restarts=6)
            if law2 is None:
                return {"available": False, "reason": "ablation refit did not converge"}
        else:
            law2 = sl.fit_truncated_power_law(y_abl)
        return {"available": True, "digit_proportions": jlist(law2.digit_proportions()),
                "params": {k: (jfloat(v) if np.isscalar(v) else jlist(v))
                           for k, v in law2.params.items()}}
    except Exception as exc:                        # noqa: BLE001 - report, never crash
        logger.warning("delta_refit failed for {}: {}", law_family, exc)
        return {"available": False, "reason": str(exc)}


def score_agreement(delta_obs: np.ndarray, delta_pred: np.ndarray, null_res: np.ndarray,
                    direction_unit: np.ndarray) -> dict:
    """Sign agreement, cosine and projection ratio of an observed shift vs a predicted one."""
    d_obs = np.asarray(delta_obs, float)
    d_pred = np.asarray(delta_pred, float)
    n_sign = int(np.sum(np.sign(d_obs) == np.sign(d_pred)))
    npred = np.linalg.norm(d_pred)
    nobs = np.linalg.norm(d_obs)
    cosine = float(np.dot(d_obs, d_pred) / (nobs * npred)) if nobs > 0 and npred > 0 else None
    proj_ratio = float(np.dot(d_obs, d_pred) / np.dot(d_pred, d_pred)) if npred > 0 else None
    # null distribution of the cosine for a shift of this magnitude, taken from
    # the matched residual null (a shift has the same nine-cell geometry)
    if null_res is not None and len(null_res):
        r = np.asarray(null_res, float)
        cos_null = r @ d_pred / (np.linalg.norm(r, axis=1) * npred + 1e-30)
        null_p97 = float(np.percentile(cos_null, 97.5))
        null_mean = float(np.mean(cos_null))
    else:
        null_p97 = null_mean = None
    return {
        "sign_agreement_count": n_sign,
        "sign_agreement_of_nine": f"{n_sign}/9",
        "cosine": jfloat(cosine),
        "cosine_null_mean": jfloat(null_mean),
        "cosine_null_p97_5": jfloat(null_p97),
        "cosine_above_null_p97_5": bool(cosine is not None and null_p97 is not None
                                        and cosine > null_p97),
        "projection_ratio": jfloat(proj_ratio),
        "projection_ratio_in_0_5_1_5": bool(proj_ratio is not None and 0.5 <= proj_ratio <= 1.5),
        "observed_shift_norm": jfloat(nobs),
        "predicted_shift_norm": jfloat(npred),
    }


def ablation_suite(panel: pd.DataFrame, even: pd.DataFrame, odd: pd.DataFrame,
                   law: sl.SizeLaw, null_res_odd: np.ndarray, rng,
                   taglists: dict) -> dict:
    """Run all three pre-registered ablations, both prediction rules, both folds.

    The TAGS were computed by rule from the panel (see taglists) before any
    outcome was inspected.  The observed shift is measured on the RESERVED odd
    fold where the confirmation lives, and also on the even fold for reference.
    """
    definitions = {
        "microstates_le_1e6": lambda df: df.population.values <= 1_000_000,
        "sids": lambda df: df.is_sids.values,
        "top20": lambda df: df.is_top20.values,
    }
    results = {}
    y_even = even.log10_pop.values
    for name, fn in definitions.items():
        mask_even = fn(even)
        mask_odd = fn(odd)
        pred = ablation_predictions(law, odd, name, mask_odd)
        # refit rule on the ablated EVEN data
        refit = delta_refit(law.family, y_even[~mask_even], rng)
        if refit.get("available"):
            refit["delta_pp"] = jlist(100.0 * (np.asarray(refit["digit_proportions"]) -
                                               law.digit_proportions()))

        # observed shifts, odd fold (primary) and even fold (reference)
        obs_by_fold = {}
        for fold_name, df, mask in [("reserved_odd", odd, mask_odd),
                                    ("screen_even", even, mask_even)]:
            p_full = np.array(bc.digit_proportions(df.d1.values))
            p_abl = np.array(bc.digit_proportions(df.loc[~mask, "d1"].values))
            obs_by_fold[fold_name] = 100.0 * (p_abl - p_full)

        plugin = pred["rules"].get("plugin", {})
        scores = {}
        if plugin.get("available"):
            scores["plugin"] = score_agreement(obs_by_fold["reserved_odd"],
                                               np.asarray(plugin["delta_pp"]),
                                               null_res_odd, None)
            scores["plugin_even"] = score_agreement(obs_by_fold["screen_even"],
                                                    np.asarray(plugin["delta_pp"]),
                                                    None, None)
        if refit.get("available"):
            scores["refit"] = score_agreement(obs_by_fold["reserved_odd"],
                                              np.asarray(refit["delta_pp"]),
                                              null_res_odd, None)

        results[name] = {
            "definition": taglists.get(f"{name}_rule", name),
            "n_rows_deleted_even": int(mask_even.sum()),
            "n_rows_deleted_odd": int(mask_odd.sum()),
            "fraction_deleted_odd": jfloat(mask_odd.mean()),
            "plugin": plugin,
            "refit": refit,
            "observed_shift_pp": {k: jlist(v) for k, v in obs_by_fold.items()},
            "agreement": scores,
        }

    # pooled 27-cell sign count, explicitly descriptive with NO p-value
    pooled_sign = 0
    pooled_total = 0
    for name, r in results.items():
        sc = r["agreement"].get("plugin")
        if sc:
            pooled_sign += sc["sign_agreement_count"]
            pooled_total += 9
    passes = [name for name, r in results.items()
              if r["agreement"].get("plugin", {}).get("cosine_above_null_p97_5")
              and r["agreement"].get("plugin", {}).get("projection_ratio_in_0_5_1_5")]
    return {
        "ablations": results,
        "pooled_sign_agreement": {
            "count": f"{pooled_sign}/{pooled_total}" if pooled_total else "not_run",
            "numerator": pooled_sign,
            "denominator": pooled_total,
            "status": "descriptive only, no p-value attached (three ablations give a sign "
                      "test no power)",
        },
        "criterion_c_pass_count": len(passes),
        "criterion_c_passing_ablations": passes,
        "criterion_c_met": bool(len(passes) >= 2),
        "criterion_c_statement": (
            "at least 2 of 3 ablations have plug-in cosine above the matched null's 95th "
            "percentile AND projection ratio in [0.5, 1.5]"
        ),
        "taglists": taglists,
    }


# --------------------------------------------------------------------------- #
# Stage 10 -- source invariance and provenance audit
# --------------------------------------------------------------------------- #
def source_invariance(panel: pd.DataFrame, wdi: pd.DataFrame, rng,
                      B: int = 4000) -> dict:
    """Recompute |c_1|, phase and M on both sources; audit their true independence.

    The audit is the point: OWID's population series is built from WPP and WDI's
    SP.POP.TOTL largely reports the same national estimates, so 'cross-source'
    without an overlap count is a consistency check dressed as independence.  The
    report therefore counts identical and near-identical country-years and runs
    the invariance statistics on the NON-IDENTICAL subset when it is large enough.
    """
    m = panel[["code", "year", "population", "mantissa", "d1", "fold"]].merge(
        wdi[["code", "year", "population_wdi", "mantissa", "d1"]],
        on=["code", "year"], suffixes=("_owid", "_wdi"), how="inner")
    m["identical"] = np.isclose(m.population.values, m.population_wdi.values, rtol=0, atol=0)
    rel = np.abs(m.population.values - m.population_wdi.values) / np.maximum(
        m.population_wdi.values, 1.0)
    m["near_identical"] = rel < 1e-6
    m["rel_diff"] = rel

    overlap_frac = float(m.identical.mean()) if len(m) else 0.0
    non_ident = m[~m.near_identical]

    def stats_of(sub, mcol, dcol, label):
        if len(sub) < 30:
            return {"label": label, "status": "insufficient_rows", "n": int(len(sub))}
        c1 = bc.c_harmonic(sub[mcol].values, 1)
        c2 = bc.c_harmonic(sub[mcol].values, 2)
        counts = np.bincount(sub[dcol].values.astype(int), minlength=10)[1:]
        idx = bc.ClusterIndex.build(sub.code.values, sub[dcol].values, sub[mcol].values)
        boot = idx.boot_c1(rng, B)
        return {
            "label": label,
            "status": "run",
            "n": int(len(sub)),
            "n_entities": int(sub.code.nunique()),
            "c1_re": jfloat(c1.real),
            "c1_im": jfloat(c1.imag),
            "c1_modulus": jfloat(abs(c1)),
            "c1_phase_deg": jfloat(np.degrees(np.angle(c1))),
            "c2_modulus": jfloat(abs(c2)),
            "MAD_vs_benford": jfloat(bc.mad_stat(counts / counts.sum())),
            "M_vs_benford": jfloat(np.max(np.abs(counts / counts.sum() - bc.BENFORD_P))),
            "modulus_ci_low": jfloat(np.percentile(np.abs(boot), 2.5)),
            "modulus_ci_high": jfloat(np.percentile(np.abs(boot), 97.5)),
        }

    owid_all = stats_of(m, "mantissa_owid", "d1_owid", "OWID full overlap")
    wdi_all = stats_of(m, "mantissa_wdi", "d1_wdi", "WDI full overlap")
    owid_ni = stats_of(non_ident, "mantissa_owid", "d1_owid", "OWID non-identical subset")
    wdi_ni = stats_of(non_ident, "mantissa_wdi", "d1_wdi", "WDI non-identical subset")

    # per-year |c1| rank correlation across sources
    per_year = []
    for yr in sorted(m.year.unique()):
        s = m[m.year == yr]
        if len(s) < 30:
            continue
        per_year.append({
            "year": int(yr),
            "abs_c1_owid": jfloat(abs(bc.c_harmonic(s.mantissa_owid.values, 1))),
            "abs_c1_wdi": jfloat(abs(bc.c_harmonic(s.mantissa_wdi.values, 1))),
        })
    if len(per_year) > 5:
        a = np.array([p["abs_c1_owid"] for p in per_year])
        b = np.array([p["abs_c1_wdi"] for p in per_year])
        rho = sstats.spearmanr(a, b)
        rank_corr = {"spearman_rho": jfloat(rho.statistic), "p": jfloat(rho.pvalue),
                     "n_years": len(per_year), "pearson": jfloat(np.corrcoef(a, b)[0, 1])}
    else:
        rank_corr = {"status": "insufficient_years"}

    # sign agreement of Re(c1) across digits and across sources
    if owid_all.get("status") == "run" and wdi_all.get("status") == "run":
        sign_re_agree = bool(np.sign(owid_all["c1_re"]) == np.sign(wdi_all["c1_re"]))
        sign_im_agree = bool(np.sign(owid_all["c1_im"]) == np.sign(wdi_all["c1_im"]))
        amp_diff = abs(owid_all["c1_modulus"] - wdi_all["c1_modulus"])
    else:
        sign_re_agree = sign_im_agree = None
        amp_diff = None

    independent = len(non_ident)
    return {
        "overlap_country_years": int(len(m)),
        "overlap_entities": int(m.code.nunique()),
        "fraction_exactly_identical": jfloat(overlap_frac),
        "fraction_near_identical_1e_6": jfloat(m.near_identical.mean()),
        "median_relative_difference": jfloat(np.median(rel)) if len(rel) else None,
        "p95_relative_difference": jfloat(np.percentile(rel, 95)) if len(rel) else None,
        "provenance":
            "OWID population-unwpp is a redistribution of the UN World Population "
            "Prospects; World Bank SP.POP.TOTL reports national estimates from the same "
            "upstream statistical system. The two labels are therefore NOT independent "
            "measurements of the same quantity, and the invariance check below is a "
            "consistency check unless the non-identical subset is large.",
        "owid_overlap_subset": owid_all,
        "wdi_overlap_subset": wdi_all,
        "owid_non_identical_subset": owid_ni,
        "wdi_non_identical_subset": wdi_ni,
        "sign_agreement_re_c1": sign_re_agree,
        "sign_agreement_im_c1": sign_im_agree,
        "modulus_difference": jfloat(amp_diff),
        "per_year_rank_correlation": rank_corr,
        "n_genuinely_non_identical": int(independent),
        "genuinely_independent_subset_usable": bool(independent >= 30),
        "criterion": {
            "stated_as": "sign agreement of Re(c1) and per-year rank correlation >= 0.7",
            "sign_agreement_met": sign_re_agree,
            "rank_correlation_met": bool(
                isinstance(rank_corr, dict) and rank_corr.get("spearman_rho") is not None
                and rank_corr["spearman_rho"] >= 0.7),
        },
        "verdict": ("consistency across sources within the shared upstream series, "
                    "NOT demonstrated independence"),
    }
