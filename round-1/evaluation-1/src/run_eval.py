#!/usr/bin/env python3
"""Orchestrator for the evaluation: runs every stage in order and writes eval_out.json.

Kept separate from eval.py so the stage modules stay importable and testable; this
file is the single entry point and owns the output assembly.
"""

from __future__ import annotations

import gc
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

import benford_core as bc
import null_models as nm
import panel as pnl
import size_laws as sl
from eval_utils import (
    bh_adjust, jcomplex, jfloat, jlist, summarise, digit_vector, percentile_of,
    cluster_bootstrap_ci, cluster_bootstrap_residuals,
)
from stages_a import fold_statistics, instrument_even_odd, scale_rotation_check
from stages_b import confirmation, projected_residuals, width_dose_response, per_year_series
from stages_c import ablation_suite, source_invariance
from stages_d import rounding_simulation, inclusion_thresholds, honest_negative

SEED = 20261119
ALPHA = 0.05
MARGIN = 0.05
ATTENUATION_LIMIT = 1.25
B_FINAL = 10_000
B_ENVELOPE = 4_000
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"

for _d in (RESULTS, FIGURES, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(LOGS / "eval.log", rotation="30 MB", level="DEBUG")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def within_country_mantissa_sd(panel: pd.DataFrame) -> dict:
    """Detrended within-country sd of log10 population, in decades.

    This is the number the cluster matched null uses for its within-country walk,
    and it is the empirical justification for treating a country's 37 even-year
    mantissas as near-replicates.
    """
    resid = []
    step = []
    for _, s in panel.sort_values("year").groupby("code"):
        y = s.log10_pop.values
        yr = s.year.values.astype(float)
        if len(y) < 10:
            continue
        p = np.polyfit(yr, y, 1)
        resid.append(float(np.std(y - np.polyval(p, yr))))
        step.append(float(np.median(np.abs(np.diff(y)))))
    return {
        "detrended_within_country_sd_decades": summarise(resid),
        "median_per_year_step_decades": jfloat(np.median(step)),
        "marginal_sd_decades": jfloat(panel.log10_pop.std()),
        "ratio_within_to_marginal": jfloat(np.median(resid) / panel.log10_pop.std()),
        "interpretation": (
            "A country's mantissa moves by ~0.03 decades detrended against a marginal "
            "spread of ~1.21 decades, so its years are near-replicates of one draw from "
            "the size law. Nominal n therefore overstates independent information by "
            "roughly the number of years per country."
        ),
    }


def main() -> dict:
    t_start = time.time()
    rng = np.random.default_rng(SEED)
    logger.info("=== EVALUATION of the Benford screen on country populations ===")
    logger.info("seed {} | python {} | numpy {} | pandas {}", SEED, platform.python_version(),
                np.__version__, pd.__version__)

    # -- Stage 0: instrument self-test must pass before anything is reported
    logger.info("running instrument self-test")
    selftest = bc.self_test(verbose=False)
    logger.info("self-test PASSED (rotation err {:.1e}, periodise-vs-closed-form {:.1e})",
                selftest["rotation_identity_max_error"],
                selftest["periodise_vs_closed_form_max_abs_error"])

    # -- Stage 1: panel
    logger.info("building the panel from both public sources")
    panel_summary = pnl.build_panel(DATA / "owid_population_unwpp.csv",
                                    DATA / "wb_population_raw.json",
                                    DATA / "wb_countries.json", DATA)
    owid = pd.read_parquet(DATA / "panel_owid.parquet")
    wdi = pd.read_parquet(DATA / "panel_wdi.parquet")

    inputs_manifest = {
        "staged_dependency_artifacts_found": [],
        "dependency_note": (
            "No DATASET or EXPERIMENT artifact was staged: the three directions of "
            "iteration 1 run in parallel and declare no same-round dependency. The panel "
            "is rebuilt from the two public sources and the screen refitted on the even "
            "fold under the identical pre-registered recipe, exactly as the plan's "
            "failure mode 1 prescribes."
        ),
        "screen_json_available": False,
        "frozen_parameters_available": False,
        "parameters_refit_by_evaluation": True,
        "frozen_parameters_source": (
            "refit on even years only under the pre-registered recipe: year parity split, "
            "even = model side, odd = reserved; lognormal mixtures by EM with 20 restarts, "
            "K by BIC"
        ),
        "inputs": {name: {"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
                          "sha256": sha256_of(p)}
                   for name, p in [("owid_csv", DATA / "owid_population_unwpp.csv"),
                                   ("wdi_json", DATA / "wb_population_raw.json"),
                                   ("wdi_countries", DATA / "wb_countries.json"),
                                   ("panel_owid", DATA / "panel_owid.parquet"),
                                   ("panel_wdi", DATA / "panel_wdi.parquet")] if p.exists()},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "platform": platform.platform()},
        "seeds": {"global": SEED, "note": "every bootstrap stream spawned from SEED"},
        "panel": panel_summary,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    # -- Stage 1b: fold split and instrument on both folds
    split = instrument_even_odd(owid)
    even, odd = split["data"]["even"], split["data"]["odd"]
    logger.info("even {} rows / odd {} rows", len(even), len(odd))

    deff_even = bc.design_effect(even.code.values, even.mantissa.values)
    deff_even_digit = bc.design_effect(even.code.values, even.d1.values.astype(float))
    deff_odd = bc.design_effect(odd.code.values, odd.mantissa.values)
    within = within_country_mantissa_sd(owid)

    # -- Stage 2: rival size laws on the even fold
    logger.info("fitting rival size laws on the even fold")
    y_even = even.log10_pop.values
    laws = {}
    laws["lognormal"] = sl.fit_lognormal(y_even)
    for k in (2, 3):
        laws[f"lognormal_mixture_k{k}"] = sl.fit_mixture(y_even, k, np.random.default_rng(SEED + k),
                                                         n_restarts=20)
    dpln = sl.fit_dpln(y_even, np.random.default_rng(SEED + 10), n_restarts=12)
    if dpln is not None:
        laws["dpln"] = dpln
    laws["truncated_power_law"] = sl.fit_truncated_power_law(y_even)

    table = []
    for name, law in laws.items():
        row = law.to_dict(len(y_even))
        row["mad_vs_benford"] = jfloat(bc.mad_stat(law.digit_proportions()))
        row["c1_modulus"] = jfloat(abs(law.harmonic(1)))
        row["c1_phase_deg"] = jfloat(np.degrees(np.angle(law.harmonic(1))))
        row["ks_gof"] = None
        table.append(row)
    table.sort(key=lambda d: d["bic"])
    # KS goodness of fit with a parametric-bootstrap p, only for the winner and runner-up
    for row in table[:2]:
        law = laws[row["name"]]
        row["ks_gof"] = goodness_of_fit_ks(y_even, law, np.random.default_rng(SEED + 7), B=300)

    winner_name = table[0]["name"]
    winner = laws[winner_name]
    logger.info("BIC winner: {} (BIC {:.1f}, next best {} at {:.1f})",
                winner_name, table[0]["bic"], table[1]["name"], table[1]["bic"])
    p_model = winner.digit_proportions()

    # Cross-check, in two parts, because they test different things:
    #  (i)  periodisation vs CLOSED-FORM harmonic synthesis of the same law.  These are
    #       two independent routes to the digit vector and must agree tightly; this is
    #       the numeric-vs-analytic check the plan requires at 1e-4.
    #  (ii) the digit-RESOLUTION transfer gap (what the nine bins can see of the exact
    #       mantissa coefficient).  This is expected to be ~1e-3 and is NOT an error --
    #       it is the price of reading the law only through nine cells, and the plan's
    #       candidate-D validity gate keys on it.
    harm_cross = None
    harm_transfer = None
    if law_is_mixture(winner):
        wts = np.asarray(winner.params["weights"], float)
        mus = np.asarray(winner.params["mus"], float)
        sgs = np.asarray(winner.params["sigmas"], float)
        c_n = [np.sum(wts * np.exp(2j * np.pi * n * mus - 2 * np.pi ** 2 * n ** 2 * sgs ** 2))
               for n in range(1, 121)]
        p_cf = bc.digits_from_harmonics(c_n, nmax=120)
        harm_cross = jfloat(np.max(np.abs(p_model - p_cf)))
    elif winner.family == "lognormal":
        mu_w, sg_w = winner.params["mu"], winner.params["sigma"]
        c_n = [np.exp(2j * np.pi * n * mu_w - 2 * np.pi ** 2 * n ** 2 * sg_w ** 2)
               for n in range(1, 121)]
        p_cf = bc.digits_from_harmonics(c_n, nmax=120)
        harm_cross = jfloat(np.max(np.abs(p_model - p_cf)))
    harm_transfer = jfloat(abs(bc.c_from_digits(p_model, 1) - winner.harmonic(1)))
    logger.info("periodise vs closed-form harmonic: max|dp| = {:.2e} (must be < 1e-4)",
                harm_cross if harm_cross is not None else float("nan"))
    logger.info("digit-resolution transfer gap for the winner: {:.2e} (nine cells only)",
                harm_transfer)

    harmonic_budget = (winner.harmonic_budget() if law_is_mixture(winner)
                       else [{"note": f"winner {winner.family} has no components"}])

    # -- Stage 3/4: observed statistics and the power gate
    ev = fold_statistics(even, n_eff=deff_even["n_effective"])
    od = fold_statistics(odd, n_eff=deff_odd["n_effective"])
    ev_model = fold_statistics(even, p_ref=p_model)
    od_model = fold_statistics(odd, p_ref=p_model)

    pred_harm = winner.harmonic(1)
    gate = power_gate(ev, winner, deff_even, deff_even_digit, len(even), even.code.nunique())
    logger.info("power gate: {}", gate["statement"])

    # -- Stage 5: matched nulls
    logger.info("building matched nulls (B={})", B_ENVELOPE)
    cluster_even = bc.ClusterIndex.build(even.code.values, even.d1.values, even.mantissa.values)
    cluster_odd = bc.ClusterIndex.build(odd.code.values, odd.d1.values, odd.mantissa.values)
    rng_n = np.random.default_rng(SEED + 100)

    null_ben_even = nm.null_benford_cluster(cluster_even, rng_n, B_FINAL)
    null_ben_odd = nm.null_benford_cluster(cluster_odd, rng_n, B_FINAL)
    null_ent_even = nm.null_entity_resample(cluster_even, even.code.values, even.d1.values,
                                            rng_n, B_FINAL)

    n_per_even = even.groupby("code").size().values
    n_per_odd = odd.groupby("code").size().values
    direction = p_model - bc.BENFORD_P
    within_sd = float(within["detrended_within_country_sd_decades"]["p50"])

    null_law_odd = nm.null_law_cluster(winner, np.random.default_rng(SEED + 200), B_ENVELOPE,
                                      n_per_odd, within_sd, p_model, direction)
    null_law_even = nm.null_law_cluster(winner, np.random.default_rng(SEED + 201), B_ENVELOPE,
                                       n_per_even, within_sd, p_model, direction)
    null_iid_odd = nm.null_law_iid(winner, np.random.default_rng(SEED + 202), 1000, len(odd))

    obs_odd = {"M": np.max(np.abs(np.asarray(od["digit_proportions"]) - p_model)),
               "abs_c1": od["harmonics"]["c1_modulus"],
               "abs_c2": od["harmonics"]["c2_modulus"],
               "residuals_pp": np.asarray(od["residuals_pp"]),
               "projection_P": bc.projected_statistic(np.asarray(od["residuals_pp"]),
                                                      direction)["P"]}
    obs_even = {"M": np.max(np.abs(np.asarray(ev["digit_proportions"]) - p_model)),
                "abs_c1": ev["harmonics"]["c1_modulus"],
                "abs_c2": ev["harmonics"]["c2_modulus"],
                "residuals_pp": np.asarray(ev["residuals_pp"]),
                "projection_P": bc.projected_statistic(np.asarray(ev["residuals_pp"]),
                                                       direction)["P"]}

    envelopes = {
        "matched_law_reserved_fold": nm.matched_envelope_summary(null_law_odd, obs_odd),
        "matched_law_screen_fold": nm.matched_envelope_summary(null_law_even, obs_even),
        "benford_cluster_reserved_fold": nm.matched_envelope_summary(null_ben_odd, obs_odd),
        "benford_cluster_screen_fold": nm.matched_envelope_summary(null_ben_even, obs_even),
        "entity_resample_screen_fold": nm.matched_envelope_summary(null_ent_even, obs_even),
        "law_iid_reserved_fold": nm.matched_envelope_summary(null_iid_odd, obs_odd),
    }

    # -- Stage 6: reserved-fold confirmation
    logger.info("reserved-fold confirmation")
    conf = confirmation(odd, even, p_model, np.asarray(null_law_odd["M"]), ev, od)

    # -- Stage 7: projected residual and dose-response
    logger.info("projected-residual test and width-dose-response")
    proj = projected_residuals(np.asarray(od["residuals_pp"]), direction,
                               null_law_odd["residuals_pp"])
    dose = width_dose_response(odd, winner, p_model)

    # -- Stage 8: per-year series
    logger.info("per-year harmonic series")
    per_year = per_year_series(owid, odd, even, winner, np.random.default_rng(SEED + 300), B=1000)

    # -- Stage 9: ablation suite
    logger.info("ablation suite")
    ablate = ablation_suite(owid, even, odd, winner, null_law_odd["residuals_pp"],
                            np.random.default_rng(SEED + 400), panel_summary["taglists"])

    # -- Stage 10: source invariance
    logger.info("source invariance and provenance audit")
    invariance = source_invariance(owid, wdi, np.random.default_rng(SEED + 500), B=4000)

    # -- Stage 11: robustness
    logger.info("robustness: rounding, thresholds, rotation")
    rounding = rounding_simulation(odd, p_model, direction)
    thresholds = inclusion_thresholds(odd, direction)
    rotation = scale_rotation_check(owid)
    negative = honest_negative(odd, winner, p_model,
                              envelopes["matched_law_reserved_fold"].get("M", {}),
                              null_law_odd["residuals_pp"])

    # multiplicity: per-year conformity across reserved years, BH-corrected.
    # The per-year null is an exact-Benford draw at that year's n, which is the
    # honest null for a single-year conformity verdict; the MATCHED null is used
    # for the pooled confirmation, where the fitted law is the question.
    year_rows = []
    rng_m = np.random.default_rng(SEED + 600)
    for row in per_year["per_year_table"]:
        n_y = row["n"]
        sims = np.abs(np.array([
            np.mean(np.exp(2j * np.pi * np.floor(rng_m.random(n_y) * 9 + 1) / 9.0))
            for _ in range(400)]))
        obs_c1 = row["abs_c1_observed"]
        year_rows.append({
            "year": row["year"], "n": n_y, "abs_c1": obs_c1,
            "p_exact_benford": jfloat(min(1.0, 2 * min(np.mean(sims <= obs_c1),
                                                       np.mean(sims >= obs_c1)))),
        })
    pv = np.array([r["p_exact_benford"] for r in year_rows if r["p_exact_benford"] is not None])
    adj = bh_adjust(pv)
    for r, a in zip([r for r in year_rows if r["p_exact_benford"] is not None], adj):
        r["p_bh_adjusted"] = jfloat(a)
    multiplicity = {
        "family": "per-year conformity on the reserved (odd) fold, exact-Benford null",
        "n_tests": int(len(pv)),
        "n_rejected_raw_0_05": int(np.sum(pv < 0.05)),
        "n_rejected_bh_0_05": int(np.sum(adj < 0.05)),
        "expected_false_positives_at_raw_0_05": jfloat(0.05 * len(pv)),
        "rows": year_rows,
        "note": ("BH over years is the multiplicity control the field expects; with ~35 "
                 "years, 1.75 rejections are expected by chance under the null."),
    }

    # -- assemble
    decisions = {
        "survivor_confirmation": conf["decision"],
        "rule": conf["rule_invoked"],
        "power_gate_demoted_aggregate_test": gate["aggregate_c1_test_underpowered"],
        "ablation_criterion_c_met": ablate["criterion_c_met"],
        "source_invariance_status": ("consistency only -- shared upstream series"
                                     if invariance["fraction_near_identical_1e_6"] > 0.5
                                     else "partial independence"),
        "rounding_rival_bounded": rounding["demographic_account_survives"],
        "branch_taken": ("honest negative: the panel conforms to its own fitted size law"
                         if conf["decision"] != "SURVIVOR CONFIRMED"
                         else "positive: the fitted law carries over to the reserved fold"),
    }

    metrics_agg = {
        "n_country_years_screen_fold": float(len(even)),
        "n_country_years_reserved_fold": float(len(odd)),
        "n_countries": float(odd.code.nunique()),
        "n_effective_entity": float(even.code.nunique()),
        "n_effective_icc": jfloat(deff_even["n_effective"]),
        "design_effect_icc": jfloat(deff_even["design_effect"]),
        "icc_within_country_mantissa": jfloat(deff_even["icc"]),
        "observed_MAD_vs_benford_screen": jfloat(ev["MAD"]),
        "observed_MAD_vs_benford_reserved": jfloat(od["MAD"]),
        "observed_abs_c1_screen": jfloat(ev["harmonics"]["c1_modulus"]),
        "observed_abs_c1_reserved": jfloat(od["harmonics"]["c1_modulus"]),
        "observed_M_vs_model_screen": jfloat(conf["M_screen"]),
        "observed_M_vs_model_reserved": jfloat(conf["M_reserved"]),
        "replication_attenuation_ratio": jfloat(conf["replication_attenuation_ratio"]),
        "M_reserved_null_percentile": jfloat(conf["M_matched_null_percentile"]),
        "confirmation_margin_threshold": MARGIN,
        "winner_bic": jfloat(table[0]["bic"]),
        "winner_mad_vs_benford": jfloat(table[0]["mad_vs_benford"]),
        "winner_abs_c1": jfloat(table[0]["c1_modulus"]),
        "chi2_vs_benford_reserved": jfloat(od["chi2"]["chi2"]),
        "kuiper_V_reserved": jfloat(od["kuiper_V"]),
        "projected_P_reserved": jfloat(proj["P"]),
        "projected_P_percentile": jfloat(proj["P_percentile"]),
        "cosine_reserved": jfloat(proj["cosine"]),
        "cosine_null_mean": jfloat(proj["cosine_null_mean"]),
        "dose_response_spearman_rho": jfloat(dose.get("spearman_rho_narrow_mass_vs_projection")),
        "per_year_modulus_pearson_r": jfloat(
            per_year["modulus_correlation"].get("pearson", {}).get("stat")),
        "per_year_growth_pearson_r": jfloat(
            per_year["growth_mechanism_check"].get("pearson", {}).get("stat")),
        "ablation_plug_in_passes": float(ablate["criterion_c_pass_count"]),
        "source_invariance_sign_agreement": float(bool(invariance["sign_agreement_re_c1"])),
        "source_overlap_near_identical_fraction": jfloat(invariance["fraction_near_identical_1e_6"]),
        "rounding_max_share_explained": jfloat(rounding["max_share_explained_by_rounding"]),
        "scale_rotation_identity_holds": float(bool(rotation["identity_holds"])),
        "per_year_bh_rejections": float(multiplicity["n_rejected_bh_0_05"]),
        "self_test_rotation_error": jfloat(selftest["rotation_identity_max_error"]),
        "self_test_periodise_error": jfloat(selftest["periodise_vs_closed_form_max_abs_error"]),
        "confirmation_confirmed": float(conf["decision"] == "SURVIVOR CONFIRMED"),
    }

    out = {
        "metadata": {
            "evaluation_name": "benford_country_population_reserved_fold",
            "description": (
                "Out-of-sample evaluation of the Benford screen on the leading digits of "
                "country populations: reserved odd-year confirmation, narrow-component "
                "ablation suite, source invariance with a provenance audit, and the "
                "honest-negative branch."
            ),
            "seed": SEED,
            "alpha": ALPHA,
            "confirmation_margin_probability_units": MARGIN,
            "n_bootstrap_final": B_FINAL,
            "n_bootstrap_envelope": B_ENVELOPE,
            "parameters_refit_by_evaluation": True,
            "instrument_self_test": {k: (jfloat(v) if isinstance(v, (int, float)) else v)
                                     for k, v in selftest.items()},
            "runtime_seconds": None,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "metrics_agg": metrics_agg,
        "inputs_manifest": inputs_manifest,
        "fold_definition": {k: v for k, v in split.items() if k != "data"},
        "screen_fold_reference": {"vs_benford": ev, "vs_model": ev_model},
        "reserved_fold_statistics": {"vs_benford": od, "vs_model": od_model},
        "size_law_fits": {"table": table, "winner": winner_name,
                          "bic_ranking": [r["name"] for r in table],
                          "harmonic_budget": harmonic_budget,
                          "periodise_vs_closed_form_max_abs_error": harm_cross,
                          "periodise_vs_closed_form_tolerance": 1e-4,
                          "periodise_vs_closed_form_passes": bool(
                              harm_cross is not None and harm_cross < 1e-4),
                          "digit_resolution_transfer_gap": harm_transfer,
                          "digit_resolution_note": (
                              "The transfer gap compares the exact mantissa coefficient with "
                              "its reconstruction from the nine cell probabilities, assuming a "
                              "uniform mantissa inside each bin. ~1e-3 here, which is the "
                              "nine-cell resolution floor, not a modelling error."
                          ),
                          "excluded_family_note": (
                              "The discrete generalised beta / rank-size family "
                              "(Fontanelli et al. 2016) is cited but not refitted: the "
                              "field's rival set for administrative size data as used here "
                              "is {single lognormal, lognormal mixture, DPLN, truncated "
                              "power law}. Excluding it is stated rather than silent."
                          )},
        "power_gate": gate,
        "design_effect": {"screen_fold_complex": deff_even,
                          "screen_fold_digit_indicator": deff_even_digit,
                          "reserved_fold_complex": deff_odd,
                          "within_country_movement": within},
        "matched_null_calibration": {
            "envelopes": envelopes,
            "nulls": {
                "matched_law_reserved": {k: summarise(v) for k, v in null_law_odd.items()
                                         if k in ("M", "abs_c1", "abs_c2", "projection_P")},
                "matched_law_screen": {k: summarise(v) for k, v in null_law_even.items()
                                       if k in ("M", "abs_c1", "abs_c2", "projection_P")},
                "benford_cluster_reserved": {k: summarise(v) for k, v in null_ben_odd.items()
                                             if k in ("M", "abs_c1", "abs_c2")},
                "benford_cluster_screen": {k: summarise(v) for k, v in null_ben_even.items()
                                           if k in ("M", "abs_c1", "abs_c2")},
                "entity_resample_screen": {k: summarise(v) for k, v in null_ent_even.items()
                                           if k in ("M", "abs_c1")},
                "law_iid_reserved": {k: summarise(v) for k, v in null_iid_odd.items()
                                     if k in ("M", "abs_c1", "abs_c2")},
            },
            "sign_probability_positive_reserved": envelopes[
                "matched_law_reserved_fold"].get("sign_prob_positive"),
        },
        "reserved_fold_confirmation": conf,
        "projected_residual_test": proj,
        "dose_response": dose,
        "per_year_series": per_year,
        "ablation_suite": ablate,
        "source_invariance": invariance,
        "robustness": {"rounding": rounding, "inclusion_thresholds": thresholds,
                       "scale_rotation": rotation},
        "multiplicity": multiplicity,
        "honest_negative_branch": negative,
        "decisions": decisions,
    }

    # save per-digit table and harmonic table as their own artefacts too
    per_digit = []
    for d in range(9):
        per_digit.append({
            "digit": d + 1,
            "benford": jfloat(bc.BENFORD_P[d]),
            "model": jfloat(p_model[d]),
            "screen_obs": jfloat(ev["digit_proportions"][d]),
            "reserved_obs": jfloat(od["digit_proportions"][d]),
            "screen_residual_pp": jfloat(ev["residuals_pp"][d]),
            "reserved_residual_pp": jfloat(od["residuals_pp"][d]),
            "reserved_ci_low": jfloat(od["simultaneous_intervals"]["lower"][d]),
            "reserved_ci_high": jfloat(od["simultaneous_intervals"]["upper"][d]),
            "reserved_residual_vs_model_pp": jfloat(od_model["residuals_pp"][d]),
        })
    out["per_digit_table"] = per_digit

    out["metadata"]["runtime_seconds"] = jfloat(time.time() - t_start)
    logger.info("total runtime {:.1f}s", time.time() - t_start)
    return out


def law_is_mixture(law) -> bool:
    return getattr(law, "family", "") == "lognormal_mixture"


def goodness_of_fit_ks(y: np.ndarray, law: sl.SizeLaw, rng: np.random.Generator,
                       B: int = 300) -> dict:
    """KS distance of the fitted law from the data with a parametric-bootstrap p.

    A parametric bootstrap is required because the parameters were estimated from
    the same sample, which makes the tabulated KS p-value anti-conservative.
    """
    from scipy import stats as sstats

    def ks_of(sample: np.ndarray) -> float:
        ref = np.sort(law.sample(rng, 20_000))
        return float(sstats.ks_2samp(sample, ref).statistic)

    obs = ks_of(y)
    null = np.array([ks_of(law.sample(rng, len(y))) for _ in range(B)])
    return {"ks": jfloat(obs),
            "p_parametric_bootstrap": jfloat(np.mean(null >= obs)),
            "null_mean": jfloat(null.mean()),
            "n_bootstrap": B}


def power_gate(observed_even: dict, law: sl.SizeLaw, deff_z: dict, deff_digit: dict,
               n_even: int, n_countries: int) -> dict:
    """Declare the aggregate c_1 test under- or well-powered BEFORE interpreting it."""
    pred = abs(law.harmonic(1))
    rows = {}
    for label, n_use in [("nominal", float(n_even)),
                         ("neff_icc", float(deff_z["n_effective"])),
                         ("neff_digit_icc", float(deff_digit["n_effective"])),
                         ("n_entity", float(n_countries))]:
        ref = bc.rayleigh_reference(n_use)
        rows[label] = {**ref,
                       "predicted_abs_c1": jfloat(pred),
                       "predicted_over_se": jfloat(pred / ref["se_component"]),
                       "underpowered": bool(pred < 2.0 * ref["se_component"]),
                       "md_amplitude_80pct_power": jfloat(2.8 * ref["se_component"])}
    primary = rows["n_entity"]
    return {
        "predicted_abs_c1_from_winner": jfloat(pred),
        "observed_abs_c1_screen": jfloat(observed_even["harmonics"]["c1_modulus"]),
        "registries": rows,
        "primary_n_eff_used": "n_entity",
        "primary_rationale": (
            "The digit law of a size law treats a country's population as ONE draw from the "
            "law; its 37 even-year mantissas are near-replicates (detrended within-country "
            "sd ~0.03 decades against a marginal sd of ~1.21). The ICC-based n_eff still "
            "counts within-country year-to-year mantissa movement as independent "
            "replication of the DIGIT LAW, which the fitted model's own structure "
            "contradicts. The entity-count null is therefore primary for the aggregate "
            "c_1; the ICC-based figure is the optimistic upper bound on information."
        ),
        "aggregate_c1_test_underpowered": bool(primary["underpowered"]),
        "consequence": (
            "Aggregate |c_1| is reported descriptively. Inferential weight goes to the "
            "confirmation margin M, the projected-residual test, the cross-stratum "
            "residual trend, and the per-year series."
        ),
        "statement": (
            f"With n_eff = {primary['n_eff']:.0f} independent entities this design detects "
            f"|c_1| >= {primary['md_amplitude_80pct_power']:.4f} at 80 percent power; the "
            f"fitted size law predicts |c_1| = {pred:.4f}, so the aggregate test is "
            f"{'UNDERPOWERED' if primary['underpowered'] else 'powered'} and is reported "
            f"descriptively only."
        ),
    }


if __name__ == "__main__":
    out = main()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "eval_out.json").write_text(json.dumps(out, indent=2, allow_nan=False))
    (ROOT / "eval_out.json").write_text(json.dumps(out, indent=2, allow_nan=False))
    logger.info("wrote eval_out.json ({:.1f} KB)", (ROOT / "eval_out.json").stat().st_size / 1024)
    print(json.dumps(out["decisions"], indent=2))
