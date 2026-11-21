"""Orchestrator: runs every pre-registered stage and writes results/*.json.

Stages, in the pre-registered order:

  0  pre-registration already on disk, hashed
  1  panel load + integrity gate
  2  the five statistics (analysis/stats.py)
  3  reconciliation: two-row estimator table, algebra identity, kappa calibration
  4  derivation validation on synthetic panels (V1-V5) + calibration/power suite
  5  the battery on both folds through every instrument
  6  entity-disjoint folds, stability, power bound
  7  entity-level provenance identification lattice
  8  harmonic-value canonicalization of the mixture budget
  9  independent re-derivation
  10 schema-shaped method_out.json + figures + audit

Run with ``uv run method.py`` (or ``.venv/bin/python method.py``).  The ``--mini``
flag shrinks every replicate count for a fast end-to-end gate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from analysis import battery as bat  # noqa: E402
from analysis.canonical import resolve_recorded_values, select_and_fit  # noqa: E402
from analysis.panel_io import load_panel  # noqa: E402
from analysis.provenance import (  # noqa: E402
    circular_mean_candidates,
    declare_canonical,
    match_recorded,
    snapshot_candidates,
)
from analysis.reconcile import (  # noqa: E402
    estimator_table,
    kappa_calibration,
    pair_identifiability,
    threshold_table,
    within_country_wander,
)
from analysis.stats import (  # noqa: E402
    BENFORD,
    STAT_NAMES,
    RotationNull,
    rayleigh_thresholds,
    statistic_battery,
)
from analysis.synthetic import PanelSpec, generate  # noqa: E402
from analysis import validation as val  # noqa: E402
from analysis.verify import assert_matches, independent_report  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"
PANEL_PATH = ROOT / "data" / "full_data_out.json"
PREREG = RESULTS / "PRE_REGISTRATION_RECONCILIATION.md"

# The recorded targets this artifact reconciles against.  Every one is used only
# as a TEXT TARGET: nothing here is read from a prior workspace.
RECORDED = {
    "rho_defective": 0.9575,
    "DE_rho": 35.47,
    "n_eff_rho": 246.2,
    "ICC1": 0.2206,
    "DE_ICC": 8.94,
    "n_eff_ICC": 976.7,
    "abs_c1_even": 0.0749,
    "abs_c1_odd": 0.0752,
    "phase_even_rad": -1.5733,
    "threshold_screen": 0.0901,
    "threshold_eval": 0.0453,
    "mad_even": 0.00919,
    "mad_odd": 0.00935,
    "chi_even": 83.2,
    "chi_odd": 91.69,
    "supm_recorded": 0.03065,
    "kuiper_even": 0.0561,
    "kuiper_odd": 0.04205,
    "entity_abs_c1": 0.0736,
    "budget_a": 0.035767,
    "budget_b": 0.035474,
    "prov_a_mad": 0.01729,
    "prov_a_chi": 9.45,
    "prov_a_c1": 0.0736,
    "prov_b_mad": 0.02029,
    "prov_b_chi": 10.18,
    "prov_b_c1": 0.1043,
    "prov_vector": [65, 33, 35, 23, 17, 17, 21, 10, 15],
    "wander_sd": 0.028,
    "ablation_pass_prior": "0/3",
}

SCALE = {
    "mini": {"B_rotation": 300, "B_placebo": 200, "n_panels_v1": 20, "n_panels_v2": 20,
             "n_panels_power": 15, "n_panels_cal": 8, "n_boot": 500, "B_disjoint": 200},
    "full": {"B_rotation": 10_000, "B_placebo": 4_000, "n_panels_v1": 500,
             "n_panels_v2": 200, "n_panels_power": 200, "n_panels_cal": 100,
             "n_boot": 10_000, "B_disjoint": 2_000},
}


# --------------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, complex):
        return {"re": o.real, "im": o.imag}
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def fmt(x, nd=5):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.{nd}f}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------ stage 3
def stage3_reconciliation(panel, names, n_eff_icc_holder: dict) -> dict:
    logger.info("STAGE 3 reconciliation")
    idx = {n: i for i, n in enumerate(names)}
    per_fold = {}
    for fold in ("even", "odd"):
        rows = panel.by_fold(fold)
        m = np.array([r.mantissa for r in rows], float)
        ei = np.array([idx[r.entity] for r in rows])
        per_fold[fold] = estimator_table(m, ei, len(names))
        n_eff_icc_holder[fold] = per_fold[fold]["family_variate"]["n_eff_ICC_pooled"]

    groups = panel.entity_groups(None)
    wander = within_country_wander(groups)
    kc = kappa_calibration(groups, wander["median_sd_decades"], len(names))
    pair = pair_identifiability(37.0, RECORDED["rho_defective"], RECORDED["ICC1"])
    thresh = threshold_table(
        kc["observed_kappa2_mean"], len(names),
        per_fold["even"]["family_variate"]["n_eff_ICC_pooled"],
        per_fold["even"]["family_variate"]["n_eff_rho_pooled"],
    )
    # are the two recorded thresholds one formula at two n?
    consistency = []
    for label, neff, recorded_val in (
        ("screen", RECORDED["n_eff_rho"], RECORDED["threshold_screen"]),
        ("evaluation", RECORDED["n_eff_ICC"], RECORDED["threshold_eval"]),
    ):
        r2 = math.sqrt(2.0 / neff)
        consistency.append({
            "label": label, "n": neff,
            "r_2sigma_from_closed_form": r2,
            "recorded": recorded_val,
            "abs_diff": abs(r2 - recorded_val),
            "reproduced": bool(abs(r2 - recorded_val) < 5e-4),
        })
    return {
        "per_fold": per_fold,
        "wander": wander,
        "kappa_calibration": kc,
        "pair_identifiability": pair,
        "threshold_table": thresh,
        "threshold_consistency": {
            "rows": consistency,
            "all_reproduced": all(r["reproduced"] for r in consistency),
            "reading": "both recorded thresholds are r_2 = 2*sqrt(0.5/n) with kappa = 1 at the "
                       "two design-effect n values; they are ONE formula at two n, so the "
                       "contradiction is a choice-of-n problem and not two different theories. "
                       "Only n = n_countries carries a degrees-of-freedom meaning for a circular "
                       "mean.",
        },
        "recorded_targets": {k: RECORDED[k] for k in
                             ("rho_defective", "DE_rho", "n_eff_rho", "ICC1", "DE_ICC", "n_eff_ICC")},
    }


# ------------------------------------------------------------------ stage 5
def stage5_battery(panel, names, scale, wander) -> dict:
    logger.info("STAGE 5 battery on both folds, all instruments")
    n_eff_icc = {}
    out = {}
    for fold in ("even", "odd"):
        rows = panel.by_fold(fold)
        t0 = time.perf_counter()
        res = bat.battery_with_instruments(
            rows, names, B_rotation=scale["B_rotation"], B_placebo=scale["B_placebo"],
            seed=20240101 + (0 if fold == "even" else 1), wander=wander,
            n_eff_icc=1200.0,      # replaced below with this fold's own value
        )
        # recompute the analytic rows with this fold's own n_eff
        est = estimator_table(np.array([r.mantissa for r in rows], float),
                             np.array([names.index(r.entity) for r in rows]),
                             len(names))
        n_eff = est["family_variate"]["n_eff_ICC_pooled"]
        from analysis.stats import analytic_thresholds
        res["analytic"] = analytic_thresholds(res["observed_full"], res["sigma_R"],
                                             n_eff, float(res["n"]))
        res["n_eff_ICC"] = n_eff
        res["seconds"] = round(time.perf_counter() - t0, 1)
        res["verdict"] = bat.verdict_for(res)
        out[fold] = res
        gc.collect()

    # pooled
    t0 = time.perf_counter()
    pooled = bat.battery_with_instruments(
        panel.rows, names, B_rotation=scale["B_rotation"], B_placebo=scale["B_placebo"],
        seed=20240201, wander=wander, n_eff_icc=1200.0,
    )
    est = estimator_table(panel.mantissas(None),
                         np.array([names.index(r.entity) for r in panel.rows]), len(names))
    from analysis.stats import analytic_thresholds
    pooled["analytic"] = analytic_thresholds(pooled["observed_full"], pooled["sigma_R"],
                                             est["family_variate"]["n_eff_ICC_pooled"],
                                             float(pooled["n"]))
    pooled["n_eff_ICC"] = est["family_variate"]["n_eff_ICC_pooled"]
    pooled["seconds"] = round(time.perf_counter() - t0, 1)
    pooled["verdict"] = bat.verdict_for(pooled)
    out["pooled"] = pooled
    gc.collect()
    return out


# --------------------------------------------------------------- verdict text
def verdict_sentences(battery_out: dict) -> dict:
    """The quotable verdict sentence from whichever pre-committed branch is taken."""
    out = {}
    for label in ("even", "odd", "pooled"):
        r = battery_out[label]
        v = r["verdict"]
        a = v["abs_c1"]
        t2 = v["threshold_2sigma"]
        t5 = v["threshold_5pct"]
        mda = v["threshold_mda80"]
        if v["abs_c1_exceeds_2sigma"]:
            freq = ("all four frequency-type statistics are inside the same null"
                    if all(v["per_statistic"][k]["consistent"]
                           for k in ("MAD", "chi_square", "sup_distance", "kuiper_V"))
                    else "at least one frequency-type statistic is also outside it")
            s = (
                f"[{label}] The panel EXCEEDS its cluster envelope on the circular-mean statistic: "
                f"|c1| = {a:.5f} against a corrected two-sigma threshold of {t2:.5f} "
                f"(5% threshold {t5:.5f}; 80%-power MDA {mda:.5f}), with observed within-country "
                f"concentration mean kappa = {r['kappa_mean']:.4f}. {freq.capitalize()}, so the "
                f"departure is specific to the BETWEEN-country phase alignment and not to the "
                f"pooled digit histogram. The phase is clustered tightly near "
                f"{r['observed_full']['c1_phase_deg']:.1f} degrees (that is, near a mantissa of "
                f"{(r['observed_full']['c1_phase_deg'] % 360.0) / 360.0:.3f}), which is the "
                f"signature of a shared multiplicative scale: countries that sit at the same "
                f"point within their decade, decade after decade, for reasons that are "
                f"arithmetic (a size law whose mass sits near a fixed mantissa) rather than "
                f"stochastic."
            )
        else:
            s = (
                f"[{label}] The panel is INSIDE its cluster envelope on the circular-mean statistic: "
                f"|c1| = {a:.5f} against a corrected two-sigma threshold of {t2:.5f} "
                f"(5% threshold {t5:.5f}; 80%-power MDA {mda:.5f}), with observed within-country "
                f"concentration mean kappa = {r['kappa_mean']:.4f}. All five frequency-type and "
                f"circular statistics are consistent with exact Benford."
                if v["all_consistent"] else
                f"[{label}] The panel is INSIDE its cluster envelope on the circular-mean statistic "
                f"(|c1| = {a:.5f} vs two-sigma {t2:.5f}) but at least one frequency-type statistic "
                f"is not consistent with exact Benford under the rotation null."
            )
        out[label] = {
            "sentence": s,
            "branch_taken": "EXCEEDS" if v["abs_c1_exceeds_2sigma"] else "WITHIN",
            "abs_c1": a,
            "threshold_2sigma": t2,
            "threshold_5pct": t5,
            "mda80": mda,
            "all_consistent": v["all_consistent"],
            "untaken_branch": (
                f"had |c1| exceeded {t2:.5f} the EXCEEDS verdict would have been reported; "
                f"it is {a:.5f}, so the WITHIN branch is taken"
                if not v["abs_c1_exceeds_2sigma"] else
                f"had |c1| stayed at or below {t2:.5f} the WITHIN verdict would have been "
                f"reported; it is {a:.5f}, so the EXCEEDS branch is taken"
            ),
        }
    return out


# ------------------------------------------------------------------- figures
def write_figure_specs(battery_out: dict, recon: dict, power: dict,
                       thresholds: dict) -> dict:
    """Chart specs (generator-valid) plus the raw values behind every panel.

    Each spec is written to ``figures/<name>.spec.json`` in exactly the shape
    ``chart_gen.py --spec`` accepts -- ONLY keys that type reads, because the
    generator refuses (rather than drops) an unread key.  The numbers behind the
    figure are written beside it to ``figures/<name>.values.json``, so every
    caption claim can be re-checked from a file without reading the picture.
    """
    FIGURES.mkdir(exist_ok=True)
    specs: dict[str, dict] = {}
    values_by_fig: dict[str, dict] = {}

    # (1) Rayleigh null density with the observed |c1| marked as an extra series
    rows = []
    for label in ("even", "odd"):
        r = battery_out[label]
        rows.append({"label": label, "sigma_R": r["sigma_R"],
                     "observed": r["observed"]["abs_c1"],
                     "t2": r["thresholds"]["r_2sigma"],
                     "t5": r["thresholds"]["r_5pct"]})
    rs = np.linspace(0.0, max(x["t5"] for x in rows) * 1.9, 240)
    series = []
    for x in rows:
        dens = (rs / x["sigma_R"] ** 2) * np.exp(-(rs**2) / (2 * x["sigma_R"] ** 2))
        series.append({"label": f"{x['label']} fold null",
                       "x": rs.tolist(), "values": dens.tolist(),
                       "linestyle": "-"})
    # the observed amplitudes as flat markers on their own small series
    for x in rows:
        series.append({
            "label": f"{x['label']} observed |c1| = {x['observed']:.4f}",
            "x": [x["observed"]], "values": [0.0],
            "marker": "v",
        })
    specs["fig1_rayleigh_null"] = {
        "type": "line",
        "title": "Rayleigh cluster null for the first circular harmonic",
        "xlabel": "|c1| (circular-mean amplitude on mantissas)",
        "ylabel": "Rayleigh density",
        "series": series,
    }
    values_by_fig["fig1_rayleigh_null"] = {"folds": rows}

    # (2) p-value heatmap: 5 statistics x 2 folds, three instruments
    stat_order = ["MAD", "chi_square", "sup_distance", "kuiper_V", "abs_c1"]
    values, row_labels = [], []
    for label in ("even", "odd"):
        r = battery_out[label]
        for stat in stat_order:
            p_r = r["rotation"][stat]["p"]
            if stat == "abs_c1":
                p_a = r["analytic"]["circular"]["rayleigh_p"]
            elif stat == "chi_square":
                p_a = r["analytic"]["effective"]["chi_square_p"]
            elif stat == "kuiper_V":
                p_a = r["analytic"]["effective"]["kuiper_p"]
            elif stat == "sup_distance":
                p_a = r["analytic"]["effective"]["sup_distance_p"]
            else:
                p_a = float("nan")
            p_c = r["placebo"][stat]["p"] if r["placebo"] else float("nan")
            values.append([p_r, p_a if np.isfinite(p_a) else 0.0,
                           p_c if np.isfinite(p_c) else 0.0])
            row_labels.append(f"{label} | {stat}")
    specs["fig2_pvalue_heatmap"] = {
        "type": "heatmap",
        "title": "p-values by statistic, fold and instrument",
        "xlabel": "instrument",
        "ylabel": "fold and statistic",
        "row_labels": row_labels,
        "col_labels": ["rotation null (operative)", "analytic cluster null",
                       "parametric placebo (legacy)"],
        "matrix": values,
        "cbar_label": "p-value",
    }
    values_by_fig["fig2_pvalue_heatmap"] = {
        "cells": [{"row": l, "p_rotation": v[0], "p_analytic": v[1],
                   "p_placebo": v[2]} for l, v in zip(row_labels, values)],
        "columns": ["rotation null (operative)", "analytic cluster null",
                    "parametric placebo (legacy)"],
    }

    # (3) power curve, with the observed amplitude as a labelled marker
    curve = power["curve"]
    amp_series = [{"label": "country-rotation null",
                   "x": [c["planted_amplitude"] for c in curve],
                   "values": [c["rejection_rate"] for c in curve],
                   "linestyle": "-"}]
    obs_amp = battery_out["even"]["observed"]["abs_c1"]
    specs["fig3_power_curve"] = {
        "type": "line",
        "title": "Rejection rate of |c1| against planted amplitude",
        "xlabel": "planted first-harmonic amplitude A",
        "ylabel": "rejection rate at alpha = 0.05",
        "series": amp_series
        + [{"label": f"observed amplitude = {obs_amp:.4f}",
            "x": [obs_amp], "values": [0.0], "marker": "v"}],
    }
    values_by_fig["fig3_power_curve"] = {"curve": curve,
                                         "observed_amplitude": obs_amp}

    # (4) reconciliation: r_2 against the n it is divided by
    tt = recon["threshold_table"]
    specs["fig4_threshold_vs_n"] = {
        "type": "line",
        "title": "Two-sigma threshold against the n it is divided by",
        "xlabel": "n used in r_2 = 2 sqrt(0.5 kappa^2 / n)",
        "ylabel": "two-sigma threshold on |c1|",
        "logx": True,
        "series": [{"label": "closed form",
                    "x": [r["n"] for r in tt],
                    "values": [r["r_2sigma"] for r in tt],
                    "linestyle": "-"}],
    }
    values_by_fig["fig4_threshold_vs_n"] = {"thresholds": tt}

    for name, spec in specs.items():
        write_json(FIGURES / f"{name}.spec.json", spec)
        write_json(FIGURES / f"{name}.values.json", values_by_fig[name])
    return specs


def sup_distance_definition_scan(panel) -> dict:
    """Which sup-distance definition reproduces the recorded 0.03065?

    A pure hygiene check, run because a cumulative sup-distance and a maximum
    single-digit deviation are different statistics with different nulls and the
    earlier rounds reported only the number.  Every candidate definition is
    computed on both folds; the recorded value is compared to each, and the
    closest is named.
    """
    from analysis.stats import BENFORD, BENFORD_CUM, DIGIT_LOWER, DIGIT_UPPER

    out: dict = {"recorded_target": 0.03065, "folds": {}}
    for fold in ("even", "odd"):
        m = np.sort(panel.mantissas(fold))
        n = m.size
        cnt = np.array([(np.searchsorted(DIGIT_LOWER, m, side="right") - 1 == j).sum()
                        for j in range(9)], float)
        p = cnt / n
        cum = np.cumsum(p)
        f_up = np.arange(1, n + 1) / n
        f_lo = np.arange(0, n) / n
        d_plus = float(np.max(f_up - m))
        d_minus = float(np.max(m - f_lo))
        out["folds"][fold] = {
            "ks_two_sided_on_mantissas": max(d_plus, d_minus),
            "ks_d_plus": d_plus,
            "ks_d_minus": d_minus,
            "kuiper_on_mantissas": d_plus + d_minus,
            "cumulative_digit_sup_vs_benford_cum": float(np.max(np.abs(cum - BENFORD_CUM))),
            "cumulative_digit_sup_vs_decade_upper": float(
                np.max(np.abs(cum - np.log10(np.arange(2.0, 11.0))))),
            "cumulative_digit_sup_vs_decade_lower": float(
                np.max(np.abs(cum - np.log10(np.arange(1.0, 10.0))))),
            "digit_kuiper_vs_benford_cum": float(
                np.max(cum - BENFORD_CUM) + np.max(BENFORD_CUM - cum)),
            "max_single_digit_deviation": float(np.max(np.abs(p - BENFORD))),
        }
    best = []
    for fold, vals in out["folds"].items():
        for name, val in vals.items():
            best.append((abs(val - out["recorded_target"]) / out["recorded_target"],
                         fold, name, val))
    best.sort()
    out["closest"] = [{"rel_err": round(r, 5), "fold": f, "definition": nm, "value": v}
                      for r, f, nm, v in best[:4]]
    closest = out["closest"][0]
    if closest["rel_err"] < 5e-3:
        out["verdict"] = (
            f"the recorded {out['recorded_target']} is NOT a cumulative sup-distance: it is the "
            f"{closest['fold']} fold's {closest['definition']} ({closest['value']:.6f}, "
            f"{closest['rel_err']:.2e} relative). The cumulative sup-distance (KS to uniform on "
            f"mantissas) is {out['folds'][closest['fold']]['ks_two_sided_on_mantissas']:.5f} on "
            f"that fold. Reported as a definitional finding, not tuned to match."
        )
    else:
        out["verdict"] = (
            f"no candidate definition reproduces the recorded {out['recorded_target']}; the "
            f"closest is {closest['definition']} on the {closest['fold']} fold at "
            f"{closest['value']:.6f}. The recorded value is superseded and unverifiable."
        )
    return out


def _nominal_p_for(stat: str, r: dict):
    """The conventional uncorrected p for a statistic, from its fold's battery."""
    if stat == "chi_square":
        return r["nominal"]["chi_square_p_df8"]
    if stat in ("abs_c1", "MAD"):
        return r["nominal"]["rayleigh_p_at_n_rows"]
    return None


def _fallback_rotation(row_type: str, meta: dict) -> str:
    """The operative method's answer, per row type, when none is passed."""
    m = meta or {}
    if row_type in ("statistic", "statistic_circular"):
        return (f"rotation-null p = {fmt(m.get('p_rotation'), 4)}; "
                f"consistent with exact Benford: {m.get('consistent')}")
    if row_type == "statistic_nominal":
        return "not applicable to the nominal-n baseline row itself"
    if row_type == "verdict":
        return (f"branch {m.get('branch')}; |c1| = {fmt(m.get('abs_c1'), 5)} vs "
                f"two-sigma {fmt(m.get('threshold_2sigma'), 5)}")
    if row_type == "disjoint_fold":
        return (f"|c1| = {fmt(m.get('abs_c1'), 5)}; rotation p = "
                f"{fmt(m.get('p_rotation'), 4)}")
    if row_type == "reconciliation_estimator":
        return f"declared value {fmt(m.get('value'), 6)}; governs: {m.get('statistic_class_governed')}"
    if row_type == "calibration_suite":
        return f"pass = {m.get('pass_flag')}"
    if row_type == "derivation_validation":
        return f"pass = {m.get('pass_flag')}; achieved {m.get('achieved')}"
    if row_type == "power_bound":
        return f"required n_countries {fmt(m.get('required_n_countries'), 1)} vs {m.get('available_n_countries')}"
    if row_type == "provenance":
        return f"{m.get('status')}"
    if row_type == "canonical_harmonic":
        return f"A_canon = {fmt(m.get('A_canon'), 6)}"
    if row_type == "case_audit":
        return f"kappa = {fmt(m.get('kappa'), 5)}; influence {fmt(m.get('influence'), 5)}"
    if row_type == "robustness":
        return f"passes {m.get('passes_count')}/{m.get('denominator')}"
    return "see the output row"


def _fallback_nominal(row_type: str, meta: dict) -> str:
    """The conventional baseline's answer, per row type, when none is passed."""
    m = meta or {}
    if row_type == "statistic":
        return (f"nominal chi-square p (df 8, no correction) = see the fold's "
                f"statistic_nominal row; the conventional verdict applies the "
                f"nominal p without any dependence correction")
    if row_type == "verdict":
        return "the conventional battery rejects on the uncorrected nominal p"
    if row_type == "reconciliation_estimator":
        return "the ratio of mean squares rho = MSB/(MSB+MSW), the conventional screen"
    return "not applicable to this row type"


def ensure_predict_fields(doc: dict) -> dict:
    """Guarantee every example carries the method-versus-baseline predictions.

    The pipeline emits ``predict_*`` fields as it builds each row, but a schema
    check also demands them on any row that predates that code, so this pass
    fills in any that are missing from the row's own metadata.  It is idempotent
    and derives nothing that is not already in the row, so it cannot disagree
    with the prose ``output``.
    """
    for ds in doc.get("datasets", []):
        for ex in ds.get("examples", []):
            meta = {k[len("metadata_"):]: v for k, v in ex.items()
                    if k.startswith("metadata_")}
            row_type = str(meta.get("row_type", ""))
            if "predict_country_rotation_null" not in ex:
                ex["predict_country_rotation_null"] = _fallback_rotation(row_type, meta)
            if "predict_nominal_baseline" not in ex:
                ex["predict_nominal_baseline"] = _fallback_nominal(row_type, meta)
            if ("predict_analytic_cluster_null" not in ex
                    and meta.get("p_analytic") is not None
                    and str(meta.get("p_analytic")) not in ("None", "nan")):
                ex["predict_analytic_cluster_null"] = f"p = {meta['p_analytic']}"
            if ("predict_parametric_placebo" not in ex
                    and meta.get("p_placebo") is not None
                    and str(meta.get("p_placebo")) not in ("None", "nan")):
                ex["predict_parametric_placebo"] = f"p = {meta['p_placebo']}"
    return doc


# ---------------------------------------------------------------- method_out
def build_method_out(panel, battery_out, recon, validation, calib, disjoint,
                     stability, pbound, prov, canon, verdicts, robust, source,
                     audit, panel_gate) -> dict:
    """Assemble the exp_gen_sol_out-shaped output: one example per reportable row."""
    examples: list[dict] = []
    names = panel.entities()

    def add(row_type: str, inp: str, out: str, predict: dict | None = None, **meta):
        """Append one reportable row.

        Every row carries the operative method's answer and the conventional
        baseline's answer as ``predict_*`` fields, so the method-versus-baseline
        comparison is machine-readable and not only in the prose.  Statistic rows
        additionally carry the analytic cluster null and the legacy parametric
        placebo, which are the other two instruments that answer the same
        question.
        """
        payload = {"input": inp, "output": out,
                   "metadata_row_type": row_type}
        for k, v in meta.items():
            payload[f"metadata_{k}"] = v
        p = predict or {}
        payload["predict_country_rotation_null"] = str(
            p.get("rotation") or _fallback_rotation(row_type, meta)
        )
        payload["predict_nominal_baseline"] = str(
            p.get("nominal") or _fallback_nominal(row_type, meta)
        )
        if "analytic" in p:
            payload["predict_analytic_cluster_null"] = str(p["analytic"])
        if "placebo" in p:
            payload["predict_parametric_placebo"] = str(p["placebo"])
        examples.append(payload)

    # --- reconciliation estimator rows
    for fold in ("even", "odd"):
        t = recon["per_fold"][fold]
        for row in t["rows"]:
            add(
                "reconciliation_estimator",
                f"{fold}|{row['name']}",
                f"{row['name']} = {fmt(row['value'], 6)}; DE = {fmt(row['de'], 4)}; "
                f"n_eff = {fmt(row['n_eff'], 2)}; governs: {row['governs']}. {row['note']}",
                fold=fold, estimator=row["name"], value=row["value"], design_effect=row["de"],
                effective_n=row["n_eff"], statistic_class_governed=row["governs"],
                note=row["note"],
            )
        add("reconciliation_algebra", f"{fold}|algebra identity",
            f"rho - ICC1 = {fmt(t['algebra']['gap_empirical_mean_over_digits'], 6)}; "
            f"closed form at the mean x = {fmt(t['algebra']['gap_closed_form_at_mean_x'], 6)}; "
            f"identity verified per digit: {t['algebra']['gap_identity_verified_per_digit']}. "
            f"Sign: {t['algebra']['sign']}.",
            fold=fold, gap=t["algebra"]["gap_empirical_mean_over_digits"],
            gap_closed=t["algebra"]["gap_closed_form_at_mean_x"],
            verified=t["algebra"]["gap_identity_verified_per_digit"],
            pooled_n_eff_deflated_by=t["algebra"]["pooled_n_eff_deflated_by_factor"])

    # --- threshold consistency
    tc = recon["threshold_consistency"]
    for r in tc["rows"]:
        add("reconciliation_threshold_consistency", f"{r['label']}|r_2 at n",
            f"r_2 = 2 sqrt(0.5/n) at n = {fmt(r['n'], 1)} gives {fmt(r['r_2sigma_from_closed_form'], 5)}, "
            f"recorded {fmt(r['recorded'], 5)} (|delta| = {r['abs_diff']:.2e}); reproduced: "
            f"{r['reproduced']}. {tc['reading']}",
            label=r["label"], n=r["n"],
            computed=r["r_2sigma_from_closed_form"], recorded=r["recorded"],
            reproduced=r["reproduced"])

    # --- pair identifiability
    p = recon["pair_identifiability"]
    add("reconciliation_pair_identifiability", "recorded rho 0.9575 and ICC1 0.2206",
        f"rho implies x = {fmt(p['x_implied_by_rho'], 6)} hence ICC1 = "
        f"{fmt(p['ICC1_implied_by_rho'], 5)}; ICC1 implies x = {fmt(p['x_implied_by_ICC1'], 6)} "
        f"hence rho = {fmt(p['rho_implied_by_ICC1'], 5)}. Jointly identifiable: "
        f"{p['jointly_identifiable']}. {p['reading']}.",
        jointly_identifiable=p["jointly_identifiable"],
        ICC1_implied_by_rho=p["ICC1_implied_by_rho"],
        rho_implied_by_ICC1=p["rho_implied_by_ICC1"])

    # --- kappa calibration
    kc = recon["kappa_calibration"]
    add("kappa_calibration", "rotation null vs parametric placebo null width",
        f"observed mean kappa = {fmt(kc['observed_kappa_mean'], 5)} "
        f"(mean kappa^2 = {fmt(kc['observed_kappa2_mean'], 5)}); placebo implied kappa = "
        f"{fmt(kc['placebo_implied_kappa'], 5)} at wander sd {fmt(kc['wander_sd_decades'], 4)} "
        f"decades; sigma_R observed {fmt(kc['sigma_R_observed'], 5)} vs placebo "
        f"{fmt(kc['sigma_R_placebo'], 5)}; two-sigma ratio placebo/observed = "
        f"{fmt(kc['threshold_ratio_placebo_over_observed'], 4)}. "
        f"{kc['reading']}.",
        observed_kappa_mean=kc["observed_kappa_mean"],
        observed_kappa2_mean=kc["observed_kappa2_mean"],
        placebo_implied_kappa=kc["placebo_implied_kappa"],
        sigma_R_observed=kc["sigma_R_observed"],
        sigma_R_placebo=kc["sigma_R_placebo"],
        threshold_ratio=kc["threshold_ratio_placebo_over_observed"])

    # --- the battery: 5 statistics x 2 folds x instruments
    for fold in ("even", "odd"):
        r = battery_out[fold]
        v = r["verdict"]
        for stat in STAT_NAMES:
            pr = r["rotation"][stat]
            if stat == "abs_c1":
                pa = r["analytic"]["circular"]["rayleigh_p"]
                governs = "n_countries"
            elif stat == "chi_square":
                pa = r["analytic"]["effective"]["chi_square_p"]
                governs = "n_eff"
            elif stat == "kuiper_V":
                pa = r["analytic"]["effective"]["kuiper_p"]
                governs = "n_eff"
            elif stat == "sup_distance":
                pa = r["analytic"]["effective"]["sup_distance_p"]
                governs = "n_eff"
            else:
                pa = None
                governs = "n_eff (interval width only; no closed-form p)"
            pc = r["placebo"][stat]["p"] if r["placebo"] else None
            add("statistic", f"{fold}|{stat}",
                f"{stat} = {fmt(r['observed'][stat], 5)}; rotation p = {fmt(pr['p'], 4)} "
                f"(operative); analytic p = {fmt(pa, 4) if pa is not None else 'n/a'}; "
                f"legacy placebo p = {fmt(pc, 4) if pc is not None else 'n/a'}; "
                f"Holm(rotation) = {fmt(r['holm_rotation'][stat], 4)}; "
                f"observed/null-median = {fmt(pr['observed_over_median'], 3)}; "
                f"consistent with exact Benford: {v['per_statistic'][stat]['consistent']}.",
                fold=fold, statistic=stat, value=r["observed"][stat],
                p_rotation=pr["p"], p_analytic=pa, p_placebo=pc,
                holm_rotation=r["holm_rotation"][stat],
                observed_over_null_median=pr["observed_over_median"],
                null_median=pr["null_median"], null_q025=pr["null_q025"],
                null_q975=pr["null_q975"], n_governs=governs,
                consistent=v["per_statistic"][stat]["consistent"],
                n=int(r["n"]), n_entities=r["n_entities"])
        add("statistic_circular", f"{fold}|abs_c1 detail",
            f"|c1| = {fmt(r['observed']['abs_c1'], 5)} at phase "
            f"{fmt(r['observed_full']['c1_phase_deg'], 2)} degrees; Rayleigh sigma_R = "
            f"{fmt(r['sigma_R'], 5)}; two-sigma {fmt(r['thresholds']['r_2sigma'], 5)}; "
            f"5% {fmt(r['thresholds']['r_5pct'], 5)}; median null "
            f"{fmt(r['thresholds']['median'], 5)}; 80%-power MDA "
            f"{fmt(r['thresholds']['mda80'], 5)} (noncentrality "
            f"{fmt(r['thresholds']['mda80_noncentrality'], 4)}).",
            fold=fold, abs_c1=r["observed"]["abs_c1"],
            phase_deg=r["observed_full"]["c1_phase_deg"], sigma_R=r["sigma_R"],
            threshold_2sigma=r["thresholds"]["r_2sigma"],
            threshold_5pct=r["thresholds"]["r_5pct"],
            rayleigh_median=r["thresholds"]["median"],
            mda80=r["thresholds"]["mda80"],
            degrees_of_freedom="n_countries")
        add("statistic_nominal", f"{fold}|conventional nominal-n battery",
            f"nominal chi-square p (df 8, uncorrected) = "
            f"{fmt(r['nominal']['chi_square_p_df8'], 4)}; Kuiper 5% critical = "
            f"{fmt(r['nominal']['kuiper_5pct_critical'], 5)} vs V = "
            f"{fmt(r['observed']['kuiper_V'], 5)} (exceeds: {r['nominal']['kuiper_exceeds']}); "
            f"|c1| Rayleigh p at the row count n (the mis-application) = "
            f"{fmt(r['nominal']['rayleigh_p_at_n_rows'], 4)}; Nigrini band on MAD: "
            f"{r['observed_full']['mad_nigrini_band']} (named, never used as the verdict).",
            fold=fold, chi_square_p_nominal=r["nominal"]["chi_square_p_df8"],
            kuiper_critical_5pct=r["nominal"]["kuiper_5pct_critical"],
            kuiper_exceeds=r["nominal"]["kuiper_exceeds"],
            rayleigh_p_at_n_rows=r["nominal"]["rayleigh_p_at_n_rows"],
            nigrini_band=r["observed_full"]["mad_nigrini_band"])
        # digit intervals, both conventions
        gw = r["digit_intervals_goodman"]
        bw = r["digit_intervals_bonferroni_wilson"]
        add("digit_intervals", f"{fold}|simultaneous per-digit intervals",
            "Goodman (recommended) coverage: "
            + ", ".join(f"{d['digit']}:{'Y' if d['covers_benford'] else 'N'}" for d in gw)
            + " | Bonferroni-Wilson coverage: "
            + ", ".join(f"{d['digit']}:{'Y' if d['covers_benford'] else 'N'}" for d in bw)
            + ". Lesperance, Reed, Stephens, Tsao & Wilton (2016, PLoS ONE 11(4):e0151235) show "
              "nine separate binomial intervals reject Benford too often for truly Benford data, "
              "so Goodman's simultaneous intervals are the recommendation.",
            fold=fold,
            goodman_covered=int(sum(d["covers_benford"] for d in gw)),
            bonferroni_wilson_covered=int(sum(d["covers_benford"] for d in bw)),
            goodman_intervals=json.dumps(gw), bonferroni_wilson_intervals=json.dumps(bw))

    # --- validation cells
    for stage, res in validation.items():
        for cell in res.get("cells", []):
            add("derivation_validation", f"{stage}|{cell['cell']}",
                f"{cell['cell']}: expected {cell['expected']}; achieved {cell['achieved']}; "
                f"tolerance {cell['tolerance']}; pass {cell['pass']}.",
                stage=stage, cell=cell["cell"], expected=str(cell["expected"]),
                achieved=str(cell["achieved"]), tolerance=str(cell["tolerance"]),
                pass_flag=bool(cell["pass"]))

    # --- calibration suite
    for cell in calib["cells"]:
        add("calibration_suite", cell["cell"],
            f"{cell['cell']}: expected {cell['expected']}; achieved {cell['achieved']}; "
            f"pass {cell['pass']}.",
            cell=cell["cell"], expected=str(cell["expected"]),
            achieved=str(cell["achieved"]), pass_flag=bool(cell["pass"]))
    add("calibration_suite", "monte_carlo_resolution",
        f"Rejection threshold alpha = {calib['alpha']} with B = {calib['B']} rotation "
        f"replicates, so the smallest attainable p is {1/(calib['B']+1):.2e}.",
        alpha=calib["alpha"], B=calib["B"], p_floor=1 / (calib["B"] + 1))

    # --- disjoint folds
    for r in disjoint["fold_rows"]:
        add("disjoint_fold", f"entity-disjoint fold {r['fold']}",
            f"fold {r['fold']}: n_countries = {r['n_countries']}, n_rows = {r['n_rows']}, "
            f"|c1| = {fmt(r['abs_c1'], 5)} with 95% Rayleigh interval "
            f"[{fmt(r['rayleigh_ci_low'], 5)}, {fmt(r['rayleigh_ci_high'], 5)}], "
            f"sigma_R = {fmt(r['sigma_R'], 5)}, 5% threshold {fmt(r['threshold_5pct'], 5)}, "
            f"80%-power MDA {fmt(r['mda80'], 5)}, rotation p = {fmt(r['p_rotation'], 4)}, "
            f"covers the pooled estimate: {r['covers_pooled']}.",
            fold=r["fold"], n_countries=r["n_countries"], n_rows=r["n_rows"],
            abs_c1=r["abs_c1"], ci_low=r["rayleigh_ci_low"], ci_high=r["rayleigh_ci_high"],
            sigma_R=r["sigma_R"], threshold_5pct=r["threshold_5pct"], mda80=r["mda80"],
            p_rotation=r["p_rotation"], covers_pooled=r["covers_pooled"])
    add("disjoint_fold_homogeneity", "Cochran Q on the four per-fold |c1|",
        f"Cochran Q = {fmt(disjoint['cochran_q'], 4)} on df = {disjoint['cochran_q_df']}, "
        f"p = {fmt(disjoint['cochran_q_p'], 4)}; homogeneous at 5%: "
        f"{disjoint['homogeneous_at_5pct']}. All folds cover the pooled estimate: "
        f"{disjoint['all_courses_cover_pooled']}. {disjoint['reading']}.",
        cochran_q=disjoint["cochran_q"], df=disjoint["cochran_q_df"],
        p=disjoint["cochran_q_p"], homogeneous=disjoint["homogeneous_at_5pct"])

    # --- stability
    add("stability", "country bootstrap and delete-one jackknife",
        f"pooled |c1| = {fmt(stability['pooled_abs_c1'], 5)}; country-bootstrap 95% interval "
        f"[{fmt(stability['bootstrap_ci_low'], 5)}, {fmt(stability['bootstrap_ci_high'], 5)}] "
        f"from {stability['bootstrap_n']} resamples ({stability['bootstrap_note']}); "
        f"jackknife range [{fmt(stability['jackknife_range_low'], 5)}, "
        f"{fmt(stability['jackknife_range_high'], 5)}]. Most influential: "
        + ", ".join(f"{d['entity']} ({d['delta']:+.5f})"
                    for d in stability["jackknife_most_influential"][:5]) + ".",
        pooled_abs_c1=stability["pooled_abs_c1"],
        bootstrap_ci_low=stability["bootstrap_ci_low"],
        bootstrap_ci_high=stability["bootstrap_ci_high"],
        bootstrap_n=stability["bootstrap_n"],
        jackknife_low=stability["jackknife_range_low"],
        jackknife_high=stability["jackknife_range_high"],
        jackknife_top=json.dumps(stability["jackknife_most_influential"]))

    # --- power bound
    for r in pbound["rows"]:
        add("power_bound", f"budget A = {fmt(r['budget_A'], 4)}",
            f"to separate exact Benford (amplitude 0) from a budget A = {fmt(r['budget_A'], 4)} "
            f"at 80% power needs n_countries = {fmt(r['required_n_countries'], 1)} against the "
            f"available {r['available_n_countries']} -- a shortfall factor of "
            f"{fmt(r['shortfall_factor'], 2)}. Detectable at the available n: {r['detectable']}. "
            f"{pbound['reading']}.",
            budget_A=r["budget_A"], required_n_countries=r["required_n_countries"],
            available_n_countries=r["available_n_countries"],
            shortfall_factor=r["shortfall_factor"], detectable=r["detectable"],
            nu80=pbound["nu80"], mean_kappa2=pbound["mean_kappa2"])

    # --- provenance lattice
    for r in prov["results"]:
        rec = r["recorded"]
        add("provenance", f"recorded triple {rec['label']}",
            f"{rec['label']}: recorded MAD {rec['MAD']}, chi-square {rec['chi_square']}, "
            f"|c1| {rec['abs_c1']} -> {r['status']}"
            + (f"; identified as {r['hits'][0]['candidate_id']} "
               f"({r['hits'][0]['definition']})" if r["identified"] else "")
            + f". Match rule: {prov['match_rule']['rule']}.",
            label=rec["label"], recorded_mad=rec["MAD"],
            recorded_chi_square=rec["chi_square"], recorded_abs_c1=rec["abs_c1"],
            identified=r["identified"], status=r["status"],
            hit_ids=", ".join(h["candidate_id"] for h in r["hits"]))
    add("provenance_summary", "lattice candidate summary",
        f"{prov['n_candidates']} pre-registered candidates "
        f"(74 single-year snapshots + 6 circular-mean variants). "
        + "; ".join(f"{c['candidate_id']}: MAD {fmt(c['MAD'], 5)} chi "
                    f"{fmt(c['chi_square'], 3)} c1 {fmt(c['abs_c1'], 4)}"
                    for c in prov["candidate_summary"]),
        n_candidates=prov["n_candidates"],
        candidate_summary=json.dumps(prov["candidate_summary"]))
    can = prov["canonical"]
    add("provenance_canonical", "declared canonical entity-level definition",
        f"canonical definition: {can['canonical_definition_id']} -- "
        f"{can['canonical_definition']}. {can['reading']}",
        canonical_id=can["canonical_definition_id"],
        canonical_definition=can["canonical_definition"],
        superseded=json.dumps(can.get("superseded", []))[:4000])

    # --- canonical harmonic
    fit = canon["fit"]
    harm = fit["harmonics"]
    add("canonical_harmonic", "mixture budget canonicalization",
        f"BIC selects k = {fit['selected_k']} components (BIC by k: "
        f"{ {k: round(v, 2) for k, v in fit['bic_by_k'].items()} }). "
        f"A_canon (modulus of the complex sum) = {fmt(harm['A_canon'], 6)}; "
        f"A_nocancel (sum of moduli) = {fmt(harm['A_nocancel'], 6)}; "
        f"cancellation ratio = {fmt(canon['resolution']['cancellation_ratio'], 4)}. "
        f"{canon['resolution']['verdict']}",
        selected_k=fit["selected_k"], A_canon=harm["A_canon"],
        A_nocancel=harm["A_nocancel"],
        cancellation_ratio=canon["resolution"]["cancellation_ratio"],
        bic_by_k=json.dumps(fit["bic_by_k"]),
        verdict=canon["resolution"]["verdict"])
    for label, test in canon["resolution"]["tests"].items():
        add("canonical_harmonic_test", f"recorded budget {label}",
            f"recorded {label} = {test['recorded'] if 'recorded' in test else canon['resolution']['recorded'][label]}: "
            f"|delta| from A_canon = {test['abs_diff_from_A_canon']:.2e}, "
            f"from A_nocancel = {test['abs_diff_from_A_nocancel']:.2e}; "
            f"matches A_canon (2e-3): {test['matches_A_canon_2e-3']}, "
            f"matches A_nocancel (2e-3): {test['matches_A_nocancel_2e-3']}.",
            label=label, abs_diff_canon=test["abs_diff_from_A_canon"],
            abs_diff_nocancel=test["abs_diff_from_A_nocancel"],
            matches_canon=test["matches_A_canon_2e-3"],
            matches_nocancel=test["matches_A_nocancel_2e-3"])
    add("canonical_harmonic_context", "budget against the observed amplitude and threshold",
        f"A_canon = {fmt(harm['A_canon'], 6)}; observed pooled |c1| = "
        f"{fmt(battery_out['pooled']['observed']['abs_c1'], 5)} "
        f"(ratio {fmt(battery_out['pooled']['observed']['abs_c1'] / harm['A_canon'], 3)}); "
        f"corrected two-sigma threshold = "
        f"{fmt(battery_out['pooled']['thresholds']['r_2sigma'], 5)}; 5% threshold = "
        f"{fmt(battery_out['pooled']['thresholds']['r_5pct'], 5)}. The budget is "
        f"{'resolvable' if harm['A_canon'] > battery_out['pooled']['thresholds']['r_5pct'] else 'NOT resolvable'} "
        f"at this ensemble's resolution.",
        A_canon=harm["A_canon"],
        observed_pooled_abs_c1=battery_out["pooled"]["observed"]["abs_c1"],
        threshold_2sigma=battery_out["pooled"]["thresholds"]["r_2sigma"],
        threshold_5pct=battery_out["pooled"]["thresholds"]["r_5pct"])

    # --- verdicts
    for label, v in verdicts.items():
        add("verdict", f"{label} fold verdict",
            v["sentence"] + f" Branch taken: {v['branch_taken']}. {v['untaken_branch']}.",
            fold=label, sentence=v["sentence"], branch=v["branch_taken"],
            abs_c1=v["abs_c1"], threshold_2sigma=v["threshold_2sigma"],
            threshold_5pct=v["threshold_5pct"], mda80=v["mda80"],
            all_five_consistent=v["all_consistent"])

    # --- robustness ablations
    for label, r in robust["variants"].items():
        if "error" in r:
            add("robustness", f"ablation {label}", f"{label}: {r['error']}",
                variant=label, error=r["error"])
            continue
        add("robustness", f"ablation {label}",
            f"{label}: n_rows {r['n_rows']}, n_entities {r['n_entities']}, |c1| "
            f"{fmt(r['abs_c1'], 5)} vs two-sigma {fmt(r['threshold_2sigma'], 5)}, MAD "
            f"{fmt(r['mad'], 5)}, rotation p(|c1|) {fmt(r['p_rotation_abs_c1'], 4)}, "
            f"passes {r['passes_count']}/{r['denominator']} statistics.",
            variant=label, n_rows=r["n_rows"], n_entities=r["n_entities"],
            abs_c1=r["abs_c1"], sigma_R=r["sigma_R"],
            threshold_2sigma=r["threshold_2sigma"], mad=r["mad"],
            p_rotation_abs_c1=r["p_rotation_abs_c1"],
            passes_count=r["passes_count"], denominator=r["denominator"],
            exceeds_2sigma=r["exceeds_2sigma"])
    add("robustness_summary", "ablation pass count under the corrected null",
        f"{robust['ablation_pass_count']}/{robust['ablation_denominator']} ablations pass all "
        f"five statistics under the corrected null. {robust['reading']}. Compare the prior "
        f"recorded {RECORDED['ablation_pass_prior']} (recorded, not recomputed).",
        pass_count=robust["ablation_pass_count"],
        denominator=robust["ablation_denominator"],
        prior_recorded=RECORDED["ablation_pass_prior"])

    # --- source invariance
    for label in ("owid_unwpp", "worldbank_wdi"):
        r = source[label]
        add("source_invariance", f"source {label}",
            f"{label}: n_rows {r['n_rows']}, n_entities {r['n_entities']}, MAD "
            f"{fmt(r['mad'], 5)}, chi-square {fmt(r['chi_square'], 3)}, sup-distance "
            f"{fmt(r['sup_distance'], 5)}, Kuiper V {fmt(r['kuiper_V'], 5)}, |c1| "
            f"{fmt(r['abs_c1'], 5)} vs two-sigma {fmt(r['threshold_2sigma'], 5)}.",
            source=label, n_rows=r["n_rows"], n_entities=r["n_entities"],
            mad=r["mad"], chi_square=r["chi_square"], sup_distance=r["sup_distance"],
            kuiper_V=r["kuiper_V"], abs_c1=r["abs_c1"], sigma_R=r["sigma_R"],
            threshold_2sigma=r["threshold_2sigma"],
            p_rotation_abs_c1=r["p_rotation_abs_c1"],
            p_rotation_mad=r["p_rotation_mad"])
    if "comparison" in source:
        c = source["comparison"]
        add("source_invariance", "WDI versus OWID comparison",
            f"|c1| ratio WDI/OWID = {fmt(c['abs_c1_ratio_wdi_over_owid'], 3)}; MAD ratio "
            f"{fmt(c['mad_ratio_wdi_over_owid'], 3)}; same side of the two-sigma threshold: "
            f"{c['same_side_of_threshold']}. {c['reading']}.",
            abs_c1_ratio=c["abs_c1_ratio_wdi_over_owid"],
            mad_ratio=c["mad_ratio_wdi_over_owid"],
            same_side=c["same_side_of_threshold"])

    # --- independent re-derivation
    add("independent_verification", "second code path",
        f"verify.py re-derived every headline quantity with plain numpy and a different "
        f"digit-indexing loop; mismatches versus the main path: "
        f"{len(audit['verify_mismatches'])}"
        + ("" if not audit["verify_mismatches"]
           else " -- " + "; ".join(audit["verify_mismatches"][:5]))
        + f". Fresh-seed rotation p for |c1| (even fold): {fmt(audit['fresh_seed_p'], 4)} vs main "
          f"{fmt(audit['main_p'], 4)}, |delta| = {audit['fresh_seed_delta']:.2e} "
          f"(tolerance {audit['fresh_seed_tolerance']:.2e}).",
        n_mismatches=len(audit["verify_mismatches"]),
        mismatches=json.dumps(audit["verify_mismatches"][:10]),
        fresh_seed_p=audit["fresh_seed_p"], main_p=audit["main_p"],
        fresh_seed_delta=audit["fresh_seed_delta"],
        fresh_seed_tolerance=audit["fresh_seed_tolerance"])

    # --- panel integrity
    add("panel_integrity", "integrity gate",
        f"{panel_gate['n_rows']} primary rows over {panel_gate['n_entities']} entities; folds "
        f"{panel_gate['n_per_fold']}; 2023 population sum over all 237 primary entities = "
        f"{panel_gate['sum_2023_all_primary_237']} (the 236-entity analysis set sums to "
        f"{panel_gate['sum_2023']}, the {panel_gate['sum_gap_2023_236_vs_237']}-person "
        f"difference being Kosovo); string-derived d1 agreement "
        f"{fmt(panel_gate['d1_agreement_rate'], 4)} with "
        f"{panel_gate['decade_boundary_cases']} whitelisted exact-decade boundary case; "
        f"gate pass {panel_gate['gate_pass']}.",
        n_rows=panel_gate["n_rows"], n_entities=panel_gate["n_entities"],
        sum_2023_all_primary=panel_gate["sum_2023_all_primary_237"],
        sum_2023_236=panel_gate["sum_2023"], gate_pass=panel_gate["gate_pass"],
        md5=panel_gate.get("md5", ""))

    # --- novelty positioning
    for row in NOVELTY:
        add("novelty_positioning", f"delta over {row['neighbour']}",
            f"{row['neighbour']} ({row['reference']}): {row['their_move']} "
            f"Delta here: {row['our_delta']}",
            neighbour=row["neighbour"], reference=row["reference"],
            their_move=row["their_move"], our_delta=row["our_delta"])

    # --- case audit: 236 rows
    groups = panel.entity_groups(None)
    gnames = sorted(groups)
    zbar = np.array([np.exp(2j * np.pi * groups[g]).mean() for g in gnames])
    kk = np.array([groups[g].size for g in gnames], float)
    wts = kk / kk.sum()
    z1 = complex(np.sum(wts * zbar))
    jack = []
    for i in range(len(gnames)):
        keep = np.ones(len(gnames), bool)
        keep[i] = False
        w2 = kk[keep]
        jack.append(abs(np.sum((w2 / w2.sum()) * zbar[keep])) - abs(z1))
    ent_meta = {}
    for r in panel.rows:
        ent_meta.setdefault(r.entity, r)
    order = np.argsort(-np.abs(np.array(jack)))
    rank_of = {gnames[j]: int(k) + 1 for k, j in enumerate(order)}
    for i, g in enumerate(gnames):
        row = ent_meta[g]
        add("case_audit", f"{row.iso_code}|entity audit",
            f"{g} ({row.iso_code}): kappa = {fmt(abs(zbar[i]), 5)}, phase "
            f"{fmt(math.degrees(np.angle(zbar[i])), 2)} degrees, weight "
            f"{fmt(wts[i], 5)}, influence on |c1| {jack[i]:+.5f} "
            f"(rank {rank_of[g]} of {len(gnames)}); top-20 {row.is_top20}, "
            f"pop <= 1M {row.pop_le_1m}, SIDS {row.is_sids}.",
            entity=g, iso_code=str(row.iso_code), entity_class=row.entity_class,
            kappa=float(abs(zbar[i])), theta_deg=float(math.degrees(np.angle(zbar[i]))),
            w_i=float(wts[i]), influence=float(jack[i]),
            influence_rank=rank_of[g], is_top20=row.is_top20,
            pop_le_1m=row.pop_le_1m, is_sids=row.is_sids,
            n_years=int(kk[i]))

    return {
        "metadata": {
            "description": (
                "Pre-registered re-derivation of how closely Benford's law fits the leading "
                "digits of world country populations, on the 236-country x 74-year UN WPP panel "
                "(17,464 country-years, 8,732 per fold). Establishes that the intraclass "
                "correlation, not a ratio of mean squares, governs the frequency-type digit "
                "statistics; that the circular-mean statistic's degrees of freedom are the "
                "country count rather than any effective row count; and that a country-rotation "
                "randomization null supplies an exact cluster null whose kappa_i are the "
                "observed ones by construction."
            ),
            "method_name": "country-rotation randomization null for Benford conformity on "
                           "clustered panel data",
            "panel_md5": panel_gate.get("md5", ""),
            "n_examples": len(examples),
            "scale": {k: v for k, v in SCALE.items() if k in ("mini", "full")}["full"],
            "created_utc": now_iso(),
            "openrouter_spend_usd": 0.0,
        },
        "datasets": [
            {"dataset": "benford_population_panel_1950_2023", "examples": examples}
        ],
    }


NOVELTY = [
    {
        "neighbour": "Lesperance, Reed, Stephens, Tsao & Wilton (2016)",
        "reference": "PLoS ONE 11(4):e0151235, DOI 10.1371/journal.pone.0151235",
        "their_move": "Compare discrete Cramer-von Mises and Pearson chi-square tests and "
                      "recommend Goodman simultaneous confidence intervals, showing that many "
                      "binomial intervals reject Benford too often.",
        "our_delta": "They establish the interval and goodness-of-fit conventions for "
                     "INDEPENDENT rows. We show what those conventions do to a clustered panel: "
                     "the frequency-type p-values need a design effect, and the circular-mean "
                     "statistic needs the country count instead, so the two cannot share one "
                     "correction. Both Goodman and Bonferroni-Wilson intervals are reported.",
    },
    {
        "neighbour": "Kish (1965) design effect",
        "reference": "Survey Sampling, Wiley",
        "their_move": "Deflate an effective sample size by DE = 1 + (k-1) ICC for a mean of "
                      "cluster-correlated observations.",
        "our_delta": "The design effect is correct for a mean of row-level values and wrong for a "
                     "circular mean of cluster-level phases: the latter's variance is "
                     "0.5 sum w_i^2 kappa_i^2 / n_countries, so its degrees of freedom are the "
                     "cluster count. Applying DE to |c1| shrinks the threshold by "
                     "sqrt(n_countries / n_eff), which is exactly the factor that made two prior "
                     "instrument readings contradict each other.",
    },
    {
        "neighbour": "McGraw & Wong (1996) intraclass correlation conventions",
        "reference": "Psychological Methods 1(1):30-46, DOI 10.1037/1082-989X.1.1.30",
        "their_move": "Fix the family of ICC estimators and specify which form corresponds to "
                      "which design and variate.",
        "our_delta": "We use their one-way ANOVA ICC(1) as the declared estimator and show "
                     "algebraically that the ratio-of-mean-squares rho = MSB/(MSB+MSW) exceeds it "
                     "by ((k-1)x + x^2)/((1+x)(1+(k-1)x)) >= 0, so a screen using rho always "
                     "overstates dependence and understates the effective n. The sign of the "
                     "bias is derived, not observed.",
    },
    {
        "neighbour": "Mardia & Jupp (2000) Rayleigh test",
        "reference": "Directional Statistics, Wiley",
        "their_move": "Give the Rayleigh test for a uniform circular distribution with its "
                      "exact null.",
        "our_delta": "Applied conditionally on the observed per-country concentrations, the "
                     "Rayleigh null becomes a cluster null whose scale is fixed by the data's own "
                     "kappa_i -- which removes the need to model within-country wandering at all.",
    },
    {
        "neighbour": "Nigrini (2012) mean absolute deviation bands",
        "reference": "Benford's Law: Applications for Forensic Accounting, Auditing and Fraud "
                     "Detection, Wiley",
        "their_move": "Give fixed MAD conformity bands for first-digit tests.",
        "our_delta": "The bands are NAMED for every MAD reported here and never used as the "
                     "verdict, because a fixed band cannot depend on the dependence structure. "
                     "The verdict comes from a randomization null that does.",
    },
    {
        "neighbour": "Stephens (1970) Kuiper critical values",
        "reference": "Journal of the Royal Statistical Society B 32:115-122",
        "their_move": "Give asymptotic critical values for the Kuiper statistic.",
        "our_delta": "Reported at both the nominal n and the design-effect n_eff so the reader "
                     "sees what the independence assumption buys; the operative verdict does not "
                     "rest on the asymptotic critical value.",
    },
    {
        "neighbour": "Sandron (2002) descriptive study of this ensemble",
        "reference": "Population 57:755-762",
        "their_move": "Describe the digit distribution of national populations without a "
                      "cluster-aware test.",
        "our_delta": "The same standard ensemble, 1950-2023, with the first cluster-aware "
                     "randomization null and an explicit discrimination-power bound: what 236 "
                     "countries can and cannot resolve.",
    },
]


# ------------------------------------------------------------------ main
@logger.catch(reraise=True)
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mini", action="store_true", help="shrink replicate counts for a gate")
    ap.add_argument("--skip-heavy", action="store_true", help="skip validation/power stages")
    args = ap.parse_args()
    scale = SCALE["mini" if args.mini else "full"]

    for d in (RESULTS, FIGURES, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    logger.add(LOGS / "run.log", rotation="30 MB", level="DEBUG")
    run_start = now_iso()
    t_start = time.perf_counter()
    logger.info(f"run start {run_start}  scale={scale}")

    # --- pre-registration audit
    if not PREREG.exists():
        raise RuntimeError("pre-registration missing; it must be written before any statistic")
    prereg_sha = sha256_file(PREREG)
    prereg_mtime = PREREG.stat().st_mtime
    logger.info(f"pre-registration sha256 {prereg_sha[:16]}... mtime {prereg_mtime}")

    audit: dict = {"prereg_sha256": prereg_sha,
                   "prereg_mtime_epoch": prereg_mtime,
                   "run_start_utc": run_start,
                   "prereg_precedes_panel_read": None}

    # --- STAGE 1
    logger.info("STAGE 1 panel load + integrity gate")
    panel_load_ts = time.time()
    panel = load_panel(PANEL_PATH)
    names = panel.entities()
    audit["panel_first_read_epoch"] = panel_load_ts
    audit["prereg_precedes_panel_read"] = bool(prereg_mtime <= panel_load_ts)
    logger.info(f"panel loaded: {len(panel.rows)} rows, {len(names)} entities")

    panel_gate = dict(_panel_gate_recap(panel))
    panel_gate["md5"] = panel.md5

    # --- STAGE 3
    recon = stage3_reconciliation(panel, names, {})
    recon["recorded_targets"] = {k: RECORDED[k] for k in
                                 ("rho_defective", "DE_rho", "n_eff_rho",
                                  "ICC1", "DE_ICC", "n_eff_ICC")}
    write_json(RESULTS / "reconciliation.json", recon)

    # --- STAGE 4 (validation)
    validation: dict = {}
    power: dict = {}
    calib: dict = {}
    obs_kappa = recon["kappa_calibration"]["observed_kappa_mean"]
    if not args.skip_heavy:
        logger.info(f"STAGE 4a V1 rotation-null exactness (kappa matched to {obs_kappa:.4f})")
        validation["V1"] = val.v1_exactness(n_panels=scale["n_panels_v1"],
                                           B=min(2_000, scale["B_rotation"]),
                                           kappa=obs_kappa)
        logger.info(f"  V1 pass={validation['V1']['pass']} fpr={validation['V1']['fpr']}")
        logger.info("STAGE 4b V2 Rayleigh vs simulation")
        validation["V2"] = val.v2_rayleigh(n_panels=scale["n_panels_v2"],
                                           B=min(4_000, scale["B_rotation"]))
        logger.info(f"  V2 pass={validation['V2']['pass']}")
        logger.info("STAGE 4c V4 time-order insensitivity")
        mm, ee = generate(PanelSpec(236, 37), 5)
        validation["V4"] = val.v4_time_order(mm, ee)
        logger.info(f"  V4 pass={validation['V4']['pass']}")
        logger.info("STAGE 4d V3 n_eff-threshold miscalibration")
        mean_k2 = recon["kappa_calibration"]["observed_kappa2_mean"]
        v3 = val.v3_neff_threshold_miscalibration(
            kappa2_mean=mean_k2,
            n_c=len(names),
            neff_rows={"n_eff from defective rho":
                       recon["per_fold"]["even"]["family_variate"]["n_eff_rho_pooled"],
                       "n_eff from declared ICC1":
                       recon["per_fold"]["even"]["family_variate"]["n_eff_ICC_pooled"]},
        )
        validation["V3"] = v3
        logger.info(f"  V3 pass={v3['pass']}")
        logger.info("STAGE 4e V5 power curve (kappa matched)")
        power = val.v5_power_curve(n_panels=scale["n_panels_power"],
                                   B=min(2_000, scale["B_rotation"]), kappa=obs_kappa)
        validation["V5"] = {"cells": power["cells"], "pass": power["pass"],
                            "setting": power["setting"], "seconds": power["seconds"],
                            "curve": power["curve"]}
        logger.info(f"  V5 pass={power['pass']} "
                    f"curve={[(c['planted_amplitude'], round(c['rejection_rate'], 2)) for c in power['curve']]}")
        logger.info("STAGE 4f calibration suite")
        calib = val.calibration_suite(n_panels=scale["n_panels_cal"],
                                      B=min(10_000, scale["B_rotation"]), kappa=obs_kappa)
        logger.info(f"  calibration pass={calib['pass']} "
                    f"cells={[(c['cell'], c['pass']) for c in calib['cells']]}")
        gc.collect()

    # --- STAGE 5
    wander = recon["wander"]["median_sd_decades"]
    battery_out = stage5_battery(panel, names, scale, wander)
    verdicts = verdict_sentences(battery_out)
    write_json(RESULTS / "verdict.json", {
        "verdicts": verdicts,
        "battery": {k: {
            "n": v["n"], "n_entities": v["n_entities"],
            "observed": v["observed"], "sigma_R": v["sigma_R"],
            "thresholds": v["thresholds"], "kappa_mean": v["kappa_mean"],
            "rotation": v["rotation"], "analytic": v["analytic"],
            "placebo": v["placebo"], "holm_rotation": v["holm_rotation"],
            "nominal": v["nominal"], "verdict": v["verdict"],
            "n_eff_ICC": v["n_eff_ICC"],
        } for k, v in battery_out.items()},
    })

    # --- STAGE 6
    logger.info("STAGE 6 disjoint folds, stability, power bound")
    disjoint = bat.disjoint_folds(panel, k=4, B=scale["B_disjoint"])
    stability = bat.stability(panel, n_boot=scale["n_boot"])
    pbound = bat.power_bound(
        sum_w2_kappa2=battery_out["pooled"]["sum_w2_kappa2"],
        n_c=len(names),
        budgets=[RECORDED["budget_b"], battery_out["pooled"]["observed"]["abs_c1"]],
        nu80=battery_out["pooled"]["thresholds"]["mda80_noncentrality"],
    )
    write_json(RESULTS / "stability.json",
               {"disjoint_folds": disjoint, "stability": stability, "power_bound": pbound})
    gc.collect()

    # --- STAGE 7
    logger.info("STAGE 7 provenance lattice")
    snaps = snapshot_candidates(panel)
    cms = circular_mean_candidates(panel)
    lattice = match_recorded(snaps + cms, [
        {"label": "provenance triple A",
         "digit_vector": RECORDED["prov_vector"],
         "MAD": RECORDED["prov_a_mad"], "chi_square": RECORDED["prov_a_chi"],
         "abs_c1": RECORDED["prov_a_c1"]},
        {"label": "provenance triple B",
         "MAD": RECORDED["prov_b_mad"], "chi_square": RECORDED["prov_b_chi"],
         "abs_c1": RECORDED["prov_b_c1"]},
    ])
    canonical = declare_canonical(lattice, {
        "MAD": battery_out["pooled"]["observed"]["MAD"],
        "chi_square": battery_out["pooled"]["observed"]["chi_square"],
        "abs_c1": battery_out["pooled"]["observed"]["abs_c1"],
        "definition": "pooled row-level digit table over all 17,464 primary rows",
    })
    # a year snapshots that reproduce the recorded vector, if any
    vector_hits = [c for c in snaps if c["digit_vector"] == RECORDED["prov_vector"]]
    lattice_out = dict(lattice)
    lattice_out["canonical"] = canonical
    lattice_out["recorded_vector_exact_hits"] = [
        {"candidate_id": c["candidate_id"], "definition": c["definition"],
         "MAD": c["MAD"], "chi_square": c["chi_square"], "abs_c1": c["abs_c1"]}
        for c in vector_hits
    ]
    write_json(RESULTS / "provenance.json", lattice_out)

    # --- sup-distance definition scan (hygiene)
    sup_scan = sup_distance_definition_scan(panel)
    write_json(RESULTS / "supm_definition_scan.json", sup_scan)
    logger.info(f"sup-distance scan: {sup_scan['verdict'][:160]}")

    # --- STAGE 8
    logger.info("STAGE 8 harmonic canonicalization")
    logs10 = np.log10(np.array([r.population for r in panel.rows], float))
    fit = select_and_fit(logs10, max_k=3, n_starts=20 if not args.mini else 4)
    resolution = resolve_recorded_values(
        fit, {"recorded 0.035767": RECORDED["budget_a"],
              "recorded 0.035474": RECORDED["budget_b"]})
    canon_out = {"fit": fit, "resolution": resolution}
    write_json(RESULTS / "canonical.json", canon_out)

    # --- robustness and source invariance
    logger.info("robustness ablations + source invariance")
    robust = bat.robustness_ablations(
        panel, names, n_eff_icc=battery_out["pooled"]["n_eff_ICC"],
        B=min(2_000, scale["B_rotation"]), wander=wander)
    source = bat.source_invariance(panel, B=min(2_000, scale["B_rotation"]),
                                   n_eff_icc=battery_out["pooled"]["n_eff_ICC"])
    gc.collect()

    # --- STAGE 9
    logger.info("STAGE 9 independent re-derivation")
    indep = independent_report(PANEL_PATH)
    # verify.py computes the ICC on the POOLED 17,464 rows for digit 1 and for
    # the continuous mantissa, so the comparison baseline must be this run's own
    # pooled estimator on those same variates -- k is 74 there, not 37.
    pooled_est = estimator_table(
        panel.mantissas(None),
        np.array([names.index(r.entity) for r in panel.rows]), len(names))
    d1_row = pooled_est["family_variate"]["per_digit"][0]
    main_cmp = {
        "n_rows": len(panel.rows), "n_entities": len(names),
        "kappa_mean": battery_out["pooled"]["kappa_mean"],
        "sigma_R": battery_out["pooled"]["sigma_R"],
        "threshold_2sigma": battery_out["pooled"]["thresholds"]["r_2sigma"],
        "threshold_5pct": battery_out["pooled"]["thresholds"]["r_5pct"],
        "ICC1_digit1": d1_row["ICC1"],
        "rho_digit1": d1_row["rho_defective"],
        "DE_ICC_digit1": d1_row["DE_ICC"],
        "DE_rho_digit1": d1_row["DE_rho"],
        "n_eff_ICC_digit1": len(panel.rows) / d1_row["DE_ICC"],
        "n_eff_rho_digit1": len(panel.rows) / d1_row["DE_rho"],
        "folds": {
            "even": {**battery_out["even"]["observed"], "n": battery_out["even"]["n"]},
            "odd": {**battery_out["odd"]["observed"], "n": battery_out["odd"]["n"]},
            "None": {**battery_out["pooled"]["observed"], "n": battery_out["pooled"]["n"]},
        },
    }
    mismatches = assert_matches(main_cmp, indep)

    # fresh-seed rotation p for |c1| on the even fold
    rows_even = panel.by_fold("even")
    m_even = np.array([r.mantissa for r in rows_even], float)
    ei_even = np.array([names.index(r.entity) for r in rows_even])
    obs_even = statistic_battery(m_even)
    rot2 = RotationNull(m_even, ei_even, len(names))
    nulls2 = rot2.replicate(np.random.default_rng(999_331), scale["B_rotation"])
    p2 = rot2.pvalues(obs_even, nulls2)
    p_main = battery_out["even"]["rotation"]["abs_c1"]["p"]
    tol = 3.0 * math.sqrt(max(p_main * (1 - p_main), 1e-9) / scale["B_rotation"])
    audit.update({
        "verify_mismatches": mismatches,
        "fresh_seed_p": p2["abs_c1"]["p"],
        "main_p": p_main,
        "fresh_seed_delta": abs(p2["abs_c1"]["p"] - p_main),
        "fresh_seed_tolerance": tol,
        "fresh_seed_agrees": bool(abs(p2["abs_c1"]["p"] - p_main) <= tol),
        "independent_report": indep,
    })

    # --- figures
    specs = write_figure_specs(battery_out, recon, power, recon["threshold_table"])

    # --- STAGE 10
    logger.info("STAGE 10 method_out")
    method_out = build_method_out(
        panel, battery_out, recon, validation, calib, disjoint, stability, pbound,
        lattice_out, canon_out, verdicts, robust, source, audit, panel_gate)
    # append the sup-distance definition scan as its own reportable row
    method_out["datasets"][0]["examples"].append({
        "input": "sup-distance definition scan",
        "output": sup_scan["verdict"],
        "metadata_row_type": "definition_scan",
        "metadata_recorded_target": sup_scan["recorded_target"],
        "metadata_closest_definition": sup_scan["closest"][0]["definition"],
        "metadata_closest_fold": sup_scan["closest"][0]["fold"],
        "metadata_closest_value": sup_scan["closest"][0]["value"],
        "metadata_relative_error": sup_scan["closest"][0]["rel_err"],
        "metadata_candidates": json.dumps(sup_scan["folds"]),
    })
    method_out["metadata"]["n_examples"] = len(method_out["datasets"][0]["examples"])
    ensure_predict_fields(method_out)
    write_json(ROOT / "method_out.json", method_out)
    logger.info(f"method_out.json written with {len(method_out['datasets'][0]['examples'])} rows")

    # --- audit
    audit.update({
        "run_end_utc": now_iso(),
        "wall_seconds": round(time.perf_counter() - t_start, 1),
        "validation_pass": {k: v.get("pass") for k, v in validation.items()},
        "calibration_pass": calib.get("pass"),
        "scale": scale,
        "files": {},
    })
    for rel in ("analysis/stats.py", "analysis/panel_io.py", "analysis/reconcile.py",
                "analysis/synthetic.py", "analysis/validation.py", "analysis/battery.py",
                "analysis/provenance.py", "analysis/canonical.py", "analysis/verify.py",
                "method.py"):
        fp = ROOT / rel
        if fp.exists():
            audit["files"][rel] = sha256_file(fp)
    write_json(RESULTS / "audit.json", audit)
    logger.info(f"done in {audit['wall_seconds']}s; mismatches={len(mismatches)}")
    return 0


def _panel_gate_recap(panel) -> dict:
    """The gate numbers, taken from the gate that actually ran."""
    c = dict(panel.gate_checks)
    return {
        "n_rows": c["n_rows"],
        "n_entities": c["n_entities"],
        "n_per_fold": c["n_per_fold"],
        "sum_2023_all_primary_237": c["sum_2023_all_primary_237"],
        "sum_2023": c["sum_2023"],
        "sum_gap_2023_236_vs_237": c["sum_gap_2023_236_vs_237"],
        "d1_agreement_rate": c["d1_agreement_rate"],
        "decade_boundary_cases": c["decade_boundary_cases"],
        "gate_pass": c["gate_pass"],
        "sum_matches_published": c["sum_2023_matches"],
        "all_checks": {k: v for k, v in c.items()
                      if not isinstance(v, (list, dict))},
    }


if __name__ == "__main__":
    raise SystemExit(main())
