#!/usr/bin/env python3
"""Stage 8.5 - figures, rendered from the saved numbers via the chart generator.

Every figure is a chart of values that already exist in
``results/screen_results.json``; nothing is drawn by hand and nothing is
recomputed here.  The spec for each figure is written to
``figures/specs/*.json`` so a reader can re-render any figure from its
numbers, and the raw values behind each one are saved beside it.

Usage:
    uv run make_figures.py
    uv run make_figures.py --results results/screen_results.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
SPEC_DIR = FIG_DIR / "specs"
DATA_DIR = FIG_DIR / "data"

#: Locate the chart generator.  Order: an explicit env var, the run machine's
#: skill path, then a couple of relative guesses -- so a reader on another
#: machine can point it at their own copy without editing this file:
#:     export AII_CHART_GEN=/path/to/aii-data-fig-gen/scripts/chart_gen.py
import os as _os


def _find_chart_gen() -> Path:
    from_env = _os.environ.get("AII_CHART_GEN")
    cands = [Path(from_env)] if from_env else []
    cands += [
        Path("/Users/Shared/aii_data/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py"),
        Path.home() / ".claude/skills/aii-data-fig-gen/scripts/chart_gen.py",
        ROOT.parent.parent.parent.parent / ".claude/skills/aii-data-fig-gen/scripts/chart_gen.py",
    ]
    for c in cands:
        if c.exists():
            return c
    return cands[1]


SKILL = _find_chart_gen()

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")


def _write_spec(name: str, spec: dict) -> Path:
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    p = SPEC_DIR / f"{name}.json"
    p.write_text(json.dumps(spec, indent=2))
    return p


def _render(spec_path: Path, out_stem: Path) -> bool:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SKILL), "--spec", str(spec_path), "--out", str(out_stem)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logger.error(f"render failed for {spec_path.name}: "
                     f"{(r.stderr or r.stdout).strip()[:600]}")
        return False
    logger.info(f"  rendered {out_stem.name}.pdf/.png")
    return True


# ---------------------------------------------------------------------------
# fig 1 - the harmonic budget: |v_j| per component
# ---------------------------------------------------------------------------
def fig_harmonic_budget(res: dict) -> None:
    bud = res["harmonic_budget"]["P"]
    comps = bud.get("components") or []
    if not comps:
        logger.warning("no decomposable harmonic budget; skipping fig 1")
        return
    labels = [f"component {c['component']}" for c in comps] + ["resultant |c₁|"]
    amps = [c["contribution_amp"] for c in comps] + [bud["predicted_amp"]]
    obs = res["observed"]["P"]["c1_amp"]
    spec = {
        "type": "bar",
        "title": "Harmonic budget of the BIC-selected size law",
        "xlabel": "Component",
        "ylabel": "Contribution to |c₁| (amplitude)",
        "aspect": "16:9",
        "categories": labels,
        "series": [{"label": "Each component's |v_j| (final bar: vector sum)",
                    "values": [float(v) for v in amps]}],
        "show_title": True,
    }
    _write_spec("fig1_harmonic_budget", spec)
    _render(SPEC_DIR / "fig1_harmonic_budget.json", FIG_DIR / "fig1_harmonic_budget")
    (DATA_DIR / "fig1_harmonic_budget.json").write_text(json.dumps({
        "components": comps, "predicted_amp": bud["predicted_amp"],
        "observed_amp_P": obs,
        "note": ("the resultant (0.0024) is two orders of magnitude below the observed "
                 "|c1| = 0.075, so the static-mixture mechanism under-predicts the panel's "
                 "harmonic amplitude"),
    }, indent=2))


# ---------------------------------------------------------------------------
# fig 2 - observed digit residuals at the pooled even fold
# ---------------------------------------------------------------------------
def fig_digit_residuals(res: dict) -> None:
    bat = res["observed"]["P"]
    resid = bat["residuals_pp"]
    cats = [f"{d}" for d in range(1, 10)]
    spec = {
        # `bar`, not `diverging`: the digits must stay in digit order, and
        # `diverging` sorts by magnitude, which would scatter 1..9 down the
        # axis and destroy the only ordering the figure is about.
        "type": "bar",
        "title": "Leading-digit residuals, pooled even fold",
        "xlabel": "Leading digit",
        "ylabel": "Observed minus Benford (percentage points)",
        "aspect": "16:9",
        "categories": cats,
        "series": [{"label": "Residual (percentage points)",
                    "values": [float(v) for v in resid]}],
        "show_title": True,
    }
    _write_spec("fig2_digit_residuals", spec)
    _render(SPEC_DIR / "fig2_digit_residuals.json", FIG_DIR / "fig2_digit_residuals")
    (DATA_DIR / "fig2_digit_residuals.json").write_text(json.dumps({
        "p_hat": bat["p_hat"], "benford": bat["benford"], "residuals_pp": resid,
        "n": bat["n"], "n_eff": res["power_statement"]["n_eff"],
        "intervals_neff": bat.get("per_digit_intervals_neff"),
        "mad": bat["mad"], "mad_band": bat["mad_nigrini_band"],
        "kuiper_V": bat["kuiper_V"], "c1_amp": bat["c1_amp"], "c1_phase": bat["c1_phase"],
    }, indent=2))


# ---------------------------------------------------------------------------
# fig 3 - per-year observed vs predicted |c1|
# ---------------------------------------------------------------------------
def fig_per_year(res: dict) -> None:
    fd = res["extras"]["figure_data"]["per_year"]["P"]
    years = fd["years"]
    obs = fd["amp"]
    pred = fd["pred_amp"]
    # pred is ~1e-13; scale it to the observed mean so the SHAPE is visible and
    # say so on the axis, rather than plotting a flat line at the axis
    obs_mean = float(np.mean(obs))
    pred_mean = float(np.mean(pred)) or 1.0
    pred_scaled = [float(v / pred_mean * obs_mean) for v in pred]
    spec = {
        "type": "line",
        "title": "Per-year |c₁|: observed and the mechanism's predicted shape",
        "xlabel": "Year (even fold)",
        "ylabel": "|c₁| (predicted series scaled to the observed mean)",
        "aspect": "16:9",
        "series": [
            {"label": "Observed |c₁(t)|", "x": [int(y) for y in years],
             "values": [float(v) for v in obs]},
            {"label": "Predicted shape exp(−2π²σₜ²), scaled", "x": [int(y) for y in years],
             "values": pred_scaled},
        ],
        "show_title": True,
    }
    _write_spec("fig3_per_year_c1", spec)
    _render(SPEC_DIR / "fig3_per_year_c1.json", FIG_DIR / "fig3_per_year_c1")
    (DATA_DIR / "fig3_per_year_c1.json").write_text(json.dumps({
        "years": years, "amp": obs, "pred_amp_absolute": pred,
        "pred_amp_scaled_to_observed_mean": pred_scaled,
        "observed_mean": obs_mean, "predicted_mean_absolute": float(np.mean(pred)),
        "r_correlation": float(np.corrcoef(obs, pred)[0, 1]),
        "note": ("the predicted series is drawn scaled to the observed mean because its "
                 "ABSOLUTE level (about 1e-13) is twelve orders of magnitude below the data; "
                 "the unscaled values are in this file under pred_amp_absolute"),
    }, indent=2))


# ---------------------------------------------------------------------------
# fig 4 - the rotation family: MAD over the unit rescaling, with |c1| flat
# ---------------------------------------------------------------------------
def fig_rotation(res: dict) -> None:
    rot = res["extras"]["figure_data"]["rotation_family"]
    grid = rot["grid"]
    spec = {
        "type": "line",
        "title": "First-digit MAD and harmonic amplitude under unit rescaling",
        "xlabel": "Rescaling factor c",
        "ylabel": "First-digit MAD",
        "aspect": "16:9",
        "series": [
            {"label": "MAD(c) - unit-dependent", "x": [float(v) for v in grid],
             "values": [float(v) for v in rot["mad"]]},
            {"label": "|c₁| variation (exactly zero)", "x": [float(v) for v in grid],
             "values": [float(v) for v in rot["amp"]]},
        ],
        "show_title": True,
    }
    _write_spec("fig4_rotation_family", spec)
    _render(SPEC_DIR / "fig4_rotation_family.json", FIG_DIR / "fig4_rotation_family")
    (DATA_DIR / "fig4_rotation_family.json").write_text(json.dumps({
        "grid": grid, "mad": rot["mad"], "kuiper": rot["kuiper"], "amp": rot["amp"],
        "mad_range": rot["mad_range"], "mad_at_1": rot["mad_at_1"],
        "amp_max_abs_dev_from_amp_at_1": rot["amp_max_abs_dev_from_amp_at_1"],
        "best_case_c": rot["best_case"], "worst_case_c": rot["worst_case"],
    }, indent=2))


# ---------------------------------------------------------------------------
# fig 5 - candidate comparison: matched-null p at all levels
# ---------------------------------------------------------------------------
def fig_candidates(res: dict) -> None:
    verd = res["verdict"]
    levels = ["P", "S1", "S2"]
    names = {"A": "static mixture", "B": "temporal desync.", "C": "rounding", "D": "rotation"}
    series = []
    for c in "ABCD":
        vals = []
        for lv in levels:
            v = verd["p_values"][lv].get(c)
            vals.append(float(v) if v is not None and np.isfinite(v) else 1.0)
        series.append({"label": f"{c}: {names[c]}", "values": vals})
    spec = {
        "type": "bar",
        "title": "Matched-null p per candidate at each pre-registered level",
        "xlabel": "Observation level",
        "ylabel": "p (smaller = mechanism's signal present)",
        "aspect": "16:9",
        "categories": ["pooled even fold", "split-half S1", "split-half S2"],
        "series": series,
        "show_title": True,
    }
    _write_spec("fig5_candidates", spec)
    _render(SPEC_DIR / "fig5_candidates.json", FIG_DIR / "fig5_candidates")
    (DATA_DIR / "fig5_candidates.json").write_text(json.dumps({
        "p_values": verd["p_values"], "status": verd["status"],
        "ranking_at_P": verd["ranking_at_P"],
        "values_plotted_note": "non-finite or undefined p is drawn as 1.0 and is null in p_values",
    }, indent=2))


# ---------------------------------------------------------------------------
# fig 6 - the c_1 instrument against its noise floor
# ---------------------------------------------------------------------------
def fig_power(res: dict) -> None:
    ps = res["power_statement"]
    obs = res["observed"]["P"]["c1_amp"]
    md80 = res.get("synthetic_validation", {}).get("V3_power", {}).get("md_amp_80pct_power")
    bud = res["harmonic_budget"]["P"]
    cats = ["observed |c₁|", "cluster noise floor", "2σ threshold",
            "80% power minimum", "law's own prediction"]
    vals = [float(obs), float(ps["noise_floor_E_abs_c1"]), float(ps["md_amp_2sigma"]),
            float(md80) if md80 is not None and np.isfinite(md80) else float(ps["md_amp_80pct_power"]),
            float(bud.get("predicted_amp", 0.0))]
    spec = {
        "type": "bar",
        "title": "How close is the panel's |c₁| to what the design can resolve?",
        "xlabel": "Quantity",
        "ylabel": "|c₁| amplitude",
        "aspect": "16:9",
        "categories": cats,
        "series": [{"label": "Amplitude", "values": vals}],
        "show_title": True,
    }
    _write_spec("fig6_power", spec)
    _render(SPEC_DIR / "fig6_power.json", FIG_DIR / "fig6_power")
    (DATA_DIR / "fig6_power.json").write_text(json.dumps({
        "observed_c1_amp": obs, "noise_floor": ps["noise_floor_E_abs_c1"],
        "md_amp_2sigma": ps["md_amp_2sigma"], "md_amp_80pct": md80,
        "law_predicted_amp": bud.get("predicted_amp"),
        "n_eff": ps["n_eff"], "design_effect": ps["design_effect"], "rho_hat": ps["rho_hat"],
        "labels": cats, "values": vals,
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "screen_results.json"))
    args = ap.parse_args()
    rp = Path(args.results)
    if not rp.exists():
        raise FileNotFoundError(f"results file not found: {rp}")
    res = json.loads(rp.read_text())
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for fn in (fig_harmonic_budget, fig_digit_residuals, fig_per_year,
               fig_rotation, fig_candidates, fig_power):
        logger.info(f"{fn.__name__}")
        try:
            fn(res)
        except (KeyError, ValueError, TypeError) as exc:
            logger.error(f"  {fn.__name__} failed: {exc}")
            raise
    logger.info(f"figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
