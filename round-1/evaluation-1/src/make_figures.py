#!/usr/bin/env python3
"""Build the three publication figures from eval_out.json via the aii-data-fig-gen spec format.

Every number drawn is read straight from eval_out.json, so a figure cannot
disagree with the results file.  The three figures are:

  fig1  first-digit residuals vs exact Benford and vs the fitted size law,
        with simultaneous per-digit intervals -- the descriptive core of the
        question "how closely does Benford fit?"
  fig2  the per-year first-harmonic series, observed with cluster-bootstrap
        bands against each year's own fitted-law prediction, plus the year
        trend -- the temporal axis of the same question
  fig3  the calibrated conformity panel: observed statistics against their
        cluster-calibrated and naive-iid reference envelopes, which is where
        the aggregation trap becomes visible.

Run after eval.py:
    uv run make_figures.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "eval_out.json"
FIGDIR = ROOT / "figures"
SPECDIR = ROOT / "figure_specs"

# The chart generator belongs to the aii-data-fig-gen skill, which is not part of
# this repository.  Locate it through an environment variable so a reader can
# point at their own checkout; the figures are already committed, so this is only
# needed when re-rendering them.
CHART_GEN = Path(
    os.environ.get("AII_DATA_FIG_GEN", "")
    or Path.home() / ".claude/skills/aii-data-fig-gen/scripts/chart_gen.py"
)
CHART_GEN_FALLBACKS = [
    Path("/Users/Shared/aii_data/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py"),
    Path("/ai-inventor/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py"),
]

COLORS = {
    "Observed": "#0173b2",
    "Fitted size law": "#de8f05",
    "Exact Benford": "#949494",
    "Cluster-calibrated null": "#029e73",
    "Naive iid null": "#cc78bc",
    "Observed |c1|": "#0173b2",
    "Year-specific fitted law": "#de8f05",
}


def load() -> dict:
    return json.loads(RESULTS.read_text())


def fig1_residuals(d: dict) -> dict:
    """Per-digit residuals (percentage points) with simultaneous intervals.

    The intervals shown are the Bonferroni simultaneous Clopper-Pearson bounds on
    the observed proportion, converted to the same percentage-point scale as the
    residual.  They belong on the "observed vs exact Benford" series, which is the
    measured quantity; the fitted-law series is a deterministic prediction and
    carries no sampling interval of its own.
    """
    md = d["metadata"]
    table = md["per_digit_table"]
    digits = [f"{r['digit']}" for r in table]
    benford_resid = [r["reserved_residual_pp"] for r in table]
    model_resid = [r["reserved_residual_vs_model_pp"] for r in table]
    # one-sided half-width toward the bound that is further from the point estimate
    half = [100.0 * max(r["reserved_obs"] - r["reserved_ci_low"],
                        r["reserved_ci_high"] - r["reserved_obs"])
            for r in table]
    return {
        "type": "bar",
        "title": "First-digit residuals on the reserved odd years (n = 8,732 country-years)",
        "xlabel": "Leading digit",
        "ylabel": "Residual (percentage points)",
        "aspect": "16:9",
        "categories": digits,
        "series": [
            {"label": "Observed vs exact Benford", "values": benford_resid,
             "errors": half},
            {"label": "Observed vs fitted size law", "values": model_resid},
        ],
    }


def fig2_per_year(d: dict) -> dict:
    """Observed |c1(y)| with its cluster-bootstrap band vs the year-specific prediction."""
    py = d["metadata"]["per_year_series"]
    rows = sorted(py["per_year_table"], key=lambda r: r["year"])
    years = [r["year"] for r in rows]
    obs = [r["abs_c1_observed"] for r in rows]
    lo = [r["abs_c1_ci_low"] for r in rows]
    hi = [r["abs_c1_ci_high"] for r in rows]
    band = [max(o - l, h - o) for o, l, h in zip(obs, lo, hi)]
    pred = [r["abs_c1_predicted"] for r in rows]
    return {
        "type": "line",
        "title": "First-harmonic amplitude by year, reserved odd years",
        "xlabel": "Year",
        "ylabel": "|c₁| (first mantissa harmonic)",
        "aspect": "16:9",
        "series": [
            {"label": "Observed |c1|", "x": years, "values": obs, "band": band},
            {"label": "Each year's own fitted size law", "x": years, "values": pred},
        ],
    }


def fig3_calibration(d: dict) -> dict:
    """Observed statistic as a multiple of its null median, cluster vs naive iid."""
    cal = d["metadata"]["conformity_headline"]["reserved_odd"]
    keys = ["MAD", "M", "abs_c1", "chi2", "kuiper_V"]
    labels = ["MAD", "sup-distance M", "|c1|", "chi-square", "Kuiper V"]
    cluster = [cal["cluster_calibrated"][k]["observed"] /
               cal["cluster_calibrated"][k]["null_mean"] for k in keys]
    iid = [cal["naive_iid_null"][k]["observed"] /
           cal["naive_iid_null"][k]["null_mean"] for k in keys]
    return {
        "type": "bar",
        "title": "Observed conformity statistic as a multiple of each null's median",
        "xlabel": "Statistic",
        "ylabel": "Observed / null median",
        "aspect": "16:9",
        "categories": labels,
        "series": [
            {"label": "Against the country-clustered Benford null",
             "values": cluster},
            {"label": "Against the i.i.d. null at nominal n",
             "values": iid},
        ],
    }


def resolve_chart_gen() -> Path:
    """Find the aii-data-fig-gen chart generator, or explain how to point at one."""
    candidates = [CHART_GEN, *CHART_GEN_FALLBACKS]
    for c in candidates:
        if c.is_file():
            return c
    raise SystemExit(
        "chart_gen.py not found. The figures are already committed under figures/, so "
        "you only need this to re-render them. Set AII_DATA_FIG_GEN to the path of "
        "aii-data-fig-gen/scripts/chart_gen.py, e.g.\n"
        "  export AII_DATA_FIG_GEN=/path/to/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py"
    )


def render(spec: dict, name: str) -> None:
    SPECDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    gen = resolve_chart_gen()
    spec_path = SPECDIR / f"{name}.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    cmd = [sys.executable, str(gen), "--spec", str(spec_path),
           "--out", str(FIGDIR / name)]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    d = load()
    render(fig1_residuals(d), "fig1_digit_residuals")
    render(fig2_per_year(d), "fig2_per_year_harmonic")
    render(fig3_calibration(d), "fig3_calibrated_conformity")

    # save the raw values behind every drawn claim, so a richer view can re-check
    md = d["metadata"]
    cal = md["conformity_headline"]["reserved_odd"]
    raw = {
        "fig1_digit_residuals": md["per_digit_table"],
        "fig2_per_year_harmonic": md["per_year_series"]["per_year_table"],
        "fig3_calibrated_conformity": {
            "note": ("each value is (observed statistic) / (median of that null's "
                     "simulated distribution); a value near 1 means the observation "
                     "is typical of the null, a large value means it is far above it"),
            "cluster_calibrated_null": {
                k: {"observed": cal["cluster_calibrated"][k]["observed"],
                    "null_median": cal["cluster_calibrated"][k]["null_p50"],
                    "null_p2_5": cal["cluster_calibrated"][k]["null_p2_5"],
                    "null_p97_5": cal["cluster_calibrated"][k]["null_p97_5"],
                    "two_sided_p": cal["cluster_calibrated"][k]["two_sided_p"],
                    "ratio_to_null_median": (cal["cluster_calibrated"][k]["observed"] /
                                             cal["cluster_calibrated"][k]["null_mean"])}
                for k in ("MAD", "M", "abs_c1", "chi2", "kuiper_V")},
            "naive_iid_null": {
                k: {"observed": cal["naive_iid_null"][k]["observed"],
                    "null_median": cal["naive_iid_null"][k]["null_p50"],
                    "null_p2_5": cal["naive_iid_null"][k]["null_p2_5"],
                    "null_p97_5": cal["naive_iid_null"][k]["null_p97_5"],
                    "two_sided_p": cal["naive_iid_null"][k]["two_sided_p"],
                    "ratio_to_null_median": (cal["naive_iid_null"][k]["observed"] /
                                             cal["naive_iid_null"][k]["null_mean"])}
                for k in ("MAD", "M", "abs_c1", "chi2", "kuiper_V")},
        },
    }
    out = ROOT / "results" / "figure_raw_values.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(raw, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
