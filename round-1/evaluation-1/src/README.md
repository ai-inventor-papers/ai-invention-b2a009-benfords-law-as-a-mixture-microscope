# Does Benford's law fit the leading digits of world-country populations?

A reserved-fold evaluation of a Benford screen over the country-population size
distribution, with the calibration the question actually requires.

**Short answer.** The panel is far from exact Benford under the null that almost
every published study implicitly uses (nominal-*n* i.i.d. draws), and it is
**inside** its exact-Benford envelope on all five conformity readouts once the
country clustering is respected — because the 8,732 country-years are 236
countries observed 37 times each, with a design effect of 8.9. And the digits
cannot separate exact Benford from the fitted size law at all: the two nulls
overlap on 89% of their support. Full argument in [`narrative.md`](narrative.md).

## Results at a glance

| | |
|---|---|
| reserved-fold rows / countries | 8,732 / 236 |
| ICC / design effect | 0.221 / 8.94 |
| MAD vs Benford | 0.00935 |
| sup-distance *M* vs Benford | 0.03065 |
| \|c₁\| / phase | 0.07521 / −89.8° |
| χ² (df 8, nominal *n*) | 91.69 |
| cluster-calibrated *p* (MAD, *M*, \|c₁\|, χ², Kuiper V) | 0.47, 0.93, 0.49, 0.36, 0.89 |
| naive i.i.d. *p* (all five) | < 0.0005 |
| χ² observed / naive-null median | 12.3× |
| entity-level *n* = 236: χ² / *p* | 12.31 / 0.138 |
| two-null discrimination overlap | 88.7% |
| size law out of sample: *M*<sub>res</sub> / ratio / percentile | 0.01425 / 1.145 / 3.55 |
| rounding rival's share of projection | 5.8% (< 50% bound) |
| per-year \|c₁\| vs year-specific law | r = 0.636 [0.237, 0.849] |
| per-year \|c₁\| vs calendar year | r = −0.717 |
| ablation criterion C | **FAILED** (0 of 3) |

Every number above is in `eval_out.json` under `metrics_agg` or
`metadata.*`; the narrative cites each one.

## Layout

| path | what it is |
|---|---|
| `eval.py` | **the entry point** — runs every stage and writes `eval_out.json` (~2 min, CPU only) |
| `benford_core.py` | first-digit statistics, harmonics, cluster-bootstrap machinery, and the self-test that must pass before anything is reported |
| `calibrated.py` | **the headline instrument** — cluster-calibrated exact-Benford nulls and the year-specific law fit |
| `null_models.py` | matched parametric nulls (size-law cluster, Benford cluster, entity resample, iid) |
| `size_laws.py` | the four rival size families and their exact leading-digit laws by periodisation |
| `panel.py` | rebuilds the country-year panel from both public sources, tags sub-ensembles by rule |
| `stages_a.py` | fold split, full instrument on one row set, scale-rotation check |
| `stages_b.py` | confirmation margin, projected residual, width-dose-response, per-year series |
| `stages_c.py` | ablation suite, source invariance with a provenance audit |
| `stages_d.py` | rounding-artifact simulation, inclusion thresholds, honest-negative branch |
| `stages_e.py` | entity-disjoint sensitivity check |
| `eval_utils.py` | JSON-safe conversion, percentiles, BH adjustment, block bootstrap |
| `make_figures.py` | renders the three figures from `eval_out.json` via the `aii-data-fig-gen` spec format |
| `figure_specs/` | the chart spec JSON for each figure |
| `figures/` | fig1–fig3 as vector PDF + PNG |
| `results/eval_out.json` | the full run output (schema `exp_eval_sol_out`) |
| `results/figure_raw_values.json` | every number drawn in a figure, so a richer view can re-check |
| `narrative.md` | the written report: answer, evidence, failures, departures |
| `verify_narrative.py` | mechanically checks every number quoted in `narrative.md` against `eval_out.json` (69 checks, all pass) |
| `verify_headline.py` | independent re-derivation of the headline numbers through a separate code path, plus the placebo/diagnostic suite (`results/headline_recheck.json`) |
| `make_variants.py` | writes `full_/mini_/preview_eval_out.json` |
| `reproducibility.md` | step-by-step Ubuntu reproduction |
| `data/` | the two public source files plus the derived panels (parquet) |

## Run it

```bash
uv sync                       # numpy, pandas, scipy, pyarrow, loguru, matplotlib
uv run eval.py                # writes results/eval_out.json and ./eval_out.json
uv run make_figures.py        # writes figures/*.pdf, *.png and results/figure_raw_values.json
uv run make_variants.py        # writes full_/mini_/preview_eval_out.json
uv run verify_narrative.py     # asserts every quoted number matches the run (69 checks)
uv run verify_headline.py      # independent re-derivation + placebo suite
```

`make_figures.py` needs the `aii-data-fig-gen` chart generator, which is a separate
skill and not vendored here; point at a copy with `AII_DATA_FIG_GEN`. The rendered
figures are already committed, so this only matters for re-rendering.

`eval.py` is deterministic (seed 20261119). It sets a 6 GB address-space cap and a
1-hour CPU-time cap at startup; actual peak memory is ≈ 1 GB and runtime ≈ 110 s
on one core.

No DATASET or EXPERIMENT artifact was staged for this round, so the panel is
rebuilt from `data/owid_population_unwpp.csv` and `data/wb_population_raw.json`
and the screen is refitted here on the even years. Every confirmation number is
therefore flagged `parameters_refit_by_evaluation: true` — a refit-transfer, not
a frozen carry-over.

## What this run fixed in its own instrument

Six defects were found and corrected during this evaluation; each is called out
where it matters in `narrative.md` and in the module docstring that fixes it.

1. **`chi2` and Kuiper V ignored the reference vector.** Both were always measured
   against Benford, so "distance from the fitted law" silently reported the
   distance from Benford. Fixed in `stages_a.fold_statistics` (χ² 91.69 vs the
   same reference, 33.45 vs the fitted law).
2. **The rounding simulation measured a tautology.** It divided the rounded data's
   own projection by the raw data's projection — a ratio of ~1 by construction,
   reported as "102% explained". It now measures the *change* rounding makes:
   **5.8%**. The old number would have falsely killed the demographic account.
3. **The per-year prediction was a constant.** A pooled mixture's first harmonic is
   attenuated to ≈3×10⁻⁵, so correlating the observed series with it measured the
   level trend. Each year's *own* fitted law is now the predictor, and the constant
   is reported alongside (`pooled_prediction_abs_c1_replaced`).
4. **Kuiper V had no reference distribution.** It is now calibrated on the same
   replicates as the other four statistics.
5. **The entity-disjoint check existed but was never called.** It is wired in and
   reports 2 of 4 folds confirmed.
6. **The naive i.i.d. null was drawn from a uniform-digit law, not Benford.** A
   uniform *mantissa* is exact Benford (`P(d) = log₁₀(1+1/d)` follows directly);
   the tempting inverse-CDF `log₁₀(1+9u)` instead makes every digit equally likely.
   Getting that backwards swapped the null and made the calibration contrast
   meaningless. It is now checked numerically at every run
   (`metadata.null_sampler_self_test`: uniform-mantissa MAD 0.0007 vs the
   uniform-digit law's 0.0594) and the run aborts if the check fails.

A seventh finding is a correctness issue in the previous output rather than a code
bug: `eval_out.json` did not satisfy the `exp_eval_sol_out` schema (no `datasets`,
all detail at the top level). Detail now nests under `metadata`, and the per-digit
rows are the `datasets` entries.

## Restoring removed files

The only path marked for removal is the Python environment, which is a cache of
the pinned dependencies and is fully reproducible:

```bash
uv sync      # rebuilds .venv/ (359 MB) from pyproject.toml and uv.lock
```

If `uv` is unavailable, the equivalent is `python3.12 -m venv .venv` followed by
`uv pip install` (or `pip install`) of the exact pinned list in `pyproject.toml`.

Everything else stays in place. `data/panel_owid.parquet` and
`data/panel_wdi.parquet` are derived but kept as well (they total under 1 MB);
`eval.py` regenerates them from the two public source files on every run, so
they can be deleted safely at any time. Provenance for the sources is in
`eval_out.json → metadata.inputs_manifest.inputs` (bytes and SHA-256).
