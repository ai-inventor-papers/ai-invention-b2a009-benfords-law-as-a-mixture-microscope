# Reproducing this evaluation

Complete, step-by-step instructions to reproduce the exact numbers in `eval_out.json`
on Ubuntu. Every path here is **relative to this folder** — the repository publishes it
as one directory, so clone the repo and `cd` into it. Nothing depends on a server path.

Verified platform for these instructions: macOS (arm64), Python 3.12.10, runtime
**≈ 105 s on one core**, peak RSS **≈ 1 GB**. Ubuntu behaves the same; only the OS
lines differ.

---

## 1. Get the artifact

```bash
git clone <repository-url>
cd <repository-root>/round-1/evaluation-1/src
```

The folder contains `eval.py`, its stage modules, the two public source files in
`data/`, and the recorded outputs. `eval.py` anchors every path on
`Path(__file__).resolve().parent`, so it runs from any working directory.

## 2. System packages and Python

Ubuntu 22.04 or 24.04:

```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv build-essential
```

Python **3.12** is required (`requires-python = ">=3.12,<3.13"` in `pyproject.toml`;
the code uses PEP 604 `X | None` annotations and `from __future__ import annotations`
throughout, so 3.10+ would work but only 3.12 was tested).

## 3. Environment and exact library versions

[`uv`](https://docs.astral.sh/uv/) is what this run used; plain `venv` + `pip` also works.

```bash
pip install --user uv          # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                        # creates .venv/ and installs the pinned set
```

`pyproject.toml` pins every package to the exact version this run installed:

| package | version | why |
|---|---|---|
| numpy | 2.5.5 | all array arithmetic |
| pandas | 3.1.1 | panel assembly, parquet I/O |
| scipy | 1.18.1 | `norm`, `chi2`, `beta`, `ks_2samp`, `spearmanr` |
| pyarrow | 26.0.1 | parquet backend for pandas |
| loguru | 0.8.0 | logging |
| matplotlib | 3.11.2 | figures only (`make_figures.py`) |
| contourpy 1.4.0, cycler 0.12.1, fonttools 4.68.0, kiwisolver 1.5.1, packaging 26.3, pillow 13.0.0, pyparsing 3.3.3, python-dateutil 4.0.0, tzdata 2026.6 | | matplotlib's own dependencies, pinned for byte-comparable figures |

Equivalent explicit install:

```bash
uv venv .venv --python=3.12
source .venv/bin/activate
uv pip install numpy==2.5.5 pandas==3.1.1 scipy==1.18.1 pyarrow==26.0.1 \
               loguru==0.8.0 matplotlib==3.11.2
```

## 4. Data, downloads, keys

**No downloads and no API keys are needed.** The two public sources are already in
`data/` and are small enough to ship:

| file | bytes | what it is |
|---|---|---|
| `data/owid_population_unwpp.csv` | 552,397 | OWID `population-unwpp` (a redistribution of the UN World Population Prospects 2024 revision) |
| `data/wb_population_raw.json` | 3,528,198 | World Bank WDI indicator `SP.POP.TOTL`, all countries and years |
| `data/wb_countries.json` | 113,954 | World Bank country metadata, used only to drop regional aggregates |

Their SHA-256 digests are recorded in `eval_out.json →
metadata.inputs_manifest.inputs`, so a reader can confirm they have the same bytes.
`data/panel_owid.parquet` and `data/panel_wdi.parquet` are **derived** — `eval.py`
regenerates both from the CSVs/JSON above on every run, so they need not be present.

To refetch the sources yourself:

- OWID: the `population-unwpp` grapher CSV from <https://ourworldindata.org/population>
- WDI: `https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?format=json&per_page=20000&page=<n>`
  (this run's copy was pulled 2026-11-11, 1 page, 17,556 rows)

Only `{OWID, WB}` filenames are read; there is no environment variable and no key.

## 5. Run it, in order

```bash
cd <this folder>            # the one containing eval.py
source .venv/bin/activate   # or prefix everything with `uv run`

python eval.py              # ≈ 105 s, 1 core, peak RSS ≈ 1 GB
python make_figures.py      # ≈ 25 s, writes figures/*.pdf, *.png
python make_variants.py     # writes full_/mini_/preview_eval_out.json
python verify_narrative.py  # asserts 69 claims in narrative.md match eval_out.json
python verify_headline.py   # independent re-derivation + placebo suite (≈ 90 s)
```

`eval.py` is deterministic: the single seed `20261119` (constant `SEED` at the top of
`eval.py`) drives every bootstrap stream. Two consecutive runs produced **byte-identical
`metrics_agg` and byte-identical `decisions`**.

**Hardware:** CPU only, one core. No GPU is used or needed — the workload is ~17,500
numbers and a few 10⁴-perturbation bootstrap loops. The script sets a 6 GB
`RLIMIT_AS` cap and a 3,600 s `RLIMIT_CPU` cap at startup (macOS refuses to lower
`RLIMIT_AS`; the cap is harmless there). `eval.py` also sets no environment variables
and writes only inside this folder.

**Seeds and bootstrap counts**, all fixed at the top of `eval.py`:

| name | value | purpose |
|---|---|---|
| `SEED` | 20261119 | root seed; every stream is `SEED + <offset>` |
| `B_CALIBRATED` | 10,000 | cluster-calibrated exact-Benford null |
| `B_IID` | 2,000 | naive i.i.d. null (calibration contrast) |
| `B_ENVELOPE` | 4,000 | matched size-law envelope |
| `B_FINAL` | 10,000 | reported cluster-bootstrap CIs |
| `B_YEAR` | 400 | per-year `|c₁|` bootstrap |
| `N_RESTARTS` | 20 | EM restarts per mixture fit |
| `MARGIN` | 0.05 | confirmation margin, probability units |

## 6. What you should get

`eval_out.json` (~170 KB) with `metrics_agg` holding 66 numbers. The values a reader
should check against the paper:

| field | expected | where in the paper |
|---|---|---|
| `n_country_years_reserved_fold` / `n_countries` | 8732 / 236 | Table 1, §3 |
| `design_effect_icc` / `icc_within_country_mantissa` | 8.942 / 0.2206 | §1, the aggregation argument |
| `observed_MAD_vs_benford_reserved` | 0.0093455 | Table 2 |
| `observed_M_vs_benford_reserved` | 0.030645 | Table 2 |
| `observed_chi2_vs_benford_reserved` | 91.691 | Table 2, the 12× headline |
| `observed_abs_c1_reserved` / `observed_phase_deg_reserved` | 0.075213 / −89.776 | Table 2 |
| `p_cluster_MAD` … `p_cluster_kuiper_V` | 0.470, 0.927, 0.493, 0.365, 0.887 | Table 3 — **the answer** |
| `p_naive_iid_*` | all < 0.0005 | Table 3, the trap |
| `entity_level_chi2` / `entity_level_chi2_p` | 12.308 / 0.1380 | §1 corroboration |
| `discrimination_overlap_share` | 0.8866 | §2, the two-null overlap |
| `observed_M_vs_model_reserved` / `replication_attenuation_ratio` | 0.014251 / 1.1448 | Table 4, confirmation |
| `per_year_modulus_pearson_r` | 0.6359 [0.2366, 0.8492] | §5 |
| `per_year_year_trend_r` | −0.7167 | §5 |
| `rounding_max_share_explained` | 0.057694 | §6 |
| `ablation_plug_in_passes` | 0.0 (criterion fails) | Table 5 |
| `entity_disjoint_folds_confirmed` | 2 of 4 | §3, the stated limit |
| `source_per_year_rank_correlation` / `source_overlap_near_identical_fraction` | 0.9902 / 0.3585 | §6 |

Also produced, and read by the paper's figures:

- `figures/fig1_digit_residuals.{pdf,png}` — nine per-digit residuals with simultaneous
  intervals (`metadata.per_digit_table`)
- `figures/fig2_per_year_harmonic.{pdf,png}` — observed vs year-specific predicted
  `|c₁(y)|` (`metadata.per_year_series.per_year_table`)
- `figures/fig3_calibrated_conformity.{pdf,png}` — observed/null-median ratios
  (`metadata.conformity_headline.reserved_odd`)
- `results/figure_raw_values.json` — every number behind each figure
- `results/headline_recheck.json` — the independent audit's output

If a number differs from the table above, the most likely causes are a different
pandas/numpy major version changing a float reduction order (the tolerances in
`verify_narrative.py` are 0.3–5 %), or different source bytes — compare the SHA-256
digests first.

## 7. Provenance and what is NOT reproducible here

- **No DATASET or EXPERIMENT artifact was staged** for this round, so the panel is
  rebuilt from the two public sources and the screen is **refitted inside `eval.py`**
  on the even years. Every confirmation number therefore carries
  `parameters_refit_by_evaluation = true`: the confirmation is a *refit-transfer*, not a
  frozen carry-over from another artifact. `eval_out.json →
  metadata.inputs_manifest` records this.
- Everything under `data/` is public and shipped. Nothing private is used.
- `narrative.md` cites only numbers present in `eval_out.json`; `verify_narrative.py`
  checks this mechanically (69 assertions).
