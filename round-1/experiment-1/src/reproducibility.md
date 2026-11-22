# Reproducing this artifact

Everything here was run on a single machine with **no GPU**: numpy/scipy on
about 8,700 values. The run machine was a macOS arm64 guest with 4 vCPUs and
4 GB for the step; the same commands work unchanged on Ubuntu.

## 1. Get the artifact

This folder is published as one directory of a public GitHub repository. Clone
the repository, then `cd` into this artifact's folder (the one holding
`method.py`):

```bash
git clone <repository-url> aii-run
cd aii-run/round-1/experiment-1/src
```

Every path in the code, in this file and in the scripts is **relative to that
folder**. The code anchors all paths on `Path(__file__)`, so nothing depends on
where the clone lives.

## 2. System packages, Python, virtualenv, exact versions

- Python **3.12** (the run machine had 3.12.10).
- `uv` for environment and package management — `pip` is not present in this
  environment and must not be used.
- On a bare Ubuntu image: `sudo apt-get install -y python3.12 python3.12-venv`
  then install `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- No compiler, no CUDA, no system BLAS tuning is needed; the wheels below are
  all manylinux/arm64 prebuilt.

```bash
uv venv .venv --python=3.12
uv pip install --python=.venv/bin/python -r pyproject.toml
```

`pyproject.toml` pins the exact versions that were installed and run:

| package | version |
|---|---|
| numpy | 2.5.5 |
| scipy | 1.18.1 |
| pandas | 3.1.1 |
| matplotlib | 3.11.2 |
| loguru | 0.8.0 |
| psutil | 7.2.2 |
| requests | 2.34.2 |

Transitive pins (`certifi`, `contourpy`, `cycler`, `fonttools`, `idna`,
`kiwisolver`, `packaging`, `pillow`, `pyparsing`, `python-dateutil`, `tzdata`,
`urllib3`, `charset-normalizer`) are listed under `[tool.aii.pinned_transitive]`
in `pyproject.toml`.

`make_figures.py` and `audit_rederive.py` need only the same environment.
`make_figures.py` shells out to the `aii-data-fig-gen` chart generator, whose
path is the module-level constant `SKILL` at the top of that file — point it at
your copy of that skill's `scripts/chart_gen.py`. Every figure spec is written
to `figures/specs/` first, so the figures can also be re-rendered by any
matplotlib with those numbers.

## 3. Data

No model weights, no checkpoints, no API keys. The panel is acquired at run
time from two credential-free public sources:

- Our World in Data, `population-unwpp` grapher CSV export:
  `https://ourworldindata.org/grapher/population-unwpp.csv?v=1&csvType=full&useColumnShortNames=true`
  → cached at `data/raw/owid_population_unwpp.csv`
- World Bank WDI indicator `SP.POP.TOTL` via
  `https://api.worldbank.org/v2/`, plus the country list used to drop
  aggregates (`region.id == "NA"`) → cached at `data/raw/wb_population.csv`

Both are already downloaded in this workspace, so a re-run is offline. If you
delete them, `src/panel.py` re-downloads them (three retries with exponential
backoff); if the network is unavailable the run stops with the HTTP response
rather than fabricating values.

`data/panel.csv` (17,464 rows, 236 entities, 1950–2023) is the analysis panel
and is committed, so the numbers can be reproduced without any download.

## 4. Exact commands, in order

All runs use the same seed: `SEED = 20261119` (`src/config.py`).

```bash
# (a) gate 1 - smoke: 6 years x 20 entities, ~6 s
uv run --python=.venv/bin/python method.py --smoke

# (b) gate 2 - quick: real panel, reduced replicates (~3 min)
uv run --python=.venv/bin/python method.py --quick

# (c) the run whose numbers the paper reports: full replicate counts
uv run --python=.venv/bin/python method.py
```

Replicate counts in the full run, all from `src/config.py`:
`REPS_POOLED = 10000` (level P), `REPS_HALF = 4000` (levels S1, S2, X),
`N_HARMONIC = 1`, `CHUNK = 500`, `KMAX = 12`, `V1_PANELS = 200`,
`V2_REPS = 50`, `V3_REPS = 200`, ablation `n_boot = 100`.

Runtime on the run machine (4 vCPUs, no GPU), measured:

| stage | wall time |
|---|---|
| panel load + fold tagging | < 1 s |
| instrument identities + interval coverage | 1 s |
| V1 calibration, 200 Benford panels | ~8 min |
| V2 known mixture, 50 repetitions | ~2 min |
| Stage 3, five laws at four levels (EM is the cost) | ~4 min |
| V3 power curve | 1 s |
| Screen, four candidates x four levels | ~3 min |
| Stage 8 extras incl. ablation bootstrap (4 worker processes) | ~4 min |
| **total** | **~22 min** |

Peak resident memory stayed near **1 GB**: every null streams replicates in
blocks of 500, and `null_law_cluster_mantissas` refuses to allocate more than
~35 MB at once.

```bash
# (d) figures, from the saved numbers (not recomputing anything)
.venv/bin/python make_figures.py

# (e) independent re-derivation of the headline numbers + placebo check
.venv/bin/python audit_rederive.py

# (f) the schema-shaped output and its mini/preview variants
#     (any Python with jsonschema-free stdlib works: the script is self-contained)
.venv/bin/python tools/aii_json_format_patched.py --format exp_gen_sol_out --input results/method_out.json
```

`tools/aii_json_format_patched.py` is the `aii-json` skill's
`aii_json_format_mini_preview.py` with exactly one line changed: its sandbox
check compared a **resolved** path against the **unresolved** `Path("/tmp")`,
and on macOS (where `/tmp` is a symlink to `/private/tmp`) that refused every
legitimate input with "input_file must be under the project directory or
/tmp". The fix resolves the allowed directories too; the formatting logic is
byte-for-byte the skill's. The three outputs it writes are
`full_method_out.json` (verbatim), `mini_method_out.json` (first 3 examples of
the single dataset) and `preview_method_out.json` (same, strings truncated to
200 characters), each mirrored at the workspace root.

## 5. Output files and where the numbers go

| file | what it holds | where in the paper |
|---|---|---|
| `results/screen_results.json` | the complete analysis: panel summary, instrument checks, V1/V2/V3, all five size-law fits at four levels, the harmonic budget, the observed battery per level, the power statement, every candidate's envelopes and scores, the verdict, and the post-selection extras | every number quoted |
| `results/method_out.json` (+ `full_`, `mini_`, `preview_`) | the `exp_gen_sol_out`-schema view, with `metadata.headline_answer` carrying the answer to the title question and each level's example carrying the **exact-Benford baseline** beside our harmonic line | the abstract's headline numbers |
| `results/PRE_REGISTRATION.md` | candidate definitions, scoring levels, survivor rule, the sha256 of `screen.py` and `panel.csv`, and the wall-clock timestamp — written before any statistic on the real panel | the pre-registration paragraph |
| `results/audit_rederive.json` | every headline number recomputed by a second, independent code path, plus the placebo results | the reproducibility paragraph |
| `figures/fig1..fig6.pdf/.png` | harmonic budget; digit residuals; per-year `|c₁|` observed vs predicted; rotation family; candidate p-values; power/floor comparison | Figures 1–6 |
| `figures/specs/*.json`, `figures/data/*.json` | each figure's chart spec and the raw values behind it | figure captions |
| `data/panel.csv` | the analysis panel with fold tags, mantissas, digits, precision and the ablation tags | the data section |

Headline numbers a reader should get (all in `results/screen_results.json`):

- `observed.P.n = 8732`, `observed.P.n_entities = 236`
- `observed.P.mad` and `observed.P.mad_nigrini_band` — the conventional verdict
- `observed.P.c1_amp`, `observed.P.c1_phase` — the harmonic reading
- `power_statement.design_effect`, `.n_eff`, `.noise_floor_E_abs_c1`,
  `.md_amp_2sigma` — why the conventional verdict cannot carry weight
- `harmonic_budget.P.predicted_amp` — what the fitted size law predicts
- `verdict.*` and `candidates.*` — the four mechanisms' scores and status
- `extras.odd_fold_for_evaluation` — the reserved half, untouched by selection

## 6. Where this artifact sits among the run's artifacts

`depends_on` is empty by construction: this is the run's first experiment, so it
acquires its own panel rather than reading a staged dataset. Nothing outside
this folder is needed to reproduce it. The **evaluation artifact** consumes
`results/screen_results.json` → `extras.odd_fold_for_evaluation` and
`extras.ablation_predictions`, and it lives in the sibling folder
`3_invention_loop/iter_1/gen_art/gen_art_evaluation_1`; it finds this one
through that relative path. No user-uploaded private data is used anywhere in
this artifact.
