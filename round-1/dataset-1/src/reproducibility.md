# Reproducibility

Step-by-step instructions to reproduce this artifact's exact outputs on Ubuntu.
Everything below is what was **actually run**, not an idealized version. Where a
step only worked because of this server's environment, that is said explicitly.

The task this artifact serves: *how closely does Benford's law fit the leading
digits of the populations of world countries?* The artifact is the **data
foundation** for that question — two country-year population panels with per-row
leading-digit encodings — and contains no derived digit statistics itself.

---

## 1. Getting this artifact

This workspace is published as **one folder of a public GitHub repository**. The
folder is `round-1/dataset-1/src` from the repository
root.

```bash
git clone <REPOSITORY-URL> repo
cd repo/round-1/dataset-1/src
```

All paths below are **relative to that folder**. No path in `data.py`,
`build_panel.py`, `restore.sh` or `pyproject.toml` is absolute — each script
anchors on `Path(__file__).resolve().parent`, so the folder can live anywhere.

No other artifact of this run is needed. This is iteration 1's first artifact and
it declares no dependencies. `restore.sh` and the curl commands in §3 are the only
inputs; every one of them is a public URL.

No user-uploaded private data is used. The `user_uploads` folder was inspected and
contains nothing this artifact reads.

## 2. System packages, Python, venv, and pinned libraries

**Python 3.12** (the build ran on CPython 3.12.14). On Ubuntu 24.04 the system
Python is already 3.12; otherwise install it.

```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv unzip curl git
```

Nothing else is needed at the system level. No compiler, no BLAS, no GPU — this is
pure-stdlib parsing of a few MB of CSV. **No GPU or CUDA is used; the whole build
runs on one CPU core in well under a minute.** Everything was executed on a
CPU-only guest (4 vCPU, 4 GB RAM) — no GPU type or VRAM is relevant here.

Create the environment and install the pinned versions from `pyproject.toml`:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install \
  "loguru==0.8.0" \
  "jsonschema==4.26.0" \
  "jsonschema-specifications==2025.9.1" \
  "referencing==0.37.0" \
  "rpds-py==2026.9.1" \
  "attrs==26.1.0" \
  "typing_extensions==4.16.0"
```

Or, equivalently, from the pinned manifest:

```bash
.venv/bin/python -m pip install .
```

The exact versions actually installed (`.venv/bin/pip freeze` equivalent):

| package | version | role |
|---|---|---|
| `loguru` | 0.8.0 | logging in `build_panel.py` and `data.py` |
| `jsonschema` | 4.26.0 | schema validation in `build/validate_direct.py` |
| `jsonschema-specifications` | 2025.9.1 | transitive dependency of `jsonschema` |
| `referencing` | 0.37.0 | transitive dependency of `jsonschema` |
| `rpds-py` | 2026.9.1 | transitive dependency of `jsonschema` |
| `attrs` | 26.1.0 | transitive dependency of `jsonschema` |
| `typing_extensions` | 4.16.0 | transitive dependency of `jsonschema` |

`data.py` itself imports only the standard library plus `loguru`. `pycountry` was
installed during exploration but is **not** imported by the final code (entity
classification uses an explicit frozen name/code table instead), so it is not a
declared dependency.

> Note on the build environment: the original run created the venv with `uv venv`
> and installed with `uv pip install`. Plain `python -m venv` + `pip` is
> equivalent and is what the commands above use, so no `uv` install is required.
> Network access is needed for the downloads in §3 only.

## 3. Data downloads

Five public files, no authentication, no API key, no environment variable:

| file | URL | md5 (recorded at download) |
|---|---|---|
| `raw/owid_population_unwpp_with_projections.csv` | `https://ourworldindata.org/grapher/population-with-un-projections.csv?v=1&csvType=full&useColumnShortNames=true` | `1a60926a766611e5df3251cedef98590` |
| `raw/owid_population_unwpp_with_projections.metadata.json` | `https://ourworldindata.org/grapher/population-with-un-projections.metadata.json` | `575b2724b3bdb76ebd9e43b2c6fd2d76` |
| `raw/wdi_sp_pop_totl.zip` | `https://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=csv` | `31a5240161839c6ba54747877e8e9071` |
| `raw/wb_countries.json` | `https://api.worldbank.org/v2/country?format=json&per_page=400` | `cd7b5ab1a05991689305d1bbd11db3f5` |
| `raw/sids_ohrlls.html` | `https://www.un.org/ohrlls/content/list-sids` | `8b01cac23d48280843d2d8c5426fd174` |

One command fetches all five:

```bash
./restore.sh
```

`restore.sh` also unzips the WDI archive into `raw/wdi/` and stages everything into
`temp/datasets/`, which is where `data.py` reads from.

**Caveat, stated honestly:** the OWID and World Bank endpoints are live and are
periodically revised (the OWID metadata recorded `lastUpdated 2024-07-12`; the
World Bank archive stamp was `2026-11-11`). If a later download differs, the md5
checks above will not match and the row counts may drift slightly. The published
`full_*.json` files are the frozen record of the vintage this artifact was built
from; `out/manifest.json` stores the same checksums machine-readably. The SIDS
list is additionally frozen as a literal 39-name set inside `build_panel.py`
(`SIDS_MEMBERS`) with its URL and access date, so the SIDS tag cannot silently
degrade if that page changes.

No model or checkpoint downloads. No API keys. No environment variables are read
by any script in this artifact.

## 4. Exact commands, in order

Run from the artifact folder.

```bash
# 0. one-time: environment (see §2) and data (see §3)
./restore.sh

# 1. build the panels: download/normalize/encode/crosscheck/QC/emit intermediates
#    --strip stages if raw/ is already present: use --stage normalize onward
.venv/bin/python build_panel.py --stage all

# 2. build the deliverable: one example per entity-year, grouped by dataset,
#    and write full/mini/preview for each dataset AND for the combined file
.venv/bin/python data.py

# 3. verify: schema + format for all nine emitted files
.venv/bin/python build/validate_direct.py

# 4. verify: 14 integrity assertions per panel, over the EMITTED examples
.venv/bin/python build/qc_emitted.py

# 5. verify: every required row field present on every example
.venv/bin/python build/acceptance.py

# 6. optional: prove the encodings carry signal; audit for prohibited statistics
.venv/bin/python build/sanity_emitted.py
```

**Seeds.** None are used and none are needed: the build is fully deterministic
(pure parsing, sorting and arithmetic; no sampling, no randomness, no model). A
re-run produces a **byte-identical** `full_data_out.json` — verified in this run by
wiping every derived file and rebuilding from `raw/`, which reproduced
`full_data_out.json` at md5 `863a26ce9fe0a07cb77376af4f762ef2` before and after.

**Runtime and hardware.** Steps 1–6 together take roughly 20–30 seconds on one CPU
core of the 4-vCPU CPU-only guest used here, dominated by JSON serialisation. No
GPU. `build_panel.py` is resumable: intermediates persist in `build/state.json` (a
regenerable scratch file, excluded from the published repo), and
`--stage qc` or `--stage emit` can be re-run alone.

**Expected exit codes.** 0 for every command above. `build/qc_emitted.py` and
`build/acceptance.py` exit non-zero on failure by design — they are gates, not
reports. If `build_panel.py --stage all` exits non-zero, its QC gate rejected the
panel; read the `QC FAIL` lines and `out/qc_report.json` rather than proceeding.

## 5. Output files and the numbers a reader should get

Nine JSON deliverables, all validated against
`build/exp_sel_data_out.schema.json`:

| file | datasets | examples | size |
|---|---|---|---|
| `full_data_out.json` | 2 | 32,304 | 34 MB |
| `mini_data_out.json` | 2 | 6 (3 per dataset) | 7 KB |
| `preview_data_out.json` | 2 | 20 (10 per dataset, strings ≤200 chars) | 21 KB |
| `full_population_panel.json` | 1 | 18,012 | 21 MB |
| `mini_population_panel.json` | 1 | 3 | 4 KB |
| `preview_population_panel.json` | 1 | 10 | 11 KB |
| `full_wdi_population.json` | 1 | 14,292 | 13 MB |
| `mini_wdi_population.json` | 1 | 3 | 3.5 KB |
| `preview_wdi_population.json` | 1 | 10 | 10 KB |

Supporting outputs: `out/qc_report.json` (integrity verdict + per-year entity
counts + boundary cases + spot checks), `out/crosscheck.json` (cross-source
agreement buckets, disagreement quantiles, digit-flip counts),
`out/entities.json` (the 237-entity audited table), `out/manifest.json`
(machine-readable provenance with md5s), `build/sanity_emitted.txt`.

**Numbers a reader should be able to confirm** — these are the load-bearing values,
and each is asserted by a script in §4:

| quantity | expected value |
|---|---|
| primary panel examples / entities | 18,012 / 237 |
| companion panel examples / entities | 14,292 / 217 |
| primary year span | 1950–2025, analysis window 1950–2023 |
| entity classes in the primary panel (2023) | `sovereign` 195, `territory` 40, `other` 2 |
| SIDS-tagged entities / rows | 39 / 2,964 |
| projection rows (primary / WDI) | 474 / 434 |
| fold balance (primary) | 9,006 even / 9,006 odd |
| duplicate `(iso_code, year)` and `(entity, year)` keys | 0 and 0, both panels |
| mantissa recomputation disagreements | 0 of 18,012 |
| `d1` string-vs-float agreement (primary) | 0.999944, 1 boundary case (Nauru 1962, `5000`) |
| `d1_reciprocal` agreement | 1.000000 |
| 2023 world total over in-analysis entities | 8,090,655,899 persons (+1.13% vs ~8.0e9) |
| years below 190 entities, 1990–2023 | none (237 every year) |
| largest in 1950 | China 544,044,355; India 346,278,818; United States 154,202,683 |
| largest in 2023 | India 1,438,069,597; China 1,422,584,930; United States 343,477,330 |
| cross-source joined entity-years | 14,226 (of which 5,106 agree within 1e-6) |
| leading-digit flip rate between sources | 0.0225 (320 pairs) |

**Where these appear in the paper.** This artifact is the data foundation for the
paper's Benford's-law analysis of country populations:

- The **primary panel** (`full_population_panel.json`) is the analysis dataset for
  every digit-frequency table, MAD, chi-square (df = 8), Kuiper statistic and
  simultaneous confidence interval the paper reports. Its `in_primary_analysis`
  mask defines the 1950–2023 analysis window.
- The **companion panel** (`full_wdi_population.json`) supplies the paper's
  source-invariance check: the same statistic recomputed on an independently
  sourced series. It is never pooled with the primary panel.
- `out/crosscheck.json` supplies the quantified source cross-check the paper's
  methods section reports, and the per-country disagreement figures (e.g. Moldova's
  ~49.5% split) explained there.
- `metadata_fold` implements the pre-registered split: **even years are the screen
  half, odd years the reserved confirmation half**. Every pooled statistic in the
  paper is computed on the screen half first and only then on the confirmation half.
- `metadata_iso_code` is the cluster key for the entity-level bootstrap the paper
  reports intervals from; resampling must be by entity, not by row.
- `metadata_pop_le_1m`, `metadata_is_sids`, `metadata_is_top20` and
  `metadata_in_primary_analysis_sovereign_only` drive the paper's ablation table
  (microstates ≤ 1M, SIDS membership, the top-20 sparse tail, sovereign-only).
- `metadata_mantissa` is the object every mantissa-based statistic is computed on;
  `metadata_d1_reciprocal` supplies the reciprocal-digit mirror control.
- **No number in this artifact is a paper result.** Digit frequencies, MAD,
  chi-square, Kuiper and any fitted size law are computed downstream by the
  experiment step, from these rows.

## 6. Known environment-specific notes

Recorded so a reader is not surprised:

1. **The `aii-json` skill's CLI is not used by the verification scripts.** Its
   `validate` and `format` entry points refuse any path that does not resolve
   inside the repo root or `/tmp`; this workspace lives on a separate volume and
   `/tmp` is a symlink, so both report `file_path must be under the project
   directory or /tmp` before reading the file. `build/validate_direct.py` therefore
   performs the same `jsonschema.validate` call against the same schema, now
   vendored at `build/exp_sel_data_out.schema.json` so the check is self-contained.
   `data.py` builds its own mini/preview variants using the skill's constants
   (3 items, 200-character strings) rather than calling the skill's script.
2. **`pycountry` is deliberately not a dependency.** Entity classification uses an
   explicit frozen table of sovereign / territory / other names in
   `build_panel.py`, derived from the full 262-name listing during the build and
   eyeballed once, as the artifact plan requires. This removes a data dependency
   that could silently change the entity set when its own version changes.
3. **The World Bank bulk archive's own `Metadata_Country` CSV is not used to
   identify aggregates.** It leaves the `Region` field blank for all 48 aggregate
   series instead of labelling them `Aggregates`. Aggregates are instead identified
   from `raw/wb_countries.json`, where `region.value == "Aggregates"` names 79
   codes. Using the CSV's Region column would have silently admitted every
   aggregate into the country set.
