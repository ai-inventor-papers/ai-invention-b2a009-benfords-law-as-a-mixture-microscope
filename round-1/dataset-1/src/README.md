# Country population panel 1950–2025

Two country-year panels of total population, built to test how well Benford's law
describes the leading digits of national populations. **One example per
entity-year**, two independently sourced datasets with an identical schema, and
per-row encodings of every quantity the digit analysis needs — and deliberately
**no** digit statistics.

| dataset | source | examples | entities | years |
|---|---|---|---|---|
| `country_population_panel_1950_2023` | UN World Population Prospects 2024 via Our World in Data | **18,012** | 237 | 1950–2025 (analysis window 1950–2023) |
| `worldbank_wdi_sp_pop_totl` | World Bank WDI `SP.POP.TOTL` | **14,292** | 217 | 1960–2025 (analysis window 1960–2023) |

Read `PROVENANCE.md` before using either panel: it states the source vintages,
the aggregate-exclusion rule, the projection boundary and its evidence, the SIDS
decision, and every deviation from the original data request.

## Layout

| path | what it is |
|---|---|
| **`full_data_out.json`** | **the canonical deliverable** — `{"metadata":…, "datasets":[…]}` with both panels, 32,304 examples, 43 MB |
| `mini_data_out.json` | first 3 examples of each panel (6 total), written by the aii-json formatter |
| `preview_data_out.json` | same 3 examples per panel, strings truncated to 200 chars |
| `full_population_panel.json` | the primary panel alone, as its own one-dataset file |
| `mini_population_panel.json` / `preview_population_panel.json` | its mini / preview variants |
| `full_wdi_population.json` | the companion panel alone, as its own one-dataset file |
| `mini_wdi_population.json` / `preview_wdi_population.json` | its mini / preview variants |
| `data.py` | builds every dataset file above from `temp/datasets/`; one example per row |
| `build_panel.py` | the underlying build: `acquire → normalize → tag → crosscheck → qc → emit`, resumable per stage |
| `pyproject.toml` | exact pinned dependency versions |
| `reproducibility.md` | step-by-step Ubuntu reproduction, with the values a reader should get |
| `out/qc_report.json` | integrity gate over the **emitted examples**, per-year entity counts, boundary cases, spot checks |
| `out/crosscheck.json` | cross-source agreement buckets, disagreement quantiles, digit-flip counts and the flip-distance control |
| `out/entities.json` | the audited entity table: name, code, class for all 237 primary entities |
| `out/manifest.json` | machine-readable provenance: row counts, year spans, exclusions, raw md5s |
| `PROVENANCE.md` | sources, vintages, deviations, integrity results |
| `build/sanity_emitted.txt` | one-off proof that the emitted encodings carry signal |
| `build/exp_sel_data_out.schema.json` | vendored schema, so the validation check runs without the skill installed |
| `temp/datasets/` | staged copies of the raw downloads that `data.py` reads |
| `raw/` | untouched downloads plus `MD5SUMS.txt` (re-downloadable — see below) |
| `build/verify_outputs.py` | the five check groups: schema, qc, acceptance, sanity, audit |
| `build/eval_datasets.py` | per-panel sample-size accounting and MAD (the dataset-suitability check) |
| `build/check_submission.py` | validates the artifact submission file and that every referenced path exists |
| `build/` | the three verification scripts above plus the vendored schema; all text |
| `logs/` | `build_panel.log`, `data.log` |

## Example shape

Each example carries `input`, `output`, `metadata_fold`, and **flat**
`metadata_<name>` fields (the schema only permits keys matching
`^metadata_[a-zA-Z_][a-zA-Z0-9_]*$`):

```json
{"input": "AFG|1950", "output": "7776180", "metadata_fold": "even",
 "metadata_entity": "Afghanistan", "metadata_iso_code": "AFG",
 "metadata_entity_class": "sovereign", "metadata_year": 1950,
 "metadata_population": 7776180, "metadata_mantissa": 0.890766,
 "metadata_d1": 7, "metadata_d2": 77, "metadata_d1_reciprocal": 1, ...}
```

| field | meaning |
|---|---|
| `input` | deterministic locator, `"<ISO3>|<YEAR>"` (e.g. `FRA|1995`); `"<slug>|<YEAR>"` when no ISO code exists |
| `output` | the published value **verbatim** from the source — preserves rounding grain for the rounding-artifact hypothesis |
| `metadata_fold` | `"even"` / `"odd"` on year parity. Pre-registered: **even = screen half, odd = reserved confirmation half** |
| `metadata_entity`, `metadata_iso_code`, `metadata_entity_class` | name; ISO-3166 alpha-3 or `null` (only Kosovo); `sovereign` (195) \| `territory` (40) \| `other` (2: Taiwan, Kosovo) |
| `metadata_year`, `metadata_population` | integer year, integer persons |
| `metadata_value_str` | same string as `output` |
| `metadata_source_variant`, `metadata_source_release` | series id and the full release string |
| `metadata_is_projection` | true beyond the source's estimate horizon (2024–2025) |
| `metadata_in_primary_analysis` | `not is_projection` and year ≤ 2023 |
| `metadata_in_primary_analysis_sovereign_only` | the above **and** `entity_class == "sovereign"` |
| `metadata_pop_le_1m` | microstate tag, population ≤ 1,000,000 |
| `metadata_is_sids`, `metadata_sids_source` | UN OHRLLS Small Island Developing States, 39 member States, frozen list |
| `metadata_is_top20`, `metadata_rank_in_year` | dense rank by descending population within the year, over in-analysis rows only |
| `metadata_mantissa` | fractional part of log10(population), 6 dp, in [0, 1) |
| `metadata_d1`, `metadata_d2` | first significant digit (1–9); first two significant digits (10–99) |
| `metadata_d1_reciprocal` | leading digit of 1/population — the pre-registered mirror control |
| `metadata_n_sig_figs`, `metadata_trailing_zeros` | significant figures implied by the string; trailing zeros in the integer part |
| `metadata_entity_year_rank_source`, `metadata_notes` | rank provenance; free text |

## How to run

```bash
uv venv .venv --python=3.12    # or: python3.12 -m venv .venv
.venv/bin/python -m pip install .   # installs the pinned deps from pyproject.toml

.venv/bin/python build_panel.py --stage all   # download, normalize, encode, crosscheck, QC, emit diagnostics
uv run data.py                                # build full_data_out.json and per-panel full/mini/preview
.venv/bin/python build/verify_outputs.py      # schema + qc + acceptance + sanity + audit (non-zero exit on failure)
```

`data.py` writes all nine dataset files — the combined `full_data_out.json` plus
each panel's own full/mini/preview trio.
`build_panel.py` stages are `acquire`, `normalize`, `tag`, `crosscheck`, `qc`,
`emit`. Intermediate state is written to `build/state.json` (26 MB, a regenerable
scratch file that is excluded from the published repo), so a run resumes without
re-downloading. Its `qc` stage **fails loudly** rather than emitting a panel that
violates an assertion — including a 2023 world total more than 2% from ~8.0e9,
the check that catches a persons-vs-thousands unit error.

Verification scripts (run once, not part of the deliverable):

```bash
uv run build/verify_outputs.py     # all five check groups below; exits non-zero on failure
uv run build/eval_datasets.py      # per-panel sample-size accounting and MAD
```

`build/verify_outputs.py` runs five groups and prints a PASS/FAIL line for each:
**schema** (all nine published files against the vendored `exp_sel_data_out`),
**qc** (uniqueness, positivity, digit agreement, the reciprocal control, the 2023
world total, fold parity — per panel), **acceptance** (all 27 required fields on
every example), **sanity** (all nine digits present, MAD under 5 pp, and no
prohibited derived statistic on any row), and **audit** (ISO-code validity, no
leaked aggregates, file size, and a reader's-eye sample).

### How the mini / preview variants are produced

`data.py` reproduces the `aii-json` skill formatter's behaviour exactly, and this is
verified: `data.py`'s `mini_data_out.json` and `preview_data_out.json` compare **equal
as parsed JSON** to the output of

```bash
python aii_json_format_mini_preview.py --input full_data_out.json --format exp_sel_data_out
```

on the same `full_data_out.json`. Both variants hold the first 3 examples of each
panel; preview additionally truncates every string (top-level metadata included) to
200 characters. `data.py` serialises compactly (`separators=(',',':')`) where the
skill script uses `indent=2`; `data.py` writes the same `indent=2` form, so its output is byte-identical to
the skill formatter's. The file is 43 MB either way, well under the 100 MB limit.

A full clean-room rebuild — delete every output, re-download all five raw files
over HTTP, rerun the two commands above — reproduces `full_data_out.json` at
md5 `a338376ff319aa913078df4669fde237`, byte for byte. See `reproducibility.md`.

## What this artifact does not contain

No per-digit frequency table, MAD, chi-square, Kuiper statistic, harmonic
coefficient, fitted size law, bootstrap envelope, or predicted digit probability
is stored in any row. Those belong to the experiment stage —
`build/sanity_emitted.txt` prints such numbers once to prove the encodings carry
signal, but they are not in the artifact. Verified: no key on any example matches
a banned-statistic name.

## Two rules for downstream steps

1. **Never pool the two datasets.** WDI is sourced from national statistical
   offices for roughly a third of economies, so the companion tests *source*
   invariance, not the same numbers twice. See `PROVENANCE.md` §2.
2. **Resample entities, not rows.** Pooling recycles the same 237 entities across
   76 years, so the effective sample size for a cross-country claim is the number
   of entities. `metadata_iso_code` is on every row as the cluster key, and the
   even/odd fold is pre-registered so the confirmation half stays untouched.

## Restoring removed files

**Nothing needs restoring to use this artifact.** Every dataset file is plain UTF-8
JSON — text — so all of it is always kept, at any size, including the 43 MB
`full_data_out.json`. `raw/` (2.2 MB) and `temp/datasets/` (1.5 MB) sit under the
10 MB auto-keep floor and are kept too. `out/`, `logs/`, `build/provenance_acquire.json`
and every `.py` and `.md` file are text and are kept.

Two entries in `.aii/manifest.yaml` need a decision. Both are regenerable
environment caches — neither is a result:

| path | decision | restore |
|---|---|---|
| `.venv/` | `delete: regenerable` | `uv venv .venv --python=3.12 && uv pip install --python=.venv/bin/python .` — dependencies are pinned in `pyproject.toml`, so this reproduces the exact environment |
| `__pycache__/` | `delete: regenerable` | `uv run data.py` — the interpreter recreates the bytecode cache automatically |

`build/state.json` (25 MB) is also kept, not deleted: it is JSON, so it is text,
and text files need no storage decision at any size. It is a resumable
intermediate — rebuild it with
`uv run build_panel.py --stage normalize && uv run build_panel.py --stage tag`.

Everything else stays in place on the run's storage volume, at the absolute
workspace path it already has. To rebuild the raw downloads from scratch anyway
(they are not deleted):

```bash
./restore.sh    # re-fetches all five public files and re-stages temp/datasets/
```

`raw/MD5SUMS.txt` and `out/manifest.json` record the md5 of every raw file, so a
re-download can be checked against the vintage this artifact was built from.

`full_data_out.json` (43 MB) is the file every later step reads. It is well under
the 100 MB per-file limit, so it travels with the repository.

### Full rebuild from a fresh clone

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install .
./restore.sh                                   # fetch the five source files
.venv/bin/python build_panel.py --stage all    # normalize, encode, crosscheck, QC
.venv/bin/python data.py                       # write the nine dataset files
```

That sequence reproduces `full_data_out.json` byte for byte at md5
`a338376ff319aa913078df4669fde237`.
