# PROVENANCE

Country population panel 1950–2025, built <2026-11-19>. Every claim below was
checked against the downloaded files in this workspace, not against a
documentation page read from memory.

## 1. Sources

### Primary panel — `datasets[0]` of `full_data_out.json`, named `country_population_panel_1950_2023`

| item | value |
|---|---|
| Series | Population — Sex: all, Age: all |
| Publisher | United Nations, World Population Prospects, **2024 revision** (28th edition) |
| Distribution | Our World in Data grapher slug `population-with-un-projections` |
| CSV URL | `https://ourworldindata.org/grapher/population-with-un-projections.csv?v=1&csvType=full&useColumnShortNames=true` |
| Metadata URL | `https://ourworldindata.org/grapher/population-with-un-projections.metadata.json` |
| Column (estimates) | `population__sex_all__age_all__variant_estimates` — timespan **1950–2023** |
| Column (projections) | `population__sex_all__age_all__variant_medium__projected` — timespan **2024–2100** |
| OWID `lastUpdated` | **2024-07-12** (both columns) |
| OWID `nextUpdate` | 2027-07-12 |
| Unit | people, integer, "measured on 1 July of the year shown" (mid-year) |
| md5 | `1a60926a766611e5df3251cedef98590` |
| rows downloaded / kept | 39,562 / **18,012** |

Two separate variables, not one series: the estimates column and the
projections column **never overlap** — no entity-year carries both (verified: 0
rows with both columns populated, intersection of their year sets is empty).
The estimate/projection boundary is therefore a property of the source, not an
assumption of this build.

### Companion panel — `datasets[1]` of `full_data_out.json`, named `worldbank_wdi_sp_pop_totl`

| item | value |
|---|---|
| Indicator | `SP.POP.TOTL` — Population, total |
| Publisher | World Bank, World Development Indicators |
| URL | `https://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=csv` |
| Archive stamp | "Last Updated Date", **2026-11-11** |
| Coverage | 1960–2025, 66 year columns |
| Unit | persons, mid-year (per the indicator's own `SOURCE_NOTE`) |
| md5 | `31a5240161839c6ba54747877e8e9071` |
| rows kept | **14,292** (217 entities) |

### Supporting files

| file | role | md5 |
|---|---|---|
| `raw/wb_countries.json` | World Bank country metadata, `region.value == "Aggregates"` flag | `cd7b5ab1a05991689305d1bbd11db3f5` |
| `raw/sids_ohrlls.html` | UN OHRLLS List of SIDS, 39 member States | `8b01cac23d48280843d2d8c5426fd174` |

All five raw files were fetched on 2026-11-19 with a browser-like User-Agent
and HTTP 200 confirmed at download time; `raw/MD5SUMS.txt` holds the checksums.

## 2. The one provenance finding that changes how the companion panel may be used

**World Bank `SP.POP.TOTL` is not simply the UN WPP series.** The World Bank
documents that WDI population figures come from (1) the UN WPP and (2) national
statistical offices, collected directly or via Eurostat, and that the NSO
source is preferred when it meets four stated criteria (complete birth/death
registration, a census within the past 15 years or administrative registers,
annual NSO *estimates* — not projections — in a time series, and a
territory definition matching the World Bank's). The World Bank states these
criteria are met for **roughly one third of all economies**. (Source: World
Bank Open Data blog, "Understanding population estimates in the World Development
Indicators", E. Suzuki and D. G. Mahler; the indicator's own metadata
`SOURCE_ORGANIZATION` field names `World Population Prospects, United Nations`.)
The indicator's own metadata also defines the values as **midyear estimates**.

This is measurable in the panel itself, and it is why the two panels must never
be pooled. Of 14,226 joined entity-years:

| agreement with the WPP figure | rows | share |
|---|---|---|
| exact | 554 | 3.9% |
| within 1e-6 | 5,106 | 35.9% |
| within 1e-4 | 8,644 | 60.8% |
| within 1e-2 | 12,674 | 89.1% |
| within 5e-2 | 13,851 | 97.4% |
| beyond 5e-2 | 375 | 2.6% |

Countries at relative difference ≈ 0 for every year (Nigeria, India, Brazil,
Syria, Venezuela) are the WPP-republished tail; countries with a stable
systematic offset across all decades are the NSO-sourced tail (France ≈ +2.1%,
Japan +0.06% to +3.4%, China −?/+1.8%, United States +0.2% to +2.0%,
Germany +0.005% to +1.5%). The largest disagreement is **Moldova**, ≈ +49.5% on
the WPP side for every year 1960–1995 — a definitional split (Transnistria),
not a data error. Full details in `out/crosscheck.json`.

## 3. Deviations from the artifact direction, stated explicitly

### (a) The primary source is WPP **estimates only** through 2023; 2024–2025 are projections

The direction asked for a 1950–2025 panel. WPP 2024 publishes **estimates for
1950–2023** and **projections from 2024**. Rows in 2024–2025 are therefore
carried but flagged `is_projection = true` and `in_primary_analysis = false`.
**The primary analysis window is 1950–2023 (74 years)**. Projected digits are
partly an output of the UN's own projection model and are scientifically a
different object; including them unflagged would contaminate the quantity being
measured. The alternative ("population-unwpp", estimates only) was also
downloaded and confirmed to end at 2023 — but it lacks the 2024–2025 rows the
direction asked for, so the projections-carrying slug was chosen and the flag
does the work.

The same rule applies to the companion: WDI publishes values for 2024 and 2025
(265 series each), all of which lie beyond the WPP estimate horizon, so they are
flagged `is_projection = true` and excluded from `in_primary_analysis`.

### (b) Two panels, not one merged panel

Source invariance is only meaningful as a comparison between independently
sourced panels. They are `datasets[0]` and `datasets[1]` of `full_data_out.json`, share an
identical schema and were produced by the same enrichment function. They are not
columns of one table and must not be concatenated.

### (c) Territories are kept and tagged, not dropped

The direction's "≥ 195 countries" target would silently drop dependent
territories, and many microstates are territories — exactly the units the
microstate ablation needs. All WPP areas are kept and classified, so the
headline entity count is **237**, not ~195:

| `entity_class` | entities | meaning |
|---|---|---|
| `sovereign` | 195 | ISO-3166 alpha-3 assigned, plus UN observer States (Vatican, Palestine) |
| `territory` | 40 | dependent territories the WPP reports separately |
| `other` | 2 | Taiwan, Kosovo (no UN/ISO assignment; Kosovo has no ISO-3166 code) |

A reviewer wanting the narrow comparison uses
`in_primary_analysis_sovereign_only`, which is true only for `sovereign` rows.
The one null `iso_code` is **Kosovo**; its OWID code is `OWID_KOS`, an internal
OWID identifier, so it is recorded as null rather than passed through as if it
were ISO.

### (d) No derived statistics

No per-digit frequency, MAD, chi-square, Kuiper statistic, harmonic coefficient,
fitted lognormal or mixture parameter, bootstrap envelope, or predicted digit
probability appears in any emitted file. The permitted aggregates are all
provenance/QA: entity counts per year, cross-source disagreement quantiles and
digit-flip counts, all in `out/qc_report.json` and `out/crosscheck.json`.

### (e) SIDS uses the 39 member States, not the ~58 figure

The direction cited "~58 SIDS", which counts associate members of UN Regional
Commissions as well. The UN OHRLLS list gives **39 member States** and, separately,
17 associate members. The build tags the 39 member States from a frozen list —
that yields 2,964 rows. Associate members (American Samoa, Anguilla, Aruba,
Curacao, French Polynesia, Guadeloupe, Puerto Rico, Sint Maarten, Turks and
Caicos Islands, and the rest of that block) are **not** tagged `is_sids`.
The frozen list is stored in `SIDS_MEMBERS` in `build_panel.py` with the URL and
access date, so the ablation cannot silently degrade if the page changes.

## 4. Aggregate exclusion — the single highest-consequence rule

25 aggregate rows are in the WPP CSV and none survives. They are excluded by an
explicit, auditable rule, not by trusting a code column:

1. **OWID internal aggregate codes** (`OWID_*`): Africa, Asia, Europe,
   North America, Oceania, South America, World, High-income countries,
   Low-income countries, Lower-middle-income countries, Upper-middle-income
   countries. Note `Kosovo` also carries an `OWID_KOS` code and is a real place,
   so the rule is "starts with `OWID_` **and** not in the non-aggregate set".
2. **UN regional aggregate codes** (`UN_*`): Africa (UN), Asia (UN), Europe (UN),
   Northern America (UN), Oceania (UN), Latin America and the Caribbean (UN).
3. **WPP group aggregates carrying no code at all**: Americas (UN), Land-locked
   developing countries (LLDC), Least developed countries, Less developed
   regions, Less developed regions excluding China, Less developed regions
   excluding least developed countries, More developed regions, and
   **Small island developing states (SIDS)** — note the last is an aggregate in
   the WPP file and its own entity; the per-country SIDS tag comes from OHRLLS
   instead, never from this row.
4. **World Bank** rows whose `region.value == "Aggregates"` in
   `raw/wb_countries.json` — **79 codes** (AFE, AFR, AFW, ARB, … WLD, XZN).
   The bulk archive's own `Metadata_Country_*.csv` leaves the Region field blank
   for all 48 aggregate series rather than labelling them "Aggregates", so the
   country-API flag is the reliable one; using the CSV's `Region` column would
   have let every aggregate through.

Why it matters: aggregate rows are perfectly collinear with the country values
that compose them, they concentrate mass at the largest magnitudes, and they
would inflate n while pulling the leading-digit distribution toward 1 — a
contamination that looks like signal.

A hard assertion enforces this: no entity classed `sovereign` may be an
aggregate name, and the build raises if one is.

## 5. Cross-source digit flips — measured, not assumed

320 joined pairs (2.25% of 14,226) have a **different leading digit** in the
two panels. The control is informative: a flipped pair whose mantissa sits near
a decade boundary is a last-digit rounding artifact; one far from the boundary
is genuine source disagreement. The flipped rows' median distance to the nearest
decade boundary is **0.0256**, against **0.0526** for all joined rows — flips do
cluster closer to boundaries, as rounding artifacts must, while the digit-flip
rate as a whole is driven by the NSO-sourced countries whose values genuinely
differ. Both numbers are in `out/crosscheck.json`.

## 6. Integrity gate

`out/qc_report.json` records every check. Summary of what passed:

| check | result |
|---|---|
| duplicate `(iso_code, year)` / `(entity, year)` keys | 0 / 0 in both panels |
| population > 0 and integer | all rows |
| mantissa in [0, 1) | all rows; recomputation from `population` disagrees on 0 / 18,012 |
| string-derived `d1` vs float-derived `d1` | **0.999944** agreement, 1 boundary case listed (Nauru 1962, `5000`, string 5 vs float-derived 4 — an exact-decade case) |
| `d1_reciprocal` vs `10^(1−m)` | **1.000000** agreement |
| 2023 sum over `in_primary_analysis` | **8,090,655,899** persons, **+1.13%** vs the ~8.0e9 world total (catches the persons-vs-thousands error) |
| years with < 190 entities, 1990–2023 | **none** |
| per-year entity counts 1950–2025 | 237 every year |
| fold balance | 9,006 even / 9,006 odd rows; 38 even years, 38 odd years |
| locator format | every row equals `<iso_code or slug>|<year>` |
| spot checks | France, Japan, Nigeria, India, China, United States, Brazil, Tuvalu, Nauru, Vatican, Kosovo, Taiwan all present in all 76 years |
| largest in 1950 | China 544,044,355; India 346,278,818; United States 154,202,683 |
| largest in 2023 | India 1,438,069,597; China 1,422,584,930; United States 343,477,330 |

### 6a. Two defects found in review, and what fixed them

Both were caught after the first build and are fixed in the shipped files. They
are recorded here because the md5 of `full_data_out.json` changed as a result:
the original build was `863a26ce…`, the bug fix moved it to `db01c3db…`, and it
is now `a338376f…` because the file is serialised with `indent=2` (the aii-json
formatter's format) instead of compactly. The last change is formatting only —
the parsed JSON values are unchanged, and `data.py` reproduces the current file
byte for byte. Anyone holding an earlier copy should re-pull.

**(i) The WDI release stamp carried a trailing `",`.** The World Bank bulk
archive's header block is quoted CSV with CRLF endings —
`"Last Updated Date","2026-11-11",` — and the original extraction split the line
on the first comma and stripped quote characters, which left
`World Bank WDI SP.POP.TOTL (Population, total), bulk download last updated 2026-11-11",`
in `metadata_source_release` on all 14,292 companion rows. The stamp is now read
with `csv.reader`, which parses the quoted field correctly. An audit of every
string-valued `metadata_*` field across both panels confirms this was the only
field affected; the value is now the clean `…last updated 2026-11-11`.

**(ii) `uv run data.py` could not build.** `pyproject.toml` declared
`build-system.requires = []` with the setuptools backend, so uv's build step
failed with `ModuleNotFoundError: No module named 'setuptools'`. It now requires
`setuptools>=68` and declares the two flat modules explicitly, so
`uv run data.py` works from a clean environment. `uv run` is what the README
documents, so this was a broken instruction, not just a broken convenience.

## 7. What the panel contains, and what it deliberately does not

The panel carries published values plus deterministic per-row encodings only:
the verbatim value string, base-10 mantissa, leading digit, first two digits,
reciprocal leading digit, significant-figure and trailing-zero counts, the
pre-registered even/odd year-parity fold, and the microstate / SIDS / top-20
ablation tags. `build/sanity_emitted.txt` holds a one-off check confirming the
encodings carry signal (MAD 0.89 pp overall; 0.47 pp for non-microstates;
2.49 pp for microstates ≤ 1M; even and odd halves within 0.02 pp of each other).
That file is a construction check, not a result, and no digit frequency table
was written into any deliverable.

## 8. On-disk layout as the exchange format requires

`full_data_out.json` is a single object, `{"metadata": …, "datasets": [ … ]}`,
with exactly two entries, each `{"dataset": "<name>", "examples": [ … ]}` and one
example per entity-year (18,012 + 14,292 = 32,304 examples, 34 MB). Each example
carries `input`, `output`, `metadata_fold` and **flat** `metadata_<name>` keys —
the pipeline schema only admits per-example keys matching
`^metadata_[a-zA-Z_][a-zA-Z0-9_]*$`, so the per-row fields recorded in §4 of the
artifact direction appear here as `metadata_entity`, `metadata_mantissa`,
`metadata_d1` and so on rather than inside a nested `metadata` object. Nothing is
lost in the flattening. `build/verify_outputs.py` asserts every one of
the 27 required row fields is present on all 32,304 examples, re-runs the per-panel
integrity assertions, checks that all nine leading digits occur and that MAD is
under 5 pp, and audits ISO codes and leaked aggregates — exit code 0 only if every
group passes. `out/qc_report.json` holds the build-time gate's own verdict.

A note on the `aii-json` skill's two CLIs, since both were used to produce and check
the shipped files. They refuse a path that does not resolve inside the repo root or
`/tmp`, and on this machine `/tmp` is a symlink to `/private/tmp`, so both report
`must be under the project directory or /tmp` for a workspace on another volume.
The workaround used, and the one a reader should use, is to copy the file and the
skill's `schemas/` directory into a plain directory under `/tmp` and run the CLIs
from there with the *real* resolved path — after which
`aii_json_validate_schema.py --format exp_sel_data_out` reports **`Validation
PASSED`** on `full_data_out.json`, `full_population_panel.json` and
`full_wdi_population.json`. The skill's formatter was run the same way; `data.py`
reproduces its output and the two are **equal as parsed JSON** for both mini and
preview. `build/verify_outputs.py` is the in-repo gate that needs no such
workaround: it validates all nine files against the vendored copy of the same
schema (`build/exp_sel_data_out.schema.json`).
