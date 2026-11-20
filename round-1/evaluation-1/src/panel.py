"""Stage 1: rebuild the country-year population panel from the two public sources.

No dependency artifact exists in this round (DATASET, EXPERIMENT and EVALUATION
run in parallel), so the panel is rebuilt here under the recipe stated in the
experiment plan, and every confirmation number carries the provenance flag
`parameters_refit_by_evaluation = true`.

Sources
  OWID  `population-unwpp` grapher CSV  -- the UN World Population Prospects
        redistribution; the analysis panel.
  WDI   World Bank `SP.POP.TOTL`        -- the invariance panel.

The digit encoding is attached here, once, so every later stage uses the same
rule: first significant digit of the PUBLISHED figure, base 10, no truncation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# UN OHRLLS membership: the 39 UN member States on the Small Island Developing
# States list (the ~58 figure that also counts associate members is NOT used).
SIDS_39 = {
    "ATG", "BHS", "BRB", "BLZ", "CPV", "COM", "COG", "COK", "CUB", "DMA", "DOM", "FJI",
    "FSM", "GRD", "GNB", "GUY", "HTI", "JAM", "KIR", "MDV", "MHL", "MUS", "NRU", "NIU",
    "PLW", "PNG", "STP", "KNA", "LCA", "VCT", "WSM", "SGP", "SLB", "SUR", "TLS", "TON",
    "TTO", "TUV", "VUT",
}

PRIMARY_YEARS = (1950, 2023)     # WPP 2024 revision: estimates only through 2023
MICROSTATE_THRESHOLD = 1_000_000


def load_owid(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read the OWID UN-WPP grapher CSV and normalise its columns."""
    raw = pd.read_csv(path)
    logger.info("OWID header verbatim: {}", list(raw.columns))
    cols = {c.lower(): c for c in raw.columns}
    entity_c, code_c, year_c = cols.get("entity"), cols.get("code"), cols.get("year")
    pop_c = cols.get("population")
    if pop_c is None:
        numeric = [c for c in raw.columns
                   if c not in (entity_c, code_c, year_c)
                   and pd.api.types.is_numeric_dtype(raw[c])]
        logger.warning("no 'population' column; numeric candidates: {}", numeric)
        if not numeric:
            raise ValueError("no numeric per-entity column in the OWID CSV")
        pop_c = numeric[0]
    logger.info("resolved OWID columns -> entity={} code={} year={} population={}",
                entity_c, code_c, year_c, pop_c)
    df = raw[[entity_c, code_c, year_c, pop_c]].copy()
    df.columns = ["entity", "code", "year", "population"]
    meta = {"source_file": str(path), "header": list(raw.columns),
            "resolved": {"entity": entity_c, "code": code_c, "year": year_c,
                         "population": pop_c}}
    return df, meta


def filter_entities(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Keep 3-letter ISO codes only; log every dropped entity with its row count."""
    df = df.copy()
    df["code"] = df.code.fillna("").astype(str)
    df["year"] = pd.to_numeric(df.year, errors="coerce").astype("Int64")
    df["population"] = pd.to_numeric(df.population, errors="coerce")

    counts = df.groupby("entity").size()
    keep_mask = df.code.str.fullmatch(r"[A-Z]{3}")
    dropped = [
        {"entity": ent, "code": code, "rows": int(counts[ent]),
         "reason": "non-ISO3 code (aggregate/region/rollup)"}
        for ent, code in df[["entity", "code"]].drop_duplicates().itertuples(index=False)
        if not (len(code) == 3 and code.isalpha() and code.isupper())
    ]
    out = df[keep_mask].copy()
    return out, dropped


def add_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Attach log10 population, mantissa, first/second digit and grain diagnostics."""
    out = df.copy()
    out["population"] = out.population.astype(float)
    out = out[(out.population > 0) & out.population.notna()].copy()
    y = np.log10(out.population.values)
    out["log10_pop"] = y
    m = np.clip(y - np.floor(y), 0.0, 1.0 - 1e-15)
    out["mantissa"] = m
    out["d1"] = np.clip(np.floor(10.0 ** m + 1e-12).astype(int), 1, 9)
    two = np.clip(np.floor(100.0 * 10.0 ** m + 1e-9).astype(int), 10, 99)
    out["d2"] = two % 10
    out["is_integer_value"] = np.isclose(out.population, np.round(out.population))

    # grain: significant figures and divisibility of the PUBLISHED value
    def sig_figs(v: float) -> int:
        s = f"{v:.10g}".replace("-", "").replace(".", "").lstrip("0")
        stripped = s.rstrip("0")
        return max(1, len(stripped) if stripped else 1)

    out["significant_figures"] = [sig_figs(v) for v in out.population.values]
    for k in (1, 3, 6):
        out[f"divisible_1e{k}"] = (np.mod(out.population.values, 10.0 ** k) == 0)
    # reciprocal leading digit (the mirror control)
    rec = 1.0 / out.population.values
    ym = np.log10(rec) % 1.0
    out["d1_reciprocal"] = np.clip(np.floor(10.0 ** ym + 1e-12).astype(int), 1, 9)
    return out


def tag_subensembles(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Pre-registered sub-ensemble tags, computed BY RULE and logged before outcomes.

    microstate: published population <= 1,000,000 in the row's own year (a
      row-level tag, so a country can enter and leave the set over time)
    sids:       UN OHRLLS 39-member list
    top20:      the 20 largest countries by MEAN population over the primary
      window (a country-level tag fixed once, so the ablation contrast does not
      move year to year)
    """
    out = df.copy()
    out["is_microstate_row"] = out.population <= MICROSTATE_THRESHOLD
    out["is_sids"] = out.code.isin(SIDS_39)
    mean_pop = out.groupby("code").population.mean()
    top20 = list(mean_pop.nlargest(20).index)
    out["is_top20"] = out.code.isin(top20)

    taglists = {
        "microstate_rule": f"published population <= {MICROSTATE_THRESHOLD:,} in the row's year",
        "microstate_countries_ever": sorted(out.loc[out.is_microstate_row, "code"].unique().tolist()),
        "microstate_row_fraction": float(out.is_microstate_row.mean()),
        "sids_rule": "UN OHRLLS 39 UN member States",
        "sids_countries_present": sorted(set(out.loc[out.is_sids, "code"].unique()) & set(out.code.unique())),
        "sids_row_fraction": float(out.is_sids.mean()),
        "top20_rule": "20 largest countries by mean population over the primary window",
        "top20_countries": top20,
        "top20_row_fraction": float(out.is_top20.mean()),
    }
    return out, taglists


def load_wdi(raw_json: Path, countries_json: Path) -> tuple[pd.DataFrame, dict]:
    """Read the World Bank WDI SP.POP.TOTL panel, keeping only real countries."""
    import json

    payload = json.loads(raw_json.read_text())
    rows = payload[1]
    meta_top = payload[0]
    w = pd.DataFrame([
        {"code": r.get("countryiso3code", ""), "year": int(r["date"]),
         "population_wdi": r["value"], "country_name": r["country"]["value"]}
        for r in rows
    ])
    logger.info("WDI raw rows={} pages={} lastupdated={}", len(w),
                meta_top.get("pages"), meta_top.get("lastupdated"))
    w = w[w.code.str.fullmatch(r"[A-Z]{3}", na=False)]
    w = w.dropna(subset=["population_wdi"])

    # drop regional / income aggregates using the WB country metadata
    cmeta = json.loads(countries_json.read_text())[1]
    aggregates = {c["id"] for c in cmeta if c["region"]["id"] == "NA"}
    n_before = w.code.nunique()
    w = w[~w.code.isin(aggregates)]
    logger.info("WDI: dropped {} aggregate codes; entities {} -> {}",
                len(aggregates & set(w.code.unique())), n_before, w.code.nunique())

    w = w[w.population_wdi > 0].copy()
    y = np.log10(w.population_wdi.values.astype(float))
    w["log10_pop"] = y
    m = np.clip(y - np.floor(y), 0.0, 1.0 - 1e-15)
    w["mantissa"] = m
    w["d1"] = np.clip(np.floor(10.0 ** m + 1e-12).astype(int), 1, 9)
    meta = {"rows": int(len(w)), "entities": int(w.code.nunique()),
            "years": [int(w.year.min()), int(w.year.max())],
            "aggregate_codes_dropped": sorted(aggregates), "lastupdated": meta_top.get("lastupdated")}
    return w, meta


def build_panel(owid_path: Path, wdi_json: Path, wdi_countries: Path,
                out_dir: Path) -> dict:
    """Assemble both panels, verify the split, write CSVs and a manifest."""
    owid_raw, owid_meta = load_owid(owid_path)
    owid, dropped = filter_entities(owid_raw)
    logger.info("OWID: dropped {} non-ISO3 entities ({} rows)",
                len(dropped), sum(d["rows"] for d in dropped))

    owid = owid[(owid.year >= PRIMARY_YEARS[0]) & (owid.year <= PRIMARY_YEARS[1])].copy()
    owid["year"] = owid.year.astype(int)
    owid = add_encodings(owid)
    owid["fold"] = np.where(owid.year % 2 == 0, "even", "odd")
    owid["is_projection"] = False          # WPP estimates only, through 2023
    owid["source"] = "OWID/UN-WPP"
    owid, taglists = tag_subensembles(owid)

    # sanity assertions from the plan
    assert owid.population.median() < 1e7, "median population implausibly large"
    assert owid.population.max() < 2e9, "max population implausibly large"
    assert not (owid.year >= 2024).any(), "projection years leaked into the primary window"

    odd_keys = set(map(tuple, owid.loc[owid.fold == "odd", ["code", "year"]].values))
    even_keys = set(map(tuple, owid.loc[owid.fold == "even", ["code", "year"]].values))
    overlap = odd_keys & even_keys
    assert not overlap, f"folds share {len(overlap)} country-years"

    wdi, wdi_meta = load_wdi(wdi_json, wdi_countries)

    out_dir.mkdir(parents=True, exist_ok=True)
    owid.to_parquet(out_dir / "panel_owid.parquet", index=False)
    wdi.to_parquet(out_dir / "panel_wdi.parquet", index=False)

    per_year = owid.groupby("year").size()
    summary = {
        "owid": owid_meta,
        "wdi": wdi_meta,
        "n_rows": int(len(owid)),
        "n_entities": int(owid.code.nunique()),
        "years": [int(owid.year.min()), int(owid.year.max())],
        "n_even": int((owid.fold == "even").sum()),
        "n_odd": int((owid.fold == "odd").sum()),
        "n_even_country_years": len(even_keys),
        "n_odd_country_years": len(odd_keys),
        "fold_overlap": len(overlap),
        "entities_in_1950": int((owid.year == 1950).sum()),
        "entities_in_2023": int((owid.year == 2023).sum()),
        "per_year_count_min": int(per_year.min()),
        "per_year_count_median": float(per_year.median()),
        "per_year_count_max": int(per_year.max()),
        "population_min": float(owid.population.min()),
        "population_max": float(owid.population.max()),
        "is_projection_boundary": 2023,
        "dropped_entities": dropped,
        "taglists": taglists,
        "grain": {
            "median_significant_figures": float(owid.significant_figures.median()),
            "sigfig_distribution": {int(k): int(v) for k, v in
                                    owid.significant_figures.value_counts().sort_index().items()},
            "fraction_integer": float(owid.is_integer_value.mean()),
            "fraction_divisible_1e1": float(owid.divisible_1e1.mean()),
            "fraction_divisible_1e3": float(owid.divisible_1e3.mean()),
            "fraction_divisible_1e6": float(owid.divisible_1e6.mean()),
        },
    }
    logger.info("panel: {} rows, {} entities, {} even / {} odd, overlap {}",
                summary["n_rows"], summary["n_entities"], summary["n_even"],
                summary["n_odd"], summary["fold_overlap"])
    return summary
