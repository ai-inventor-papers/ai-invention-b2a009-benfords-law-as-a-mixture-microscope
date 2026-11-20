"""Stage 1 - panel acquisition, entity filtering, fold tagging.

Two credential-free sources:

* Our World in Data grapher export of UN World Population Prospects (the
  convenient redistribution; 1950-2023 estimates plus projections).
* World Bank WDI SP.POP.TOTL via the v2 API (1960-2025), used only as the
  independent-source cross-check in Stage 8.3.

The OWID file mixes countries with aggregates (World, continents, World Bank
income groups, UN regions).  An unfiltered download silently inserts values
five orders of magnitude above any country and destroys both the size-law fit
and the digit distribution, so the filter is explicit and logged.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger

from config import (AGGREGATE_NAME_BLOCKLIST, MICROSTATE_MAX_POP, OWID_CSV,
                    PROJECTION_START, SIDS_ISO3, TOP_N, WB_CSV, YEAR_MIN)

OWID_URL = ("https://ourworldindata.org/grapher/population-unwpp.csv"
            "?v=1&csvType=full&useColumnShortNames=true")
WB_COUNTRY_URL = "https://api.worldbank.org/v2/country?format=json&per_page=400"
WB_INDICATOR_URL = ("https://api.worldbank.org/v2/country/all/indicator/"
                    "SP.POP.TOTL?format=json&per_page=20000&date=1960:2025&page={page}")
SOURCES_NOTE = {"owid": OWID_URL, "wdi_template": WB_INDICATOR_URL}


# ---------------------------------------------------------------------------
# HTTP with retries
# ---------------------------------------------------------------------------
def http_get(url: str, *, retries: int = 3, timeout: int = 120, backoff: float = 2.0) -> requests.Response:
    """GET with exponential backoff; raises on repeated failure."""
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout,
                                headers={"User-Agent": "benford-panel-screen/1.0"})
            if resp.status_code == 200:
                return resp
            last_exc = RuntimeError(f"HTTP {resp.status_code} for {url}: {resp.text[:300]}")
        except requests.RequestException as exc:      # network-level failure
            last_exc = exc
        wait = backoff ** attempt
        logger.warning(f"GET attempt {attempt}/{retries} failed for {url}: {last_exc}; "
                       f"retrying in {wait:.0f}s")
        time.sleep(wait)
    raise RuntimeError(f"data acquisition failed after {retries} attempts for {url}: {last_exc}")


# ---------------------------------------------------------------------------
# OWID
# ---------------------------------------------------------------------------
def download_owid(path: Path = OWID_CSV) -> Path:
    """Download the OWID UN WPP population grapher export once."""
    if path.exists() and path.stat().st_size > 100_000:
        logger.info(f"OWID export already present: {path} ({path.stat().st_size} bytes)")
        return path
    resp = http_get(OWID_URL)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    logger.info(f"Downloaded OWID export -> {path} ({len(resp.content)} bytes, HTTP 200)")
    return path


def _resolve_population_column(columns: list[str]) -> str:
    """Find the single numeric per-entity-per-year population column."""
    lower = {c.lower(): c for c in columns}
    if "population" in lower:
        return lower["population"]
    cands = [c for c in columns if c.lower().startswith("population")]
    if len(cands) == 1:
        logger.warning(f"no exact 'population' column; selected {cands[0]!r} "
                       f"from {len(columns)} columns: {columns}")
        return cands[0]
    raise KeyError(f"cannot resolve a population column; columns are {columns}")


def load_owid(path=OWID_CSV) -> pd.DataFrame:
    """Load and entity-filter the OWID export (downloading it if needed)."""
    path = download_owid(path)
    raw_header = path.read_text().splitlines()[0]
    logger.info(f"OWID header, verbatim: {raw_header}")

    df = pd.read_csv(path, dtype=str)
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    need = {}
    for want in ("entity", "code", "year"):
        if want not in lower:
            raise KeyError(f"OWID export is missing column {want!r}; got {cols}")
        need[want] = lower[want]
    pop_col = _resolve_population_column(cols)
    need["population"] = pop_col
    logger.info(f"resolved columns: {need}")

    df = df.rename(columns={v: k for k, v in need.items()})
    df = df[["entity", "code", "year", "population"]].copy()
    df["entity"] = df["entity"].astype(str).str.strip()
    df["code"] = df["code"].fillna("").astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    n_before = len(df)

    # --- filter 1: ISO3 code only, never an OWID_/UN_ pseudo-code
    def is_iso3(code: str) -> bool:
        return len(code) == 3 and code.isalpha() and code.upper() == code and not code.startswith("OWID_")

    iso3_mask = df["code"].map(is_iso3)
    dropped_non_iso3 = (df.loc[~iso3_mask, ["entity", "code"]]
                        .drop_duplicates().sort_values("entity").values.tolist())
    df = df[iso3_mask].copy()

    # --- filter 2: explicit aggregate blocklist for anything that survived
    blocked = df["entity"].isin(AGGREGATE_NAME_BLOCKLIST)
    dropped_blocked = (df.loc[blocked, ["entity", "code"]]
                       .drop_duplicates().sort_values("entity").values.tolist())
    df = df[~blocked].copy()

    # --- coerce
    df["raw_str"] = df["population"].astype(str)
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    n_bad = int(df["population"].isna().sum() + (df["population"] <= 0).sum())
    df = df[df["population"].notna() & (df["population"] > 0)].copy()
    df["year"] = df["year"].astype(int)
    df = df[df["year"] >= YEAR_MIN].copy()

    # --- sanity assertions
    med = float(df["population"].median())
    mx = float(df["population"].max())
    if not (med < 1e7 and mx < 2e9):
        raise AssertionError(f"entity filter failed the sanity check: median={med:.3g}, max={mx:.3g}")

    dropped_all = [{"entity": e, "code": c, "reason": "non-iso3-code"} for e, c in dropped_non_iso3]
    dropped_all += [{"entity": e, "code": c, "reason": "aggregate-blocklist"} for e, c in dropped_blocked]
    logger.info(f"OWID rows {n_before} -> {len(df)}; dropped {len(dropped_all)} aggregate entities, "
                f"{n_bad} empty/non-positive values; median pop {med:,.0f}, max {mx:,.0f}")
    logger.info(f"dropped aggregates: {sorted({d['entity'] for d in dropped_all})}")

    df.attrs["dropped_entities"] = dropped_all
    df.attrs["owid_header"] = raw_header
    df.attrs["owid_population_column"] = pop_col
    return df


def build_panel(owid: pd.DataFrame) -> pd.DataFrame:
    """Derive fold, mantissa, digit and tag columns; attach the panel summary."""
    df = owid.copy()
    df["is_projection"] = (df["year"] >= PROJECTION_START).astype(int)
    df["fold"] = np.where(df["year"] % 2 == 0, "even", "odd")
    df["log10_pop"] = np.log10(df["population"].to_numpy(dtype=float))
    frac = df["log10_pop"] - np.floor(df["log10_pop"])
    df["mantissa"] = np.clip(frac, 0.0, 1.0 - 1e-15)
    d1 = np.floor(10.0 ** df["mantissa"].to_numpy() + 1e-12).astype(int)
    df["d1"] = np.clip(d1, 1, 9)
    df["is_integer_value"] = df["raw_str"].map(
        lambda s: bool(s.replace("-", "").replace("+", "").replace(".", "").isdigit()
                       and "." not in s and "e" not in s.lower()))
    df["sig_digits"] = df["raw_str"].map(significant_digits_of_string)

    # --- pre-registered sub-ensemble tags (used by the Stage 8.4 ablation only)
    pop_median_over_years = df.groupby("code")["population"].median()
    micro = set(pop_median_over_years[pop_median_over_years <= MICROSTATE_MAX_POP].index)
    top = set(pop_median_over_years.nlargest(TOP_N).index)
    df["tag_microstate"] = df["code"].isin(micro).astype(int)
    df["tag_sids"] = df["code"].isin(SIDS_ISO3).astype(int)
    df["tag_top20"] = df["code"].isin(top).astype(int)

    summary = panel_summary(df, owid.attrs)
    df.attrs.update(summary)
    df.attrs["dropped_entities"] = owid.attrs.get("dropped_entities", [])
    df.attrs["owid_header"] = owid.attrs.get("owid_header", "")
    df.attrs["owid_population_column"] = owid.attrs.get("owid_population_column", "")
    return df


def significant_digits_of_string(s: str) -> int:
    """Count significant decimal digits in a printed integer or decimal string.

    Trailing zeros in an integer with no decimal point are *not* significant
    (they carry the scale, not the measurement); trailing zeros after a
    decimal point are.
    """
    s = str(s).strip().lstrip("+-")
    if not s or s.lower() in {"nan", "none"}:
        return 0
    if "e" in s.lower():
        mant = s.lower().split("e")[0]
        exp_digits = 0
    else:
        mant, exp_digits = s, 0
    has_point = "." in mant
    core = mant.replace(".", "").lstrip("0")
    if not core:
        return 0
    if has_point:
        return len(core)
    return len(core.rstrip("0")) or 1


def panel_summary(df: pd.DataFrame, attrs: dict) -> dict:
    """Everything Stage 1.8 asks to print, in one dict."""
    even, odd = df[df.fold == "even"], df[df.fold == "odd"]
    e50 = set(df.loc[df.year == df.year.min(), "code"])
    e_max = set(df.loc[df.year == df.year.max(), "code"])
    per_year = (df.groupby("year").size().to_dict())
    return {
        "n_rows": int(len(df)),
        "n_entities": int(df["code"].nunique()),
        "n_even": int(len(even)),
        "n_odd": int(len(odd)),
        "n_entities_even": int(even["code"].nunique()),
        "n_entities_odd": int(odd["code"].nunique()),
        "year_min": int(df.year.min()),
        "year_max": int(df.year.max()),
        "n_years": int(df.year.nunique()),
        "n_projection_rows": int(df.is_projection.sum()),
        "projection_start": PROJECTION_START,
        "entities_gained_over_window": sorted(e_max - e50),
        "entities_lost_over_window": sorted(e50 - e_max),
        "min_population": float(df.population.min()),
        "max_population": float(df.population.max()),
        "median_population": float(df.population.median()),
        "per_year_entity_count": {int(k): int(v) for k, v in sorted(per_year.items())},
        "dropped_entities": attrs.get("dropped_entities", []),
        "n_dropped_entities": len(attrs.get("dropped_entities", [])),
        "owid_header": attrs.get("owid_header", ""),
        "owid_population_column": attrs.get("owid_population_column", ""),
        "sources": SOURCES_NOTE,
    }


# ---------------------------------------------------------------------------
# World Bank WDI (cross-check only)
# ---------------------------------------------------------------------------
def download_wdi(iso3_keep: set[str], path: Path = WB_CSV) -> pd.DataFrame:
    """Fetch SP.POP.TOTL for the real-country ISO3 set only.

    ``country/all`` returns 296 rows of which 217 are actual economies
    (``region.id != 'NA'`` excludes aggregates such as 'Africa Eastern and
    Southern').  Two filters are applied: the API-level aggregate filter and
    the panel's own ISO3 set.
    """
    if path.exists() and path.stat().st_size > 10_000:
        logger.info(f"WDI cache present: {path}; reusing")
        return pd.read_csv(path, dtype={"code": str})

    resp = http_get(WB_COUNTRY_URL)
    meta = resp.json()
    countries = meta[1]
    real = [c for c in countries if c.get("region", {}).get("id") != "NA"]
    api_iso3 = {c["id"] for c in real}
    logger.info(f"WDI country list: {len(countries)} rows, {len(real)} non-aggregate economies")

    all_rows: list[dict] = []
    page = 1
    while True:
        r = http_get(WB_INDICATOR_URL.format(page=page))
        payload = r.json()
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            break
        info = payload[0]
        all_rows.extend(payload[1])
        logger.info(f"WDI page {page}/{info.get('pages')}: {len(payload[1])} rows")
        if page >= int(info.get("pages", 1)):
            break
        page += 1

    keep = api_iso3 & set(iso3_keep)
    rows = []
    for r in all_rows:
        code = r.get("countryiso3code") or r.get("country", {}).get("id", "")
        val = r.get("value")
        yr = r.get("date")
        if code in keep and val is not None and yr is not None:
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            if v > 0:
                rows.append({"code": code, "year": int(yr), "population": v})
    df = pd.DataFrame(rows).drop_duplicates(subset=["code", "year"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"WDI panel written: {len(df)} rows, {df.code.nunique()} economies, "
                f"years {df.year.min()}-{df.year.max()}, iso3 intersection {len(keep)}")
    return df


def write_panel(df: pd.DataFrame, path) -> None:
    """Persist the analysis panel (text CSV; no binary cache)."""
    cols = ["entity", "code", "year", "population", "raw_str", "is_projection", "fold",
            "log10_pop", "mantissa", "d1", "is_integer_value", "sig_digits",
            "tag_microstate", "tag_sids", "tag_top20"]
    path.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(path, index=False, float_format="%.17g")
    logger.info(f"panel written: {path} ({len(df)} rows)")


def load_panel(path) -> pd.DataFrame:
    """Read the panel back, restoring numeric dtypes."""
    df = pd.read_csv(path, dtype={"code": str, "raw_str": str})
    return df
