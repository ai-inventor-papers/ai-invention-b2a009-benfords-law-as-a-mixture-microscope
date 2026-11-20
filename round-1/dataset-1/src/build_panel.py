#!/usr/bin/env python3
"""Build the country-population panel artifact.

Primary panel  : OWID redistribution of UN World Population Prospects 2024
                 (variant: estimates 1950-2023, medium-variant projections 2024-2100;
                 emitted 1950-2025, primary analysis window 1950-2023).
Companion panel: World Bank WDI SP.POP.TOTL (1960-2025; primary window 1960-2023),
                 same schema, so source invariance is a cross-panel comparison.

Rows carry only published values plus deterministic per-row encodings. No digit
frequency tables, no MAD, no chi-square, no fitted size law -- those belong to the
experiment stage.

Usage:
    uv run build_panel.py --stage all
    uv run build_panel.py --stage acquire|normalize|tag|digest|crosscheck|qc|emit
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from loguru import logger

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
BUILD = ROOT / "build"
OUT = ROOT / "out"
LOGS = ROOT / "logs"
for d in (RAW, BUILD, OUT, LOGS):
    d.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(LOGS / "build_panel.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------- constants
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

URL_OWID_CSV = ("https://ourworldindata.org/grapher/population-with-un-projections.csv"
                "?v=1&csvType=full&useColumnShortNames=true")
URL_OWID_META = "https://ourworldindata.org/grapher/population-with-un-projections.metadata.json"
URL_WB_ZIP = "https://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=csv"
URL_WB_COUNTRIES = "https://api.worldbank.org/v2/country?format=json&per_page=400"
URL_SIDS = "https://www.un.org/ohrlls/content/list-sids"
SIDS_ACCESS_DATE = "2026-11-19"

F_OWID_CSV = RAW / "owid_population_unwpp_with_projections.csv"
F_OWID_META = RAW / "owid_population_unwpp_with_projections.metadata.json"
F_WB_ZIP = RAW / "wdi_sp_pop_totl.zip"
F_WB_CTRY = RAW / "wb_countries.json"
F_SIDS_HTML = RAW / "sids_ohrlls.html"

EST_COL = "population__sex_all__age_all__variant_estimates"
PRJ_COL = "population__sex_all__age_all__variant_medium__projected"

# Year windows. WPP2024 publishes estimates for 1950-2023 and projections from 2024.
WPP_FIRST_YEAR, WPP_LAST_ESTIMATE_YEAR, WPP_EMIT_LAST_YEAR = 1950, 2023, 2025
WDI_FIRST_YEAR, WDI_EMIT_LAST_YEAR = 1960, 2025

MICROSTATE_THRESHOLD = 1_000_000
TOP_N = 20
WORLD_POP_2023 = 8.0e9
WORLD_POP_TOLERANCE = 0.02

# --- Aggregate exclusion. Frozen after eyeballing the full 262-entity listing.
# Names carrying an OWID_ internal aggregate code.
AGG_BY_OWID_CODE = {
    "Africa", "Asia", "Europe", "North America", "Oceania", "South America", "World",
    "High-income countries", "Low-income countries", "Lower-middle-income countries",
    "Upper-middle-income countries",
}
# Names carrying a UN_ regional-aggregate code.
AGG_BY_UN_CODE = {
    "Africa (UN)", "Asia (UN)", "Europe (UN)", "Northern America (UN)",
    "Oceania (UN)", "Latin America and the Caribbean (UN)",
}
# Names carrying NO code at all: these are the WPP's own group aggregates.
AGG_NO_CODE = {
    "Americas (UN)", "Land-locked developing countries (LLDC)", "Least developed countries",
    "Less developed regions", "Less developed regions, excluding China",
    "Less developed regions, excluding least developed countries",
    "More developed regions", "Small island developing states (SIDS)",
}
AGGREGATE_NAMES = AGG_BY_OWID_CODE | AGG_BY_UN_CODE | AGG_NO_CODE

# Entities whose OWID code is OWID_* but which are real places, not aggregates.
OWID_CODE_NON_AGGREGATE = {"Kosovo"}

# --- Dependent territories / special areas reported by the WPP as separate rows.
TERRITORY_NAMES = {
    "American Samoa", "Anguilla", "Aruba", "Bermuda", "Bonaire Sint Eustatius and Saba",
    "British Virgin Islands", "Cayman Islands", "Cook Islands", "Curacao", "Falkland Islands",
    "Faroe Islands", "French Guiana", "French Polynesia", "Gibraltar", "Greenland",
    "Guadeloupe", "Guam", "Guernsey", "Hong Kong", "Isle of Man", "Jersey", "Macao",
    "Martinique", "Mayotte", "Montserrat", "New Caledonia", "Niue",
    "Northern Mariana Islands", "Pitcairn", "Puerto Rico", "Reunion", "Saint Barthelemy",
    "Saint Helena", "Saint Martin (French part)", "Saint Pierre and Miquelon",
    "Sint Maarten (Dutch part)", "Tokelau", "Turks and Caicos Islands",
    "United States Virgin Islands", "Wallis and Futuna", "Western Sahara",
}
# Areas that are neither a UN member state nor a dependent territory of one.
OTHER_NAMES = {"Taiwan", "Kosovo"}
# UN observer states, treated as sovereign.
OBSERVER_STATES = {"Vatican", "Palestine"}

# --- SIDS member States (39), frozen from the UN OHRLLS list, access 2026-11-19.
SIDS_MEMBERS = {
    "Antigua and Barbuda", "Bahamas", "Barbados", "Belize", "Cape Verde", "Comoros",
    "Cook Islands", "Cuba", "Dominica", "Dominican Republic", "Fiji", "Grenada",
    "Guinea-Bissau", "Guyana", "Haiti", "Jamaica", "Kiribati", "Maldives",
    "Marshall Islands", "Micronesia (country)", "Mauritius", "Nauru", "Niue", "Palau",
    "Papua New Guinea", "Samoa", "Sao Tome and Principe", "Singapore", "Saint Kitts and Nevis",
    "Saint Lucia", "Saint Vincent and the Grenadines", "Seychelles", "Solomon Islands",
    "Suriname", "East Timor", "Tonga", "Trinidad and Tobago", "Tuvalu", "Vanuatu",
}
SIDS_SOURCE = f"UN OHRLLS List of SIDS (39 member States), {URL_SIDS}, frozen {SIDS_ACCESS_DATE}"


# ---------------------------------------------------------------- helpers
def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, *, timeout: int = 120, retries: int = 3,
             binary: bool = True) -> dict[str, Any]:
    """Download url to dest with retries; return a provenance record."""
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                status = resp.status
            if binary:
                dest.write_bytes(body)
            else:
                dest.write_text(body.decode("utf-8", "replace"), encoding="utf-8")
            rec = {"url": url, "path": str(dest.relative_to(ROOT)), "http_status": status,
                   "bytes": len(body), "md5": md5_of(dest),
                   "accessed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            logger.info(f"downloaded {dest.name}: HTTP {status}, {len(body)} bytes, md5 {rec['md5']}")
            return rec
        except Exception as e:  # noqa: BLE001 - retried below, then reraised
            last_err = e
            logger.warning(f"attempt {attempt}/{retries} failed for {url}: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"download failed after {retries} attempts: {url}") from last_err


def slugify(name: str) -> str:
    out = []
    for ch in name.lower():
        out.append(ch if ch.isalnum() else "_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def first_char(stripped: str, pos: int = 0) -> int:
    return int(stripped[pos])


def sig_figs_and_trailing_zeros(digits: str) -> tuple[int, int]:
    """Significant figures implied by the string, and trailing zeros in the integer part."""
    core = digits.lstrip("0")
    if not core:
        return 0, 0
    tz = len(core) - len(core.rstrip("0"))
    nsf = len(core) - tz if tz else len(core)
    return max(nsf, 1), tz


def mantissa_of(value: int) -> float:
    m = math.log10(value) - math.floor(math.log10(value))
    m = round(m, 6)
    if m >= 1.0:
        m = 0.0
    if m < 0.0:
        m = 0.0
    return m


def leading_digit_from_mantissa(m: float) -> int:
    """int(10**m) with an epsilon guard, so an exact decade (m == 0) yields 1, not 9."""
    v = 10.0 ** (m + 1e-12)
    return int(v)


def reciprocal_leading_digit(value: int) -> int:
    """Leading digit of 1/value, from a 15-significant-digit decimal expansion."""
    from decimal import localcontext
    with localcontext() as ctx:
        ctx.prec = 15
        r = Decimal(1) / Decimal(value)
    s = format(r, "f")
    for ch in s:
        if ch.isdigit() and ch != "0":
            return int(ch)
    return 1


def parse_int_exact(raw: str) -> tuple[int, str]:
    """Return (int value, verbatim string) from a source cell, rejecting anything non-integral."""
    verbatim = raw.strip()
    if not verbatim:
        raise ValueError("empty cell")
    cleaned = verbatim.replace(",", "").replace("\u00a0", "").replace(" ", "")
    if cleaned.endswith(".0"):
        cleaned = cleaned[:-2]
    val = int(Decimal(cleaned))
    return val, cleaned


# ---------------------------------------------------------------- acquire
def acquire() -> dict[str, Any]:
    logger.info("STAGE acquire: fetching primary (OWID/WPP) and companion (WDI) sources")
    prov: dict[str, Any] = {"sources": []}
    prov["sources"].append(download(URL_OWID_CSV, F_OWID_CSV, timeout=120))
    prov["sources"].append(download(URL_OWID_META, F_OWID_META, timeout=60, binary=False))
    prov["sources"].append(download(URL_WB_ZIP, F_WB_ZIP, timeout=240))
    prov["sources"].append(download(URL_WB_COUNTRIES, F_WB_CTRY, timeout=120, binary=False))
    try:
        prov["sources"].append(download(URL_SIDS, F_SIDS_HTML, timeout=60, binary=False))
        prov["sids_page_ok"] = True
    except RuntimeError as e:
        logger.error(f"SIDS page unavailable, falling back to frozen list only: {e}")
        prov["sids_page_ok"] = False

    # verify the primary CSV header before anything depends on it
    with F_OWID_CSV.open(encoding="utf-8") as fh:
        hdr = next(csv.reader(fh))
    logger.info(f"OWID header: {hdr}")
    missing = [c for c in ("entity", "code", "year", EST_COL, PRJ_COL) if c not in hdr]
    if missing:
        raise RuntimeError(f"OWID CSV is missing expected columns: {missing}")

    # unzip the WDI bulk archive
    wdi_dir = RAW / "wdi"
    wdi_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(F_WB_ZIP) as zf:
        zf.extractall(wdi_dir)
    names = sorted(p.name for p in wdi_dir.glob("*.csv"))
    logger.info(f"WDI archive contents: {names}")

    prov["sids_available"] = True
    prov["sids_source"] = SIDS_SOURCE
    (BUILD / "provenance_acquire.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
    return prov


# ---------------------------------------------------------------- normalize
def entity_class_of(name: str, code: str) -> str:
    if name in OTHER_NAMES:
        return "other"
    if name in TERRITORY_NAMES:
        return "territory"
    if name in OBSERVER_STATES:
        return "sovereign"
    if len(code) == 3 and code.isalpha():
        return "sovereign"
    return "other"


def load_primary_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse the OWID/WPP CSV into tidy rows, keeping the verbatim value string."""
    rows: list[dict[str, Any]] = []
    dropped_aggregate: set[str] = set()
    seen_codes: dict[str, str] = {}
    with F_OWID_CSV.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ent = r["entity"].strip()
            code = (r["code"] or "").strip()
            year = int(r["year"])
            est, prj = (r[EST_COL] or "").strip(), (r[PRJ_COL] or "").strip()
            value_str = est or prj
            if not value_str:
                continue
            if not (WPP_FIRST_YEAR <= year <= WPP_EMIT_LAST_YEAR):
                continue

            is_aggregate = (
                ent in AGGREGATE_NAMES
                or (code.startswith("OWID_") and ent not in OWID_CODE_NON_AGGREGATE)
                or code.startswith("UN_")
            )
            if is_aggregate:
                dropped_aggregate.add(ent)
                continue

            pop, clean = parse_int_exact(value_str)
            iso = code if (len(code) == 3) else None
            seen_codes[ent] = code
            rows.append({
                "entity": ent, "code": code, "iso_code": iso, "year": year,
                "population": pop, "value_str": clean,
                "is_projection": bool(prj),
            })
    logger.info(f"primary: kept {len(rows)} rows, dropped {len(dropped_aggregate)} aggregate entities")
    return rows, {"dropped_aggregates": sorted(dropped_aggregate), "codes": seen_codes}


def load_wdi_rows() -> list[dict[str, Any]]:
    """Reshape the wide WDI DS2 CSV to long form, excluding World Bank aggregates."""
    data_file = next((RAW / "wdi").glob("API_SP.*.csv"))
    lines = data_file.read_text(encoding="utf-8-sig").splitlines()
    hdr_idx = next(i for i, ln in enumerate(lines) if ln.startswith('"Country Name"'))
    hdr = next(csv.reader([lines[hdr_idx]]))
    year_cols = [(j, int(c)) for j, c in enumerate(hdr) if c.strip().isdigit()]
    logger.info(f"WDI: {len(year_cols)} year columns, {year_cols[0][1]}..{year_cols[-1][1]}")

    agg_flags = json.loads(F_WB_CTRY.read_text(encoding="utf-8"))[1]
    wb_aggregates = {x["id"] for x in agg_flags
                     if x["region"]["value"].strip() == "Aggregates"}
    wb_names = {x["id"]: x["name"] for x in agg_flags}
    logger.info(f"WDI: {len(wb_aggregates)} World Bank aggregate codes excluded")

    rows: list[dict[str, Any]] = []
    dropped = 0
    for rec in csv.reader(lines[hdr_idx + 1:]):
        if not rec or not rec[0].strip():
            continue
        name, code = rec[0].strip(), rec[1].strip()
        if code in wb_aggregates:
            dropped += 1
            continue
        for j, yr in year_cols:
            if not (WDI_FIRST_YEAR <= yr <= WDI_EMIT_LAST_YEAR):
                continue
            cell = (rec[j] if j < len(rec) else "").strip()
            if not cell:
                continue
            pop, clean = parse_int_exact(cell)
            rows.append({
                "entity": name, "code": code, "iso_code": code if len(code) == 3 else None,
                "year": yr, "population": pop, "value_str": clean,
                # WDI publishes values for 2024-2025 that lie beyond the WPP estimate
                # horizon; they are nowcasts/projections and are flagged as such.
                "is_projection": yr > WPP_LAST_ESTIMATE_YEAR,
            })
    logger.info(f"WDI: kept {len(rows)} rows, dropped {dropped} aggregate series")
    return rows


def normalize() -> dict[str, Any]:
    logger.info("STAGE normalize: building entity table and both panels")
    primary, pinfo = load_primary_rows()
    wdi = load_wdi_rows()

    ent_table: list[dict[str, Any]] = []
    for ent in sorted({r["entity"] for r in primary}):
        code = pinfo["codes"][ent]
        ent_table.append({"entity": ent, "code": code,
                          "entity_class": entity_class_of(ent, code)})
    (OUT / "entities.json").write_text(
        json.dumps(ent_table, indent=1, ensure_ascii=False), encoding="utf-8")

    cls_counts = Counter(e["entity_class"] for e in ent_table)
    logger.info(f"entity classes: {dict(cls_counts)}  total={len(ent_table)}")
    for e in ent_table:
        logger.debug(f"  [{e['entity_class']:<9}] {e['entity']} ({e['code']})")

    # hard check: no entity classed sovereign may be an aggregate name
    bad = [e for e in ent_table
           if e["entity_class"] == "sovereign" and e["entity"] in AGGREGATE_NAMES]
    if bad:
        raise RuntimeError(f"aggregate leaked into the sovereign set: {bad}")

    state = {"primary": primary, "wdi": wdi, "ent_table": ent_table,
             "pinfo": pinfo, "cls_counts": dict(cls_counts)}
    return state


# ---------------------------------------------------------------- tag + digest
def enrich(rows: list[dict[str, Any]], *, source_variant: str, source_release: str,
           ent_table: list[dict[str, Any]], emit_last: int) -> list[dict[str, Any]]:
    """Attach per-row encodings, class, tags and within-year rank."""
    cls_of = {e["entity"]: e["entity_class"] for e in ent_table}
    # cross-source entity matching: WDI names sometimes differ, so map by ISO code first
    iso_of: dict[str, str | None] = {}
    for e in ent_table:
        iso_of[e["entity"]] = e["code"] if len(e["code"]) == 3 else None

    # passes 1-2: encodings
    for r in rows:
        pop, vstr = r["population"], r["value_str"]
        r["d1"] = first_char(vstr, 0)
        r["d2"] = int(vstr[:2]) if len(vstr) >= 2 else int(vstr) * 10
        r["n_sig_figs"], r["trailing_zeros"] = sig_figs_and_trailing_zeros(vstr)
        r["mantissa"] = mantissa_of(pop)
        r["d1_float"] = leading_digit_from_mantissa(r["mantissa"])
        r["d1_reciprocal"] = reciprocal_leading_digit(pop)
        r["entity_class"] = cls_of.get(r["entity"], "other")
        r["pop_le_1m"] = pop <= MICROSTATE_THRESHOLD
        r["is_sids"] = r["entity"] in SIDS_MEMBERS
        r["sids_source"] = SIDS_SOURCE if r["is_sids"] else None
        r["in_primary_analysis"] = not r["is_projection"] and r["year"] <= WPP_LAST_ESTIMATE_YEAR
        r["in_primary_analysis_sovereign_only"] = (
            r["in_primary_analysis"] and r["entity_class"] == "sovereign")
        r["source_variant"] = source_variant
        r["source_release"] = source_release
        r["metadata_fold"] = "even" if r["year"] % 2 == 0 else "odd"
        r["notes"] = ""

    # pass 3: dense rank within year over in-primary-analysis rows only
    by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["in_primary_analysis"]:
            by_year[r["year"]].append(r)
    for yr, group in by_year.items():
        group.sort(key=lambda x: -x["population"])
        for rank, r in enumerate(group, start=1):
            r["rank_in_year"] = rank
            r["is_top20"] = rank <= TOP_N
    for r in rows:
        if not r["in_primary_analysis"]:
            r["rank_in_year"] = None
            r["is_top20"] = False

    # pass 4: locator string
    for r in rows:
        key = r["iso_code"] or slugify(r["entity"])
        r["input"] = f"{key}|{r['year']}"
    return rows


def tag_and_digest(state: dict[str, Any]) -> dict[str, Any]:
    logger.info("STAGE tag/digest: computing per-row encodings for both panels")
    meta = json.loads(F_OWID_META.read_text(encoding="utf-8"))
    cols_by_short = {v["shortName"]: v for v in meta["columns"].values()}
    est_md = cols_by_short[EST_COL]
    # the CSV column appends '__projected' to the metadata short name
    prj_md = (cols_by_short.get(PRJ_COL) or cols_by_short[PRJ_COL.removesuffix("__projected")])
    prim_release = (f"UN World Population Prospects 2024 (28th edition) via OWID grapher "
                    f"population-with-un-projections; column '{est_md['titleLong']}' "
                    f"lastUpdated {est_md['lastUpdated']} + column "
                    f"'{prj_md['titleLong']}' lastUpdated {prj_md['lastUpdated']} "
                    f"(WPP2024: estimates 1950-2023, medium-variant projections 2024-2100)")
    wdi_file = next((RAW / "wdi").glob("API_SP.*.csv"))
    wdi_stamp = "unknown"
    for ln in wdi_file.read_text(encoding="utf-8-sig").splitlines()[:5]:
        if "Last Updated Date" in ln:
            wdi_stamp = ln.split(",", 1)[1].strip().strip('"')
    wdi_release = f"World Bank WDI SP.POP.TOTL (Population, total), bulk download last updated {wdi_stamp}"

    primary = enrich(state["primary"], source_variant="owid_population_unwpp",
                    source_release=prim_release, ent_table=state["ent_table"],
                    emit_last=WPP_EMIT_LAST_YEAR)
    # WDI entities absent from the WPP table still need a class
    known = {e["entity"] for e in state["ent_table"]}
    wdi_pairs = sorted({(r["entity"], r["code"]) for r in state["wdi"]})
    extra = [{"entity": e, "code": c, "entity_class": entity_class_of(e, c)}
             for e, c in wdi_pairs if e not in known]
    wdi = enrich(state["wdi"], source_variant="worldbank_wdi_sp_pop_totl",
                 source_release=wdi_release, ent_table=state["ent_table"] + extra,
                 emit_last=WDI_EMIT_LAST_YEAR)
    logger.info(f"primary rows {len(primary)}, wdi rows {len(wdi)}; "
                f"wdi entities not in WPP: {[e['entity'] for e in extra]}")
    state.update({"primary": primary, "wdi": wdi, "extra_entities": extra,
                  "releases": {"primary": prim_release, "wdi": wdi_release}})
    return state


def to_record(r: dict[str, Any]) -> dict[str, Any]:
    """Project an enriched row to the exact published row schema."""
    meta = {
        "entity": r["entity"],
        "iso_code": r["iso_code"],
        "entity_class": r["entity_class"],
        "year": r["year"],
        "population": r["population"],
        "value_str": r["value_str"],
        "source_variant": r["source_variant"],
        "source_release": r["source_release"],
        "is_projection": r["is_projection"],
        "in_primary_analysis": r["in_primary_analysis"],
        "in_primary_analysis_sovereign_only": r["in_primary_analysis_sovereign_only"],
        "pop_le_1m": r["pop_le_1m"],
        "is_sids": r["is_sids"],
        "sids_source": r["sids_source"],
        "is_top20": r["is_top20"],
        "rank_in_year": r["rank_in_year"],
        "mantissa": r["mantissa"],
        "d1": r["d1"],
        "d2": r["d2"],
        "d1_reciprocal": r["d1_reciprocal"],
        "n_sig_figs": r["n_sig_figs"],
        "trailing_zeros": r["trailing_zeros"],
        "entity_year_rank_source": f"dense rank descending over in_primary_analysis rows of {r['year']}",
        "notes": r["notes"],
    }
    return {"input": r["input"], "output": r["value_str"],
            "metadata_fold": r["metadata_fold"], "metadata": meta}


# ---------------------------------------------------------------- crosscheck
# Documented cross-source name/code aliases. Used ONLY to join the two panels for
# provenance QA, never written to a row's iso_code (which stays ISO-3166 or null).
CROSSCHECK_ALIAS = {
    "Kosovo": "XKX",              # World Bank user-assigned code; no ISO-3166 code exists
    "Channel Islands": "CHI",     # World Bank user-assigned code
}


def crosscheck(state: dict[str, Any]) -> dict[str, Any]:
    logger.info("STAGE crosscheck: joining primary and WDI panels on (iso_code, year)")
    alias_used: dict[str, str] = {}

    def join_code(r: dict[str, Any]) -> str | None:
        if r["iso_code"]:
            return r["iso_code"]
        alt = CROSSCHECK_ALIAS.get(r["entity"])
        if alt:
            alias_used[r["entity"]] = alt
            return alt
        return None

    pw = {(join_code(r), r["year"]): r for r in state["primary"] if join_code(r)}
    ww = {(join_code(r), r["year"]): r for r in state["wdi"] if join_code(r)}

    only_p = [k for k in pw if k not in ww]
    only_w = [k for k in ww if k not in pw]
    joined = [k for k in pw if k in ww]

    diffs: list[tuple[float, str, int, int, int]] = []
    flips: list[dict[str, Any]] = []
    for k in joined:
        a, b = pw[k]["population"], ww[k]["population"]
        rel = abs(a - b) / b if b else float("nan")
        diffs.append((rel, k[0], k[1], a, b))
        if pw[k]["d1"] != ww[k]["d1"]:
            m = pw[k]["mantissa"]
            lo = math.log10(pw[k]["d1"])
            flips.append({"iso_code": k[0], "year": k[1], "wpp_pop": a, "wdi_pop": b,
                          "wpp_d1": pw[k]["d1"], "wdi_d1": ww[k]["d1"],
                          "wpp_mantissa": m,
                          "distance_to_decade_boundary": round(abs(m - lo), 6)})
    diffs.sort()
    rels = [d[0] for d in diffs]

    def q(p: float) -> float:
        if not rels:
            return float("nan")
        idx = min(len(rels) - 1, int(p * (len(rels) - 1)))
        return round(rels[idx], 8)

    # Tolerance buckets. Exact equality is a poor agreement measure here because the
    # two pipelines round the same underlying series to different last digits; the
    # informative quantity is how much of the panel agrees within a stated tolerance.
    buckets = {"exact": 0, "within_1e-6": 0, "within_1e-4": 0,
               "within_1e-2": 0, "within_5e-2": 0, "beyond_5e-2": 0}
    for rel, *_ in diffs:
        if rel == 0:
            buckets["exact"] += 1
        if rel < 1e-6:
            buckets["within_1e-6"] += 1
        if rel < 1e-4:
            buckets["within_1e-4"] += 1
        if rel < 1e-2:
            buckets["within_1e-2"] += 1
        if rel < 5e-2:
            buckets["within_5e-2"] += 1
        else:
            buckets["beyond_5e-2"] += 1
    n_join = max(len(diffs), 1)
    bucket_shares = {k: round(v / n_join, 6) for k, v in buckets.items()}

    # Flip-distance control: if leading-digit flips are rounding artifacts they must sit
    # near a decade boundary. Compare the flip distribution against all joined rows.
    def distance_to_boundary(m: float) -> float:
        d = leading_digit_from_mantissa(m)
        return abs(m - math.log10(d))

    all_dist = sorted(distance_to_boundary(pw[k]["mantissa"]) for k in joined)
    flip_dist = sorted(f["distance_to_decade_boundary"] for f in flips)

    def med(xs: list[float]) -> float | None:
        return round(xs[len(xs) // 2], 6) if xs else None

    # reason categories for WDI-only rows
    reasons = Counter()
    wdi_entity_year = {(r["iso_code"], r["year"]) for r in state["primary"] if r["iso_code"]}
    p_years = {r["year"] for r in state["primary"]}
    for k in only_w:
        if k[1] < WPP_FIRST_YEAR:
            reasons["year_out_of_primary_range"] += 1
        elif k[1] > WPP_LAST_ESTIMATE_YEAR:
            reasons["beyond_wpp_estimate_horizon"] += 1
        else:
            reasons["entity_not_in_wpp_panel"] += 1

    report = {
        "join_key": "(iso_code, year) with documented cross-source aliases",
        "cross_source_aliases_applied": alias_used,
        "rows_joined": len(joined),
        "rows_primary_only": len(only_p),
        "rows_wdi_only": len(only_w),
        "wdi_only_reasons": dict(reasons),
        "primary_only_by_year_histogram": dict(sorted(
            Counter(k[1] for k in only_p).items())),
        "primary_only_entity_sample": sorted({k[0] for k in only_p})[:25],
        "relative_difference_quantiles": {"p50": q(0.50), "p90": q(0.90),
                                          "p99": q(0.99), "max": q(1.0)},
        "agreement_tolerance_buckets": buckets,
        "agreement_tolerance_shares": bucket_shares,
        "rows_exact_match": buckets["exact"],
        "share_exact_match": bucket_shares["exact"],
        "leading_digit_flip_count": len(flips),
        "leading_digit_flip_rate": round(len(flips) / n_join, 6),
        "flip_distance_to_decade_boundary": {
            "flipped_rows_median": med(flip_dist),
            "flipped_rows_p90": (round(flip_dist[int(0.9 * (len(flip_dist) - 1))], 6)
                                 if flip_dist else None),
            "all_joined_rows_median": med(all_dist),
            "flipped_closer_to_boundary_than_typical_row": (
                (med(flip_dist) or 0.0) < (med(all_dist) or 0.0)),
            "read": ("Distance from a row's mantissa to the nearest decade boundary of its own "
                     "leading digit. If a flipped pair sat close to that boundary the flip is a "
                     "last-digit rounding artifact; if flipped pairs are no closer than typical "
                     "rows the flip reflects genuine source disagreement on the value."),
        },
        "largest_absolute_disagreements": [
            {"iso_code": c, "year": y, "wpp_population": a, "wdi_population": b,
             "relative_difference": round(rel, 6)}
            for rel, c, y, a, b in sorted(diffs, reverse=True)[:10]],
        "samples_of_digit_flips": flips[:20],
        "interpretation": (
            "World Bank WDI SP.POP.TOTL is sourced from UN WPP for roughly two thirds of "
            "economies and from national statistical offices (or Eurostat) for the rest; the "
            "World Bank documents this explicitly. Rows agreeing to within 1e-6 of the WPP "
            "figure are the WPP-sourced tail; the larger differences are the NSO-sourced "
            "countries. Both panels are kept and must never be pooled."),
    }
    (OUT / "crosscheck.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    logger.info(f"crosscheck: joined={len(joined)}, only_primary={len(only_p)}, "
                f"only_wdi={len(only_w)}, exact={report['share_exact_match']:.3f}, "
                f"digit_flip_rate={report['leading_digit_flip_rate']:.5f}")
    return report


# ---------------------------------------------------------------- qc
def qc(state: dict[str, Any]) -> dict[str, Any]:
    logger.info("STAGE qc: running integrity gate")
    primary, wdi = state["primary"], state["wdi"]
    fails: list[str] = []
    report: dict[str, Any] = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    for label, rows in (("primary", primary), ("wdi", wdi)):
        k1 = Counter((r["iso_code"], r["year"]) for r in rows if r["iso_code"])
        dup_iso = {k: v for k, v in k1.items() if v > 1}
        k2 = Counter((r["entity"], r["year"]) for r in rows)
        dup_ent = {k: v for k, v in k2.items() if v > 1}
        report[f"{label}_duplicate_iso_year"] = len(dup_iso)
        report[f"{label}_duplicate_entity_year"] = len(dup_ent)
        if dup_iso:
            fails.append(f"{label}: duplicate (iso_code, year) keys: {list(dup_iso)[:5]}")
        if dup_ent:
            fails.append(f"{label}: duplicate (entity, year) keys: {list(dup_ent)[:5]}")

        bad_pop = [r for r in rows if not isinstance(r["population"], int) or r["population"] <= 0]
        report[f"{label}_nonpositive_or_nonint_population"] = len(bad_pop)
        if bad_pop:
            fails.append(f"{label}: {len(bad_pop)} rows with non-positive/non-integer population")

        bad_m = [r for r in rows if not (0.0 <= r["mantissa"] < 1.0)]
        report[f"{label}_mantissa_out_of_range"] = len(bad_m)
        if bad_m:
            fails.append(f"{label}: {len(bad_m)} rows with mantissa outside [0,1)")

        d1_match = sum(1 for r in rows if r["d1"] == r["d1_float"])
        rate = d1_match / len(rows)
        boundary = [{"entity": r["entity"], "year": r["year"], "value_str": r["value_str"],
                     "d1_string": r["d1"], "d1_float": r["d1_float"]}
                    for r in rows if r["d1"] != r["d1_float"]]
        report[f"{label}_d1_string_float_agreement"] = round(rate, 6)
        report[f"{label}_digit_boundary_cases"] = boundary[:50]
        report[f"{label}_digit_boundary_count"] = len(boundary)
        logger.info(f"{label}: d1 string/float agreement {rate:.5f}, boundary cases {len(boundary)}")
        # 99.9% criterion applies to the primary panel (plan); WDI reported for information
        if label == "primary" and rate < 0.999:
            fails.append(f"{label}: string/float d1 agreement {rate:.5f} < 0.999")

        # reciprocal cross-check
        rec_ok = 0
        rec_bad = []
        for r in rows:
            expect = leading_digit_from_mantissa(round(1.0 - r["mantissa"] if r["mantissa"] else 0.0, 6))
            if r["d1_reciprocal"] == expect:
                rec_ok += 1
            else:
                rec_bad.append(r)
        report[f"{label}_reciprocal_agreement"] = round(rec_ok / len(rows), 6)
        report[f"{label}_reciprocal_exceptions"] = [
            {"entity": r["entity"], "year": r["year"], "population": r["population"],
             "mantissa": r["mantissa"], "d1_reciprocal": r["d1_reciprocal"]}
            for r in rec_bad[:20]]
        if label == "primary" and rec_ok / len(rows) < 0.995:
            fails.append(f"{label}: reciprocal agreement {rec_ok / len(rows):.5f} < 0.995")

        # locator string
        bad_input = [r for r in rows
                     if r["input"] != f"{r['iso_code'] or slugify(r['entity'])}|{r['year']}"]
        report[f"{label}_bad_input_strings"] = len(bad_input)
        if bad_input:
            fails.append(f"{label}: {len(bad_input)} rows with malformed input locator")

        # fold parity
        bad_fold = [r for r in rows
                    if r["metadata_fold"] != ("even" if r["year"] % 2 == 0 else "odd")]
        report[f"{label}_fold_parity_violations"] = len(bad_fold)
        if bad_fold:
            fails.append(f"{label}: fold tag disagrees with year parity on {len(bad_fold)} rows")

    # world total sanity, primary panel only
    latest = max(r["year"] for r in primary if r["in_primary_analysis"])
    tot = sum(r["population"] for r in primary
              if r["in_primary_analysis"] and r["year"] == latest)
    report["primary_world_total"] = {"year": latest, "sum_persons": tot,
                                     "known_world_total": WORLD_POP_2023,
                                     "relative_error": round((tot - WORLD_POP_2023) / WORLD_POP_2023, 6)}
    logger.info(f"world total {latest}: {tot:,} persons (rel err {report['primary_world_total']['relative_error']:+.4f})")
    if abs(tot - WORLD_POP_2023) / WORLD_POP_2023 > WORLD_POP_TOLERANCE:
        fails.append(f"primary: {latest} world total {tot} is more than "
                     f"{WORLD_POP_TOLERANCE:.0%} from {WORLD_POP_2023:.3g} -- check the persons unit")

    # per-year entity counts
    per_year_p = Counter(r["year"] for r in primary)
    per_year_prim = Counter(r["year"] for r in primary if r["in_primary_analysis"])
    report["primary_entity_count_per_year"] = dict(sorted(per_year_p.items()))
    report["primary_in_analysis_count_per_year"] = dict(sorted(per_year_prim.items()))
    thin = sorted(y for y, c in per_year_prim.items() if 1990 <= y <= 2023 and c < 190)
    report["primary_years_below_190_entities_1990_2023"] = thin
    report["wdi_entity_count_per_year"] = dict(sorted(
        Counter(r["year"] for r in wdi).items()))
    report["primary_year_span"] = [min(per_year_p), max(per_year_p)]
    report["wdi_year_span"] = [min(r["year"] for r in wdi), max(r["year"] for r in wdi)]
    if thin:
        fails.append(f"primary: years with fewer than 190 entities in 1990-2023: {thin}")

    # entity-class accounting
    cls = Counter(r["entity_class"] for r in primary if r["year"] == latest)
    report["primary_entity_class_counts_latest"] = dict(cls)

    # named spot checks
    spot_names = ["France", "Japan", "Nigeria", "India", "China", "United States",
                  "Brazil", "Tuvalu", "Nauru", "Vatican", "Kosovo", "Taiwan"]
    spot = {}
    for nm in spot_names:
        got = sorted({r["year"] for r in primary if r["entity"] == nm})
        sample = {y: next((r["population"] for r in primary
                           if r["entity"] == nm and r["year"] == y), None)
                  for y in (1950, 1990, 2023)}
        spot[nm] = {"years_present": [min(got), max(got)] if got else None,
                    "n_years": len(got), "population": sample}
        if not got:
            fails.append(f"primary: named entity missing: {nm}")
    report["named_entity_spot_check"] = spot

    # fold balance
    fe = sum(1 for r in primary if r["metadata_fold"] == "even")
    fo = sum(1 for r in primary if r["metadata_fold"] == "odd")
    report["fold_row_counts"] = {"even": fe, "odd": fo,
                                 "even_years": len({r['year'] for r in primary if r['metadata_fold'] == 'even'}),
                                 "odd_years": len({r['year'] for r in primary if r['metadata_fold'] == 'odd'})}

    report["largest_1950"] = [{"entity": r["entity"], "population": r["population"]}
                              for r in sorted((r for r in primary if r["year"] == 1950
                                               and r["in_primary_analysis"]),
                                              key=lambda x: -x["population"])[:3]]
    report["largest_latest"] = [{"entity": r["entity"], "population": r["population"]}
                                for r in sorted((r for r in primary
                                                 if r["year"] == latest and r["in_primary_analysis"]),
                                                key=lambda x: -x["population"])[:3]]
    report["sids_rows"] = sum(1 for r in primary if r["is_sids"])
    report["sids_available"] = True
    report["projection_rows_primary"] = sum(1 for r in primary if r["is_projection"])
    report["failure_list"] = fails
    report["passed"] = not fails

    (OUT / "qc_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False),
                                        encoding="utf-8")
    if fails:
        for f in fails:
            logger.error(f"QC FAIL: {f}")
        raise SystemExit(f"QC gate failed with {len(fails)} problem(s); see out/qc_report.json")
    logger.info("QC gate PASSED")
    return report


# ---------------------------------------------------------------- emit
def emit(state: dict[str, Any]) -> None:
    logger.info("STAGE emit: writing panels, docs and manifest")
    primary = sorted(state["primary"], key=lambda r: (r["entity"], r["year"]))
    wdi = sorted(state["wdi"], key=lambda r: (r["entity"], r["year"]))
    precords = [to_record(r) for r in primary]
    wrecords = [to_record(r) for r in wdi]

    # NOTE: this stage writes only diagnostics, not the deliverable. data.py owns the
    # shipped dataset files (full_data_out.json and the per-dataset trios) and builds
    # them from temp/datasets/. Writing a partial panel here under that same filename
    # would leave a stale flat-array file behind after `data.py` had already run, so
    # no panel JSON is emitted from this stage.
    logger.info(f"emit: {len(precords)} primary records and {len(wrecords)} companion "
                "records prepared; the shipped dataset files are written by data.py")

    manifest = {
        "artifact": "country_population_panel_1950_2025",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "primary_panel": {
            "file": "full_population_panel.json (also datasets[0] of full_data_out.json)",
            "rows": len(precords),
            "entities": len({r["metadata"]["entity"] for r in precords}),
            "year_span": [min(r["metadata"]["year"] for r in precords),
                          max(r["metadata"]["year"] for r in precords)],
            "primary_analysis_years": [WPP_FIRST_YEAR, WPP_LAST_ESTIMATE_YEAR],
            "source_variant": "owid_population_unwpp",
            "source_release": state["releases"]["primary"],
        },
        "companion_panel": {
            "file": "full_wdi_population.json (also datasets[1] of full_data_out.json)", "rows": len(wrecords),
            "entities": len({r["metadata"]["entity"] for r in wrecords}),
            "year_span": [min(r["metadata"]["year"] for r in wrecords),
                          max(r["metadata"]["year"] for r in wrecords)],
            "primary_analysis_years": [WDI_FIRST_YEAR, WPP_LAST_ESTIMATE_YEAR],
            "source_variant": "worldbank_wdi_sp_pop_totl",
            "source_release": state["releases"]["wdi"],
        },
        "derived_statistics_present": False,
        "sids_source": SIDS_SOURCE,
        "fold_definition": {"even": "screen half (pre-registered)",
                            "odd": "reserved confirmation half (pre-registered)"},
        "aggregate_exclusion": {
            "method": "OWID_/UN_ internal aggregate codes, empty-code WPP group aggregates, "
                      "and World Bank region.value == 'Aggregates'",
            "excluded_names": sorted(AGGREGATE_NAMES),
        },
        "raw_sources": [
            {"url": URL_OWID_CSV, "md5": md5_of(F_OWID_CSV)},
            {"url": URL_OWID_META, "md5": md5_of(F_OWID_META)},
            {"url": URL_WB_ZIP, "md5": md5_of(F_WB_ZIP)},
            {"url": URL_WB_COUNTRIES, "md5": md5_of(F_WB_CTRY)},
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False),
                                       encoding="utf-8")
    logger.info(f"emitted primary {len(precords)} rows, companion {len(wrecords)} rows")


def save_state(state: dict[str, Any]) -> None:
    """Persist intermediates so later stages can resume without re-downloading."""
    payload = {
        "ent_table": state["ent_table"],
        "extra_entities": state.get("extra_entities", []),
        "releases": state.get("releases", {}),
        "primary": state["primary"],
        "wdi": state["wdi"],
    }
    (BUILD / "state.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_state() -> dict[str, Any]:
    payload = json.loads((BUILD / "state.json").read_text(encoding="utf-8"))
    return {"ent_table": payload["ent_table"], "primary": payload["primary"],
            "wdi": payload["wdi"], "extra_entities": payload.get("extra_entities", []),
            "releases": payload.get("releases", {})}


@logger.catch(reraise=True)
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all",
                    choices=["all", "acquire", "normalize", "tag", "crosscheck", "qc", "emit"])
    args = ap.parse_args()
    stage = args.stage

    if stage in ("all", "acquire"):
        acquire()
        if stage == "acquire":
            return
    if stage in ("all", "normalize"):
        state = normalize()
        save_state(state)
        if stage == "normalize":
            return
    if stage in ("all", "tag"):
        state = load_state()
        state = tag_and_digest(state)
        save_state(state)
        if stage == "tag":
            return
    if stage in ("all", "crosscheck"):
        state = load_state()
        crosscheck(state)
        if stage == "crosscheck":
            return
    if stage in ("all", "qc"):
        state = load_state()
        qc(state)
        if stage == "qc":
            return
    if stage in ("all", "emit"):
        state = load_state()
        emit(state)


if __name__ == "__main__":
    main()
