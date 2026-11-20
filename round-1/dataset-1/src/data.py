#!/usr/bin/env python3
"""Build the experiment's data artifact from the staged country-population sources.

Reads from temp/datasets/ (staged copies of the raw downloads), reuses the
normalization and per-row encoding logic in build_panel.py, and emits
full_data_out.json in the pipeline's exp_sel_data_out layout:

    {"datasets": [{"dataset": "<name>", "examples": [<one example per data ROW>]}, ...]}

Every example carries FLAT metadata_<name> fields (not a nested metadata object),
because that is what the schema permits.

Two datasets, each independently sourced, identical schema:
  * country_population_panel_1950_2023  -- UN WPP 2024 via Our World in Data
  * worldbank_wdi_sp_pop_totl          -- World Bank WDI SP.POP.TOTL

Usage:
    uv run data.py
    .venv/bin/python data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import build_panel as bp  # noqa: E402  (path insert must come first)

DATA_DIR = ROOT / "temp" / "datasets"
OUT = ROOT / "out"
LOGS = ROOT / "logs"
OUT.mkdir(exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(LOGS / "data.log", rotation="30 MB", level="DEBUG")

DATASET_PRIMARY = "country_population_panel_1950_2023"
DATASET_WDI = "worldbank_wdi_sp_pop_totl"

# Per-dataset output stems, so each dataset ships its own full/mini/preview trio.
STEM_PRIMARY = "population_panel"
STEM_WDI = "wdi_population"

MINI_ITEMS = 3          # matches the aii-json formatter's MAX_ARRAY_ITEMS
PREVIEW_ITEMS = 10      # the exchange format asks for 10 preview examples
MAX_STRING = 200        # matches the aii-json formatter's MAX_STRING_LENGTH


def assert_staged() -> None:
    """Fail loudly if the staged inputs are missing or the wrong shape."""
    required = [
        DATA_DIR / "owid_population_unwpp_with_projections.csv",
        DATA_DIR / "owid_population_unwpp_with_projections.metadata.json",
        DATA_DIR / "wb_countries.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"staged source missing: {missing}; run build_panel.py --stage acquire")
    wdi = sorted((DATA_DIR / "wdi").glob("API_SP.*.csv"))
    if not wdi:
        raise FileNotFoundError(
            f"no WDI data CSV under {DATA_DIR / 'wdi'}; run build_panel.py --stage acquire")
    logger.info(f"staged sources present; WDI file {wdi[0].name}")


def to_example(r: dict[str, Any]) -> dict[str, Any]:
    """Project an enriched row to one flat-metadata example.

    Accepts the enriched intermediate dict (fields at top level), which is what
    build_panel.tag_and_digest returns, or an already-projected record.
    """
    m = r["metadata"] if "metadata" in r else r
    rank_source = (m.get("entity_year_rank_source")
                   or f"dense rank descending over in_primary_analysis rows of {m['year']}")
    return {
        "input": r["input"],
        "output": r["output"] if "output" in r else m["value_str"],
        "metadata_fold": r["metadata_fold"],
        # --- flat per-example metadata (schema requires ^metadata_<name>$) ---
        "metadata_entity": m["entity"],
        "metadata_iso_code": m["iso_code"],
        "metadata_entity_class": m["entity_class"],
        "metadata_year": m["year"],
        "metadata_population": m["population"],
        "metadata_value_str": m["value_str"],
        "metadata_source_variant": m["source_variant"],
        "metadata_source_release": m["source_release"],
        "metadata_is_projection": m["is_projection"],
        "metadata_in_primary_analysis": m["in_primary_analysis"],
        "metadata_in_primary_analysis_sovereign_only": m["in_primary_analysis_sovereign_only"],
        "metadata_pop_le_1m": m["pop_le_1m"],
        "metadata_is_sids": m["is_sids"],
        "metadata_sids_source": m["sids_source"],
        "metadata_is_top20": m["is_top20"],
        "metadata_rank_in_year": m["rank_in_year"],
        "metadata_mantissa": m["mantissa"],
        "metadata_d1": m["d1"],
        "metadata_d2": m["d2"],
        "metadata_d1_reciprocal": m["d1_reciprocal"],
        "metadata_n_sig_figs": m["n_sig_figs"],
        "metadata_trailing_zeros": m["trailing_zeros"],
        "metadata_entity_year_rank_source": rank_source,
        "metadata_notes": m.get("notes", ""),
    }


def build_state() -> dict[str, Any]:
    """Re-run normalization + encoding straight from temp/datasets/."""
    assert_staged()
    # point the shared build at the staged copies so this script is self-contained
    bp.F_OWID_CSV = DATA_DIR / "owid_population_unwpp_with_projections.csv"
    bp.F_OWID_META = DATA_DIR / "owid_population_unwpp_with_projections.metadata.json"
    bp.F_WB_CTRY = DATA_DIR / "wb_countries.json"
    bp.RAW = DATA_DIR

    logger.info("normalizing primary (OWID/WPP) and companion (WDI) panels")
    state = bp.normalize()
    state = bp.tag_and_digest(state)
    return state


def truncate(value: Any) -> Any:
    """Recursively truncate strings to MAX_STRING, and lists to PREVIEW_ITEMS."""
    if isinstance(value, list):
        return [truncate(v) for v in value]
    if isinstance(value, str) and len(value) > MAX_STRING:
        return value[:MAX_STRING] + "..."
    if isinstance(value, dict):
        return {k: truncate(v) for k, v in value.items()}
    return value


def write_variants(payload: dict[str, Any], stem: str) -> dict[str, Path]:
    """Write full / mini / preview variants of one datasets-grouped payload."""
    def with_n(n: int | None, do_truncate: bool = False) -> dict[str, Any]:
        ds = []
        for d in payload["datasets"]:
            exs = d["examples"] if n is None else d["examples"][:n]
            if do_truncate:
                exs = [truncate(e) for e in exs]
            ds.append({**d, "examples": exs})
        return {**payload, "datasets": ds}

    paths = {
        "full": ROOT / f"full_{stem}.json",
        "mini": ROOT / f"mini_{stem}.json",
        "preview": ROOT / f"preview_{stem}.json",
    }
    bodies = {
        "full": with_n(None),
        "mini": with_n(MINI_ITEMS),
        "preview": with_n(PREVIEW_ITEMS, do_truncate=True),
    }
    for kind, path in paths.items():
        path.write_text(json.dumps(bodies[kind], separators=(",", ":"), ensure_ascii=False),
                        encoding="utf-8")
        n = sum(len(d["examples"]) for d in bodies[kind]["datasets"])
        logger.info(f"wrote {path.name}  datasets={len(bodies[kind]['datasets'])} "
                    f"examples={n}  {path.stat().st_size/1024:.1f} KB")
    return paths


def main() -> None:
    state = build_state()
    primary = sorted(state["primary"], key=lambda r: (r["entity"], r["year"]))
    wdi = sorted(state["wdi"], key=lambda r: (r["entity"], r["year"]))

    prim_ex = [to_example(r) for r in primary]
    wdi_ex = [to_example(r) for r in wdi]
    logger.info(f"primary {len(prim_ex)} examples; companion {len(wdi_ex)} examples")

    shared_meta = {
        "fold_definition": {
            "even": "screen half (pre-registered)",
            "odd": "reserved confirmation half (pre-registered)",
        },
        "primary_analysis_years": [1950, 2023],
        "do_not_pool_panels": (
            "WDI is sourced from national statistical offices for roughly one third of "
            "economies, so the companion panel tests source invariance, not the same "
            "numbers twice. Never concatenate the two datasets."),
        "cluster_key": "metadata_iso_code (resample entities, not rows)",
    }

    # ---- per-dataset trios (each dataset ships its own full/mini/preview) ----
    written: dict[str, dict[str, Path]] = {}
    for name, stem, exs, desc in (
        (DATASET_PRIMARY, STEM_PRIMARY, prim_ex,
         "Country-year total population, UN World Population Prospects 2024 via Our World "
         "in Data. One example per entity-year, 1950-2025; estimates 1950-2023 are the "
         "primary analysis window and 2024-2025 are flagged projections."),
        (DATASET_WDI, STEM_WDI, wdi_ex,
         "Country-year total population, World Bank World Development Indicators "
         "SP.POP.TOTL. One example per entity-year, 1960-2025; the independently sourced "
         "companion panel, primary analysis window 1960-2023."),
    ):
        payload = {"metadata": {"description": desc, **shared_meta},
                   "datasets": [{"dataset": name, "examples": exs}]}
        written[name] = write_variants(payload, stem)

    # ---- combined file: the canonical cross-panel deliverable ----
    combined = {
        "metadata": {
            "description": (
                "Country-year total population panels for testing how well Benford's law "
                "describes the leading digits of national populations. One example per "
                "entity-year. Primary panel: UN World Population Prospects 2024 via Our "
                "World in Data (1950-2025; estimates 1950-2023, 2024-2025 flagged as "
                "projections). Companion panel: World Bank WDI SP.POP.TOTL (1960-2025), "
                "independently sourced. Examples carry the published value plus per-row "
                "digit encodings, the pre-registered even/odd year fold, and the "
                "microstate / SIDS / top-20 ablation tags. No derived digit statistics."),
            **shared_meta,
        },
        "datasets": [
            {"dataset": DATASET_PRIMARY, "examples": prim_ex},
            {"dataset": DATASET_WDI, "examples": wdi_ex},
        ],
    }
    write_variants(combined, "data_out")
    logger.info("combined deliverable: full_data_out.json, mini_data_out.json, "
                "preview_data_out.json")


if __name__ == "__main__":
    main()
