"""Global constants for the country-population Benford screen.

All pre-registered constants live here so that the pre-registration hash in
results/PRE_REGISTRATION.md covers them exactly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# numpy is a hard dependency of this artifact; import it at module level so
# sanitize() can test against the real scalar types.
import numpy as np

np_bool = np.bool_
np_floating = np.floating
np_integer = np.integer

# ----------------------------------------------------------------------------
# Paths (all relative to the artifact workspace root, resolved from this file)
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
LOGS_DIR = ROOT / "logs"

PANEL_CSV = DATA_DIR / "panel.csv"
OWID_CSV = RAW_DIR / "owid_population_unwpp.csv"
WB_CSV = RAW_DIR / "wb_population.csv"

# ----------------------------------------------------------------------------
# Pre-registered constants
# ----------------------------------------------------------------------------
SEED = 20261119
ALPHA = 0.05          # significance level of every matched null
MARGIN = 0.05         # the pre-registered margin (== ALPHA; stated separately)
REPS_POOLED = 10_000  # replicates for the pooled level envelope
REPS_HALF = 4_000     # replicates for the split-half level envelope
N_POST = 2_000        # posterior-predictive replicate draws per candidate
NPARAM_DRAWS = 5      # parameter draws per posterior replicate
CHUNK = 500           # replicate batch size (bounded memory)
KMAX = 8              # periodization truncation for the log-density
NGRID = 2 ** 12       # numeric grid resolution for periodization / digit laws
V1_PANELS = 200       # synthetic calibration panels
V2_REPS = 50          # synthetic truth repetitions
V3_REPS = 200         # power-curve repetitions per amplitude

# Population panel window.  WPP 2024 revision: estimates through 2023,
# projections from 2024 onward.
YEAR_MIN = 1950
YEAR_MAX_OPEN_ENDED = True   # take every year the source actually carries
PROJECTION_START = 2024      # WPP 2024: year >= this is model output

# Sub-ensemble tags for the ablation predictions (pre-registered here).
MICROSTATE_MAX_POP = 1_000_000
TOP_N = 20

# SIDS membership (UN ODA list), used only for the ablation sub-ensemble tag.
SIDS_ISO3 = [
    "ATG", "BHS", "BRB", "BLZ", "CPV", "COM", "CUB", "DMA", "DOM", "FJI",
    "GRD", "GUY", "HTI", "JAM", "KIR", "MDV", "MHL", "FSM", "MUS", "NRU",
    "PLW", "PNG", "LCA", "VCT", "WSM", "STP", "SYC", "SLE", "SLB", "SUR",
    "TLS", "TON", "TTO", "TUV", "VUT",
]

# Entity names that are aggregates rather than countries.  Anything with a
# non-ISO3 code is dropped first; this blocklist catches aggregates that
# nonetheless carry a 3-letter code in some OWID exports.
AGGREGATE_NAME_BLOCKLIST = {
    "World", "Africa", "Asia", "Europe", "North America", "South America",
    "Oceania", "European Union (27)", "World Bank income groups",
    "Low-income countries", "Lower-middle-income countries",
    "Upper-middle-income countries", "High-income countries",
    "Small Island Developing States (SIDS)", "OECD members",
    "Sub-Saharan Africa", "Americas (UN)", "Africa (UN)", "Asia (UN)",
    "Europe (UN)", "Oceania (UN)", "Northern America (UN)",
    "Latin America and the Caribbean (UN)", "Less developed regions",
    "More developed regions", "Least developed countries",
    "Land-locked developing countries (LLDC)",
}

# Candidate labels, used verbatim in every output file.
CANDIDATES = {
    "A": "static mixture harmonic fingerprint",
    "B": "temporal desynchronization",
    "C": "rounding artifact",
    "D": "phase-only rotation law",
}

BENFORD_D1 = {d: __import__("math").log10(1.0 + 1.0 / d) for d in range(1, 10)}


def ensure_dirs() -> None:
    """Create every output directory this artifact writes into."""
    for d in (DATA_DIR, RAW_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def sanitize(obj):
    """Recursively replace non-finite floats with None so the JSON is strict.

    NaN and infinity are not JSON, and silently writing them as 0 would
    invent a number.  They become ``null`` and the caller reports, beside each
    such field, that it is 'not testable' rather than a measured zero.
    """
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (bool, np_bool)):
        return bool(obj)
    if isinstance(obj, np_integer):
        return int(obj)
    if isinstance(obj, (float, np_floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def dump_json(obj: object, path: Path, indent: int = 2) -> None:
    """Write strict JSON (no NaN/infinity), stable key order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(obj), indent=indent, allow_nan=False, default=str))
