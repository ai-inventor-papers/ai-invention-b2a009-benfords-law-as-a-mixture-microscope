#!/usr/bin/env python3
"""Render the artifact's figures with the run's figure generator.

Reads ``figures/<name>.spec.json`` (the generator's own spec shape) and calls
``chart_gen.py``, so every figure shares the paper's house style and is drawn
directly from the numbers in ``figures/<name>.values.json`` beside it.

    .venv/bin/python render_figures.py [--chart-gen PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"

DEFAULT_CHART_GEN = Path(
    os.environ.get(
        "AII_CHART_GEN",
        "/Users/Shared/aii_data/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py",
    )
)


def find_chart_gen(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    cand = Path(os.environ.get("AII_CHART_GEN", "")) if os.environ.get("AII_CHART_GEN") else None
    if DEFAULT_CHART_GEN.exists():
        return DEFAULT_CHART_GEN
    for base in (Path.home() / ".claude", Path("/ai-inventor/.claude")):
        p = base / "skills" / "aii-data-fig-gen" / "scripts" / "chart_gen.py"
        if p.exists():
            return p
    if cand and cand.exists():
        return cand
    found = shutil.which("chart_gen.py")
    return Path(found) if found else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chart-gen", default=None)
    args = ap.parse_args()

    gen = find_chart_gen(args.chart_gen)
    if gen is None:
        print("chart_gen.py not found. Pass --chart-gen PATH or set AII_CHART_GEN.",
              file=sys.stderr)
        print("The specs and their raw values are still in figures/ and can be rendered "
              "with any matplotlib that has the house style applied.", file=sys.stderr)
        return 2

    interp = sys.executable
    specs = sorted(FIG.glob("*.spec.json"))
    if not specs:
        print(f"no *.spec.json in {FIG}", file=sys.stderr)
        return 1

    rc = 0
    for spec in specs:
        name = spec.name.replace(".spec.json", "")
        out = FIG / name
        print(f"rendering {name} ...", flush=True)
        proc = subprocess.run(
            [interp, str(gen), "--spec", str(spec), "--out", str(out)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"  FAILED\n{proc.stdout}\n{proc.stderr}", file=sys.stderr)
            rc = 1
            continue
        print(f"  {proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else 'ok'}")

    # report what exists, with the raw-values file beside each image
    print("\nrendered:")
    for img in sorted(FIG.glob("*.png")):
        vals = FIG / f"{img.stem}.values.json"
        print(f"  {img.name:34s} {'+ values.json' if vals.exists() else '(no values file)'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
