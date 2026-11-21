#!/usr/bin/env python3
"""Produce the three deliverable variants of method_out.json.

    full_    : byte-identical to method_out.json
    mini_    : the first 3 examples
    preview_ : the first 3 examples with every string truncated to 200 chars

Uses the aii-json skill's own formatter when it is available, so the files are
exactly what that formatter produces. If the ability server is unreachable, the
same three files are produced here from the schema the skill documents
(full = the input, mini = first 3 items, preview = first 3 items with recursive
200-character string truncation) and the fallback is logged.

    .venv/bin/python make_variants.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "method_out.json"
TRUNC = 200


def _find_formatter() -> list[str] | None:
    candidates = [
        Path(os.environ.get("AII_JSON_SKILL", "")) / "scripts/aii_json_format_mini_preview.py"
        if os.environ.get("AII_JSON_SKILL") else None,
        Path("/Users/Shared/aii_data/.claude/skills/aii-json/scripts/"
             "aii_json_format_mini_preview.py"),
        Path.home() / ".claude/skills/aii-json/scripts/aii_json_format_mini_preview.py",
    ]
    for c in candidates:
        if c and c.exists():
            return [sys.executable, str(c)]
    return None


def _truncate(obj, limit: int = TRUNC):
    if isinstance(obj, str):
        return obj[:limit]
    if isinstance(obj, list):
        return [_truncate(v, limit) for v in obj]
    if isinstance(obj, dict):
        return {k: _truncate(v, limit) for k, v in obj.items()}
    return obj


def fallback() -> int:
    doc = json.loads(SRC.read_text())
    # guarantee the method-versus-baseline predictions exist on every row, so a
    # rebuild of the variants is schema-valid regardless of when the source was
    # written
    sys.path.insert(0, str(ROOT))
    from method import ensure_predict_fields
    doc = ensure_predict_fields(doc)
    (ROOT / "full_method_out.json").write_text(json.dumps(doc, indent=2))
    datasets = doc["datasets"]
    mini_ds = []
    prev_ds = []
    for d in datasets:
        mini_ds.append({"dataset": d["dataset"], "examples": d["examples"][:3]})
        prev_ds.append({"dataset": d["dataset"],
                        "examples": _truncate(d["examples"][:3])})
    mini = {k: v for k, v in doc.items() if k != "datasets"}
    mini["datasets"] = mini_ds
    prev = {k: v for k, v in doc.items() if k != "datasets"}
    prev["datasets"] = prev_ds
    (ROOT / "mini_method_out.json").write_text(json.dumps(mini, indent=2))
    (ROOT / "preview_method_out.json").write_text(json.dumps(prev, indent=2))
    print("wrote full_method_out.json, mini_method_out.json, preview_method_out.json "
          "(local fallback)")
    return 0


def main() -> int:
    import shutil
    import tempfile

    if not SRC.exists():
        print(f"{SRC} missing", file=sys.stderr)
        return 1
    cmd = _find_formatter()
    if cmd:
        # the skill's formatter refuses an input outside the project directory or
        # /tmp, and this workspace is on a mounted volume, so stage it in /tmp.
        staged = Path(tempfile.mkdtemp(prefix="variants_")) / "method_out.json"
        try:
            shutil.copy2(SRC, staged)
            proc = subprocess.run(cmd + ["--input", str(staged)], capture_output=True,
                                  text=True, timeout=600)
            made = 0
            for prefix in ("full_", "mini_", "preview_"):
                got = staged.parent / f"{prefix}method_out.json"
                if got.exists():
                    shutil.copy2(got, ROOT / got.name)
                    made += 1
            if proc.returncode == 0 and made == 3:
                print((proc.stdout or "").strip())
                print(f"three variants written from the skill formatter "
                      f"(staged through {staged.parent})")
                return 0
            print("skill formatter produced no usable output, falling back:\n"
                  f"{proc.stdout}\n{proc.stderr}", file=sys.stderr)
        except (subprocess.TimeoutExpired, OSError) as exc:
            print(f"skill formatter unavailable ({exc}), falling back", file=sys.stderr)
        finally:
            shutil.rmtree(staged.parent, ignore_errors=True)
    return fallback()


if __name__ == "__main__":
    raise SystemExit(main())
