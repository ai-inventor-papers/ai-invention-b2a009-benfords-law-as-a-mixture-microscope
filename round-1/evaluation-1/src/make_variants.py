#!/usr/bin/env python3
"""Produce full/mini/preview variants of eval_out.json.

This reimplements the transformation the aii-json skill's
`aii_json_format_mini_preview.py` documents, because that script's input path
guard compares against the literal `/tmp` while macOS resolves `/tmp` to
`/private/tmp`, so it rejects every path on this machine.

The documented transformation is:

  full_eval_out.json     identical to the original
  mini_eval_out.json     first 3 examples per dataset (datasets-grouped schema)
  preview_eval_out.json  mini, with every string truncated to 200 characters
                         plus a truncation marker

`eval_out.json` declares two datasets -- the reserved odd-fold rows and the
screen even-fold rows -- so the mini keeps both datasets and slices each one's
examples to three, which is what the skill's `_slice_dataset_examples` does.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAX_ITEMS = 3
MAX_STRING = 200
MARKER = "... [truncated]"


def truncate(value):
    if isinstance(value, str):
        if len(value) <= MAX_STRING:
            return value
        return value[:MAX_STRING] + MARKER
    if isinstance(value, list):
        return [truncate(v) for v in value]
    if isinstance(value, dict):
        return {k: truncate(v) for k, v in value.items()}
    return value


def slice_examples(datasets, k):
    out = []
    for ds in datasets:
        if not isinstance(ds, dict):
            out.append(ds)
            continue
        d = copy.deepcopy(ds)
        ex = d.get("examples", [])
        if isinstance(ex, list):
            d["examples"] = ex[:k]
        out.append(d)
    return out


def main() -> None:
    src = ROOT / "eval_out.json"
    data = json.loads(src.read_text())
    if "datasets" not in data:
        raise SystemExit("expected a datasets-grouped eval_out.json")

    # full: identical content, normalised formatting
    (ROOT / "full_eval_out.json").write_text(json.dumps(data, indent=2))
    n_full = sum(len(ds.get("examples", [])) for ds in data["datasets"])

    # mini: every dataset kept, examples sliced
    mini = copy.deepcopy(data)
    mini["datasets"] = slice_examples(data["datasets"], MAX_ITEMS)
    (ROOT / "mini_eval_out.json").write_text(json.dumps(mini, indent=2))
    n_mini = sum(len(ds.get("examples", [])) for ds in mini["datasets"])

    # preview: mini with strings truncated
    preview = truncate(mini)
    (ROOT / "preview_eval_out.json").write_text(json.dumps(preview, indent=2))

    print(f"full_eval_out.json    ({n_full} examples total across "
          f"{len(data['datasets'])} datasets)")
    print(f"mini_eval_out.json    ({n_mini} examples)")
    print(f"preview_eval_out.json ({n_mini} examples, strings <= {MAX_STRING} chars)")
    for name in ("full_eval_out.json", "mini_eval_out.json", "preview_eval_out.json"):
        print(f"  {name}: {(ROOT / name).stat().st_size} bytes")


if __name__ == "__main__":
    main()
