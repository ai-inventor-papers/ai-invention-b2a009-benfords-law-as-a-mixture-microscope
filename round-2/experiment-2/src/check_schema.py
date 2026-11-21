"""Local schema check for exp_gen_sol_out, replicating its rules exactly.

The ability-server validator gave no diagnostic, so this states every rule from
the schema file and names the offending example index and property.

  top level : object with required "datasets"; optional "metadata" (any object)
  datasets  : non-empty array of {dataset: str, examples: non-empty array}, no
              extra properties
  examples  : object with required input (str) and output (str); every other key
              must match ^metadata_[a-zA-Z_][a-zA-Z0-9_]*$ (any type) or
              ^predict_[a-zA-Z_][a-zA-Z0-9_]*$ (string). No additional
              properties at all.

    .venv/bin/python check_schema.py [path]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

META = re.compile(r"^metadata_[a-zA-Z_][a-zA-Z0-9_]*$")
PRED = re.compile(r"^predict_[a-zA-Z_][a-zA-Z0-9_]*$")


def check(path: Path) -> list[str]:
    errs: list[str] = []
    doc = json.loads(path.read_text())

    if not isinstance(doc, dict):
        return ["top level is not an object"]
    for k in doc:
        if k not in ("metadata", "datasets"):
            # additionalProperties: false on the top level
            errs.append(f"top-level extra property {k!r} (schema allows metadata, datasets)")
    if "datasets" not in doc:
        errs.append("missing required top-level 'datasets'")
    if "metadata" in doc and not isinstance(doc["metadata"], dict):
        errs.append("metadata is not an object")

    ds = doc.get("datasets", [])
    if not isinstance(ds, list) or not ds:
        errs.append("datasets must be a non-empty array")
        return errs

    for di, d in enumerate(ds):
        if not isinstance(d, dict):
            errs.append(f"datasets[{di}] is not an object")
            continue
        for k in d:
            if k not in ("dataset", "examples"):
                errs.append(f"datasets[{di}] extra property {k!r}")
        if "dataset" not in d or not isinstance(d["dataset"], str):
            errs.append(f"datasets[{di}] missing/invalid 'dataset' (string)")
        exs = d.get("examples")
        if not isinstance(exs, list) or not exs:
            errs.append(f"datasets[{di}].examples must be a non-empty array")
            continue
        for ei, e in enumerate(exs):
            if not isinstance(e, dict):
                errs.append(f"datasets[{di}].examples[{ei}] is not an object")
                continue
            if "input" not in e or not isinstance(e["input"], str):
                errs.append(f"examples[{ei}] missing/invalid 'input' (string)")
            if "output" not in e or not isinstance(e["output"], str):
                errs.append(f"examples[{ei}] missing/invalid 'output' (string)")
            for k, v in e.items():
                if k in ("input", "output"):
                    continue
                if META.match(k):
                    continue
                if PRED.match(k):
                    if not isinstance(v, str):
                        errs.append(f"examples[{ei}].{k} must be a string "
                                    f"(predict_* is typed string)")
                    continue
                errs.append(f"examples[{ei}] additional property {k!r} "
                            f"(must be input/output, metadata_*, or predict_*)")
    return errs


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("method_out.json")
    errs = check(path)
    doc = json.loads(path.read_text())
    n = sum(len(d["examples"]) for d in doc.get("datasets", []))
    exs = [e for d in doc.get("datasets", []) for e in d["examples"]]
    n_pred = sum(1 for e in exs if any(k.startswith("predict_") for k in e))
    print(f"{path}: {len(doc.get('datasets', []))} dataset(s), {n} examples, "
          f"{n_pred} with a predict_* field")
    # the pipeline's validator requires at least one predict_* field per example
    if exs and n_pred < len(exs):
        errs.append(f"{len(exs) - n_pred} example(s) carry no predict_* field; "
                    f"every row must carry the method-versus-baseline prediction")
    if errs:
        print(f"FAILED with {len(errs)} problem(s):")
        for e in errs[:40]:
            print(f"  - {e}")
        return 1
    print("PASSED: conforms to exp_gen_sol_out (including the predict_* requirement)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
