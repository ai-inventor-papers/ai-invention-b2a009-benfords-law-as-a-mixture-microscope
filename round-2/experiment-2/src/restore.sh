#!/usr/bin/env bash
# Restore everything this artifact removes after the run.
#
#   ./restore.sh
#
# Three local build artefacts are removed: the virtual environment and the two
# byte-compiled caches. Nothing a later step reads is removed -- results/,
# figures/ and the input panel all stay.

set -u
cd "$(dirname "$0")"

PY_VERSION="3.12"
PINS=(loguru==0.8.0 numpy==2.5.5 psutil==7.2.2 scipy==1.18.1)
PANEL_MD5="a338376ff319aa913078df4669fde237"

echo "== 1. Python environment =="
if [ ! -x .venv/bin/python ]; then
    uv venv .venv --python="${PY_VERSION}" || { echo "uv not found: install it first"; exit 1; }
fi
uv pip install --python=.venv/bin/python "${PINS[@]}" || exit 1
.venv/bin/python -c "import numpy, scipy, loguru; print('imports ok')" || exit 1

echo
echo "== 2. Byte-compiled caches =="
.venv/bin/python -m compileall -q analysis method.py && echo "caches rebuilt"

echo
echo "== 3. Input panel =="
if [ -f data/full_data_out.json ]; then
    GOT=$(md5 -q data/full_data_out.json 2>/dev/null || md5sum data/full_data_out.json | cut -d' ' -f1)
    if [ "${GOT}" = "${PANEL_MD5}" ]; then
        echo "data/full_data_out.json present, md5 verified: ${GOT}"
    else
        echo "WARNING: panel md5 is ${GOT}, expected ${PANEL_MD5}"
    fi
else
    mkdir -p data
    SIBLING="../gen_art_dataset_1/full_data_out.json"
    if [ -f "${SIBLING}" ]; then
        echo "copying from the sibling dataset artifact: ${SIBLING}"
        cp "${SIBLING}" data/full_data_out.json
    else
        echo "panel missing. Take it from ../gen_art_dataset_1/full_data_out.json"
        echo "or rebuild it with analysis/dependency_data_recipe.py in that folder."
    fi
fi

echo
echo "== 4. Smoke test =="
if [ -f data/full_data_out.json ]; then
    .venv/bin/python method.py --mini
else
    echo "skipped (no panel)"
fi
