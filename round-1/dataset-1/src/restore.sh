#!/usr/bin/env bash
# Restore every file marked `delete` in .aii/manifest.yaml.
set -u
cd "$(dirname "$0")"
mkdir -p raw

echo "== raw/owid_population_unwpp_with_projections.csv"
curl -fsSL -A "Mozilla/5.0" -o raw/owid_population_unwpp_with_projections.csv \
  "https://ourworldindata.org/grapher/population-with-un-projections.csv?v=1&csvType=full&useColumnShortNames=true"

echo "== raw/owid_population_unwpp_with_projections.metadata.json"
curl -fsSL -A "Mozilla/5.0" -o raw/owid_population_unwpp_with_projections.metadata.json \
  "https://ourworldindata.org/grapher/population-with-un-projections.metadata.json"

echo "== raw/wdi_sp_pop_totl.zip + raw/wdi/"
curl -fsSL -A "Mozilla/5.0" -o raw/wdi_sp_pop_totl.zip \
  "https://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=csv"
mkdir -p raw/wdi && unzip -oq raw/wdi_sp_pop_totl.zip -d raw/wdi

echo "== raw/wb_countries.json"
curl -fsSL -A "Mozilla/5.0" -o raw/wb_countries.json \
  "https://api.worldbank.org/v2/country?format=json&per_page=400"

echo "== raw/sids_ohrlls.html"
curl -fsSL -A "Mozilla/5.0" -o raw/sids_ohrlls.html \
  "https://www.un.org/ohrlls/content/list-sids"

echo "== .venv/ (skip if you already have one)"
if [ ! -x .venv/bin/python ]; then
  python3.12 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install .
fi

echo "== staging temp/datasets/ for data.py"
mkdir -p temp/datasets/wdi
cp raw/owid_population_unwpp_with_projections.csv temp/datasets/
cp raw/owid_population_unwpp_with_projections.metadata.json temp/datasets/
cp raw/wdi_sp_pop_totl.zip temp/datasets/
cp raw/wb_countries.json temp/datasets/
cp raw/sids_ohrlls.html temp/datasets/
cp raw/wdi/*.csv temp/datasets/wdi/

echo
echo "Done. Verify against the recorded checksums in out/manifest.json,"
echo "then rebuild with:  .venv/bin/python build_panel.py --stage all && .venv/bin/python data.py"
