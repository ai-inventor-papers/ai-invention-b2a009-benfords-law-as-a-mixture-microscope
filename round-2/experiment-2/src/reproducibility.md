# Reproducibility

Complete, step-by-step instructions to reproduce every number in this artifact on a fresh
Ubuntu machine. Everything below is what was **actually run**, not an idealised recipe.

Compute used: a `cpu_basic` pod — 2 shared vCPUs, 16 GB RAM, no GPU, 40 GB disk, macOS
guest (Apple silicon). Nothing here needs a GPU, and there is no CUDA, no Docker and no
Linux binary dependency. Wall clock: about 40 minutes for the full run, about 35 seconds for
the `--mini` gate.

---

## 1. Get the code

This artifact is one folder of a public GitHub repository. Clone the repository and change
into this artifact's folder:

```bash
git clone <repository-url> aii
cd aii/<path-to>/gen_art_experiment_2
```

From here on, **every path in this document and in the code is relative to that folder**.
The code anchors itself on `Path(__file__).resolve().parent`, so it runs from any working
directory and never references a server path.

---

## 2. System packages and Python

Ubuntu 22.04 or 24.04. Only a Python 3.12 interpreter and a few system libraries are needed;
no compiler is required because every dependency has a manylinux wheel.

```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv build-essential curl
```

Install `uv` (the only package manager used; `pip` is never invoked directly):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
. "$HOME/.cargo/env"
```

Create the environment and install **the exact pinned versions** from `pyproject.toml`:

```bash
uv venv .venv --python=3.12
uv pip install --python=.venv/bin/python \
    loguru==0.8.0 numpy==2.5.5 psutil==7.2.2 scipy==1.18.1
```

These are the four versions actually installed, recorded with `uv pip freeze`:

```
loguru==0.8.0
numpy==2.5.5
psutil==7.2.2
scipy==1.18.1
```

`matplotlib` is **not** required by the analysis. It is needed only if you want to re-render
the figures (section 6), and then only in the interpreter that has the figure skill's
dependencies.

---

## 3. Data

There is **no download step and no API key**. The single input is the panel produced by the
upstream dataset artifact `art_ym9ZVhHxA-vn`, which publishes it as a sibling folder of the
same repository:

```
../gen_art_dataset_1/full_data_out.json        # 43 MiB, md5 a338376ff319aa913078df4669fde237
```

`method.py` reads it through one constant, `PANEL_PATH`:

```python
PANEL_PATH = ROOT / "data" / "full_data_out.json"
```

So either copy the sibling file into `data/`:

```bash
mkdir -p data
cp ../gen_art_dataset_1/full_data_out.json data/full_data_out.json
```

or point the constant at wherever you put the file. No model weights, no checkpoints and no
credentials of any kind are used.

**Integrity.** The artifact refuses to run on a different panel. It verifies, before any
statistic is computed:

- md5 of the file equals `a338376ff319aa913078df4669fde237`;
- the 2023 total population over **all 237 primary entities including Kosovo** equals
  `8,090,655,899`. The analysis set of 236 entities with an ISO code sums to
  `8,088,955,860`; the 1,700,039-person difference *is* Kosovo, and the gate reports both
  sums so that the difference cannot be mistaken for the classic persons-versus-thousands
  unit error;
- 17,464 primary rows over 236 entities, exactly 74 years (1950–2023) per entity, an
  8,732 / 8,732 even/odd fold split, and no duplicate `(iso_code, year)` or
  `(entity, year)` key;
- every mantissa in `[0, 1)` and a string-derived first-digit agreement rate of 1.0, with the
  one exact-decade boundary case whitelisted (Nauru 1962, population 5000, mantissa exactly
  0.69897).

Any failed check raises and the run stops with the check dictionary printed. To rebuild the
panel from source instead of copying it, run the upstream recipe (`data.py` in the sibling
dataset folder, staged here as `analysis/dependency_data_recipe.py`); it reproduces
`full_data_out.json` byte for byte.

---

## 4. Run it

### Fast end-to-end gate

```bash
.venv/bin/python method.py --mini
```

Every replicate count is shrunk (B = 300 rotation replicates, 20 synthetic panels for the
level check, 15 panels per power point, 8 panels per calibration case, 500 bootstrap
resamples). It exercises **every** stage, including the independent re-derivation, in about
35 seconds, and writes all of `method_out.json` and `results/`. Use it to confirm the
environment before committing to the full run.

### The full run

```bash
.venv/bin/python method.py
```

This is the run the reported numbers come from. In the background, with a log:

```bash
mkdir -p logs
nohup .venv/bin/python -u method.py > logs/full_run.log 2>&1 &
echo "PID=$!"
```

Replicate counts and seeds, all fixed:

| stage | what | count |
|---|---|---|
| pre-registration | `results/PRE_REGISTRATION_RECONCILIATION.md` written and hashed **before** the panel is read | n/a |
| V1 level | kappa-matched synthetic panels, rotation null, α = 0.05 | 500 panels × B = 2,000 |
| V2 Rayleigh agreement | same panels, rotation spread vs `2.44775 σ_R` | 200 panels × B = 4,000 |
| V3 identity | `r₂(n_eff)/r₂(n_c) = sqrt(n_c/n_eff)`, exact | analytic, tol 1e-12 |
| V4 time order | row permutation leaves all five statistics bit-identical | 1 panel, tol 1e-12 |
| V5 power curve | kappa-matched panels, planted `\|c1\|` ∈ {0, 0.035, 0.075, 0.11, 0.14} | 200 panels per amplitude × B = 2,000 |
| calibration | uniform digit law, narrow lognormal σ = 0.1, exactly-Benford (over-regular) | 100 panels each × B = 10,000, α = 0.001 |
| battery | both folds + pooled, all instruments | B = 10,000 rotation, 4,000 placebo |
| disjoint folds | md5(iso_code) mod 4, entity-disjoint | 4 folds × B = 2,000 |
| stability | country bootstrap / delete-one jackknife | 10,000 resamples |
| canonicalization | EM for a 1–3 component lognormal mixture on log₁₀ population | 20 starts per k |

Seeds: rotation null `20240101` (even), `20240102` (odd), `20240201` (pooled); disjoint folds
`4242 + i`; bootstrap `7777`; placebo `seed + 1_000_003`; validation seeds `20000`, `40000`,
`60000`, `80000` plus the panel index. Every one is a literal in the source, so the run is
deterministic given the same panel and dependency versions.

### Pre-registration ordering

The pre-registration must exist before `method.py` touches the panel. The script records its
sha256 and mtime and compares the mtime against the timestamp of the first panel read,
writing the comparison to `results/audit.json::prereg_precedes_panel_read`. To re-run
cleanly, keep that file in place — do not delete it and do not let `method.py` create it.

---

## 5. Outputs, and where each number appears

All under this folder:

| file | contents | paper location |
|---|---|---|
| `method_out.json` | every reportable row in the `exp_gen_sol_out` layout, one example per row | drives every table |
| `full_method_out.json`, `mini_method_out.json`, `preview_method_out.json` | the same, plus the 3-row and truncated previews | inspection only |
| `results/PRE_REGISTRATION_RECONCILIATION.md` | the pre-registered estimator, class table, null, verdict rules, both branches, integrity gate | methods section |
| `results/reconciliation.json` | two-row estimator table per fold, algebra identity, pair identifiability, kappa calibration, threshold table | reconciliation table |
| `results/verdict.json` | the five-statistic × fold × instrument matrix and the quotable verdict sentence per fold | headline result |
| `results/stability.json` | disjoint folds with Rayleigh intervals and Cochran Q, bootstrap, jackknife, power bound | stability and power paragraphs |
| `results/provenance.json` | the 80-candidate identification lattice and the declared canonical definition | data-hygiene paragraph |
| `results/canonical.json` | EM fit, `A_canon`, `A_nocancel`, resolution of the recorded budgets | budget paragraph |
| `results/audit.json` | sha256 of every source file, the independent-verification mismatches, wall clock, fresh-seed p | reproducibility statement |
| `results/supm_definition_scan.json` | which sup-distance definition reproduces the recorded 0.03065 | data-hygiene footnote |
| `logs/run.log` | DEBUG-level run log | — |

**What a reader should see.** The headline quantities are

- the pooled `|c1| = 0.07507` and the two per-fold values `0.07493` / `0.07521`, each against
  the corrected two-sigma threshold `≈ 0.0575` derived from the observed `kappa`;
- `sigma_R ≈ 0.0288` from `sqrt(0.5 Σ wᵢ²κᵢ²)` with mean `kappa = 0.5441`, i.e. the
  Rayleigh null is roughly half as wide as the `kappa = 1` reference the plan quoted
  (`0.0460`), which is why the crossing is closer than the plan's numbers implied;
- the four frequency-type statistics **consistent** (`p` from 0.07 to 0.29 on the rotation
  null) while `|c1|` is **not** on both folds — the sharpened positive, not a blanket pass;
- `A_canon = 0.0355` with `A_nocancel` within 0.1% of it, so no phase cancellation exists in
  the selected fit and the two recorded budgets are the same quantity under different EM
  starts;
- the provenance triple B reproduced exactly as the per-country circular mean over even
  years, and triple A genuinely not reproducible from this panel;
- zero mismatches between the main path and the independent path in `results/audit.json`.

---

## 6. Figures (optional)

The figure specifications and the raw values behind every panel are written by the run to
`figures/figure_specs.json`. Render them with the run's figure skill rather than by hand, so
they share the paper's style:

```bash
python <aii-data-fig-gen>/scripts/chart_gen.py \
    --spec figures/fig1_rayleigh_null.json --out figures/fig1_rayleigh_null
```

The four figures are: the Rayleigh null with the observed `|c1|` marked on both folds; the
p-value heatmap over statistic × fold × instrument; the power curve against planted
amplitude with the observed amplitude as a vertical rule; and the threshold-versus-`n` plot
showing `r₂` at n = 236, the two design-effect n values and the nominal row count. Each spec
carries its own numbers in `raw_values`, so a claim in the caption can be re-checked from the
file.

---

## 7. What is not reproducible, and why

- **The two recorded values this artifact reconciles against** (the defective `rho = 0.9575`,
  the `ICC1 = 0.2206`, the four `|c1|`/MAD/chi-square readings, the two mixture budgets and
  the two provenance triples) come from earlier iterations whose workspaces are gone. They
  are hard-coded in `RECORDED` in `method.py` as **text targets** and are labelled "recorded,
  not recomputed" wherever one is quoted without being re-derived here. Everything else is
  computed fresh.
- **`sup-M = 0.03065`.** `results/supm_definition_scan.json` shows this is not a cumulative
  sup-distance at all: it is the odd fold's **maximum single-digit deviation**, which we
  compute as `0.030645`. The cumulative sup-distance (KS to uniform on mantissas) is
  `0.04316` / `0.04443` on the two folds. Reported as a definitional finding, not tuned away.
- **The panel** is not regenerated by this artifact. It is the upstream dependency's output,
  pinned by md5; regenerating it requires that dependency's own recipe.
