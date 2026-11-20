# How closely does Benford's law fit the leading digits of country populations?

A harmonic-coordinate screen of four rival mechanisms of Benford deviation on the
UN World Population Prospects country panel (1950–2023), with the leading-digit
departure measured by one instrument — the complex first Fourier coefficient
`c₁` of the mantissa distribution — and every claim scored against a matched
null rather than against exact Benford alone.

## The research question and the shape of the answer

The conventional answer to "does Benford fit?" is a first-digit MAD and a
chi-square against exact Benford. This artifact computes that baseline in full,
then shows why it cannot carry a verdict here: the panel is a **census of
countries over 37 even years**, so a country's mantissa barely moves from year
to year and the nominal 8,732 rows are only ~246 effectively independent
observations (measured design effect 35.5). Any i.i.d. envelope manufactures
spectacular significance.

So the deliverable is:

- the full conventional battery (MAD with Nigrini's bands named but never used
  as a verdict, chi-square with its df and its small-cell caveat, Kuiper V,
  mantissa mean and MAD-from-0.5, first-two-digit MAD, Bonferroni–Wilson
  simultaneous per-digit intervals **at the effective sample size**);
- the same panel in harmonic coordinates (`c₁` amplitude and phase), with its
  entity-cluster Rayleigh noise floor and the minimum detectable amplitude at
  80% power;
- five rival size laws (single lognormal, 2- and 3-component lognormal mixtures
  by EM with BIC, double Pareto-lognormal, truncated power law), each converted
  into its exact leading-digit law by periodizing the log-density, plus a
  per-component "harmonic budget";
- four candidate mechanisms, each with its own matched parametric-bootstrap null
  and a pre-registered verdict file written before the real statistics run;
- the entity **cross-section** as a fourth observation level, because the
  static-mixture mechanism is a statement about the distribution of *countries*
  and pooling country-years averages away exactly the between-country spread it
  is about.

## Layout

```
method.py                  the run: stages 1-9, gate on gate
make_figures.py            renders every figure from results/screen_results.json
audit_rederive.py          recomputes every headline number by a different code path
                           (plain numpy, no src/ imports) and runs the placebo check
reproducibility.md         step-by-step reproduction instructions
pyproject.toml             dependencies pinned to the exact installed versions

tools/reassemble_outputs.py   rebuilds method_out.json (+prose) from a finished run
                              without recomputing any statistic
tools/aii_json_format_patched.py   the aii-json formatter with one macOS path bug fixed

src/config.py              every pre-registered constant, the blocklists, the tags
src/panel.py               Stage 1 - OWID + World Bank acquisition, filtering, fold tagging
src/instrument.py          Stage 2 - mantissas, c_n, digit reconstruction, companion battery
src/size_laws.py           Stage 3 - the five rival laws, periodization, harmonic budget
src/nulls.py               Stage 5 - matched nulls, design effect, walk calibration
src/screen.py              Stages 6-7 - candidates, scores, ranking, pre-registration
src/synthetic.py           Stage 2.5 - V1 calibration, V2 known truth, V3 power curve

data/panel.csv             the analysis panel: fold, mantissa, d1, sig_digits, ablation tags
data/raw/                  the two source CSVs exactly as downloaded
results/screen_results.json   the complete analysis (every number in the paper)
results/method_out.json    the exp_gen_sol_out-schema view of the same analysis
results/PRE_REGISTRATION.md    hashes + candidate definitions, written before scoring
figures/*.pdf, *.png       the six figures
figures/specs/*.json       each figure's chart spec, so it can be re-rendered
figures/data/*.json        the raw values behind each figure
logs/                      run logs
```

## How to run

```bash
uv venv .venv --python=3.12
uv pip install --python=.venv/bin/python -r pyproject.toml

uv run --python=.venv/bin/python method.py --smoke   # 3 years x 20 entities, <60 s
uv run --python=.venv/bin/python method.py --quick   # real panel, reduced replicates
uv run --python=.venv/bin/python method.py           # full run
uv run --python=.venv/bin/python make_figures.py     # figures from the saved results
```

All computation is numpy/scipy on ~8,700 values. No GPU. Peak RSS stays near
1 GB because every replicate stream is processed in blocks of 500 and only one
scalar (or one 9-vector) per replicate is retained.

## What was fixed relative to a naive implementation

These are recorded because each one changed a number in the results:

1. **Log-likelihood measure.** The DPLN and truncated-power-law log-likelihoods
   were computed in the density measure of `x` while the lognormal and the
   mixtures used the measure of `y = log10 x`. The two differ by a Jacobian
   summing to ~1.3×10⁵ on this panel, so BIC compared two different normalising
   constants and silently removed the DPLN from the comparison. Every law now
   reports its log-likelihood on the log10 measure, and the DPLN fit carries a
   check that the gap between the two measures equals the Jacobian exactly.
2. **DPLN bounds.** With `alpha, beta` capped at 8 the family could not reach
   its own lognormal limit (`alpha, beta → ∞`), so the optimiser settled into a
   `sigma → 0` corner and reported a converged "DPLN" with `sigma` pinned to its
   lower bound. The bounds now contain the limit, and one of the restarts starts
   at it.
3. **Entity-cluster null drift.** The null's within-entity log10 walk was a
   fixed 0.02 per year; the panel's own within-country drift is nine times that,
   so the null over-clustered and its `c₁` envelope was too wide. The step is now
   chosen by matching the panel's measured design effect (`calibrate_walk_sd`),
   with the whole search grid stored in the output for audit.
4. **Per-entity walk.** The walk accumulated across the whole flattened panel
   rather than resetting at each country, coupling different countries' mantissas
   and overstating the within-entity variance — the quantity the design effect is
   built on.
5. **Streaming.** The biggest null materialised a (10,000 × 8,732) mantissa array
   (700 MB); every null now streams in blocks of 500 and the materialising helper
   refuses to allocate more than 35 MB.
6. **Candidate B testability.** `|c₁(t)| = exp(−2π²σₜ²)` is ~10⁻¹³ in absolute
   terms for any σₜ near 1.2, so testing the *absolute* spread against the mean
   declared a clean 0.71 correlation "not testable". The test is on the relative
   spread, and the amplitude-*scale* check is reported separately — the mechanism
   can reproduce the year-to-year *shape* while being twelve orders of magnitude
   short on the *level*, and only the conjunction counts as support.
7. **Synthetic frame precision.** Synthetic Benford panels hard-coded 7
   significant digits, which made candidate C's grain hypothesis untestable on
   every calibration panel and produced a spurious "self-falsification". The
   frame now rounds to integers and counts digits the way the source prints them.
8. **Power-curve extrapolation.** A flat power curve produced a "minimum
   detectable amplitude" of 7.5×10³⁴ from a near-zero slope. An unattainable
   minimum is now reported as non-finite with its reason, never as a number.

## Key outputs for a later step

- `results/odd_fold_for_evaluation.json` (inside `screen_results.json`, key
  `extras.odd_fold_for_evaluation`): the reserved odd-year half's statistics,
  with envelopes built from the **even-fold** fits only. Nothing was ever fitted
  on odd years.
- `results/screen_results.json` → `extras.ablation_predictions`: the predicted
  `c₁` shift for removing pre-tagged sub-ensembles (microstates ≤ 10⁶, SIDS,
  top-20), with bootstrap intervals, so the odd fold can test them without
  refitting.
- `results/screen_results.json` → `size_law_fits`: every law's parameters,
  log-likelihood, AIC, BIC and parametric-bootstrap KS at every level.
- `results/screen_results.json` → `power_statement`: design effect, effective
  sample size, noise floor, 2σ threshold, 80%-power minimum detectable amplitude.
- `data/panel.csv` carries the fold tag, the digit masks and the ablation tags,
  so any of the above can be rebuilt if the extras stage did not run.

## Restoring removed files

Three cache directories are removed after the round ends. Each is rebuilt by one
command:

- `.venv/` — the Python environment with the pinned dependencies:
  ```bash
  uv venv .venv --python=3.12
  uv pip install --python=.venv/bin/python -r pyproject.toml
  ```
- `__pycache__/` and `src/__pycache__/` — Python's bytecode cache, which
  Python rebuilds on the next import; nothing to do beyond running the code:
  ```bash
  uv run --python=.venv/bin/python method.py
  ```

Everything else stays. `data/`, `results/`, `figures/`, `src/`, `tools/`,
`logs/` and the root JSON/Markdown files are all text and are published as-is;
none of them is marked `delete`, so a later step reads them where they are.

The two source CSVs under `data/raw/` are re-downloadable from
`https://ourworldindata.org/grapher/population-unwpp.csv?v=1&csvType=full&useColumnShortNames=true`
and the World Bank WDI indicator `SP.POP.TOTL`, but they are kept so the exact
source revision behind every number is on disk.

## Note on the figure and JSON tooling

`make_figures.py` renders through the `aii-data-fig-gen` chart generator; the
path is the `SKILL` constant at the top of that file. `tools/aii_json_format_patched.py`
is the `aii-json` skill's formatter with one line fixed: its sandbox check
compared a *resolved* path against the *unresolved* `Path("/tmp")`, which on
macOS (where `/tmp` is a symlink to `/private/tmp`) refused every legitimate
input. Nothing else in that file is changed; `reproducibility.md` records the
exact command that regenerates `full_`/`mini_`/`preview_method_out.json`.
