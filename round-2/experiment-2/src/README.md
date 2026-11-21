# Re-testing Benford fit with the right dependence null

*How closely does Benford's law fit the leading digits of the populations of world
countries?*

This artifact answers that question with a **country-rotation randomization null**: for
country *i* draw `phi_i ~ Uniform[0,1)` independently and set every one of that country's
mantissas `m*_it = (m_it + phi_i) mod 1`, then recompute the whole five-statistic battery.
Because `E_phi[p*_d] = b_d` exactly for any data, this is an exact level-α test of "country
decade placements are independent and uniform". It holds every within-country trajectory
fixed, so the observed per-country concentrations `kappa_i` enter the null by construction and
no design effect has to be estimated.

The question is old; what is new is that every conventional answer treats the 17,464
country-years as 17,464 independent draws, when they are **236 countries each observed 74
times**.

---

## Result in one paragraph

The four frequency-type statistics (mean absolute deviation, Pearson chi-square on df = 8,
the sup-distance to uniform on mantissas, and the Kuiper statistic) are **consistent with
exact Benford** under the rotation null on both folds, with p-values between 0.07 and 0.29.
The circular-mean statistic `|c1|` is **not** consistent: `|c1| = 0.07493` (even) and
`0.07521` (odd) against a corrected two-sigma threshold of about `0.0575`, with a rotation
p-value of `0.0365` on both folds. That is the sharpened positive the design was built to
deliver: the panel's digit *histogram* is Benford-like, but its *between-country phase
alignment* is tighter than a cluster null reproduces, because national populations pile up
near the same point within their decade for arithmetic reasons rather than stochastic ones.

---

## Layout

| path | what it is |
|---|---|
| `method.py` | the orchestrator: runs every pre-registered stage in order and writes `results/` and `method_out.json` |
| `analysis/panel_io.py` | panel load, the integrity gate, fold and entity-disjoint split construction |
| `analysis/stats.py` | the five statistics, the Rayleigh thresholds, the country-rotation null, the parametric placebo, Holm, Goodman and Wilson intervals |
| `analysis/reconcile.py` | the two-row estimator table, the algebra identity, pair identifiability, the kappa calibration and the threshold table |
| `analysis/synthetic.py` | the synthetic panel generator (base law, within-country wander, drift, AR(1)) |
| `analysis/validation.py` | derivation validation V1–V5 (level, Rayleigh agreement, threshold identity, time order, power curve) and the calibration suite; includes the kappa-matched generator and the amplitude-planting rule |
| `analysis/battery.py` | the battery at scale, verdict rules, disjoint folds, stability, power bound, robustness ablations, source invariance |
| `analysis/provenance.py` | the entity-level identification lattice (74 yearly snapshots + 6 circular-mean variants) |
| `analysis/canonical.py` | EM for the lognormal size law and the harmonic-value canonicalization |
| `analysis/verify.py` | an **independent** second code path (plain numpy, no shared helpers) that re-derives every headline quantity |
| `results/PRE_REGISTRATION_RECONCILIATION.md` | the pre-registration, written and hashed **before** the panel is read |
| `results/*.json` | reconciliation, verdict, stability, provenance, canonicalization, audit, sup-distance scan |
| `figures/*.spec.json` | chart specs in the figure generator's own shape, one file per figure |
| `figures/*.values.json` | the numbers behind each figure, beside it, so a caption can be re-checked |
| `figures/*.pdf`, `figures/*.png` | the four rendered figures |
| `check_schema.py` | local `exp_gen_sol_out` conformance check, including the `predict_*` requirement |
| `audit_headline.py` | independent re-derivation of every headline number from the raw results, with a placebo test |
| `check_figures.py` | verifies each rendered figure against its `values.json` |
| `render_figures.py` | renders the figures via the figure generator |
| `data/full_data_out.json` | the input panel, staged from the dependency artifact (not published — see *Restoring removed files*) |
| `reproducibility.md` | complete step-by-step reproduction instructions |
| `analysis/dependency_data_recipe.py` | the upstream dataset's build script, kept as the rebuild path |

---

## Run it

```bash
uv venv .venv --python=3.12
uv pip install --python=.venv/bin/python loguru==0.8.0 numpy==2.5.5 psutil==7.2.2 scipy==1.18.1

mkdir -p data
cp ../gen_art_dataset_1/full_data_out.json data/full_data_out.json

.venv/bin/python method.py --mini     # ~35 s, every stage, shrunk counts
.venv/bin/python method.py            # ~40 min, the reported numbers
```

There is no GPU, no API key, no download and **$0 OpenRouter spend**: no LLM call, no image
generation and no web request happens inside the run. `reproducibility.md` lists the exact
seeds, replicate counts and expected outputs.

---

## What the artifact establishes

1. **The estimator.** The one-way ANOVA intraclass correlation `ICC1 = (MSB-MSW)/(MSB+(k-1)MSW)`
   is declared for the frequency-type digit statistics. The ratio of mean squares
   `rho = MSB/(MSB+MSW)` used by the earlier screen exceeds it by
   `((k-1)x + x²)/((1+x)(1+(k-1)x)) >= 0` for every first-digit indicator, so the defect has a
   **sign**: it can only overstate dependence and understate the effective sample size. The
   identity is verified per digit to 1e-12.
2. **The contradiction was a choice of n.** The two recorded `|c1|` thresholds (0.0901 at
   n_eff 246, 0.0453 at n_eff 976) are one formula, `r₂ = 2 sqrt(0.5 κ²/n)`, at two different
   n. They are not two theories.
3. **The right degrees of freedom.** `c1 = Σᵢ wᵢ z̄ᵢ` is a circular mean of country phases, so
   its null is Rayleigh with `σ_R = sqrt(0.5 Σᵢ wᵢ² κᵢ²)` and its degrees of freedom are the
   **country count 236**, never an effective row count. Substituting a row-based `n_eff`
   shrinks the threshold by exactly `sqrt(n_c/n_eff)` — printed as a verified identity.
4. **An exact cluster null.** The country-rotation null needs no design-effect estimate and no
   model of within-country wandering; validated for level, Rayleigh agreement and time-order
   invariance, with a discrimination-power curve at the panel's own `kappa`.
5. **A power bound instead of an overlap statistic.** What 236 countries can resolve, and what
   `n_countries` a given amplitude would need. The ensemble is a census, so no sample-size fix
   exists and the bound is stated plainly.
6. **Where the rotation null works and where it stops.** Measured on uniform-digit-law panels
   with a controlled within-country spread, the rotation null rejects at rate 1.00 up to a
   within-country drift of 0.60 decades (mean `kappa` 0.48); only past that, where every
   country already covers most of `[0,1)`, does power collapse (0.12 at 1.0 decades, `kappa`
   0.03). This is the honest operating range of the instrument, and it is stated rather than
   discovered by a reader. The run's `--mini` log records the scan; it is reproduced by
   `analysis/validation.py::calibration_suite` at B = 10,000.

Two hygiene questions are settled on explicit definitions rather than trust:

- **The provenance lattice.** One of the two disputed entity-level triples is reproduced
  **exactly** as the per-country circular mean mantissa over even years (MAD `0.02029`,
  chi-square `10.183`, `|c1| = 0.1043`). The other is genuinely **not reproducible** from this
  panel and is reported as superseded and unverifiable rather than matched by tuning.
- **The mixture budget.** `A_canon` (modulus of the complex Fourier sum) and `A_nocancel`
  (sum of component moduli) agree to 0.1% in the selected three-component fit, so there is no
  phase cancellation and the two recorded budgets are the same quantity under different EM
  starts.

---

## Output shape

`method_out.json` is in the pipeline's `exp_gen_sol_out` layout: one top-level
`datasets` group named `benford_population_panel_1950_2023`, with one example per reportable
row (319 of them) and every field flat. Each example carries

- `input` — the row's identity (`"even|chi_square"`, `"entity-disjoint fold 2"`, …);
- `output` — the human-readable result sentence, which is what the paper quotes;
- `metadata_*` — the machine-readable numbers, typed;
- `predict_country_rotation_null` and `predict_nominal_baseline` — **the operative method's
  answer and the conventional baseline's answer on the same row**, so the method-versus-
  baseline comparison is machine-readable rather than only in prose. Statistic rows
  additionally carry `predict_analytic_cluster_null` and `predict_parametric_placebo`, the
  other two instruments that answer the same question.

`check_schema.py` validates all of that locally against the schema's rules, including the
`predict_*` requirement.

---

## Restoring removed files

Nothing a later step reads is removed. Three local build artefacts are deleted after the run
and each rebuilds in one command:

| path | why it is removed | how to restore it |
|---|---|---|
| `.venv/` | a local environment, machine-specific | `uv venv .venv --python=3.12 && uv pip install --python=.venv/bin/python loguru==0.8.0 numpy==2.5.5 psutil==7.2.2 scipy==1.18.1` |
| `analysis/__pycache__/` | byte-compiled cache, regenerated on import | `python -m compileall analysis method.py` |
| `__pycache__/` | byte-compiled cache, regenerated on import | `python -m compileall analysis method.py` |

`restore.sh` does all three plus a smoke test in one step.

**Kept, and present in the published folder:**

- `data/full_data_out.json` (43 MiB) — the input panel, under the 100 MB publish limit, so it
  travels with the repository. Verify `md5` = `a338376ff319aa913078df4669fde237`. If you fork
  from a copy that lacks it, take it from the sibling dataset artifact
  `../gen_art_dataset_1/full_data_out.json` or rebuild it with
  `analysis/dependency_data_recipe.py`.
- `results/` — every computed table. Not cheap to redo and read by later steps.
- `figures/` — the four chart specifications, the values file behind each, and the rendered
  PDF/PNG pairs.

---

## References used

Only references whose details were confirmed against the publisher or a bibliographic index
appear here. Nothing is cited from memory.

- Lesperance M, Reed WJ, Stephens MA, Tsao C, Wilton B (2016). Assessing conformance with
  Benford's law: goodness-of-fit tests and simultaneous confidence intervals. *PLoS ONE*
  11(4):e0151235. DOI 10.1371/journal.pone.0151235. — the source of the recommendation to use
  Goodman's simultaneous intervals and of the finding that many binomial intervals reject
  Benford too often for truly Benford data.
- McGraw KO, Wong SP (1996). Forming inferences about some intraclass correlation
  coefficients. *Psychological Methods* 1(1):30–46. DOI 10.1037/1082-989X.1.1.30. — the
  one-way ANOVA ICC(1) convention declared here.
- Kish L (1965). *Survey Sampling*. Wiley. — the design effect `1 + (k-1)ICC`.
- Mardia KV, Jupp PE (2000). *Directional Statistics*. Wiley. — the Rayleigh test.
- Stephens MA (1970). Use of the Kolmogorov-Smirnov, Cramér-von Mises and related statistics
  without extensive tables. *JRSS B* 32:115–122. — the Kuiper critical values `1.747/sqrt(n)`.
- Nigrini MJ (2012). *Benford's Law*. Wiley. — the MAD conformity bands, named for every MAD
  reported and never used as the verdict.
- Sandron F (2002). *Population* 57:755–762. — the founding descriptive study of this
  ensemble.

`Goodman LA (1965)`, *Technometrics* 7:247–254, supplies the simultaneous interval formula
used in `analysis/stats.py::goodman_interval`; the formula itself is reproduced verbatim in
the docstring and cross-checked against the Lesperance et al. description.
