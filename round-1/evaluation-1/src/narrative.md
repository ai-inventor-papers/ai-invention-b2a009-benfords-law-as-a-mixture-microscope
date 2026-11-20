# How closely does Benford's law fit the leading digits of world-country populations?

**Reserved-fold evaluation of the iteration's Benford screen.**
Source of all numbers: `eval_out.json` (regenerate with `uv run eval.py`, ~2 min, CPU only).
Peak memory ≈ 1 GB. Seed 20261119 throughout.

---

## Answer in one paragraph

The panel is **far from Benford under the statistic the literature reports, and completely
consistent with Benford under the statistic the data support**. Measured against exact Benford,
the reserved odd-year fold gives MAD = 0.0093, sup-distance *M* = 0.031, |c₁| = 0.075 and
χ² = 91.7 on nominal *n* = 8,732 — every one of which rejects conformity under the usual i.i.d.
null (*p* < 10⁻⁴), with χ² twelve times the null median. But the 8,732 rows are **236 countries
observed 37 times each**, with an ICC of 0.221 and a design effect of 8.94. Against a null that
draws *one* exact-Benford mantissa per country and lets that country's other years wander by the
measured within-country amplitude (0.028 decades), the same five statistics give two-sided
*p* = **0.47 (MAD), 0.93 (M), 0.49 (|c₁|), 0.36 (χ²), 0.89 (Kuiper V)** — the panel is **inside**
its exact-Benford envelope on all five readouts, and χ² sits at 0.59× the null median. An
independent entity-level check using exactly one draw per country (*n* = 236) agrees: χ² = 12.31,
*p* = 0.138; likelihood-ratio *G*² = 11.88, *p* = 0.156. The answer to the title question is
therefore **"about as closely as a 236-country sample can possibly be measured, and that is not
closely enough to call a departure."** The apparent Benford departure is the aggregation artefact
that a nominal-*n* test manufactures — and it is not marginal: it is a 12× χ² produced entirely by
counting one country's near-identical years as independent evidence.

Two further results fix what that negative means, and both are reported because they pull in
opposite directions from a naive reading:

- The fitted size law (a three-component lognormal mixture, BIC winner) **does carry over** to the
  reserved fold: *M* = 0.0143 < 0.05, replication-attenuation ratio 1.14 ≤ 1.25, matched-null
  percentile 3.55. So the digits are well described by the size distribution.
- But the **digits cannot tell the size law from Benford at all**: a likelihood-ratio comparison of
  the observed nine-vector against each null's own simulated distribution gives 88.7% overlap
  between the two null distributions, and the observed vector sits inside both. That
  indistinguishability — not a rejection of either — is the sharpest form of the negative, and it
  is exactly what a mixture whose own digit law has MAD ≈ 0.005 from Benford must produce.

---

## What was evaluated

No DATASET or EXPERIMENT artifact was staged (the three directions of iteration 1 run in
parallel), so the panel was rebuilt from the two public sources and the screen **refitted here**
under the identical pre-registered recipe. Every confirmation number carries
`parameters_refit_by_evaluation = true`: this is a **refit-transfer**, not a frozen carry-over. No
screen value is invented.

| | |
|---|---|
| Analysis panel | OWID `population-unwpp` (UN WPP redistribution), 236 ISO3 countries, 1950–2023 (WPP estimates only through 2023; no projections) |
| Invariance panel | World Bank WDI `SP.POP.TOTL`, 217 entities |
| Rows | 17,464 (8,732 even / 8,732 odd; **0 shared country-years**) |
| Split | year parity: **even = screen side (model fitted here), odd = reserved confirmation side** |
| Digit definition | first significant digit of the *published* figure, base 10, no truncation |
| Rival size laws | single lognormal, 2- and 3-component lognormal mixtures, DPLN, truncated power law |

Held constant: digit definition, inclusion threshold, source vintage per year, fitting recipe
(EM, 20 restarts, K by BIC), bootstrap resampling unit (country). Excluded and stated rather than
silent: the discrete generalised beta / rank-size family (Fontanelli et al. 2016) is cited but not
refitted.

---

## 1. Conformity, calibrated three ways

All five statistics are computed on the **same** simulated replicates, so they cannot drift apart,
and Kuiper V finally has a reference distribution rather than being reported bare.

| statistic | observed | cluster-calibrated *p* | naive i.i.d. *p* | observed / naive null median |
|---|---|---|---|---|
| MAD vs Benford | 0.00935 | **0.470** | < 0.0005 | 3.79 |
| sup-distance *M* | 0.03065 | **0.927** | < 0.0005 | 5.31 |
| \|c₁\| | 0.07521 | **0.493** | < 0.0005 | 8.53 |
| χ² (df 8, nominal *n*) | 91.69 | **0.365** | < 0.0005 | **12.33** |
| Kuiper V | 0.04205 | **0.887** | < 0.0005 | 4.99 |

*(reserved odd fold; 10,000 cluster replicates, 2,000 i.i.d. replicates.)* The last column is
the trap in one number: under the null the literature assumes, the typical χ² is 7.44 and the
observed 91.7 is **12×** that — a decisive rejection for exactly the right statistical reason,
because that null treats 8,732 rows as 8,732 independent draws. Correct the null for the 236
clusters and the same statistic sits at 0.59× the null median, inside the envelope. **Same data,
same statistic, opposite conclusion; the only thing that changed is the null the data support.**
Figure 3 shows the contrast on the statistic/null-median scale for all five readouts.

**Why the cluster null is the right one.** The digit law of a size law treats a country's
population as *one* draw from the law. A country's 37 even-year mantissas move by 0.028 decades
detrended, against a marginal spread of 1.206 decades — a ratio of 0.023. Its years are
near-replicates. Counting them as independent replication is the aggregation trap, and it inflates
χ² by the design effect of ~9. MAD and |c₁| are less sensitive but still misread at this clustering.

**Entity-level corroboration** (one draw per country, *n* = 236, the structurally correct sample
size): χ² = 12.31 (*p* = 0.138), *G*² = 11.88 (*p* = 0.156), MAD = 0.0170. Non-significant, and
consistent with the cluster-calibrated simulation. Note the expected count for digit 9 is 10.8, so
the asymptotic χ² is itself marginal here — it corroborates, it does not decide.

**Per-digit detail** (`metadata.per_digit_table`): the largest raw Benford residual is −3.06 pp on
digit 1 and +1.98 pp on digit 5. All nine simultaneous Clopper–Pearson intervals (Bonferroni over
nine digits, familywise 0.05) **cover** Benford. No single digit drives a rejection.

## 2. Can the digits discriminate the size law from Benford at all?

This is the question that decides what the negative means, and it is answered by its own
simulation rather than asserted. The statistic is the log-likelihood ratio of the nine counts under
the fitted law over exact Benford, calibrated against *each* null's own simulated distribution with
the country structure preserved (5,000 replicates each).

- Observed LLR = **28.41**; the fitted law predicts 9.96 under itself, 0 under Benford.
- **88.7%** of the Benford-null LLR distribution falls inside the model-null 2.5–97.5% band: the two
  nulls are largely the same distribution at this sample size.
- Only 10.6% of model-null draws exceed the Benford null's 97.5th percentile — the design has little
  separating power.
- Two-sided *p* against Benford = 0.075; against the fitted law = 0.766. The data do not reject
  either, and lean only weakly toward the model.

The reason is arithmetic, not noise: the BIC-winning mixture's own digit law is MAD = 0.0050 from
exact Benford (max 1.65 pp on digit 1). Two hypotheses whose predictions differ by ≤ 1.7 pp cannot
be separated by 236 independent draws whose digit-proportion standard error is ≈ 0.03.

## 3. Does the size law carry over out of sample? (yes, with a stated limit)

Pre-registered rule: confirmed iff *M*<sub>reserved</sub> < 0.05 in probability units **and** its
matched-null percentile in [2.5, 97.5] **and** *M*<sub>reserved</sub>/*M*<sub>screen</sub> ≤ 1.25.

| quantity | value |
|---|---|
| *M*<sub>screen</sub> | 0.01245 |
| *M*<sub>reserved</sub> | **0.01425** |
| replication-attenuation ratio | **1.145** (limit 1.25) |
| matched-null percentile of *M*<sub>reserved</sub> | **3.55** |
| decision | **SURVIVOR CONFIRMED** |

**Limit, stated plainly.** The percentile of 3.55 is inside the pre-registered band but close to
its lower edge: the reserved fold is *closer* to the fitted law than the null typically is, which is
expected when the law was fitted to the same countries' even years. And the year-parity split is
out-of-sample in *time* but not in *entity* — all 236 countries appear on both sides.

**Entity-disjoint sensitivity check** (never the primary decision; countries hashed into 4 folds,
each fitted on its own countries' even years and confirmed on its own countries' odd years):
**2 of 4 folds confirmed.** The two failures are informative rather than alarming — fold 1 sits at
percentile 62 (dead centre of the envelope, i.e. *M* is exactly what the null predicts, so the
"failure" is only that the fitted law is not tighter than the null) and fold 3 at percentile 0.0
(*M*<sub>test</sub> = 0.0249, comfortably under the 0.05 margin, but tighter than the null). All
four have attenuation ratios 1.02–1.06 and *M*<sub>test</sub> ≤ 0.059. The confirmation therefore
rests partly on within-country mantissa stability; that is stated, not hidden.

## 4. Ablations: the pre-registered causal criterion **fails**

Three pre-tagged sub-ensembles (microstates ≤ 10⁶ in the row's year; UN OORLLS 39-State SIDS list;
top-20 by mean population), two prediction rules each. Agreement measures: sign count of nine,
cosine against the matched-null mean, and the projection ratio (should be near 1 if the predicted
magnitude is right).

| ablation | plug-in signs | plug-in cosine | plug-in ratio | refit signs | refit cosine |
|---|---|---|---|---|---|
| microstates ≤ 10⁶ | **3/9** | **−0.809** | **−2.36** | 7/9 | −0.275 |
| SIDS | 5/9 | 0.588 | 1.04 | 8/9 | 0.855 |
| top-20 | 4/9 | 0.081 | 0.04 | **9/9** | **0.924** |

Pooled sign agreement: **12/27** (descriptive; three ablations give a sign test no power, so no
*p*-value is attached). Pre-registered criterion C — at least 2 of 3 with plug-in cosine above the
null's 95th percentile **and** projection ratio in [0.5, 1.5] — is **not met**
(`criterion_c_pass_count = 0`).

**What this means.** The *mechanism* statement (delete a component's weight, renormalise) is
falsified for microstates, where the predicted shift points the wrong way. The *observable net
effect* (refit on the ablated data) is much better — 7/9, 8/9, 9/9 signs, and a cosine of 0.92 for
top-20 — which is the expected asymmetry: refitting absorbs part of the deletion, and the sign
agreement there is nearly free because the residual vector is broadly the same shape. **The
honest reading is that the residual nine-vector's sign pattern is reproducible under ablation but
its magnitude is not predicted by simple component deletion.** The project's central "spectroscopy"
claim does not survive this test in the form the plan specified.

## 5. Graded evidence: width-dose-response **and** the per-year series

**Width-dose-response**: countries binned into five strata by posterior weight on the narrowest
mixture component; the claim is that narrow-component mass and projected residual should track.
Spearman **ρ = 0.10, *p* = 0.87**. No graded evidence. Strata 3 and 4 (highest narrow mass) do
show large projected residuals (P = −30.6 and +40.1) but with *opposite* signs, so no monotone
relation exists.

**Per-year series** (37 reserved years): observed |c₁(y)| against **each year's own fitted law**
(fitting a pooled law would be useless here — its first harmonic is attenuated to ≈ 3×10⁻⁵, a
constant to six decimals; that constant is reported as `pooled_prediction_abs_c1_replaced` so the
inflation is visible).

- observed vs year-specific predicted |c₁|: Pearson **r = 0.636** (moving-block bootstrap 95% CI
  **[0.237, 0.849]**, block length 5, 1,000 replicates); Spearman 0.623 [0.143, 0.869].
- observed |c₁(y)| vs calendar year: **r = −0.717** — the amplitude declines across the window.
- observed |c₁(y)| vs cross-country sd of log growth: r = 0.346, 95% CI **[−0.033, 0.649]** — the
  mechanism check's interval **crosses zero**, so it is not supported at this sample size.

The pre-registered criterion (r > 0 with one-sided *p* < 0.05) is **met as trend evidence** and
reported as such, never as a confirmation. Note the confound the year trend exposes: because
|c₁(y)| and year are strongly correlated, part of the obs-vs-prediction correlation is the shared
time trend. The correct statement is "the year-specific size law tracks the observed amplitude
better than a constant does", not "the size law explains the amplitude".

## 6. Bounding the non-demographic rivals

- **Reporting grain**: rounding to 3 significant figures and separately to the nearest 1,000 moves
  the predicted-direction projection by only **5.77%** of its value (criterion: < 50%). The rival is
  bounded. *(The prior version of this artifact reported 102% here; the numerator was the rounded
  data's own projection rather than the change rounding makes — a circular measurement. This run
  fixes it.)* The grain audit is in `metadata.robustness.rounding.grain_audit`: median 7 significant
  figures, 10.3% of values divisible by 10 (expected 10% under a smooth law), 0.11% by 1,000
  (expected 0.1%).
- **Inclusion thresholds**: the sign of the projected residual is stable at every floor from 0 to
  5×10⁶.
- **Scale rotation**: phase rotates exactly as 360·log₁₀ c for c ∈ {1.5, 3, 7} (error < 10⁻⁶) and
  modulus is unchanged, so the phase estimates are real, not numerical.
- **Source invariance**: OWID vs WDI, per-year |c₁| rank correlation **ρ = 0.990**; Re(c₁) and
  Im(c₁) signs agree. But only **35.8%** of overlapping country-years are near-identical
  (median relative difference 1.1×10⁻⁵) — **this is a consistency check, not demonstrated
  independence**, because both labels descend from the same upstream statistical system.
- **Multiplicity**: 37 per-year conformity tests, 0 raw rejections at 0.05, 0 after Benjamini–
  Hochberg. Nothing to control.

## 7. Instrument validity

`bc.self_test` runs before any statistic and must pass: rotation identity (8.7×10⁻¹⁷), mirror and
reciprocal identities (~10⁻¹⁸), numeric periodisation vs closed-form harmonic synthesis
(2.9×10⁻¹⁶, gate 10⁻⁴), digit-resolution reconstruction including the within-bin sinc term
(midpoint-only is measurably worse, as it must be), and simultaneous coverage of the Bonferroni
intervals on Benford draws (Clopper–Pearson ≥ 0.95 at n = 200, 500, 900; Wilson's shortfall is
measured and recorded rather than hidden). The nine-cell transfer gap is 3.4×10⁻³ — the resolution
floor of reading a law through nine bins, not a modelling error.

`cal.sampler_self_test` then verifies that the null sampler reproduces Benford *and that the
tempting alternative does not*: a uniform mantissa gives digit law MAD 0.0007 from Benford, while
the inverse-CDF `log₁₀(1+9u)` (which makes digits equally likely, not Benford) gives 0.0594.
Getting that backward would silently swap the null, so the check runs every time and the run
aborts if it fails.

---

## 8. Independent re-derivation and placebo audit

`verify_headline.py` recomputes the headline numbers through a **different code path** —
reading the raw parquet panels, taking digits by explicit threshold comparison on
`mod(log10 x, 1)` rather than the pipeline's `floor(10**m + 1e-12)`, rebuilding the
cluster null with a stratified per-country sampler instead of independent normal
steps, and re-doing the one-way ANOVA by hand. It imports nothing from
`benford_core` / `calibrated` / `null_models` / `stages_*`.

| number | re-derived | pipeline | relative difference |
|---|---|---|---|
| MAD vs Benford | 0.00934552 | 0.00934552 | 0 |
| sup-distance M | 0.0306452 | 0.0306452 | 0 |
| χ² | 91.691 | 91.691 | 1.6×10⁻¹⁶ |
| \|c₁\| | 0.0752126 | 0.0752126 | 1.9×10⁻¹⁶ |
| phase | −89.7762° | −89.7762° | 1.6×10⁻¹⁶ |
| ICC | 0.220621 | 0.220621 | 2.5×10⁻¹⁶ |
| design effect | 8.94236 | 8.94236 | 4.0×10⁻¹⁶ |
| within-country sd | 0.027982 | 0.027982 | 0 |
| clustered *p* for MAD | 0.478 | 0.470 | 1.7×10⁻² (Monte-Carlo, different sampler) |

**Placebo / diagnostic suite.** A test that passes on destroyed data proves nothing, so the
clustered test is exercised on four controls:

| control | MAD | clustered *p* | required behaviour |
|---|---|---|---|
| observed real panel | 0.00935 | 0.478 | not rejected ✓ |
| mantissa permutation | 0.00935 | 0.502 | **vacuous** — MAD identical by construction |
| i.i.d. uniform mantissa | 0.00286 | 0.000 | correctly flagged **too regular** ✓ |
| uniform digits (*p*<sub>d</sub> = 1/9) | 0.06078 | 0.000 | **must** reject ✓ |
| narrow lognormal (σ = 0.1 dex) | 0.11855 | 0.000 | **must** reject ✓ |

Three findings from the audit are worth stating rather than hiding. (i) The permutation
placebo is **vacuous** for a marginal statistic — it preserves the nine digit counts exactly,
so it can only tell you about the RNG, not about power; that trap is exactly what the audit
is meant to catch. (ii) A uniform mantissa *is* exact Benford marginally, so it is not a
placebo at all: drawn i.i.d. at 8,732 rows it carries far more independent information than
236 countries, and the two-sided clustered test rightly calls it over-regular (MAD 0.0029 vs
the observed 0.0093). That is the sharpest direct evidence that the clustered null has the
right **width** for this panel — a null too narrow would not flag it. (iii) The two genuine
non-Benford alternatives are rejected decisively, so the observed non-rejection is not a
power failure.

---

## Honest-negative branch (the deliverable)

> **Country populations show no departure from exact Benford that survives country-clustered
> calibration, and no departure from their own fitted size law that survives a matched envelope.
> The widely-quoted "populations deviate from Benford" result is reproduced only under a null the
> data cannot support.**

This recalibrates every Benford screen on official statistics: an investigator who runs a
nominal-*n* χ² on a panel of countries will reject Benford essentially always, and the rejection
carries no information.

It also reschedules the science. The residual temporal signal (|c₁(y)| correlating at 0.64 with a
year-specific(size law) and at −0.72 with the calendar) is the only place a real, mechanism-linked
signal appears, and it needs a design with genuine entity-level independence, not more readouts
over the same 236 countries.

## Where the plan's departures bite

1. **Aggregate |c₁| is demoted to descriptive**, by a pre-flight power gate computed before the
   number is interpreted (`metadata.power_gate`): at 236 independent entities the 80%-power
   detectable amplitude is 0.129 against a fitted-law prediction of 0.036, so the test is
   underpowered 3.6×. The budget went into graded evidence and calibration, not into more metrics.
2. **Ablation criterion C is reported as failed.** Three ablations cannot power a sign test, so
   only effect sizes with intervals are given.
3. **Discrete generalised beta is excluded** and labelled as such.
4. **The 0.05 margin is stated in probability units**, as the plan fixes it, and is reported
   alongside the plan-agnostic figures so the choice of statistic is auditable.
5. **Source independence is unverified**, and the overlap is measured (35.8%) rather than asserted.

## Figures

| file | what it shows |
|---|---|
| `figures/fig1_digit_residuals.pdf` | nine per-digit residuals against exact Benford and against the fitted size law (reserved fold) |
| `figures/fig2_per_year_harmonic.pdf` | observed \|c₁(y)\| with cluster-bootstrap bands vs each year's own fitted law |
| `figures/fig3_calibrated_conformity.pdf` | observed statistic as a multiple of each null's median — the aggregation trap made visible |

Raw numbers behind each figure: `results/figure_raw_values.json`.
