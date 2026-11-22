# Testing Benford's law on country populations

`demo/` — Self-contained demo (Colab-ready notebook or markdown). Run without setup.  
`src/` — Full source code, data, and outputs from the experiment execution.

**Type:** evaluation  
**ID:** `art_4wZNFo7Nh_VZ`

## Layman Summary

Re-measures how well Benford's law fits the leading digits of world-country populations, and shows that the widely reported 'deviation' disappears once the fact that each country is counted many times is properly accounted for.

## Full Summary

EVALUATION of the iteration's Benford screen on the leading digits of world-country populations, on the reserved odd-year half of a 236-country x 74-year panel (8,732 country-years, 0 shared with the screen half). No DATASET or EXPERIMENT artifact was staged, so the panel is rebuilt from OWID's UN-WPP redistribution and World Bank WDI SP.POP.TOTL and the screen refitted on even years under the identical pre-registered recipe; every confirmation number carries parameters_refit_by_evaluation = true.

HEADLINE. Under the null the literature implicitly uses (nominal-n i.i.d.) the panel is decisively far from exact Benford: MAD 0.00935, sup-distance M 0.03065, |c1| 0.07521 (phase -89.78 deg), chi-square 91.69 at nominal n = 8,732, Kuiper V 0.04205, all p < 0.0005, chi-square at 12.3x the naive null median. But those rows are 236 countries observed 37 times each (ICC 0.2206, design effect 8.94, within-country mantissa sd 0.028 decades vs a marginal 1.206). Against a null drawing one exact-Benford mantissa per country and letting that country's other years wander at the measured amplitude, the TWO-SIDED p-VALUES ARE 0.470 (MAD), 0.927 (M), 0.493 (|c1|), 0.365 (chi-square), 0.887 (Kuiper V): the panel is inside its exact-Benford envelope on all five. An entity-level check at n = 236 agrees (chi-square 12.31, p = 0.138; G-squared 11.88, p = 0.156).

TWO-NULL DISCRIMINATION. The digits cannot separate exact Benford from the fitted three-component lognormal mixture either: 88.7% overlap between the two nulls' simulated log-likelihood-ratio distributions, observed 28.41 vs model-predicted 9.96, p = 0.075 vs Benford and 0.766 vs the model. The winning mixture's digit law is MAD 0.0050 from Benford, so no 236-draw sample could separate them; that indistinguishability is the sharpened negative.

RESERVED-FOLD CONFIRMATION. M_screen 0.01245, M_reserved 0.01425 < 0.05, attenuation ratio 1.145 <= 1.25, matched-null percentile 3.55: SURVIVOR CONFIRMED. Limit printed: the percentile is near the lower band edge, and an ENTITY-DISJOINT check (4 country-hashed folds) confirms 2 of 4; all four have ratios 1.02-1.06 and M_test <= 0.059, so the confirmation partly rests on within-country mantissa stability.

PRE-REGISTERED ABLATION CRITERION FAILS. Microstates <= 1e6 (plug-in signs 3/9, cosine -0.809, ratio -2.36), SIDS (5/9, 0.588, 1.04), top-20 (4/9, 0.081, 0.04); pooled 12/27; pass count 0 of 3. The post-ablation REFIT rule does far better (7/9, 8/9, 9/9 signs; cosine 0.924 for top-20), the expected asymmetry since refitting absorbs part of the deletion. The sign pattern is reproducible under ablation but its magnitude is not predicted by component deletion, so the 'digits as spectroscopy' claim does not survive as specified.

GRADED EVIDENCE. Width-dose-response over five narrow-mass strata: Spearman rho 0.10, p = 0.87. Per-year series over 37 reserved years against EACH YEAR'S OWN fitted law (a pooled fit's first harmonic is attenuated to ~3e-5, a constant, so it cannot be the predictor): observed vs predicted |c1(y)| r = 0.636, CI [0.237, 0.849]; vs calendar year r = -0.717; vs cross-country sd of log growth r = 0.346, CI [-0.033, 0.649]. Trend evidence only.

RIVALS AND VALIDITY. Rounding to 3 significant figures or the nearest 1,000 moves the predicted-direction projection by only 5.8% (bound < 50%). Projected-residual sign stable at every population floor 0 to 5e6; scale-rotation identity exact for c in {1.5, 3, 7}. Source invariance WDI vs OWID: per-year |c1| rank correlation 0.990 and Re(c1)/Im(c1) signs agree, but only 35.8% of overlapping country-years are near-identical, so this is a consistency check, NOT independence. 37 per-year tests, 0 raw and 0 BH rejections. bc.self_test must pass first; cal.sampler_self_test confirms a uniform mantissa reproduces Benford (MAD 0.0007) while the uniform-DIGIT inverse-CDF log10(1+9u) does not (0.0594).

SEVEN DEFECTS FIXED, documented in code and README: (1) chi-square and Kuiper V ignored the reference vector; (2) the rounding simulation measured a tautology (rounded/raw ~ 1, reported as '102% explained') instead of the change rounding makes (5.8%); (3) the per-year predictor was a constant; (4) Kuiper V had no reference distribution; (5) the entity-disjoint check was never called; (6) the naive i.i.d. null was drawn from a uniform-digit law; (7) the prior eval_out.json failed the schema.

AUDIT. verify_headline.py recomputes MAD, M, chi-square, |c1|, phase, ICC, design effect and within-country sd from the raw parquet through a different code path and matches to 1e-16. Its placebo suite shows power: uniform-digit data (MAD 0.0608) and a narrow lognormal (sigma 0.1, MAD 0.1186) are rejected at p < 0.0005, while an i.i.d. uniform-mantissa sample is flagged over-regular (MAD 0.0029 vs observed 0.0093); a mantissa-permutation placebo is VACUOUS for a marginal statistic.

## Output Files

- `eval.py`
- `full_eval_out.json`
- `mini_eval_out.json`
- `preview_eval_out.json`
- `reproducibility.md`

## Demo Files

- **eval.py** — Evaluation script with metrics computation

---
*Generated by AI Inventor Pipeline*
