# Four rival explanations of country-population digit patterns

`demo/` — Self-contained demo (Colab-ready notebook or markdown). Run without setup.  
`src/` — Full source code, data, and outputs from the experiment execution.

**Type:** experiment  
**ID:** `art_04FdxASEUd1B`

## Layman Summary

Tests how closely the leading digits of every country's population follow Benford's law, and shows that most of the apparent deviation is country-to-country repetition rather than a real signal.

## Full Summary

Nine stages end to end on the UN World Population Prospects country panel (17,464 country-years, 236 countries, 1950-2023), acquired from the OWID grapher export and cross-checked against World Bank WDI SP.POP.TOTL.

BASELINE, at the same n and rows: the conventional exact-Benford battery - first-digit frequencies, MAD with Nigrini's bands named but never used as a verdict, chi-square with df and its independence caveat, Kuiper V on mantissas, mantissa mean and MAD-from-0.5, first-two-digit MAD, and Bonferroni-Wilson simultaneous per-digit intervals at the naive n and at the effective n. Pooled even fold (n = 8,732): MAD = 0.00919 (acceptable conformity), chi-square = 83.2 on 8 df with naive i.i.d. p = 1.1e-14, Kuiper V = 0.0561. The baseline rejects exact Benford, as the published literature does.

OUR METHOD: the same panel in harmonic coordinates via the complex first Fourier coefficient c_1 of the mantissa distribution (exact Benford <=> every c_n = 0). Observed |c_1| = 0.0749, phase -1.5733. The panel is a census of countries over 37 even years, so its 8,732 rows are not independent: measured intraclass correlation rho = 0.9575, design effect 35.47, hence n_eff = 246. There the entity-cluster Rayleigh floor is 0.0565 and the 2-sigma threshold 0.0901, so |c_1| = 0.0749 sits 1.33x the floor and BELOW 2 sigma. An independent re-derivation (audit_rederive.py, different code path, plain numpy) reproduces |c_1| = 0.074926 and n_eff = 246.18 exactly, and its own entity-cluster exact-Benford placebo gives p = 0.274, matching the screen's candidate-A p of 0.262.

FIVE RIVAL SIZE LAWS at four levels, each converted to its exact leading-digit law by periodizing the log-density: single lognormal, 2- and 3-component lognormal mixtures by EM with BIC, double Pareto-lognormal, truncated power law. All log-likelihoods are on one log10 measure so BIC is comparable; BIC selects the 3-component mixture (parametric-bootstrap KS p = 0.281), whose harmonic budget predicts |c_1| = 0.0355 (ratio 2.11 to the observed), so the static-mixture mechanism is not excluded by amplitude alone.

FOUR CANDIDATE MECHANISMS, each against its own matched parametric-bootstrap null, scored at three levels with a pre-registered conjunction. The pre-registration (definitions, levels, survivor rule, sha256 of screen.py and panel.csv) is written before any statistic touches the real panel, and a post-run audit confirms no odd-year row entered any screen level. Ranking by matched-null p at the pooled even fold: rotation law 0.00498 (Holm 0.0199), temporal desynchronization 0.155, static mixture 0.262, rounding 0.956. No candidate clears all pre-registered levels - the verdict is survivor-less rather than a false winner. V1 calibration on 200 synthetic exact-Benford panels with the real entity structure passes for every scored candidate (fractions rejecting at 5%: 0.025, 0.13, 0.045).

SIXTH FINDING: the entity cross-section (one row per country, circular-mean mantissa) gives |c_1| = 0.0736 at n = 236, essentially the pooled value - the harmonic structure is a between-country property, not a pooling artefact.

NINE CORRECTNESS BUGS found and fixed, each changing a number and each in README.md: (1) DPLN/power-law log-likelihoods were on the x-density measure while lognormal/mixtures used log10, so BIC compared different constants and dropped the DPLN; (2) DPLN bounds made its lognormal limit unreachable; (3) cluster-null drift was fixed 0.02/yr vs observed 0.18, now calibrated to the design effect; (4) that walk accumulated across the panel rather than per country; (5) a 700 MB array now streamed with a guard; (6) candidate B tested absolute spread of a 1e-13 series, now relative; (7) synthetic panels hard-coded 7 significant digits; (8) a flat power curve extrapolated to 7.5e34, now non-finite; (9) the viability diagnostic stored max_component_amp under a different key, so the detectability comparison read NaN and always returned False.

The schema-shaped output holds 70 examples: 4 observation levels, 4 candidate mechanisms, 37 per-year cells and a 25-country sample of the entity cross-section, each carrying predict_baseline_* (exact Benford) and predict_our_* (harmonic) fields so the two methods are compared on the same rows.

FOR THE EVALUATION ARTIFACT: the reserved odd-year half's statistics with envelopes built from EVEN-fold fits only (odd |c_1| = 0.0752, an out-of-sample near-replication of the even 0.0749); three pre-registered ablation predictions with bootstrap intervals for microstates <= 1e6, SIDS and top-20; every fitted law's parameters at every level; and the fold plus ablation row tags in data/panel.csv.

Outputs: results/screen_results.json, results/method_out.json plus full/mini/preview (schema-validated), results/PRE_REGISTRATION.md, results/audit_rederive.json, and six figures via the aii-data-fig-gen generator with chart specs and raw values beside them. Runtime 816 s on 4 vCPUs, no GPU; peak RSS near 1 GB.

## Output Files

- `method.py`
- `full_method_out.json`
- `mini_method_out.json`
- `preview_method_out.json`
- `reproducibility.md`

## Demo Files

- **method.py** — Research methodology implementation

---
*Generated by AI Inventor Pipeline*
