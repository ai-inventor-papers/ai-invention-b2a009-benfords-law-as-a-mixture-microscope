import json
d = json.load(open('results/eval_out.json'))
c = d['reserved_fold_confirmation']
for k in ['M_reserved', 'M_screen', 'replication_attenuation_ratio', 'M_matched_null_percentile',
          'M_matched_null_in_band_2_5_97_5', 'passes_margin', 'passes_attenuation', 'decision']:
    print(k, '=', c[k])
print('\nnull M (reserved):', json.dumps(d['matched_null_calibration']['nulls']['matched_law_reserved']['M']))
print('envelope M reserved:', json.dumps(d['matched_null_calibration']['envelopes']['matched_law_reserved_fold']['M']))
print('\nrounding:', json.dumps({k: v for k, v in d['robustness']['rounding'].items() if k != 'variants'}))
print('rounding variants:', json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'residuals_pp'}
                                        for k, v in d['robustness']['rounding']['variants'].items()}))
print('\ncross-check:', d['size_law_fits']['periodise_vs_harmonic_cross_check'])
print('winner digits:', [round(x, 5) for x in d['size_law_fits']['table'][0]['digit_proportions']])
print('observed reserved:', [round(x, 5) for x in d['reserved_fold_statistics']['vs_benford']['digit_proportions']])
print('\nper-year r:', d['metrics_agg']['per_year_modulus_pearson_r'],
      'growth r:', d['metrics_agg']['per_year_growth_pearson_r'])
print('dose rho:', d['metrics_agg']['dose_response_spearman_rho'])
print('proj P:', d['metrics_agg']['projected_P_reserved'], 'pct', d['metrics_agg']['projected_P_percentile'])
print('\nablation:', json.dumps(d['ablation_suite']['pooled_sign_agreement']))
for n, r in d['ablation_suite']['ablations'].items():
    ag = r['agreement']
    print(' ', n, 'plugin:', json.dumps({k: ag['plugin'][k] for k in
          ['sign_agreement_of_nine', 'cosine', 'cosine_null_p97_5', 'projection_ratio']}) if 'plugin' in ag else 'none',
          '| refit:', json.dumps({k: ag['refit'][k] for k in ['sign_agreement_of_nine', 'cosine', 'projection_ratio']}) if 'refit' in ag else 'none')
print('\ninvariance:', d['source_invariance']['fraction_near_identical_1e_6'],
      'n non-ident', d['source_invariance']['n_genuinely_non_identical'])
print('thresholds stable:', d['robustness']['inclusion_thresholds']['projection_sign_stable_across_thresholds'])
