"""Scratch: fit the rival size laws on the even fold and compare their digit laws."""
import time
import numpy as np
import pandas as pd

import benford_core as bc
import size_laws as sl

t0 = time.time()
rng = np.random.default_rng(20261119)
df = pd.read_csv('data/owid_population_unwpp.csv')
df.columns = ['entity', 'code', 'year', 'pop']
df['code'] = df.code.fillna('')
d = df[df.code.str.match(r'^[A-Z]{3}$')].copy()
d = d[(d.year >= 1950) & (d.year <= 2023) & (d['pop'] > 0)].reset_index(drop=True)
d['y'] = np.log10(d['pop'].astype(float))

even = d[d.year % 2 == 0]
y = even.y.values
print('n even =', len(y), ' mean %.4f sd %.4f' % (y.mean(), y.std()))

laws = sl.fit_all(y, np.random.default_rng(1), n_restarts=12)
print('fit time %.1fs' % (time.time() - t0))

p_obs = bc.digit_proportions(bc.first_digit(even['pop'].values))
print('\nobserved  ', np.round(p_obs, 5))
print('Benford   ', np.round(bc.BENFORD_P, 5))
print()
rows = []
for k, law in laws.items():
    p = law.digit_proportions()
    cac = sl.analytic_c_mixture([1.0], [y.mean()], [y.std()]) if k == 'lognormal' else None
    M = float(np.max(np.abs(p_obs - p)))
    rows.append((k, law.n_params, law.bic(len(y)), float(np.max(np.abs(p - bc.BENFORD_P))), M,
                 abs(law.harmonic(1))))
    print('%-22s npar=%d BIC=%10.1f  MAD_vs_Ben=%.5f  M(obs,model)=%.5f  |c1|model=%.5f' % rows[-1])

print('\nsorted by BIC:')
for r in sorted(rows, key=lambda t: t[2]):
    print('  %-22s BIC=%10.1f' % (r[0], r[2]))

km = laws['mixture_k3'] if 'mixture_k3' in laws else laws['mixture_k2']
print('\nharmonic budget for', km.name)
for row in km.harmonic_budget():
    print('  ', {k: round(v, 5) for k, v in row.items()})

print('\nsampler validation (k=20000):')
for k in ['lognormal', 'mixture_k3']:
    print(' ', k, sl.validate_sampler(laws[k], np.random.default_rng(2), n=20000))
print('total %.1fs' % (time.time() - t0))
