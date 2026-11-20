"""Scratch: design effect, cluster bootstrap, band structure."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(20261119)
df = pd.read_csv('data/owid_population_unwpp.csv')
df.columns = ['entity', 'code', 'year', 'pop']
df['code'] = df.code.fillna('')
d = df[df.code.str.match(r'^[A-Z]{3}$')].copy()
d = d[(d.year >= 1950) & (d.year <= 2023) & (d['pop'] > 0)].reset_index(drop=True)
d['logpop'] = np.log10(d['pop'].astype(float))
d['m'] = d.logpop % 1.0
d['z'] = np.exp(2j * np.pi * d.m.values)

even = d[d.year % 2 == 0]
odd = d[d.year % 2 == 1]


def c1(sub):
    return np.mean(sub['z'].values)


print('even c1', c1(even), abs(c1(even)))
print('odd  c1', c1(odd), abs(c1(odd)))

# ICC of z within country (one-way ANOVA on Re and Im)
for comp in ['re', 'im']:
    v = even['z'].apply(lambda x: getattr(x, comp)).values
    g = even.groupby('code')['z'].apply(lambda s: getattr(np.mean(s.values), comp)).values
    # between/within
    grp = even.groupby('code')
    mus = grp['z'].apply(lambda s: getattr(np.mean(s.values), comp))
    ns = grp.size()
    overall = v.mean()
    ssb = (ns * (mus - overall) ** 2).sum()
    ssw = sum(((even[even.code == c]['z'].apply(lambda x: getattr(x, comp)).values
                - mus[c]) ** 2).sum() for c in mus.index)
    msb = ssb / (len(ns) - 1)
    msw = ssw / (len(v) - len(ns))
    nbar = ns.mean()
    icc = (msb - msw) / (msb + (nbar - 1) * msw)
    print(f'  {comp}: ICC={icc:.4f} nbar={nbar:.1f} deff={1+(nbar-1)*icc:.2f}')

# cluster bootstrap: resample countries, recompute c1 on even fold
codes = even.code.unique()
groups = {c: even[even.code == c]['z'].values for c in codes}
print('\ncountries', len(codes))
B = 2000
amps = np.empty(B)
for b in range(B):
    pick = rng.choice(codes, size=len(codes), replace=True)
    zz = np.concatenate([groups[c] for c in pick])
    amps[b] = abs(zz.mean())
print('cluster bootstrap |c1|: mean %.4f sd %.4f p50 %.4f p95 %.4f p97.5 %.4f' % (
    amps.mean(), amps.std(), np.percentile(amps, 50), np.percentile(amps, 95), np.percentile(amps, 97.5)))
obs = abs(c1(even))
print('observed |c1| = %.4f, percentile %.1f' % (obs, 100 * (amps < obs).mean()))

# analytic Rayleigh check at n=236
print('rayleigh mean |c1| at n=236:', np.sqrt(np.pi / (4 * 236)), 'p95', np.sqrt(-np.log(0.05) / (2 * 236)))

# band structure: residual pattern by population decile
even = even.copy()
even['dec'] = pd.qcut(even.logpop, 5, labels=False)
BEN = np.log10(1 + 1 / np.arange(1, 10))
print('\nby logpop quintile:')
for q, s in even.groupby('dec'):
    dd = np.clip(np.floor(10 ** s.m + 1e-12), 1, 9).astype(int)
    p = np.bincount(dd, minlength=10)[1:] / len(s)
    print(' q%d n=%d meanlog=%.2f |c1|=%.3f arg=%6.1f  r(pp)=%s' % (
        q, len(s), s.logpop.mean(), abs(s.z.mean()), np.degrees(np.angle(s.z.mean())),
        np.round(100 * (p - BEN), 1)))

# how much does the mantissa move within country over time?
sp = d.groupby('code')['m'].agg(lambda x: np.std(np.unwrap(2 * np.pi * x.values) / (2 * np.pi)))
print('\nwithin-country circular sd of mantissa: median %.3f p90 %.3f' % (sp.median(), sp.quantile(0.9)))
