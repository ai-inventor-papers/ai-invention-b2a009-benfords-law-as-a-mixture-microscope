"""Scratch: design effect via one-way ANOVA on the complex contribution, cluster bootstrap."""
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
d['zr'] = np.cos(2 * np.pi * d['m'])
d['zi'] = np.sin(2 * np.pi * d['m'])

for fold, sub in [('even', d[d.year % 2 == 0]), ('odd', d[d.year % 2 == 1])]:
    print('===', fold, 'n=', len(sub))
    for comp in ['zr', 'zi']:
        grp = sub.groupby('code')[comp]
        mus = grp.mean()
        ns = grp.size()
        overall = sub[comp].mean()
        ssb = float((ns * (mus - overall) ** 2).sum())
        ssw = float(sum(((sub.loc[sub.code == c, comp].values - mus[c]) ** 2).sum() for c in mus.index))
        msb = ssb / (len(ns) - 1)
        msw = ssw / (len(sub) - len(ns))
        nbar = ns.mean()
        icc = (msb - msw) / (msb + (nbar - 1) * msw)
        deff = 1 + (nbar - 1) * icc
        print(f'  {comp}: ICC={icc:.4f} nbar={nbar:.2f} deff={deff:.2f} n_eff={len(sub)/deff:.0f}')

even = d[d.year % 2 == 0]
codes = even.code.unique()
groups = {c: (even.loc[even.code == c, 'zr'].values + 1j * even.loc[even.code == c, 'zi'].values)
          for c in codes}
B = 4000
amps = np.empty(B)
c1s = np.empty(B, dtype=complex)
for b in range(B):
    pick = rng.choice(codes, size=len(codes), replace=True)
    zz = np.concatenate([groups[c] for c in pick])
    c1s[b] = zz.mean()
    amps[b] = abs(c1s[b])
obs = (even.zr.values + 1j * even.zi.values).mean()
print('\nobserved even c1 = %.5f%+.5fj  |c1|=%.5f' % (obs.real, obs.imag, abs(obs)))
print('cluster bootstrap |c1|: mean %.4f sd %.4f p2.5 %.4f p50 %.4f p97.5 %.4f' % (
    amps.mean(), amps.std(), np.percentile(amps, 2.5), np.percentile(amps, 50), np.percentile(amps, 97.5)))
print('percentile of observed |c1| in bootstrap: %.2f' % (100 * (amps < abs(obs)).mean()))
print('Rayleigh floor n=236: mean %.4f, 95th percentile %.4f' % (
    np.sqrt(np.pi / (4 * 236)), np.sqrt(-np.log(0.05) / (2 * 236))))
print('Rayleigh floor n=8732: mean %.4f' % np.sqrt(np.pi / (4 * 8732)))

# residual-vector cluster bootstrap
BEN = np.log10(1 + 1 / np.arange(1, 10))
def resvec(sub):
    dd = np.clip(np.floor(10 ** (sub.logpop.values % 1) + 1e-12), 1, 9).astype(int)
    p = np.bincount(dd, minlength=10)[1:] / len(dd)
    return 100 * (p - BEN)
print('\nobserved even residual vector (pp)', np.round(resvec(even), 3))
mads = np.empty(B); mx = np.empty(B)
for b in range(B):
    pick = rng.choice(codes, size=len(codes), replace=True)
    sub = pd.concat([even[even.code == c] for c in pick])
    r = resvec(sub)
    mads[b] = np.abs(r).mean() / 100
    mx[b] = np.abs(r).max() / 100
ro = resvec(even) / 100
print('cluster bootstrap of MAD: mean %.5f p2.5 %.5f p97.5 %.5f ; observed %.5f -> pct %.2f' % (
    mads.mean(), np.percentile(mads, 2.5), np.percentile(mads, 97.5), np.abs(ro).mean(),
    100 * (mads < np.abs(ro).mean()).mean()))
print('cluster bootstrap of M: mean %.5f p97.5 %.5f ; observed %.5f -> pct %.2f' % (
    mx.mean(), np.percentile(mx, 97.5), np.abs(ro).max(), 100 * (mx < np.abs(ro).max()).mean()))
np.save('scratch_amp_boot.npy', amps)
