"""Scratch: the CLUSTER matched null (primary), projections, ablations, per-year series."""
import time
import numpy as np
import pandas as pd

import benford_core as bc
import size_laws as sl

t0 = time.time()
df = pd.read_csv('data/owid_population_unwpp.csv')
df.columns = ['entity', 'code', 'year', 'pop']
df['code'] = df.code.fillna('')
d = df[df.code.str.match(r'^[A-Z]{3}$')].copy()
d = d[(d.year >= 1950) & (d.year <= 2023) & (d['pop'] > 0)].reset_index(drop=True)
d['y'] = np.log10(d['pop'].astype(float))
d['m'] = d.y % 1.0
d['d1'] = bc.first_digit(d['pop'].values)
even = d[d.year % 2 == 0].reset_index(drop=True)
odd = d[d.year % 2 == 1].reset_index(drop=True)

y = even.y.values
n = len(y)
law = sl.fit_mixture(y, 3, np.random.default_rng(1), n_restarts=10)
pm = law.digit_proportions()
print('k3 params', law.params)
print('k3 digits', np.round(pm, 5), ' MAD_vs_Ben %.5f' % np.mean(np.abs(pm - bc.BENFORD_P)))

p_obs = bc.digit_proportions(even.d1.values)
r_obs = 100 * (p_obs - bc.BENFORD_P)
print('obs    ', np.round(p_obs, 5), ' M=%.4f' % np.abs(r_obs).max())

# ---- CLUSTER matched null: one mantissa per country from the law, replicated
# over that country's own even years with a small mean-reverting walk.
codes = even.code.values
uniq = np.unique(codes)
inv = np.searchsorted(uniq, codes)
n_per = np.bincount(inv, minlength=len(uniq))
print('\ncountries %d, even-years per country: min %d max %d' % (len(uniq), n_per.min(), n_per.max()))

B = 4000
rng = np.random.default_rng(31)
Mv = np.empty(B); c1v = np.empty(B); Pv = np.empty(B)

# predicted direction (unit normalised residual nine-vector)
u_dir = pm - bc.BENFORD_P


def pool_stats(ms_by_country):
    """Digit counts and c1 from a list of per-country mantissa arrays."""
    counts = np.zeros(9)
    z = 0j
    nn = 0
    for ms in ms_by_country:
        dd = np.clip(np.floor(10 ** ms + 1e-12), 1, 9).astype(int)
        counts += np.bincount(dd, minlength=10)[1:]
        z += np.sum(np.exp(2j * np.pi * ms))
        nn += len(ms)
    return counts / counts.sum(), z / nn


WALK = 0.028  # measured detrended within-country sd of log10
t1 = time.time()
for b in range(B):
    base = law.sample(rng, len(uniq))
    parts = []
    for i in range(len(uniq)):
        steps = rng.normal(0, WALK, n_per[i])
        parts.append((base[i] + steps) % 1.0)
    pb, c1b = pool_stats(parts)
    Mv[b] = np.max(np.abs(100 * (pb - pm)))
    c1v[b] = abs(c1b)
    Pv[b] = bc.projected_statistic(100 * (pb - bc.BENFORD_P), u_dir)["P"]
print('cluster-law null B=%d in %.1fs' % (B, time.time() - t1))
c1_obs = bc.c_harmonic(even.m.values, 1)
P_obs = bc.projected_statistic(r_obs, u_dir)["P"]
cos_obs = bc.projected_statistic(r_obs, u_dir)["cosine"]
print('\n  M   : obs %.4f ; null mean %.4f p2.5 %.4f p50 %.4f p97.5 %.4f -> pct %.1f' % (
    np.abs(r_obs).max(), Mv.mean(), np.percentile(Mv, 2.5), np.percentile(Mv, 50),
    np.percentile(Mv, 97.5), 100 * (Mv < np.abs(r_obs).max()).mean()))
print('  |c1|: obs %.4f ; null mean %.4f p2.5 %.4f p50 %.4f p97.5 %.4f -> pct %.1f' % (
    abs(c1_obs), c1v.mean(), np.percentile(c1v, 2.5), np.percentile(c1v, 50),
    np.percentile(c1v, 97.5), 100 * (c1v < abs(c1_obs)).mean()))
print('  P   : obs %.4f ; null mean %.4f p97.5 %.4f -> pct %.1f' % (
    P_obs, Pv.mean(), np.percentile(Pv, 97.5), 100 * (Pv < P_obs).mean()))
print('  cos : obs %.4f' % cos_obs)
print('  predicted direction u:', np.round(u_dir, 4), ' |u| %.4f' % np.linalg.norm(u_dir))

# ---- ablations
print('\n--- ablations (odd fold, so it is the reserved evidence) ---')
print('microstates <=1e6 share: %.3f of country-years' % ((odd['pop'] <= 1e6).mean()))
print('top20 share: %.3f' % odd['code'].isin(odd.groupby('code')['pop'].mean().nlargest(20).index).mean())
print('n distinct microstates', odd[odd['pop'] <= 1e6].code.nunique())
print('n distinct top20', odd.code.nunique() and odd.groupby('code')['pop'].mean().nlargest(20).size)

# ---- per-year series
print('\n--- per-year |c1| and growth dispersion ---')
rows = []
for yr in sorted(even.year.unique()):
    s = even[even.year == yr]
    rows.append((yr, abs(bc.c_harmonic(s.m.values, 1))))
    # growth over the 2-yr step needs the full panel
ser = pd.DataFrame(rows, columns=['year', 'c1'])
full = d.pivot_table(index='year', columns='code', values='y')
g = full.diff().std(axis=1)
mrg = ser.merge(g.rename('gsd'), left_on='year', right_index=True)
print('corr |c1(y)| vs growth sd: pearson %.3f spearman %.3f' % (
    mrg.c1.corr(mrg.gsd), mrg.c1.corr(mrg.gsd, method='spearman')))
print(mrg.head(6).to_string())
print('\ntotal %.1fs' % (time.time() - t0))
