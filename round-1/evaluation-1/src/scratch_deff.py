"""Scratch: design effect, cluster bootstrap timing, within-country mantissa movement."""
import time
import numpy as np
import pandas as pd

import benford_core as bc

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

print('within-country movement of log10 per year:')
mv = []
for c, s in d.sort_values('year').groupby('code'):
    dy = np.diff(s.y.values)
    mv.append(np.median(np.abs(dy)))
mv = np.array(mv)
print('  median |dlog10|/yr: median %.5f  p10 %.5f  p90 %.5f' % (
    np.median(mv), np.quantile(mv, .1), np.quantile(mv, .9)))
# equivalent per-2-year (even-fold spacing) sd of the *deviation* around the
# country's own trend: detrended residual sd
resid = []
for c, s in d.sort_values('year').groupby('code'):
    yy = s.y.values
    if len(yy) < 10:
        continue
    p = np.polyfit(s.year.values, yy, 1)
    resid.append(np.std(yy - np.polyval(p, s.year.values)))
print('  detrended within-country sd of log10: median %.4f p90 %.4f' % (
    np.median(resid), np.quantile(resid, .9)))

for name, sub in [('even', even), ('odd', odd)]:
    de = bc.design_effect(sub.code.values, sub.m.values)
    print('%s: ICC=%.4f nbar=%.2f deff=%.2f n_eff=%.1f' % (
        name, de['icc'], de['mean_rows_per_group'], de['design_effect'], de['n_effective']))
    de2 = bc.design_effect(sub.code.values, sub.d1.values.astype(float))
    print('   digit-indicator ICC=%.4f deff=%.2f n_eff=%.1f' % (
        de2['icc'], de2['design_effect'], de2['n_effective']))

ci = bc.ClusterIndex.build(even.code.values, even.d1.values, even.m.values)
print('\ncluster index: U=%d' % len(ci.codes))
t1 = time.time()
rng = np.random.default_rng(1)
B = 10000
c1b = ci.boot_c1(rng, B)
print('B=%d c1 bootstrap in %.2fs' % (B, time.time() - t1))
obs = bc.c_harmonic(even.m.values, 1)
amps = np.abs(c1b)
print('obs c1 = %.5f%+.5fj |c1|=%.5f phase %.2f deg' % (
    obs.real, obs.imag, abs(obs), np.degrees(np.angle(obs))))
print('boot |c1|: mean %.5f sd %.5f p2.5 %.5f p50 %.5f p97.5 %.5f' % (
    amps.mean(), amps.std(), np.percentile(amps, 2.5), np.percentile(amps, 50), np.percentile(amps, 97.5)))
print('percentile of obs: %.2f' % (100 * (amps < abs(obs)).mean()))
print('   (rayleigh at n=236: mean %.4f p95 %.4f)' % (
    np.sqrt(np.pi / (4 * 236)), np.sqrt(-np.log(.05) / (2 * 236))))

props = ci.boot_digit_props(rng, B)
r_boot = 100 * (props - bc.BENFORD_P)
print('\nM boot: mean %.5f p2.5 %.5f p97.5 %.5f; obs M=%.5f pct %.2f' % (
    np.abs(r_boot).max(1).mean(), np.percentile(np.abs(r_boot).max(1), 2.5),
    np.percentile(np.abs(r_boot).max(1), 97.5),
    np.max(np.abs(100 * (bc.digit_proportions(even.d1.values) - bc.BENFORD_P))),
    100 * (np.abs(r_boot).max(1) < np.max(np.abs(100 * (bc.digit_proportions(even.d1.values) - bc.BENFORD_P)))).mean()))
print('total %.1fs' % (time.time() - t0))
