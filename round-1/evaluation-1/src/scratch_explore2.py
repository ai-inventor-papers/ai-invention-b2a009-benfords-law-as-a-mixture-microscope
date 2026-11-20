"""Scratch: first look at the pooled digit distribution and the size law."""
import numpy as np
import pandas as pd

df = pd.read_csv('data/owid_population_unwpp.csv')
df.columns = ['entity', 'code', 'year', 'pop']
df['code'] = df.code.fillna('')
d = df[df.code.str.match(r'^[A-Z]{3}$')].copy()
d = d[(d.year >= 1950) & (d.year <= 2023) & (d['pop'] > 0)]

BEN = np.log10(1 + 1 / np.arange(1, 10))


def first_digit(x):
    m = np.log10(x) % 1.0
    return np.clip(np.floor(10 ** m + 1e-12), 1, 9).astype(int)


def report(sub, label):
    dd = first_digit(sub['pop'].values)
    counts = np.bincount(dd, minlength=10)[1:]
    p = counts / counts.sum()
    r = 100 * (p - BEN)
    mad = np.mean(np.abs(p - BEN))
    chi2 = len(dd) * np.sum((p - BEN) ** 2 / BEN)
    m = np.log10(sub['pop'].values) % 1.0
    c1 = np.mean(np.exp(2j * np.pi * m))
    c2 = np.mean(np.exp(4j * np.pi * m))
    print(f'--- {label}: n={len(dd)}')
    print('  counts', counts)
    print('  p     ', np.round(p, 5))
    print('  r(pp)', np.round(r, 3))
    print(f'  MAD={mad:.5f}  chi2={chi2:.1f} (df=8)  |c1|={abs(c1):.4f} arg={np.degrees(np.angle(c1)):.1f}deg |c2|={abs(c2):.4f}')


report(d, 'pooled 1950-2023 OWID')
report(d[d.year % 2 == 0], 'even years')
report(d[d.year % 2 == 1], 'odd years')
y = np.log10(d['pop'].values)
print('\nlog10 pop: mean %.4f sd %.4f min %.3f max %.3f' % (y.mean(), y.std(), y.min(), y.max()))
print('skew %.3f' % pd.Series(y).skew())
print('\nper-year |c1| (every 8th year):')
for yr in range(1950, 2024, 8):
    s = d[d.year == yr]
    m = np.log10(s['pop'].values) % 1
    print('  %d n=%d |c1|=%.4f' % (yr, len(s), abs(np.mean(np.exp(2j * np.pi * m)))))
