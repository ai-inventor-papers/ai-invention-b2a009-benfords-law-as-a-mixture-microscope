"""Scratch exploration of the two population sources."""
import json
import numpy as np
import pandas as pd

raw = json.load(open('data/wb_population_raw.json'))[1]
w = pd.DataFrame([{'code': r['countryiso3code'], 'year': int(r['date']), 'pop': r['value']} for r in raw])
print(len(w), w.code.nunique(), w.year.min(), w.year.max())
w = w.dropna(subset=['pop'])
w = w[w.code.str.len() == 3]
print('after dropna+len3:', len(w), w.code.nunique())

df = pd.read_csv('data/owid_population_unwpp.csv')
df.columns = ['entity', 'code', 'year', 'pop']
df['code'] = df.code.fillna('')
d = df[df.code.str.match(r'^[A-Z]{3}$')][['code', 'year', 'pop']]
m = d.merge(w, on=['code', 'year'], suffixes=('_owid', '_wb'))
print('overlap country-years', len(m))
m['identical'] = (m.pop_owid == m.pop_wb)
print('identical frac', m.identical.mean())
m['rel'] = (m.pop_owid - m.pop_wb).abs() / m.pop_wb
print('rel diff median', m.rel.median(), 'p95', m.rel.quantile(0.95))
print(m.head(5).to_string())
print('wb-only codes', len(set(w.code.unique()) - set(d.code.unique())),
      sorted(set(w.code.unique()) - set(d.code.unique()))[:15])
print('owid-only codes', len(set(d.code.unique()) - set(w.code.unique())),
      sorted(set(d.code.unique()) - set(w.code.unique()))[:15])

# how many WB rows are aggregates (region.id == NA)
print('\n-- WB granularity --')
wv = w['pop'].astype(float)
for k in [1, 3]:
    print('wb divisible 10^%d: %.4f' % (k, (wv % 10 ** k == 0).mean()))
print('wb sig digit-ish print:', wv.head(5).tolist())
