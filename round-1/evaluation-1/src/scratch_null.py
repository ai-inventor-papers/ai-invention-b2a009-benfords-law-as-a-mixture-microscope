"""Scratch: parametric matched nulls, DPLN convergence, ablation magnitudes."""
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
even = d[d.year % 2 == 0]
y = even.y.values
n = len(y)

# DPLN attempt
print('DPLN fit...')
dpln = sl.fit_dpln(y, np.random.default_rng(0), n_restarts=3)
print('  ->', 'dropped (did not converge / failed density check)' if dpln is None else dpln.params)

laws = {
    'lognormal': sl.fit_lognormal(y),
    'mixture_k2': sl.fit_mixture(y, 2, np.random.default_rng(1), n_restarts=8),
    'mixture_k3': sl.fit_mixture(y, 3, np.random.default_rng(1), n_restarts=8),
}
p_obs = bc.digit_proportions(bc.first_digit(even['pop'].values))
counts = bc.digit_counts(bc.first_digit(even['pop'].values))
r_obs = 100 * (p_obs - bc.BENFORD_P)

# exact-Benford null at nominal n and at n_eff
rng = np.random.default_rng(11)
for label, nn in [('nominal n=8732', n), ('n_eff=977', 977)]:
    draws = rng.multinomial(nn, bc.BENFORD_P, size=10000) / nn
    r = 100 * (draws - bc.BENFORD_P)
    print('\n-- exact-Benford null, %s' % label)
    print('   M: mean %.4f p95 %.4f ; obs |M|=%.4f pct %.2f' % (
        np.abs(r).max(1).mean(), np.percentile(np.abs(r).max(1), 95),
        np.abs(r_obs).max(), 100 * (np.abs(r).max(1) < np.abs(r_obs).max()).mean()))
    # c1
    pc = np.mean(np.exp(2j * np.pi * (np.floor(10 ** (draws)) / 10 + 0)), axis=1) if False else None

# parametric matched null per law at the observed per-year composition
years = even.year.values
codes = even.code.values
for name, law in laws.items():
    pm = law.digit_proportions()
    # simulate: same number of countries per year, drawn from the law
    B = 4000
    rng2 = np.random.default_rng(21)
    Mvals = np.empty(B)
    c1vals = np.empty(B)
    upn = np.unique(years)
    per_year = pd.Series(years).value_counts().sort_index().values
    for b in range(B):
        tot = np.zeros(9)
        zsum = 0j
        for ny in per_year:
            ys = law.sample(rng2, ny)
            ms = ys % 1.0
            dd = np.clip(np.floor(10 ** ms + 1e-12), 1, 9).astype(int)
            tot += np.bincount(dd, minlength=10)[1:]
            zsum += np.sum(np.exp(2j * np.pi * ms))
        pb = tot / tot.sum()
        Mvals[b] = np.max(np.abs(100 * (pb - pm)))
        c1vals[b] = abs(zsum / per_year.sum())
    M_obs = np.max(np.abs(100 * (p_obs - pm)))
    print('\n-- matched null under %s (|c1|_model=%.4f, MAD_vs_Ben=%.5f)' % (
        name, abs(law.harmonic(1)), np.mean(np.abs(pm - bc.BENFORD_P))))
    print('   M(obs,model)=%.4f ; null M mean %.4f p2.5 %.4f p95 %.4f p97.5 %.4f -> pct %.1f' % (
        M_obs, Mvals.mean(), np.percentile(Mvals, 2.5), np.percentile(Mvals, 95),
        np.percentile(Mvals, 97.5), 100 * (Mvals < M_obs).mean()))
    print('   |c1| null: mean %.4f p2.5 %.4f p97.5 %.4f ; obs |c1|=%.4f -> pct %.1f' % (
        c1vals.mean(), np.percentile(c1vals, 2.5), np.percentile(c1vals, 97.5),
        abs(bc.c_harmonic(even.m.values if 'm' in even else (even.y.values % 1), 1)),
        100 * (c1vals < abs(bc.c_harmonic((even.y.values % 1), 1))).mean()))
print('\ntotal %.1fs' % (time.time() - t0))
