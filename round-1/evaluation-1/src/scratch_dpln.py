"""Scratch: diagnose the DPLN fit (density normalisation + tail-index starts)."""
import numpy as np
import pandas as pd
from scipy import integrate, optimize, special

import size_laws as sl

df = pd.read_csv('data/owid_population_unwpp.csv')
df.columns = ['entity', 'code', 'year', 'pop']
df['code'] = df.code.fillna('')
d = df[df.code.str.match(r'^[A-Z]{3}$')].copy()
d = d[(d.year >= 1950) & (d.year <= 2023) & (d['pop'] > 0)]
even = d[d.year % 2 == 0]
y = np.log10(even['pop'].astype(float).values)
x = np.asarray(y) * sl.LN10

# check the density at plausible params
for pars in [(1.5, 1.5, 14.0, 2.5), (2.0, 1.0, 14.3, 2.5), (1.0, 2.0, 14.0, 2.0)]:
    a, b, mu, s = pars
    law = sl.DPLNLaw(a, b, mu, s, 0.0)
    lp = law._logpdf_ln(x)
    integ = integrate.quad(lambda t: np.exp(law._logpdf_ln(np.array([t]))[0]),
                           mu - 40 * s, mu + 40 * s, limit=500)[0]
    # analytic mean: mu + 1/a - 1/b
    print('pars', pars, 'integral %.6f' % integ, 'sample mean ln %.3f analytic %.3f'
          % (x.mean(), mu + 1 / a - 1 / b), 'loglik %.1f' % np.sum(lp), 'finite', np.all(np.isfinite(lp)))

# fit with wider bounds and fewer assumptions
def negll(theta):
    a, b, mu, s = theta
    if not (0.15 < a < 15 and 0.15 < b < 15 and 0.02 < s < 10):
        return 1e12
    law = sl.DPLNLaw(a, b, mu, s, 0.0)
    lp = law._logpdf_ln(x)
    if not np.all(np.isfinite(lp)):
        return 1e12
    return -float(np.sum(lp))

best = None
rng = np.random.default_rng(5)
for r in range(12):
    th0 = np.array([rng.uniform(0.5, 4), rng.uniform(0.5, 4), x.mean() + rng.normal(0, 1), rng.uniform(1, 4)])
    res = optimize.minimize(negll, th0, method="L-BFGS-B",
                            bounds=[(0.15, 15), (0.15, 15), (None, None), (0.02, 10)],
                            options={"maxiter": 2000})
    if best is None or res.fun < best.fun:
        best = res
print('\nbest', best.x, 'negll %.2f' % best.fun, 'success', best.success, best.message)
a, b, mu, s = best.x
law = sl.DPLNLaw(a, b, mu, s, -best.fun)
integ = integrate.quad(lambda t: np.exp(law._logpdf_ln(np.array([t]))[0]),
                       max(mu - 60 * s, -1e4), mu + 60 * s, limit=800)[0]
print('integral %.8f' % integ)
print('loglik directly %.2f vs -negll %.2f' % (np.sum(law._logpdf_ln(x)), -best.fun))
# lognormal comparison
print('lognormal loglik %.2f' % sl.fit_lognormal(y).loglik)
print('digit law', np.round(sl.periodise_from_logpdf(law._logpdf_log10, k_range=law.k_range()), 5))
