"""Scratch: integrate the DPLN in LOG space, where the integrand is well conditioned."""
import numpy as np
import pandas as pd
from scipy import integrate, optimize, stats

LN10 = np.log(10.0)
df = pd.read_csv('data/owid_population_unwpp.csv')
df.columns = ['entity', 'code', 'year', 'pop']
df['code'] = df.code.fillna('')
d = df[df.code.str.match(r'^[A-Z]{3}$')].copy()
d = d[(d.year >= 1950) & (d.year <= 2023) & (d['pop'] > 0)]
even = d[d.year % 2 == 0]
y10 = np.log10(even['pop'].astype(float).values)
x = y10 * LN10


def logpdf_u(u, a, b, mu, s):
    """log density of u = ln X (the DPLN in log space)."""
    c = np.log(a * b / (a + b))
    t1 = c - a * u + a * mu + 0.5 * a * a * s * s + np.log(stats.norm.cdf((u - mu - a * s * s) / s))
    t2 = c + b * u - b * mu + 0.5 * b * b * s * s + np.log(stats.norm.cdf((-u + mu - b * s * s) / s))
    return np.logaddexp(t1, t2)


for pars in [(1.5, 1.5, 14.0, 2.5), (2.0, 1.0, 14.3, 2.5), (1.0, 2.0, 14.0, 2.0)]:
    a, b, mu, s = pars
    lo, hi = mu - 40 * s, mu + 40 * s
    grid = np.linspace(lo, hi, 200001)
    dens = np.exp(logpdf_u(grid, a, b, mu, s))
    print('pars', pars, 'grid integral %.8f' % np.trapezoid(dens, grid),
          '| quad %.8f' % integrate.quad(lambda t: np.exp(logpdf_u(np.array([t]), a, b, mu, s)[0]),
                                        lo, hi, points=[mu - 4 * s, mu - s, mu, mu + s, mu + 4 * s], limit=400)[0])

# direct MLE in log space
def negll(theta):
    a, b, mu, s = theta
    if not (0.1 < a < 20 and 0.1 < b < 20 and 0.01 < s < 12):
        return 1e12
    lp = logpdf_u(x, a, b, mu, s)
    if not np.all(np.isfinite(lp)):
        return 1e12
    return -float(np.sum(lp))

best = None
rng = np.random.default_rng(5)
for r in range(20):
    th0 = np.array([rng.uniform(0.4, 4), rng.uniform(0.4, 4), x.mean() + rng.normal(0, 1),
                    rng.uniform(1.5, 4)])
    res = optimize.minimize(negll, th0, method="L-BFGS-B",
                            bounds=[(0.1, 20), (0.1, 20), (None, None), (0.01, 12)],
                            options={"maxiter": 3000})
    if best is None or res.fun < best.fun:
        best = res
a, b, mu, s = best.x
print('\nbest', np.round(best.x, 4), 'negll %.2f' % best.fun)
print('  analytic mean ln  %.4f  sample mean ln %.4f' % (mu + 1 / a - 1 / b, x.mean()))
print('  analytic var      %.4f  sample var      %.4f' % (
    s * s + 1 / a ** 2 + 1 / b ** 2, x.var()))
print('  lognormal negll %.2f' % (-np.sum(stats.norm.logpdf(x, x.mean(), x.std()))))
lo, hi = mu - 40 * s, mu + 40 * s
print('  integral on grid %.10f' % np.trapezoid(np.exp(logpdf_u(np.linspace(lo, hi, 400001), a, b, mu, s)),
                                                 np.linspace(lo, hi, 400001)))
