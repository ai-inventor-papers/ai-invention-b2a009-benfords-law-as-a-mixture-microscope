"""Coverage check of candidate simultaneous intervals for multinomial digit cells."""
import numpy as np
from scipy import stats

BEN = np.log10(1 + 1 / np.arange(1, 10))
rng = np.random.default_rng(3)


def wilson(k, n, z):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return np.maximum(0, c - h), np.minimum(1, c + h)


def cp(k, n, a):
    lo = np.where(k > 0, stats.beta.ppf(a / 2, k, n - k + 1), 0.0)
    hi = np.where(k < n, stats.beta.ppf(1 - a / 2, k + 1, n - k), 1.0)
    return lo, hi


for n in (200, 500, 900):
    draws = rng.multinomial(n, BEN, size=4000)
    za = stats.norm.ppf(1 - 0.05 / (2 * 9))
    for name, fn in [('wilson', lambda k: wilson(k, n, za)),
                     ('cp', lambda k: cp(k, n, 0.05 / 9))]:
        cov = np.ones(len(draws), bool)
        marg = []
        for i, row in enumerate(draws.astype(float)):
            lo, hi = fn(row)
            ok = (lo <= BEN) & (BEN <= hi)
            cov[i] = bool(ok.all())
            marg.append(ok.mean())
        print(f'n={n} {name}: simultaneous={cov.mean():.4f} marginal={np.mean(marg):.4f}')
