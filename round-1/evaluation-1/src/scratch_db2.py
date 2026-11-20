import numpy as np
from scipy import stats
import benford_core as bc

rng = np.random.default_rng(7)
mu, sigma = 6.2, 1.2
# Monte-Carlo ground truth for the digit law
y = rng.normal(mu, sigma, 4_000_000)
m = y % 1.0
d = np.clip(np.floor(10 ** m + 1e-12), 1, 9).astype(int)
p_mc = np.bincount(d, minlength=10)[1:] / len(d)
print('MC       ', np.round(p_mc, 6))

lower = np.concatenate([bc.DIGIT_EDGES, [1.0]])
p = np.zeros(9)
for k in range(-20, 21):
    p += stats.norm.cdf((lower[1:] + k - mu) / sigma) - stats.norm.cdf((lower[:-1] + k - mu) / sigma)
print('CDFdiff  ', np.round(p, 6), 'sum', p.sum())
print('Benford  ', np.round(bc.BENFORD_P, 6))

# per-k contributions for digit 1 only
forty = 0.0
for k in range(-20, 21):
    forty += stats.norm.cdf((lower[1] + k - mu) / sigma) - stats.norm.cdf((lower[0] + k - mu) / sigma)
print('digit1 via CDFdiff:', forty)
print('digit1 MC:', p_mc[0], ' Benford:', bc.BENFORD_P[0])

# check harmonics of the MC mantissa
for n in (1, 2):
    print(f'MC c{n}', np.mean(np.exp(2j * np.pi * n * m)))
print('analytic c1', np.exp(2j * np.pi * mu - 2 * np.pi ** 2 * sigma ** 2))
