import numpy as np
from scipy import stats
import benford_core as bc
import size_laws as sl

mu, sigma = 6.2, 1.2
p_num = sl.periodise_lognormal(mu, sigma)
print('p_num', np.round(p_num, 6))
print('benford', np.round(bc.BENFORD_P, 6))
print('max |p_num - benford|', np.max(np.abs(p_num - bc.BENFORD_P)))

cn = [np.exp(2j * np.pi * n * mu - 2 * np.pi ** 2 * n ** 2 * sigma ** 2) for n in range(0, 400)]
print('c1', cn[1], 'c2', cn[2])
p_cf = bc.digits_from_harmonics(cn[1:], nmax=300)
print('p_cf ', np.round(p_cf, 6))
print('sum p_cf', p_cf.sum())

# brute-force numeric periodisation by dense quadrature for comparison
lower = np.concatenate([bc.DIGIT_EDGES, [1.0]])
p_brute = np.zeros(9)
for k in range(-8, 9):
    p_brute += stats.norm.cdf((lower[1:] + k - mu) / sigma) - stats.norm.cdf((lower[:-1] + k - mu) / sigma)
p_brute /= p_brute.sum()
print('p_brute', np.round(p_brute, 6))

# check K coefficient numerically against quadrature
for n in [1, 2, 3]:
    kq = np.zeros(9, dtype=complex)
    for i in range(9):
        e, w = bc.DIGIT_EDGES[i], bc.DIGIT_WIDTHS[i]
        t = np.linspace(e, e + w, 20001)
        kq[i] = np.trapezoid(np.exp(-2j * np.pi * n * t), t)
    print(f'n={n} K err', np.max(np.abs(kq - bc.fourier_bin_coefficient(n))))
