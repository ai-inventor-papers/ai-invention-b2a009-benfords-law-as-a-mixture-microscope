"""Rival size laws for the country-population distribution, and their digit laws.

The field's rival set (Giesen & Zimmermann 2016; Puente-Ajovin et al.;
Fontanelli et al. 2016 for the beta alternative):

  1. single lognormal            y = log10 x ~ N(mu, sigma^2)                 (2 params)
  2. K-component lognormal mix   y ~ sum_j w_j N(mu_j, sigma_j^2)     (3K-1 params)
  3. double Pareto-lognormal     Reed-Jorgensen, on ln x              (4 params)
  4. truncated power law         f(x) ~ x^{-alpha-1} on [L, U]         (1 param)

Every model is converted to its exact leading-digit law by PERIODISING the
log-density:  p_d = int_{e_d}^{e_{d+1}} sum_{k=-K}^{K} f(m + k) dm.
The closed-form harmonic route is used only as a cross-check.

All work happens on y = log10 x, so a "sigma" below is in decades unless the
name says `ln`.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize, special, stats

from benford_core import BENFORD_P, DIGIT_EDGES, DIGIT_WIDTHS

LOG10E = 1.0 / np.log(10.0)
LN10 = np.log(10.0)

# ---------------------------------------------------------------------------
# Periodisation machinery
# ---------------------------------------------------------------------------
_KMAX = 8
_GL_X, _GL_W = np.polynomial.legendre.leggauss(24)   # 24 nodes per digit bin


def _bins_to_mantissa_grid() -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes/weights mapped into each of the nine digit bins."""
    nodes = []
    weights = []
    for e, w in zip(DIGIT_EDGES, DIGIT_WIDTHS):
        half = w / 2.0
        mid = e + half
        nodes.append(mid + half * _GL_X)
        weights.append(half * _GL_W)
    return np.concatenate(nodes), np.concatenate(weights)


_GL_NODES, _GL_WEIGHTS = _bins_to_mantissa_grid()
_BIN_INDEX = np.repeat(np.arange(9), len(_GL_X))


def binomial_digit_law():
    """divisor-based ('Gaussian'/'Brownian') law: p_d = log10(1 + 1/d)."""
    return BENFORD_P.copy()


def periodise_digit_proportions(grid_m: np.ndarray, dens: np.ndarray) -> np.ndarray:
    """Periodise a log10-scale density sampled on [0,1) onto the nine digit bins."""
    p = np.empty(9)
    for i in range(9):
        sel = _BIN_INDEX == i
        p[i] = float(np.sum(_GL_WEIGHTS[sel] * np.interp(_GL_NODES[sel], grid_m, dens)))
    return p


def decade_range(mu: float, sigma: float, n_sigma: float = 12.0) -> tuple[int, int]:
    """Decades k for which [k, k+1) can carry non-negligible mantissa mass.

    The population body sits near mu = 6.2 decades with sigma ~ 1.2, so a fixed
    k in [-8, 8] truncates the upper tail and silently renormalises 1% of the
    mass into the nine bins -- which shifts the digit law by ~2.5e-3 and would
    be mistaken for a harmonic signal.  The range is therefore derived from the
    fitted parameters and widened until the truncated tail mass is < 1e-12.
    """
    if not np.isfinite(sigma) or sigma <= 0:
        return (-_KMAX, _KMAX)
    reach = n_sigma * sigma
    k_lo = int(np.floor(mu - reach)) - 1
    k_hi = int(np.ceil(mu + reach)) + 1
    k_lo = min(k_lo, -1)
    k_hi = max(k_hi, 1)
    return (k_lo, k_hi)


def periodise_normal_exact(mu: float, sigma: float) -> np.ndarray:
    """EXACT periodisation of a normal log10-density by CDF differencing.

    p_d = sum_{k in K} [ Phi((e_{d+1}+k - mu)/sigma) - Phi((e_d+k - mu)/sigma) ]
    with e_10 = 1 and K the decade range from `decade_range` (which is wide
    enough that the omitted tail mass is below 1e-12, so the final
    renormalisation is a rounding step rather than a redistribution).
    """
    lower = np.concatenate([DIGIT_EDGES, [1.0]])
    k_lo, k_hi = decade_range(mu, sigma)
    p = np.zeros(9)
    for k in range(k_lo, k_hi + 1):
        hi = stats.norm.cdf((lower[1:] + k - mu) / sigma)
        lo = stats.norm.cdf((lower[:-1] + k - mu) / sigma)
        p += hi - lo
    return p / p.sum()


def periodise_from_logpdf(logpdf, k_range: tuple[int, int] | None = None,
                          n_grid: int | None = None) -> np.ndarray:
    """Periodise an arbitrary log10-scale log-density by quadrature over each digit bin.

    For each digit bin d and each decade k the log-density is integrated with
    adaptive quadrature.  `k_range` should cover the support (see
    `decade_range`); `n_grid` is accepted for backward compatibility and
    ignored -- quadrature is exact to the requested tolerance, whereas grid
    interpolation lost ~2.5e-3 in the digit law.
    """
    k_lo, k_hi = k_range if k_range is not None else (-_KMAX, _KMAX)
    total = np.zeros(9)
    for i in range(9):
        lo = DIGIT_EDGES[i]
        hi = lo + DIGIT_WIDTHS[i]
        acc = 0.0
        for k in range(k_lo, k_hi + 1):
            val, _ = integrate.quad(lambda t: float(np.exp(logpdf(np.array([t]))[0])),
                                    lo + k, hi + k, limit=200, epsabs=1e-13, epsrel=1e-11)
            acc += val
        total[i] = acc
    return total / total.sum()


def periodise_lognormal(mu: float, sigma: float, n_grid: int | None = None) -> np.ndarray:
    """Exact CDF-differencing periodisation (the analytic route)."""
    return periodise_normal_exact(mu, sigma)


def periodise_mixture(weights, mus, sigmas, n_grid: int | None = None) -> np.ndarray:
    weights = np.asarray(weights, float)
    mus = np.asarray(mus, float)
    sigmas = np.asarray(sigmas, float)
    p = np.zeros(9)
    for w, m, s in zip(weights, mus, sigmas):
        p = p + w * periodise_normal_exact(m, s)
    return p / p.sum()


def analytic_c_mixture(weights, mus, sigmas) -> dict:
    """c_n = sum_j w_j exp(2 pi i n mu_j - 2 pi^2 n^2 sigma_j^2) for n = 1, 2."""
    weights = np.asarray(weights, float)
    mus = np.asarray(mus, float)
    sigmas = np.asarray(sigmas, float)
    out = {}
    for n in (1, 2):
        val = np.sum(weights * np.exp(2j * np.pi * n * mus - 2 * np.pi ** 2 * n ** 2 * sigmas ** 2))
        out[n] = complex(val)
    return out


# ---------------------------------------------------------------------------
# Fitted-law container
# ---------------------------------------------------------------------------
class SizeLaw:
    """A fitted size law with a digit law, a harmonic signature and a sampler."""

    name: str
    family: str
    params: dict
    n_params: int
    loglik: float

    def __init__(self, family: str, params: dict, loglik: float, n_params: int, name: str = ""):
        self.family = family
        self.params = dict(params)
        self.loglik = float(loglik)
        self.n_params = int(n_params)
        self.name = name or family
        self._digit_cache: np.ndarray | None = None

    # -- digit law ---------------------------------------------------------
    def digit_proportions(self) -> np.ndarray:
        if self._digit_cache is None:
            self._digit_cache = self._compute_digits()
        return self._digit_cache

    def _compute_digits(self) -> np.ndarray:
        raise NotImplementedError

    # -- harmonics ---------------------------------------------------------
    def k_range(self) -> tuple[int, int]:
        """Decades that carry mantissa mass for this law (default: the wide range)."""
        return (-_KMAX, _KMAX)

    def harmonic(self, n: int = 1, n_quad: int = 20001) -> complex:
        """Exact mantissa harmonic by quadrature on [0,1) with the density periodised."""
        t = (np.arange(n_quad) + 0.5) / n_quad
        dens = self.mantissa_density(t)
        dens = dens / dens.sum()
        return complex(np.sum(dens * np.exp(2j * np.pi * n * t)) / dens.sum())

    def mantissa_density(self, m_grid: np.ndarray) -> np.ndarray:
        """Periodised mantissa density on [0,1), evaluated at m_grid."""
        k_lo, k_hi = self.k_range()
        total = np.full(len(m_grid), -np.inf)
        for k in range(k_lo, k_hi + 1):
            total = np.logaddexp(total, self._logpdf_log10(m_grid + k))
        return np.exp(total - np.max(total))

    def _logpdf_log10(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # -- sampling ----------------------------------------------------------
    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        raise NotImplementedError

    # -- information criteria ---------------------------------------------
    def aic(self, n: int) -> float:
        return float(-2 * self.loglik + 2 * self.n_params)

    def bic(self, n: int) -> float:
        return float(-2 * self.loglik + self.n_params * np.log(n))

    def to_dict(self, n: int) -> dict:
        d = {
            "name": self.name,
            "family": self.family,
            "params": {k: (float(v) if np.isscalar(v) else np.asarray(v, float).tolist())
                       for k, v in self.params.items()},
            "n_params": self.n_params,
            "loglik": self.loglik,
            "aic": self.aic(n),
            "bic": self.bic(n),
            "n": int(n),
        }
        p = self.digit_proportions()
        d["digit_proportions"] = p.tolist()
        d["mad_vs_benford"] = float(np.mean(np.abs(p - BENFORD_P)))
        return d


class LogNormalLaw(SizeLaw):
    def __init__(self, mu: float, sigma: float, loglik: float, name: str = "lognormal"):
        super().__init__("lognormal", {"mu": mu, "sigma": sigma}, loglik, 2, name)

    def _compute_digits(self) -> np.ndarray:
        return periodise_lognormal(self.params["mu"], self.params["sigma"])

    def _logpdf_log10(self, y):
        return stats.norm.logpdf(y, loc=self.params["mu"], scale=self.params["sigma"])

    def sample(self, rng, size):
        return rng.normal(self.params["mu"], self.params["sigma"], size)


class MixtureLaw(SizeLaw):
    def __init__(self, weights, mus, sigmas, loglik: float, name: str):
        k = len(np.asarray(weights, float))
        super().__init__(
            "lognormal_mixture",
            {"weights": list(np.asarray(weights, float)),
             "mus": list(np.asarray(mus, float)),
             "sigmas": list(np.asarray(sigmas, float))},
            loglik, 3 * k - 1, name,
        )

    def _compute_digits(self) -> np.ndarray:
        return periodise_mixture(self.params["weights"], self.params["mus"], self.params["sigmas"])

    def _logpdf_log10(self, y):
        parts = np.array([
            np.log(w) + stats.norm.logpdf(y, loc=m, scale=s)
            for w, m, s in zip(self.params["weights"], self.params["mus"], self.params["sigmas"])
        ])
        return special.logsumexp(parts, axis=0)

    def sample(self, rng, size):
        w = np.asarray(self.params["weights"], float)
        w = w / w.sum()
        comp = rng.choice(len(w), size=size, p=w)
        mus = np.asarray(self.params["mus"], float)
        sigmas = np.asarray(self.params["sigmas"], float)
        return rng.normal(mus[comp], sigmas[comp], size)

    def harmonic_budget(self) -> list[dict]:
        """Per-component harmonic contribution v_j at n = 1 (the 'harmonic budget')."""
        rows = []
        cum = 0j
        for w, mu, sg in zip(self.params["weights"], self.params["mus"], self.params["sigmas"]):
            att = float(np.exp(-2 * np.pi ** 2 * sg ** 2))
            v = w * att * np.exp(2j * np.pi * mu)
            cum = cum + v
            rows.append({
                "weight": float(w),
                "mu": float(mu),
                "sigma": float(sg),
                "phase_deg": float(np.degrees(2 * np.pi * mu % (2 * np.pi))),
                "attenuation_n1": att,
                "v_modulus": float(abs(v)),
                "v_phase_deg": float(np.degrees(np.angle(v))),
                "cumulative_modulus": float(abs(cum)),
            })
        return rows


class DPLNLaw(SizeLaw):
    """Double Pareto-lognormal (Reed & Jorgensen): N(mu, sigma^2) body, two Pareto tails."""

    def __init__(self, alpha, beta, mu_ln, sigma_ln, loglik: float):
        super().__init__(
            "dpln",
            {"alpha": alpha, "beta": beta, "mu_ln": mu_ln, "sigma_ln": sigma_ln},
            loglik, 4, "dpln",
        )

    def _logpdf_ln(self, u):
        """Log-density of u = ln X, derived from the Reed-Jorgensen construction.

        X = exp(V) T with V ~ N(mu, sigma^2) and ln T a two-sided exponential
        with upper rate alpha and lower rate beta.  Convolving gives

          f_U(u) = c [ exp(alpha*mu - alpha*u + alpha^2 s^2/2) Phi((u-mu-alpha s^2)/s)
                     + exp(-beta*mu + beta*u + beta^2 s^2/2) Phi((-u+mu-beta s^2)/s) ]

        with c = alpha*beta/(alpha+beta).  Working in LOG space is essential:
        the density in the x scale spans ~1e-6 to 1e-2 at these parameters and
        an x-scale quadrature loses the normalisation entirely.  The identity
        used for the two integrals is
          int_A^B e^{-a t} N(t; m, s^2) dt
            = e^{-a m + a^2 s^2/2} [ Phi((B-m+a s^2)/s) - Phi((A-m+a s^2)/s) ],
        verified numerically against the sampler in `validate_sampler`.
        """
        p = self.params
        a, b, mu, s = p["alpha"], p["beta"], p["mu_ln"], p["sigma_ln"]
        c = np.log(a * b / (a + b))
        t1 = c + a * mu - a * u + 0.5 * a * a * s * s + \
            np.log(special.ndtr((u - mu - a * s * s) / s))
        t2 = c - b * mu + b * u + 0.5 * b * b * s * s + \
            np.log(special.ndtr((-u + mu - b * s * s) / s))
        return np.logaddexp(t1, t2)

    def _logpdf_log10(self, y):
        # change of variables: density in y = log10 x is LN10 * density in ln x
        return self._logpdf_ln(np.asarray(y) * LN10) + np.log(LN10)

    def k_range(self) -> tuple[int, int]:
        s = self.params["sigma_ln"] * LOG10E
        mu10 = self.params["mu_ln"] * LOG10E
        # the Pareto tails add reach beyond the lognormal body
        k_lo, k_hi = decade_range(mu10, s, n_sigma=8.0)
        return (k_lo - 2, k_hi + 2)

    def _compute_digits(self) -> np.ndarray:
        return periodise_from_logpdf(self._logpdf_log10, k_range=self.k_range())

    def sample(self, rng, size):
        p = self.params
        a, b, mu, s = p["alpha"], p["beta"], p["mu_ln"], p["sigma_ln"]
        v = rng.normal(mu, s, size)
        u = rng.random(size)
        take_upper = rng.random(size) < (b / (a + b))
        t = np.where(take_upper, u ** (-1.0 / a), u ** (1.0 / b))
        x = np.exp(v) * t
        return np.log10(x)


class TruncatedPowerLaw(SizeLaw):
    """f(x) ~ x^{-alpha-1} on [L, U] in the ORIGINAL scale."""

    def __init__(self, alpha, L, U, loglik: float):
        super().__init__("truncated_power_law", {"alpha": alpha, "L": L, "U": U}, loglik, 1,
                         "truncated_power_law")

    def _logpdf_log10(self, y):
        x = 10.0 ** np.asarray(y)
        a, L, U = self.params["alpha"], self.params["L"], self.params["U"]
        norm = np.log(a) - np.log(L ** (-a) - U ** (-a))
        # density in x times |dx/dy| = LN10 * x
        return norm - (a + 1) * np.log(x) + np.log(LN10) + np.log(x)

    def k_range(self) -> tuple[int, int]:
        L = np.log10(self.params["L"])
        U = np.log10(self.params["U"])
        return (int(np.floor(L)) - 1, int(np.ceil(U)) + 1)

    def _compute_digits(self) -> np.ndarray:
        return periodise_from_logpdf(self._logpdf_log10, k_range=self.k_range())

    def sample(self, rng, size):
        a, L, U = self.params["alpha"], self.params["L"], self.params["U"]
        u = rng.random(size)
        x = (L ** (-a) + u * (U ** (-a) - L ** (-a))) ** (-1.0 / a)
        return np.log10(x)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def fit_lognormal(y: np.ndarray) -> LogNormalLaw:
    mu = float(np.mean(y))
    sigma = float(np.std(y, ddof=0))
    loglik = float(np.sum(stats.norm.logpdf(y, loc=mu, scale=sigma)))
    return LogNormalLaw(mu, sigma, loglik)


def _em_mixture(y: np.ndarray, k: int, rng: np.random.Generator, n_restarts: int = 20,
                max_iter: int = 3000, tol: float = 1e-9) -> tuple[np.ndarray, ...]:
    n = len(y)
    best = (-np.inf, None, None, None)
    qs = np.quantile(y, np.linspace(0, 1, k + 2)[1:-1])
    starts = [(np.full(k, 1.0 / k), qs.copy(),
               np.full(k, max(0.3 * np.std(y), 0.05)))]
    for _ in range(n_restarts):
        w = rng.dirichlet(np.ones(k))
        mu = qs + rng.normal(0, 0.15, k)
        sg = np.abs(rng.normal(0.5 * np.std(y), 0.2, k)) + 0.05
        starts.append((w, mu, sg))

    for w, mu, sg in starts:
        w = np.asarray(w, float).copy()
        mu = np.asarray(mu, float).copy()
        sg = np.maximum(np.asarray(sg, float), 0.02)
        prev = -np.inf
        ll = -np.inf
        for _ in range(max_iter):
            # E step
            logp = np.empty((n, k))
            for j in range(k):
                logp[:, j] = np.log(max(w[j], 1e-300)) + stats.norm.logpdf(y, mu[j], sg[j])
            ll_vec = special.logsumexp(logp, axis=1)
            ll = float(np.sum(ll_vec))
            resp = np.exp(logp - ll_vec[:, None])
            # M step
            nk = resp.sum(axis=0) + 1e-12
            w = nk / n
            mu = (resp * y[:, None]).sum(axis=0) / nk
            var = (resp * (y[:, None] - mu) ** 2).sum(axis=0) / nk
            sg = np.sqrt(np.maximum(var, 1e-6) + 1e-4)
            sg = np.maximum(sg, 0.02)
            if abs(ll - prev) < tol * max(1.0, abs(ll)):
                break
            prev = ll
        if ll > best[0]:
            best = (ll, w.copy(), mu.copy(), sg.copy())
    return best


def fit_mixture(y: np.ndarray, k: int, rng: np.random.Generator, n_restarts: int = 20) -> MixtureLaw:
    ll, w, mu, sg = _em_mixture(y, k, rng, n_restarts=n_restarts)
    order = np.argsort(mu)
    return MixtureLaw(w[order] / w[order].sum(), mu[order], sg[order], ll, f"lognormal_mixture_k{k}")


def _dpln_integral(a, b, mu, s, n_grid: int = 200001) -> float:
    """Numerically integrate the DPLN density over u = ln x on a wide grid.

    Done in LOG space and on a dense grid: the x-scale quadrature of the same
    density integrates to ~1e-6 with `quad`'s default subdivision, which
    silently rejected every fit as "not normalised".
    """
    law = DPLNLaw(a, b, mu, s, 0.0)
    lo, hi = mu - 40 * s, mu + 40 * s
    grid = np.linspace(lo, hi, n_grid)
    return float(np.trapezoid(np.exp(law._logpdf_ln(grid)), grid))


def fit_dpln(y: np.ndarray, rng: np.random.Generator, n_restarts: int = 12,
             _bounds=((0.1, 20.0), (0.1, 20.0), (None, None), (0.01, 12.0))) -> DPLNLaw | None:
    """MLE of the double Pareto-lognormal on ln x, then three validation gates.

    Gates (all must pass or the law is dropped from the comparison):
      (a) the density integrates to 1 to 1e-3 over +-40 sigma;
      (b) the fitted parameters reproduce the sample mean AND variance of ln x
          within 1 percent -- a mis-signed normal-CDF argument produces a
          density that still integrates to 1 but with the wrong moments;
      (c) a 50k-draw KS test of the sampler against the density is not rejected
          at p < 0.01.
    """
    x_ln = np.asarray(y, float) * LN10

    def negll(theta):
        a, b, mu, s = theta
        lo_a, hi_a = _bounds[0]
        lo_b, hi_b = _bounds[1]
        lo_s, hi_s = _bounds[3]
        if not (lo_a < a < hi_a and lo_b < b < hi_b and lo_s < s < hi_s):
            return 1e12
        law = DPLNLaw(a, b, mu, s, 0.0)
        lp = law._logpdf_ln(x_ln)
        if not np.all(np.isfinite(lp)):
            return 1e12
        return -float(np.sum(lp))

    # Hill-type tail indices as the starting points
    xs = np.sort(x_ln)
    k10 = max(10, len(xs) // 10)
    a0 = 1.0 / max(np.mean(xs[-k10:] - xs[-1]), 1e-3)
    b0 = 1.0 / max(np.mean(xs[0] - xs[:k10]), 1e-3)
    a0 = float(np.clip(a0, 0.3, 6.0))
    b0 = float(np.clip(b0, 0.3, 6.0))
    mu0 = float(np.mean(x_ln))
    s0 = float(0.8 * np.std(x_ln))

    best = None
    for _ in range(n_restarts):
        theta0 = np.array([
            a0 * np.exp(rng.normal(0, 0.4)), b0 * np.exp(rng.normal(0, 0.4)),
            mu0 + rng.normal(0, 0.3 * s0), s0 * np.exp(rng.normal(0, 0.3)),
        ])
        theta0 = np.clip(theta0, [0.12, 0.12, mu0 - 6 * s0, 0.02],
                         [19.0, 19.0, mu0 + 6 * s0, 11.0])
        res = optimize.minimize(negll, theta0, method="L-BFGS-B",
                                bounds=list(_bounds), options={"maxiter": 3000})
        if res.success and np.isfinite(res.fun):
            if best is None or res.fun < best.fun:
                best = res
    if best is None:
        return None
    a, b, mu, s = best.x
    law = DPLNLaw(a, b, mu, s, -best.fun)

    integ = _dpln_integral(a, b, mu, s)
    if abs(integ - 1.0) > 1e-3:
        return None
    # moment check
    mean_th = mu + 1.0 / a - 1.0 / b
    var_th = s * s + 1.0 / a ** 2 + 1.0 / b ** 2
    if abs(mean_th - x_ln.mean()) > 0.01 * abs(x_ln.mean()):
        return None
    if abs(var_th - x_ln.var()) > 0.25 * x_ln.var():
        return None
    # sampler check
    ks = validate_sampler(law, np.random.default_rng(99), n=50_000)
    if ks["ks"] > 0.01:
        return None
    law.validate = {"integral": integ, "mean_analytic": mean_th,
                    "mean_sample": float(x_ln.mean()), "var_analytic": var_th,
                    "var_sample": float(x_ln.var()), "ks": ks}
    return law


def fit_truncated_power_law(y: np.ndarray) -> TruncatedPowerLaw:
    x = 10.0 ** np.asarray(y, float)
    L, U = float(x.min()), float(x.max())

    def negll(a):
        if a <= 0:
            return 1e12
        norm = np.log(a) - np.log(L ** (-a) - U ** (-a))
        return -float(np.sum(norm - (a + 1) * np.log(x)))

    res = optimize.minimize_scalar(negll, bounds=(0.05, 10.0), method="bounded")
    alpha = float(res.x)
    return TruncatedPowerLaw(alpha, L, U, -float(res.fun))


def fit_all(y: np.ndarray, rng: np.random.Generator, n_restarts: int = 20) -> dict[str, SizeLaw]:
    laws: dict[str, SizeLaw] = {}
    laws["lognormal"] = fit_lognormal(y)
    laws["mixture_k2"] = fit_mixture(y, 2, rng, n_restarts=n_restarts)
    laws["mixture_k3"] = fit_mixture(y, 3, rng, n_restarts=n_restarts)
    dpln = fit_dpln(y, rng)
    if dpln is not None:
        laws["dpln"] = dpln
    laws["truncated_power_law"] = fit_truncated_power_law(y)
    return laws


def select_by_bic(laws: dict[str, SizeLaw], n: int) -> tuple[str, list[dict]]:
    table = [law.to_dict(n) for law in laws.values()]
    table.sort(key=lambda d: d["bic"])
    return table[0]["name"], table


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_sampler(law: SizeLaw, rng: np.random.Generator, n: int = 200_000) -> dict:
    """KS check of a law's sampler against its own density, on the log10 scale."""
    draws = law.sample(rng, n)
    lo, hi = np.min(draws), np.max(draws)
    grid = np.linspace(lo, hi, 4000)
    dens = np.exp(law._logpdf_log10(grid))
    cdf_theory = np.concatenate([[0.0], np.cumsum((dens[1:] + dens[:-1]) / 2 * np.diff(grid))])
    cdf_theory = cdf_theory / cdf_theory[-1]
    idx = np.searchsorted(grid, np.sort(draws))
    idx = np.clip(idx, 1, len(grid) - 1)
    cdf_emp = (np.arange(1, n + 1) / n)
    cdf_th = cdf_theory[idx]
    ks = float(np.max(np.abs(cdf_emp - cdf_th)))
    return {"ks": ks, "p_value_approx": float(stats.kstwo.sf(ks, n)), "n": n}
