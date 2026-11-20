"""Stage 3 - rival size laws for the population distribution, and the exact
leading-digit law each one implies.

Everything is done on the log10 scale: y = log10 x.  With Y = log10 X the
leading-digit probabilities are

    p_d = int_{log10 d}^{log10(d+1)} f_Y(y) dy           (exact; the mantissa
                                                          bin lies inside [0,1))

and the harmonic coefficients are the characteristic function of Y at 2*pi*n:

    c_n = int f_Y(y) exp(2j pi n y) dy,

because the periodization of f_Y over the integers has no effect on the
integrand (exp(2j pi n (y-k)) = exp(2j pi n y)).  For the lognormal this
characteristic function is available in closed form, which gives the
formula-verification test the plan requires.

Laws implemented
----------------
single lognormal            (mu, sigma)                 MLE closed form
lognormal mixture K=2,3     (w, mu, sigma)              EM + BIC
double Pareto-lognormal     (mu, sigma, alpha, beta)    MLE, Reed & Jorgensen
truncated power law         (alpha; L, U at the data)   MLE, Brent on the bound
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats
from scipy.special import logsumexp

from instrument import BENFORD, DIGIT_EDGES

SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------------------
# quadrature helpers
# ---------------------------------------------------------------------------
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(200)


def _bin_quadrature() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-computed quadrature points and weights over the nine digit bins."""
    pts, wts, bin_of = [], [], []
    for d in range(9):
        lo, hi = DIGIT_EDGES[d], DIGIT_EDGES[d + 1]
        pts.append(0.5 * (hi - lo) * _GL_NODES + 0.5 * (hi + lo))
        wts.append(0.5 * (hi - lo) * _GL_WEIGHTS)
        bin_of.append(np.full(_GL_NODES.size, d))
    return np.concatenate(pts), np.concatenate(wts), np.concatenate(bin_of)


_BQ_PTS, _BQ_WTS, _BQ_BIN = _bin_quadrature()


def _periodized_quadrature(kmax: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quadrature points over the nine digit bins, replicated over k = -K..K.

    The mantissa density is the periodization g(m) = sum_k f_Y(m + k), so the
    mass in digit bin d is sum_k int_{e_d+k}^{e_{d+1}+k} f_Y.  Population
    log10 values live around 2..9.4, i.e. far outside [0, 1), so the
    periodization is not optional: integrating only the [0,1) copy would
    return a digit law that does not sum to 1.
    """
    pts, wts, bin_of = [], [], []
    for k in range(-kmax, kmax + 1):
        pts.append(_BQ_PTS + k)
        wts.append(_BQ_WTS)
        bin_of.append(_BQ_BIN)
    return np.concatenate(pts), np.concatenate(wts), np.concatenate(bin_of)


_PER_PTS, _PER_WTS, _PER_BIN = _periodized_quadrature()


def digit_law_from_logpdf(logpdf, periodization_check: bool = True,
                          tol: float = 1e-3) -> tuple[np.ndarray, float]:
    """Nine digit probabilities of the periodized log10-scale log-density.

    p_d = sum_k int_{log10 d + k}^{log10(d+1) + k} f_Y(y) dy.

    Returns ``(p, normalisation_residual)``.  A law with hard support (the
    truncated power law) has edges that do not align with the quadrature
    nodes, so the raw sum can miss a sliver of mass; the residual is
    recorded and, when it is small, the vector is renormalised.  A large
    residual means the truncated periodization genuinely cannot represent the
    density and the fit must not be reported.
    """
    dens = np.exp(logpdf(_PER_PTS))
    p = np.zeros(9)
    np.add.at(p, _PER_BIN, dens * _PER_WTS)
    total = float(p.sum())
    residual = abs(total - 1.0)
    if periodization_check and residual > tol:
        raise AssertionError(
            f"periodized digit law sums to {total:.8f} (residual {residual:.2e} > {tol:.0e}): "
            f"the truncated periodization cannot represent this density")
    if total > 0:
        p = p / total
    return p, residual


def charfun_numeric(logpdf, n: int, lo: float = -60.0, hi: float = 60.0,
                    npts: int = 24001) -> complex:
    """c_n = int f_Y(y) exp(2j pi n y) dy by trapezoid over a wide support.

    The log10-scale support of every law here is far inside [-60, 60].
    """
    y = np.linspace(lo, hi, npts)
    f = np.exp(logpdf(y))
    return complex(np.trapezoid(f * np.exp(2j * np.pi * n * y), y))


# ---------------------------------------------------------------------------
# law container
# ---------------------------------------------------------------------------
@dataclass
class LawFit:
    """A fitted size law plus everything derived from it."""

    name: str
    family: str
    params: dict[str, float]
    k_params: int
    loglik: float
    n: int
    level: str = ""
    converged: bool = True
    diagnostics: dict = field(default_factory=dict)
    logpdf: object = None            # callable on log10 scale
    sampler: object = None           # callable (rng, size) -> x
    digit_law: np.ndarray | None = None
    c1_analytic: complex | None = None

    def aic(self) -> float:
        return -2.0 * self.loglik + 2.0 * self.k_params

    def bic(self) -> float:
        return -2.0 * self.loglik + self.k_params * math.log(max(self.n, 1))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "family": self.family,
            "level": self.level,
            "params": self.params,
            "k_params": self.k_params,
            "loglik": float(self.loglik),
            "aic": float(self.aic()),
            "bic": float(self.bic()),
            "n": int(self.n),
            "converged": bool(self.converged),
            "diagnostics": self.diagnostics,
            "digit_law": [float(v) for v in self.digit_law] if self.digit_law is not None else None,
            "digit_law_max_abs_dev_from_benford": (
                float(np.max(np.abs(self.digit_law - BENFORD))) if self.digit_law is not None else None),
            "c1_analytic": ({"re": float(self.c1_analytic.real), "im": float(self.c1_analytic.imag),
                             "amp": float(abs(self.c1_analytic)),
                             "phase": float(np.angle(self.c1_analytic))}
                            if self.c1_analytic is not None else None),
        }


# ---------------------------------------------------------------------------
# 1. single lognormal
# ---------------------------------------------------------------------------
def fit_lognormal(y: np.ndarray, level: str = "") -> LawFit:
    mu = float(np.mean(y))
    sigma = float(np.std(y, ddof=0))

    def logpdf(yy, mu=mu, sigma=sigma):
        return -0.5 * ((yy - mu) / sigma) ** 2 - np.log(sigma) - 0.5 * math.log(2 * math.pi)

    def sampler(rng, size, mu=mu, sigma=sigma):
        return 10.0 ** rng.normal(mu, sigma, size)

    ll = float(np.sum(logpdf(y)))
    fit = LawFit(name="lognormal", family="lognormal", params={"mu": mu, "sigma": sigma},
                 k_params=2, loglik=ll, n=int(y.size), level=level,
                 logpdf=logpdf, sampler=sampler)
    fit.digit_law, _res = digit_law_from_logpdf(logpdf)
    fit.diagnostics['digit_law_normalisation_residual'] = _res
    fit.c1_analytic = complex(np.exp(2j * np.pi * mu - 2 * np.pi ** 2 * sigma ** 2))
    return fit


# ---------------------------------------------------------------------------
# 2. lognormal mixture by EM
# ---------------------------------------------------------------------------
def em_lognormal_mixture(y: np.ndarray, k: int, *, rng: np.random.Generator,
                         max_iter: int = 3000, tol: float = 1e-8,
                         n_restarts: int = 21, sigma_floor: float = 0.02,
                         init: dict | None = None) -> dict:
    """EM for a K-component lognormal mixture on y = log10 x.

    Initialisation: one deterministic quantile split plus ``n_restarts - 1``
    random starts.  The best final log-likelihood is returned together with
    the number of non-degenerate components and the monotonicity flag.

    ``init`` supplies ``{w, mu, sigma}`` to start from instead of the quantile
    split -- used by the ablation bootstrap, where a resampled panel should be
    fitted from the full-panel solution rather than from scratch.  Starting
    there is both faster (a handful of EM iterations instead of thousands) and
    statistically the right thing: a bootstrap replicate is a perturbation of
    the same generating law, and letting it fall into an unrelated local
    optimum would widen the interval with optimisation noise rather than
    sampling noise.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    qs = np.quantile(y, np.linspace(0, 1, k + 1))
    sd0 = max(0.5 * (np.quantile(y, 0.75) - np.quantile(y, 0.25)), 0.1)

    def one_start(w, mu, sig) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, bool]:
        ll_prev = -np.inf
        monotone = True
        tiny = math.log(2 * math.pi)
        for _ in range(max_iter):
            # E-step, vectorised over components: log p_j(y_i) = log w_j -
            # 0.5*((y_i-mu_j)/sigma_j)^2 - log sigma_j - 0.5 log 2pi.  A
            # per-component Python loop here was the dominant cost of a k=3
            # refit, and the ablation bootstrap does hundreds of them.
            sig_c = np.maximum(sig, sigma_floor)
            logp = (np.log(np.maximum(w, 1e-12))[None, :]
                    - 0.5 * ((y[:, None] - mu[None, :]) / sig_c[None, :]) ** 2
                    - np.log(sig_c)[None, :] - 0.5 * tiny)
            lse = logsumexp(logp, axis=1)
            ll = float(np.sum(lse))
            R = np.exp(logp - lse[:, None])
            # M-step
            nk = R.sum(axis=0) + 1e-12
            w_new = nk / nk.sum()
            mu_new = (R * y[:, None]).sum(axis=0) / nk
            var_new = (R * (y[:, None] - mu_new) ** 2).sum(axis=0) / nk
            sig_new = np.sqrt(np.maximum(var_new, sigma_floor ** 2 + 1e-4))
            if ll < ll_prev - 1e-9:
                monotone = False
            converged = abs(ll - ll_prev) < tol
            w, mu, sig = w_new, mu_new, sig_new
            ll_prev = ll
            if converged:
                break
        return ll_prev, w, mu, sig, monotone

    best = None
    warm = None
    if init is not None:
        w_i = np.asarray(init["w"], dtype=float)[:k]
        mu_i = np.asarray(init["mu"], dtype=float)[:k]
        sig_i = np.asarray(init["sigma"], dtype=float)[:k]
        if w_i.size == k and mu_i.size == k and sig_i.size == k:
            warm = [(w_i, mu_i, sig_i)]
            for _ in range(max(min(2, n_restarts - 1), 0)):
                warm.append((rng.dirichlet(np.maximum(w_i * 20.0, 0.5)),
                             mu_i + rng.normal(0.0, 0.05, k),
                             np.maximum(sig_i * rng.uniform(0.85, 1.15, k), sigma_floor)))
            n_restarts = len(warm)
    for restart in range(n_restarts):
        if warm is not None:
            w, mu, sig = warm[restart]
        elif restart == 0:
            mu = np.array([0.5 * (qs[j] + qs[j + 1]) for j in range(k)])
            sig = np.full(k, sd0)
            w = np.full(k, 1.0 / k)
        else:
            mu = np.array([np.quantile(y, (j + 0.5) / k) for j in range(k)]) + rng.normal(0, 0.15, k)
            sig = np.abs(rng.normal(sd0, 0.2 * max(sd0, 0.2), k)) + 0.05
            w = rng.dirichlet(np.ones(k))
        ll, w_f, mu_f, sig_f, mono = one_start(w, mu, sig)
        if best is None or ll > best[0]:
            best = (ll, w_f, mu_f, sig_f, mono)

    ll, w, mu, sig = best[0], best[1], best[2], best[3]
    order = np.argsort(mu)
    w, mu, sig = w[order], mu[order], sig[order]
    degenerate = int(np.sum(sig <= sigma_floor * 1.05))
    return {"loglik": float(ll), "w": w, "mu": mu, "sigma": sig,
            "monotone": bool(best[4]), "n_degenerate": degenerate,
            "min_sigma": float(np.min(sig))}


def fit_lognormal_mixture(y: np.ndarray, k: int, *, rng: np.random.Generator,
                          level: str = "", n_restarts: int = 21,
                          init: dict | None = None) -> LawFit:
    res = em_lognormal_mixture(y, k, rng=rng, n_restarts=n_restarts, init=init)
    w, mu, sig = res["w"], res["mu"], res["sigma"]

    def logpdf(yy, w=w, mu=mu, sig=sig):
        cols = [np.log(w[j]) - 0.5 * ((yy - mu[j]) / sig[j]) ** 2
                - np.log(sig[j]) - 0.5 * math.log(2 * math.pi) for j in range(len(w))]
        return logsumexp(np.vstack(cols), axis=0)

    def sampler(rng, size, w=w, mu=mu, sig=sig):
        z = rng.choice(len(w), size=size, p=w / w.sum())
        return 10.0 ** rng.normal(mu[z], sig[z])

    fit = LawFit(name=f"lognormal_mixture_k{k}", family=f"lognormal_mixture_{k}",
                 params={"w": w.tolist(), "mu": mu.tolist(), "sigma": sig.tolist()},
                 k_params=3 * k - 1, loglik=res["loglik"], n=int(y.size), level=level,
                 logpdf=logpdf, sampler=sampler,
                 diagnostics={"monotone_loglik": res["monotone"],
                              "n_degenerate_components": res["n_degenerate"],
                              "min_sigma": res["min_sigma"],
                              "n_restarts": n_restarts})
    fit.digit_law, _res = digit_law_from_logpdf(logpdf)
    fit.diagnostics['digit_law_normalisation_residual'] = _res
    fit.c1_analytic = complex(np.sum(w * np.exp(2j * np.pi * mu - 2 * np.pi ** 2 * sig ** 2)))
    return fit


# ---------------------------------------------------------------------------
# 3. double Pareto-lognormal (Reed & Jorgensen 2004)
# ---------------------------------------------------------------------------
def dpln_logpdf_ln(x: np.ndarray, mu: float, sigma: float, alpha: float, beta: float) -> np.ndarray:
    """DPLN density of x, re-expressed in the measure of ln x.

    Derivation (this is the form the document reports).  With V ~ N(mu,
    sigma^2) and T double-Pareto, X = exp(V) * T, so u = ln X = V + S with
    S = ln T having the two-sided exponential density

        f_S(s) = C exp(-alpha s)  (s > 0),   C exp(beta s)  (s < 0),
        C = alpha*beta/(alpha+beta),

    which integrates to C(1/alpha + 1/beta) = 1.  Convolving with the normal
    and changing to the x measure gives, with the SAME constant C in both
    terms,

        f_X(x) = C * x**(beta-1)     * exp(-beta*mu + beta**2 sigma**2/2)
                     * Phi((mu - ln x - beta*sigma**2)/sigma)
               + C * x**(-alpha-1)   * exp(alpha*mu + alpha**2 sigma**2/2)
                     * Phi((ln x - mu - alpha*sigma**2)/sigma).

    A prefactor of alpha on the second term and beta on the first -- the
    natural mis-transcription -- gives each term the wrong weight, and the
    resulting density integrates to neither 1 nor 2 consistently.  The
    constant was fixed here only after a direct numerical convolution of
    f_S with the normal agreed term by term; the ratio of the wrong-prefactor
    density to the convolution ranged over exactly [alpha/C, beta/C].
    """
    x = np.asarray(x, dtype=float)
    lx = np.log(x)
    log_c = math.log(alpha) + math.log(beta) - math.log(alpha + beta)
    t1 = (log_c + (beta - 1.0) * lx - beta * mu + 0.5 * beta ** 2 * sigma ** 2
          + stats.norm.logcdf((mu - lx) / sigma - beta * sigma))
    t2 = (log_c - (alpha + 1.0) * lx + alpha * mu + 0.5 * alpha ** 2 * sigma ** 2
          + stats.norm.logcdf((lx - mu) / sigma - alpha * sigma))
    return np.logaddexp(t1, t2)


def dpln_logpdf_log10(y: np.ndarray, mu: float, sigma: float, alpha: float, beta: float) -> np.ndarray:
    """The same density expressed on the log10 scale (adds the Jacobian)."""
    y = np.asarray(y, dtype=float)
    return dpln_logpdf_ln(10.0 ** y, mu, sigma, alpha, beta) + y * math.log(10.0) + math.log(math.log(10.0))


def dpln_sampler(rng: np.random.Generator, size: int, mu: float, sigma: float,
                 alpha: float, beta: float) -> np.ndarray:
    """Exact sampler: V ~ N(mu, sigma^2); T double-Pareto; X = exp(V) * T."""
    v = rng.normal(mu, sigma, size)
    u = rng.uniform(0.0, 1.0, size)
    big = rng.uniform(0.0, 1.0, size) < beta / (alpha + beta)
    t = np.where(big, u ** (-1.0 / alpha), u ** (1.0 / beta))
    return np.exp(v) * t


def _hill_tail_index(xs: np.ndarray, upper: bool) -> float:
    """Crude Hill-type tail index for the init of alpha (upper) / beta (lower)."""
    xs = np.abs(np.asarray(xs, dtype=float))
    xs = xs[xs > 0]
    if xs.size < 20:
        return 2.0
    q = np.quantile(xs, 0.9 if upper else 0.1)
    tail = xs[xs > q] if upper else xs[xs < q]
    if tail.size < 5:
        return 2.0
    if upper:
        est = 1.0 / max(np.mean(np.log(tail / q)), 1e-3)
    else:
        est = 1.0 / max(np.mean(np.log(q / tail)), 1e-3)
    return float(np.clip(est, 0.5, 6.0))


def fit_dpln(y: np.ndarray, *, level: str = "", n_restarts: int = 5,
             seed: int = 0) -> LawFit:
    """MLE of the DPLN on the log10 scale, with the plan's three validations.

    MEASURE DISCIPLINE (a bug fixed here).  ``dpln_logpdf_ln`` returns the
    density in the measure of x, while ``dpln_logpdf_log10`` returns the
    density of y = log10 x.  Those two differ by the Jacobian
    ``log(x * ln 10)``, which is constant with respect to the parameters but
    large in magnitude (about 1.3e5 on this panel).  Reporting the x-measure
    log-likelihood beside the lognormal's y-measure one made the two BICs
    differ by ~1.1e5 for purely notational reasons, which silently removed
    the DPLN from the comparison.  Every law in this module now reports its
    log-likelihood on the log10 measure, so BIC/AIC compare like with like.

    BOUNDS.  alpha and beta are tail indices on the natural-log scale.  A
    DPLN with alpha, beta -> infinity is a single lognormal, so capping them
    at 8 makes the lognormal limit unreachable and forces the optimiser into
    the ``sigma -> 0`` corner that the first version reported as a
    converged DPLN with sigma pinned to its lower bound.  The bounds below
    let the family contain its own lognormal limit.
    """
    y = np.asarray(y, dtype=float)
    lnx = y * math.log(10.0)
    n = y.size
    rng = np.random.default_rng(seed)
    mu0 = float(np.mean(y)) * math.log(10.0)
    sig0 = float(np.std(y)) * math.log(10.0)
    a0 = _hill_tail_index(lnx, upper=False)
    b0 = _hill_tail_index(lnx, upper=True)

    ALPHA_MAX = 40.0

    def negll(theta):
        mu, sigma, alpha, beta = theta
        if not (0.05 <= sigma <= 8.0 and 0.05 <= alpha <= ALPHA_MAX and 0.05 <= beta <= ALPHA_MAX):
            return 1e12
        lp = dpln_logpdf_log10(y, mu, sigma, alpha, beta)
        if not np.all(np.isfinite(lp)):
            return 1e12
        return -float(np.sum(lp))

    bounds = [(mu0 - 5.0, mu0 + 5.0), (0.05, 8.0), (0.05, ALPHA_MAX), (0.05, ALPHA_MAX)]
    starts = [[mu0, max(float(np.clip(sig0, 0.1, 6.0)), 0.1), a0, b0],
              # the lognormal limit, so the optimiser can discover that the
              # DPLN's extra flexibility buys nothing on this panel
              [mu0, max(sig0, 0.1), 20.0, 20.0],
              [mu0, max(0.7 * sig0, 0.1), 4.0, 4.0]]
    for r in range(max(n_restarts - len(starts), 0)):
        starts.append([mu0 + rng.normal(0, 0.3),
                       float(np.clip(sig0 * rng.uniform(0.4, 1.4), 0.1, 6.0)),
                       float(np.clip(a0 * rng.uniform(0.5, 3.0), 0.1, 20.0)),
                       float(np.clip(b0 * rng.uniform(0.5, 3.0), 0.1, 20.0))])
    best = None
    for x0 in starts:
        try:
            res = optimize.minimize(negll, x0, method="L-BFGS-B", bounds=bounds,
                                    options={"maxiter": 4000, "ftol": 1e-13, "gtol": 1e-9})
        except (ValueError, FloatingPointError):
            continue
        if np.all(np.isfinite(res.x)) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError("DPLN optimisation failed from every restart")

    mu, sigma, alpha, beta = (float(v) for v in best.x)
    # report on the log10 measure, like every other law here
    ll = float(np.sum(dpln_logpdf_log10(y, mu, sigma, alpha, beta)))
    ll_x_measure = float(np.sum(dpln_logpdf_ln(np.exp(lnx), mu, sigma, alpha, beta)))

    # ---- validation (a): numerical normalisation, on both scales
    grid10 = np.linspace(-8.0, 24.0, 320001)
    dens10 = np.exp(dpln_logpdf_log10(grid10, mu, sigma, alpha, beta))
    mass10 = float(np.trapezoid(dens10, grid10))
    grid_ln = grid10 * math.log(10.0)
    dens_ln = np.exp(dpln_logpdf_ln(np.exp(grid_ln), mu, sigma, alpha, beta) + grid_ln)
    mass_ln = float(np.trapezoid(dens_ln, grid_ln))
    # ---- validation (b): sampler against the density by KS on ln x
    draws = dpln_sampler(np.random.default_rng(seed + 1), 200_000, mu, sigma, alpha, beta)
    ln_draws = np.log(draws)
    cdf_ln = np.concatenate([[0.0], np.cumsum(0.5 * (dens_ln[1:] + dens_ln[:-1]) * np.diff(grid_ln))])
    cdf_ln /= cdf_ln[-1]
    ks = stats.kstest(ln_draws, lambda t: np.interp(t, grid_ln, cdf_ln))
    # ---- validation (c): analytic vs Monte-Carlo moments of ln x
    mean_analytic = mu + 1.0 / alpha - 1.0 / beta
    var_analytic = sigma ** 2 + 1.0 / alpha ** 2 + 1.0 / beta ** 2
    mean_mc = float(np.mean(ln_draws))
    se_mc = float(np.std(ln_draws) / math.sqrt(ln_draws.size))
    checks = {
        "normalisation_mass_log10_scale": mass10,
        "normalisation_mass_ln_scale": mass_ln,
        "normalisation_ok": bool(abs(mass10 - 1.0) < 1e-5 and abs(mass_ln - 1.0) < 1e-5),
        "ks_sampler_vs_density": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "sampler_density_ok": bool(ks.pvalue > 0.01),
        "mean_lnx_analytic": mean_analytic,
        "mean_lnx_mc": mean_mc,
        "mean_lnx_se": se_mc,
        "mean_ok": bool(abs(mean_analytic - mean_mc) < 3.0 * se_mc),
        "var_lnx_analytic": var_analytic,
        "var_lnx_mc": float(np.var(ln_draws)),
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message)[:160],
        "method": ("L-BFGS-B, >=5 restarts including the lognormal limit, "
                   "bounds sigma[0.05,8] alpha,beta[0.05,40]"),
        "loglik_log10_measure": ll,
        "loglik_x_measure": ll_x_measure,
        "loglik_measure_gap_is_jacobian": bool(
            abs((ll_x_measure - ll) - float(np.sum(np.log(10.0 ** y * math.log(10.0))))) < 1e-3),
    }
    logpdf10 = lambda yy: dpln_logpdf_log10(yy, mu, sigma, alpha, beta)
    fit = LawFit(name="dpln", family="dpln",
                 params={"mu_ln": mu, "sigma_ln": sigma, "alpha": alpha, "beta": beta},
                 k_params=4, loglik=ll, n=n, level=level,
                 converged=bool(best.success), diagnostics=checks,
                 logpdf=logpdf10,
                 sampler=lambda rng, size: dpln_sampler(rng, size, mu, sigma, alpha, beta))
    fit.digit_law, _res = digit_law_from_logpdf(logpdf10)
    fit.diagnostics['digit_law_normalisation_residual'] = _res
    fit.c1_analytic = charfun_numeric(logpdf10, 1)
    return fit


# ---------------------------------------------------------------------------
# 4. truncated power law on x
# ---------------------------------------------------------------------------
def fit_truncated_power_law(x: np.ndarray, *, level: str = "") -> LawFit:
    """MLE of a truncated power law on x, reported on the log10 measure.

    MEASURE DISCIPLINE (the same bug as in ``fit_dpln``).  The density of y =
    log10 x is ``alpha ln10 10^{-alpha y} / (L^-alpha - U^-alpha)``, whose
    log-likelihood differs from the x-measure one by the constant
    ``sum(log(x ln 10))``.  Fitting on the x measure and reporting that
    log-likelihood beside a log10-measure lognormal makes BIC compare two
    different normalising constants; the fit here is done on the log10
    measure so that every entry in the comparison table is on one scale.
    """
    x = np.asarray(x, dtype=float)
    lo, hi = float(np.min(x)), float(np.max(x))
    y = np.log10(x)

    def negll(alpha):
        # log f_Y(y) = log(alpha) + log(ln10) - alpha*ln10*y - log(L^-alpha - U^-alpha)
        logZ = math.log(lo ** (-alpha) - hi ** (-alpha))
        return -(x.size * (math.log(alpha) + math.log(math.log(10.0)))
                 - logZ - alpha * math.log(10.0) * float(np.sum(y)))

    res = optimize.minimize_scalar(negll, bounds=(1e-3, 20.0), method="bounded",
                                  options={"xatol": 1e-8})
    alpha = float(res.x)
    ll = -float(res.fun)

    def logpdf(yy):
        # f_Y(y) = alpha ln10 10^{-alpha y} / (lo^-alpha - hi^-alpha), and
        # identically ZERO outside [log10 lo, log10 hi]: the power law is
        # supported only on [L, U].  Evaluating the analytic continuation
        # outside the support would give a small but non-zero density, and
        # summing it over the periodization copies inflates the digit-law
        # normalisation by a factor of ~20.
        yy = np.asarray(yy, dtype=float)
        inside = (yy >= math.log10(lo)) & (yy <= math.log10(hi))
        out = np.full(yy.shape, -np.inf)
        out[inside] = (math.log(alpha) + math.log(math.log(10.0))
                       - alpha * yy[inside] * math.log(10.0)
                       - math.log(lo ** (-alpha) - hi ** (-alpha)))
        return out

    fit = LawFit(name="truncated_power_law", family="truncated_power_law",
                 params={"alpha": alpha, "L": lo, "U": hi}, k_params=3, loglik=ll,
                 n=int(x.size), level=level, logpdf=logpdf,
                 sampler=lambda rng, size: (lo ** (-alpha)
                                            - rng.uniform(0, 1, size)
                                            * (lo ** (-alpha) - hi ** (-alpha))) ** (-1.0 / alpha),
                 diagnostics={"optimizer_success": bool(res.success), "alpha": alpha})
    fit.digit_law, _res = digit_law_from_logpdf(logpdf)
    fit.diagnostics['digit_law_normalisation_residual'] = _res
    fit.c1_analytic = charfun_numeric(logpdf, 1)
    return fit


# ---------------------------------------------------------------------------
# goodness of fit on the full data
# ---------------------------------------------------------------------------
def ks_gof(fit: LawFit, y: np.ndarray, *, n_boot: int = 2000,
           rng: np.random.Generator | None = None) -> dict:
    """KS statistic of the fitted CDF on y, with a parametric-bootstrap p."""
    rng = rng or np.random.default_rng(0)
    y = np.asarray(y, dtype=float)

    def make_cdf(f: LawFit):
        grid = np.linspace(float(np.min(y)) - 3.0, float(np.max(y)) + 3.0, 40001)
        dens = np.exp(f.logpdf(grid))
        c = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))])
        if c[-1] <= 0:
            raise RuntimeError(f"CDF construction failed for {f.name}")
        c /= c[-1]
        return lambda t: np.interp(t, grid, c)

    cdf = make_cdf(fit)
    stat = float(stats.kstest(y, cdf).statistic)
    null = np.empty(n_boot)
    for b in range(n_boot):
        sim = fit.sampler(rng, y.size)
        yy = np.log10(np.clip(sim, 1e-300, None))
        null[b] = stats.kstest(yy, cdf).statistic
    return {"D": stat, "p_value": float(np.mean(null >= stat)), "n_boot": n_boot,
            "null_q95": float(np.quantile(null, 0.95))}


# ---------------------------------------------------------------------------
# harmonic budget
# ---------------------------------------------------------------------------
def harmonic_budget(fit: LawFit, n_harm: int = 1) -> dict:
    """Per-component contribution of each mixture component to c_n.

    For a lognormal mixture, component j contributes
    v_j = w_j * exp(-2 pi^2 n^2 sigma_j^2) * exp(2j pi n mu_j), so the
    amplitude attenuation is exp(-2 pi^2 n^2 sigma_j^2) and the phase is
    2 pi n mu_j mod 2 pi.  The sum of the v_j is the mixture's exact c_n.
    """
    fam = fit.family
    out: dict = {"fit": fit.name, "family": fam, "n_harmonic": n_harm}
    if fam == "lognormal":
        w = np.array([1.0]); mu = np.array([fit.params["mu"]]); sig = np.array([fit.params["sigma"]])
    elif fam.startswith("lognormal_mixture"):
        w = np.array(fit.params["w"]); mu = np.array(fit.params["mu"]); sig = np.array(fit.params["sigma"])
    else:
        # Non-mixture families: a single implicit component with the fitted
        # density's own c_1 (no additive decomposition exists).
        out.update({"decomposable": False,
                    "c1_amp": float(abs(fit.c1_analytic)),
                    "c1_phase": float(np.angle(fit.c1_analytic)),
                    "note": "no closed-form component decomposition for this family; "
                            "the fitted law's own c_1 is reported instead"})
        return out
    att = np.exp(-2.0 * np.pi ** 2 * (n_harm ** 2) * sig ** 2)
    v = w * att * np.exp(2j * np.pi * n_harm * mu)
    cum = np.cumsum(v)
    comps = []
    running = 0j
    for j in range(len(w)):
        running += v[j]
        comps.append({
            "component": j + 1, "weight": float(w[j]), "mu": float(mu[j]), "sigma": float(sig[j]),
            "phase_rad": float((2 * np.pi * n_harm * mu[j]) % (2 * np.pi)),
            "attenuation": float(att[j]),
            "contribution_re": float(v[j].real), "contribution_im": float(v[j].imag),
            "contribution_amp": float(abs(v[j])),
            "cumulative_amp": float(abs(running)),
        })
    total = complex(np.sum(v))
    out.update({
        "decomposable": True, "components": comps,
        "sum_contribution_re": float(total.real), "sum_contribution_im": float(total.imag),
        "predicted_amp": float(abs(total)), "predicted_phase": float(np.angle(total)),
        "max_component_amp": float(np.max(np.abs(v))) if len(v) else 0.0,
        "abs_sum_minus_analytic": float(abs(total - fit.c1_analytic)) if fit.c1_analytic is not None else None,
    })
    return out


def viability_diagnostic(budget: dict, threshold: float = 0.03) -> dict:
    """Pre-fit viability: can this fitted law produce a detectable |c_1|?

    The key is stored under BOTH names on purpose.  ``max_component_amp`` is
    what the ranking code and the verdict text read (it matches the field name
    the harmonic budget itself uses); ``max_component_or_law_amp`` is the
    name this function historically emitted and is kept so an older serialised
    result still loads.  Emitting only the long name was a real defect: the
    verdict path read the missing key as NaN, so the "is the law detectable at
    this n_eff?" comparison silently returned False regardless of the data.
    """
    mx = budget.get("max_component_amp", budget.get("c1_amp", 0.0))
    if mx is None or not np.isfinite(mx):
        mx = 0.0
    mx = float(mx)
    return {"max_component_amp": mx, "max_component_or_law_amp": mx,
            "threshold": threshold,
            "viable": bool(mx >= threshold),
            "statement": ("fitted law can in principle produce a detectable amplitude"
                          if mx >= threshold else
                          "fitted law cannot produce |c_1| at or above the viability threshold "
                          "from its own fit")}
