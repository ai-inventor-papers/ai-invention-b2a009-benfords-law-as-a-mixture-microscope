"""Stage 2.5 - synthetic ground-truth validation, run BEFORE the real screen.

This is simultaneously the unit test, the null-calibration check and the
power analysis.  Three blocks:

  V1 calibration: on panels that ARE exact Benford, each candidate's p-value
     must be approximately Uniform(0,1).  A miscalibrated null is the single
     most likely way this experiment produces a false winner, so a candidate
     rejecting at more than 15% forces its null to be replaced by the
     nonparametric entity bootstrap, and the change is recorded.

  V2 known mixture: simulate a 2-component mixture with a known narrow
     component; assert EM recovers its parameters, assert the analytic c_1
     matches the empirical c_1 of the simulated sample, and assert the
     instrument detects it.

  V3 power curve: inject narrow components with a known predicted |c_1| at
     the measured effective sample size and report detection power.  The
     minimum detectable amplitude at 80% power is compared with the real
     harmonic budget BEFORE the screen runs, so the verdict cannot be spun.
"""

from __future__ import annotations

import math

import numpy as np
from loguru import logger
from scipy import stats

import instrument as I
import nulls as N
import size_laws as SL


# ---------------------------------------------------------------------------
# synthetic panel generator with the real entity structure
# ---------------------------------------------------------------------------
def synthetic_panel(counts: np.ndarray, sampler, rng: np.random.Generator,
                    within_sd: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic panel with the observed entity/year structure.

    One base value per entity from ``sampler``, then a mean-reverting log10
    random walk across the entity's years, which is what makes the design
    effect real rather than assumed.
    """
    k = counts.size
    n = int(counts.sum())
    y0 = np.log10(np.clip(sampler(rng, k), 1e-300, None))
    steps = rng.normal(0.0, within_sd, size=n)
    damp = 1.0 / np.sqrt(np.arange(1, n + 1))
    walk = np.cumsum(steps) * damp
    yy = np.repeat(y0, counts) + walk
    x = 10.0 ** yy
    codes = np.repeat(np.arange(k), counts)
    return x, codes


def benford_sampler(rng: np.random.Generator, size: int) -> np.ndarray:
    d1 = rng.choice(np.arange(1, 10), size=size, p=I.BENFORD)
    lo = np.log10(d1)
    hi = np.log10(d1 + 1)
    return 10.0 ** (np.floor(rng.uniform(2, 7, size)) + lo + rng.uniform(0, 1, size) * (hi - lo))


# ---------------------------------------------------------------------------
# V1 - null calibration
# ---------------------------------------------------------------------------
def v1_calibration(counts: np.ndarray, *, n_panels: int, reps: int,
                   rng: np.random.Generator, fit_law=None,
                   audit: dict | None = None) -> dict:
    """Run every candidate's DETECTION test on panels that ARE exact Benford.

    This deliberately calls the same scoring functions the real screen uses.
    A calibration that reimplements the test under a different null cannot
    detect a miscalibrated null, which is the one failure this check exists to
    catch.  Each synthetic panel carries the real entity structure and real
    year labels, so the entity-cluster nulls and the epoch contrast are
    exercised exactly as they will be on the panel.

    ``fit_law`` is the fitted-law factory; when None a single lognormal is
    fitted per panel (cheap, and the null only needs *a* smooth law).
    """
    import screen as SC

    p_vals = {c: [] for c in "ABC"}
    for i in range(n_panels):
        x, codes = synthetic_panel(counts, benford_sampler, rng)
        m = I.mantissa(x)
        # year labels: one per row, consecutive within each entity
        year_list = []
        for c, n_c in zip(np.unique(codes), counts):
            year_list.extend(list(range(1950, 1950 + int(n_c))))
        sub = _synthetic_frame(x, codes, np.array(year_list))
        struct = N.make_struct(codes, m)
        t_obs = float(abs(I.c_n(m, 1)))

        fit = fit_law(sub) if fit_law is not None else SL.fit_lognormal(np.log10(x))

        det = N.null_benford_cluster(struct, rng, reps)
        p_vals["A"].append(float((np.sum(det["amp"] >= t_obs) + 1.0) / (det["amp"].size + 1.0)))

        py = SC.per_year_c1(sub)
        resB = SC.score_candidate_B(sub, py, rng, max(reps // 3, 100))
        p_vals["B"].append(resB["p"] if resB["testable"] else float("nan"))

        aud = audit or SC.rounding_audit(sub)
        resC = SC.score_candidate_C(sub, fit, rng, max(reps // 3, 150), aud)
        p_vals["C"].append(resC["p"] if resC.get("testable") else float("nan"))

    out = {}
    for c, vals in p_vals.items():
        v = np.array([x for x in vals if np.isfinite(x)])
        out[c] = {
            "n_panels": int(v.size),
            "n_undefined": int(len(vals) - v.size),
            "fraction_p_below_0.05": float(np.mean(v < 0.05)) if v.size else float("nan"),
            "ks_uniform_p": float(stats.kstest(v, "uniform").pvalue) if v.size > 10 else float("nan"),
            "calibrated": bool(v.size and 0.0 <= np.mean(v < 0.05) <= 0.15),
            "note": ("a candidate whose statistic is undefined on exact-Benford panels is "
                     "self-falsifying, not miscalibrated" if v.size == 0 else
                     "target fraction below 0.05 is 0.03-0.08; above 0.15 means the null is "
                     "miscalibrated and must be replaced by the nonparametric entity bootstrap"),
        }
    return {"per_candidate": out,
            "note": ("the calibration calls the SAME scoring functions as the real screen, with "
                     "real entity structure and year labels, so a miscalibrated null shows up "
                     "here rather than as a false winner later"),
            "n_panels": n_panels, "reps": reps}


def _synthetic_frame(x: np.ndarray, codes: np.ndarray, years: np.ndarray):
    """A frame with the panel's columns, for the scoring functions.

``sig_digits`` and ``is_integer_value`` are derived from the value the way the
SOURCE prints it, not hard-coded.  The panel's figures are printed as whole
integers, so a synthetic panel is rounded to integers and its significant
digits counted from that string.  Hard-coding 7 digits made candidate C's
grain hypothesis untestable on every synthetic panel and produced a spurious
'self-falsification' in the calibration, because the calibration never saw a
value with the source's actual precision.
"""
    import pandas as pd
    from panel import significant_digits_of_string
    # round to the printed integer FIRST, then derive the mantissa from that
    # value, exactly as build_panel does from the CSV string
    xi = np.where(np.round(x) < 1.0, 1.0, np.round(x))
    y = np.log10(xi)
    frac = y - np.floor(y)
    n = xi.size
    raw = [str(int(v)) for v in xi]
    sig = np.array([significant_digits_of_string(s) for s in raw], dtype=int)
    return pd.DataFrame({
        "code": codes, "year": years, "population": xi,
        "mantissa": np.clip(frac, 0.0, 1.0 - 1e-15),
        "d1": np.clip(np.floor(10.0 ** frac + 1e-12).astype(int), 1, 9),
        "is_integer_value": np.ones(n, dtype=bool),
        "sig_digits": sig,
        "fold": np.where(years % 2 == 0, "even", "odd"),
        "tag_microstate": np.zeros(n, dtype=int), "tag_sids": np.zeros(n, dtype=int),
        "tag_top20": np.zeros(n, dtype=int),
    })


# ---------------------------------------------------------------------------
# V2 - known two-component mixture
# ---------------------------------------------------------------------------
V2_TRUE = {"w": np.array([0.15, 0.85]),
           "mu": np.array([3.2, 6.5]),
           "sigma": np.array([0.15, 1.2])}


def v2_known_mixture(n_rows: int, *, n_reps: int, rng: np.random.Generator) -> dict:
    """Simulate the known mixture; check EM recovery and the analytic c_1."""
    w, mu, sig = V2_TRUE["w"], V2_TRUE["mu"], V2_TRUE["sigma"]
    truth_c1 = complex(np.sum(w * np.exp(2j * np.pi * mu - 2 * np.pi ** 2 * sig ** 2)))

    recov, emp_gaps, detected = [], [], []
    for r in range(n_reps):
        z = rng.choice(2, size=n_rows, p=w)
        y = rng.normal(mu[z], sig[z])
        fit = SL.fit_lognormal_mixture(y, 2, rng=np.random.default_rng(1000 + r), n_restarts=9)
        p = fit.params
        # match components by mu
        order = np.argsort(p["mu"])
        recov.append({"w": [float(np.asarray(p["w"])[i]) for i in order],
                      "mu": [float(np.asarray(p["mu"])[i]) for i in order],
                      "sigma": [float(np.asarray(p["sigma"])[i]) for i in order]})
        m = np.mod(y, 1.0)
        emp_c1 = I.c_n(m, 1)
        emp_gaps.append(abs(abs(emp_c1) - abs(fit.c1_analytic)))
        # detection via entity-cluster exact-Benford null
        codes = np.arange(n_rows)
        struct = N.make_struct(codes, m)
        det = N.null_benford_cluster(struct, rng, 400)
        t_obs = float(abs(emp_c1))
        detected.append(float((np.sum(det["amp"] >= t_obs) + 1.0) / (det["amp"].size + 1.0)) < 0.05)

    rec = np.array([[r["w"][0], r["w"][1], r["mu"][0], r["mu"][1], r["sigma"][0], r["sigma"][1]]
                    for r in recov])
    truth = np.array([w[0], w[1], mu[0], mu[1], sig[0], sig[1]])
    med = np.median(rec, axis=0)
    rel_err = np.abs(med - truth) / truth
    return {
        "true": {"w": w.tolist(), "mu": mu.tolist(), "sigma": sig.tolist()},
        "median_recovered": {"w": [float(med[0]), float(med[1])],
                             "mu": [float(med[2]), float(med[3])],
                             "sigma": [float(med[4]), float(med[5])]},
        "relative_error": {"w": [float(rel_err[0]), float(rel_err[1])],
                           "mu": [float(rel_err[2]), float(rel_err[3])],
                           "sigma": [float(rel_err[4]), float(rel_err[5])]},
        "recovery_within_20pct": bool(np.all(rel_err < 0.20)),
        "truth_analytic_c1": {"re": float(truth_c1.real), "im": float(truth_c1.imag),
                              "amp": float(abs(truth_c1))},
        "median_abs_gap_empirical_vs_analytic": float(np.median(emp_gaps)),
        "analytic_formula_ok_within_10pct": bool(np.median(emp_gaps) < 0.10 * max(abs(truth_c1), 0.01)),
        "detection_rate": float(np.mean(detected)),
        "detected_at_0.05": bool(np.mean(detected) > 0.9),
        "n_reps": n_reps,
    }


# ---------------------------------------------------------------------------
# V3 - power curve
# ---------------------------------------------------------------------------
def v3_power_curve(counts: np.ndarray, targets: list[float], *, n_reps: int,
                   rng: np.random.Generator, reps: int = 400) -> dict:
    """Detection power against injected amplitude at the measured n_eff.

    The injected law is a two-component mixture whose narrow component is
    tuned so the mixture's analytic |c_1| equals the target amplitude; this
    is the cleanest way to place a known amplitude at a known phase.
    """
    out = []
    for target in targets:
        # solve for w1 given a fixed narrow component; wide component is
        # wide enough that its own contribution is numerically zero
        mu_n, sig_n = 6.0, 0.12
        mu_w, sig_w = 4.2, 1.5
        c_narrow = np.exp(2j * np.pi * mu_n - 2 * np.pi ** 2 * sig_n ** 2)
        c_wide = np.exp(2j * np.pi * mu_w - 2 * np.pi ** 2 * sig_w ** 2)
        # |w*c1n + (1-w)*c1w| = target  -> solve numerically for w
        ws = np.linspace(0.0, 0.6, 6001)
        amps = np.abs(ws * c_narrow + (1 - ws) * c_wide)
        idx = int(np.argmin(np.abs(amps - target)))
        w1 = float(ws[idx])
        pred_amp = float(abs(w1 * c_narrow + (1 - w1) * c_wide))
        hit = []
        for r in range(n_reps):
            z = (rng.uniform(0, 1, counts.sum()) < w1).astype(int)
            yy = np.where(z == 1, rng.normal(mu_n, sig_n, counts.sum()),
                          rng.normal(mu_w, sig_w, counts.sum()))
            x = 10.0 ** yy
            m = np.mod(yy, 1.0)
            codes = np.repeat(np.arange(counts.size), counts)
            struct = N.make_struct(codes, m)
            # the detection null must be the amplitude-free reference: a
            # narrow-component-free panel with the same entity structure
            x0 = 10.0 ** rng.normal(mu_w, sig_w, counts.sum())
            m0 = np.mod(np.log10(x0), 1.0)
            s0 = N.make_struct(codes, m0)
            det = N.null_benford_cluster(s0, rng, reps)
            t_obs = float(abs(I.c_n(m, 1)))
            hit.append(float((np.sum(det["amp"] >= t_obs) + 1.0) / (det["amp"].size + 1.0)) < 0.05)
        out.append({"target_amp": target, "realised_amp": pred_amp, "weight_narrow": w1,
                    "power": float(np.mean(hit)), "n_reps": n_reps})
    powers = np.array([o["power"] for o in out])
    amps = np.array([o["realised_amp"] for o in out])
    md80 = float("nan")
    ok = np.where(powers >= 0.80)[0]
    if ok.size:
        md80 = float(amps[ok[0]])
    elif powers.size and powers.max() > 0.02:
        # Extrapolate on a log-linear fit of power against log10(amplitude).
        # GUARD: a fit with few distinct power values, or one whose slope
        # implies an absurd crossing, is not an estimate.  The first version
        # returned 7.5e34 here because the smoke curve was flat at chance and
        # the slope was ~0, so (0.80 - intercept)/slope exploded.  An
        # unattainable minimum is reported as infinity with the reason, never
        # as a number.
        sel = powers > 0.02
        if sel.sum() >= 2 and np.ptp(np.log10(amps[sel])) > 0:
            slope, intercept = np.polyfit(np.log10(amps[sel]), powers[sel], 1)
            if slope > 1e-3:
                cand = 10.0 ** ((0.80 - intercept) / slope)
                if np.isfinite(cand) and cand <= 10.0 * float(amps.max()):
                    md80 = float(cand)
    return {"points": out, "md_amp_80pct_power": md80,
            "md_amp_80pct_power_attainable": bool(np.isfinite(md80)),
            "note": ("md_amp_80pct_power is NaN/inf when no injected amplitude in the tested "
                     "grid reached 80% power and no admissible log-linear extrapolation could be "
                     "made; the power curve is then read as 'underpowered at every amplitude "
                     "tested', not as a finite threshold")}


def interval_coverage(n: float, alpha: float, *, n_sims: int,
                      rng: np.random.Generator) -> dict:
    """Simultaneous coverage of the Bonferroni-Wilson per-digit intervals."""
    cover = []
    for _ in range(n_sims):
        d1 = rng.choice(np.arange(1, 10), size=int(n), p=I.BENFORD)
        p = np.bincount(d1, minlength=10)[1:10] / float(n)
        cover.append(I.simultaneous_coverage(p, n, alpha))
    cov = float(np.mean(cover))
    return {"familywise_coverage": cov, "nominal": 1 - alpha, "n_sims": n_sims,
            "n": n, "passed": bool(cov >= 1 - alpha - 0.01)}
