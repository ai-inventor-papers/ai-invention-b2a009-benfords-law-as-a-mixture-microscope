"""Stage 6-7 - the four rival mechanisms, their statistics, their matched
nulls, the two scoring levels, the ranking and the pre-registered verdict.

Two scores per candidate, and the direction of each is fixed here BEFORE any
data are read (the pre-registration file records this resolution):

  p_c  detection tail probability.  P(T_rep >= T_obs | candidate c's matched
       DETECTION null).  SMALL p_c means the mechanism's predicted signal is
       present beyond what its matched null produces.

  q_c  adequacy.  A two-sided posterior-predictive tail against the null
       implied by the candidate's OWN quantitative prediction:
           q_low  = P(T_rep <= T_obs),  small when the candidate OVER-predicts
           q_high = P(T_rep >= T_obs),  small when the candidate UNDER-predicts
           q_c    = min(1, 2 * min(q_low, q_high)).
       LARGE q_c (>= 0.05) means the candidate's own prediction is not
       rejected by the data.

The direction is two-sided on purpose.  A strictly one-sided
P(T_rep <= T_obs) can never reject a candidate whose prediction is zero,
so a law that predicts no digit anomaly at all would automatically be
declared 'adequate' against a panel that shows one - the opposite of the
intent.  Both one-sided tails are reported alongside q_c so a reader can
re-derive either reading.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
from loguru import logger
from scipy import stats

import instrument as I
import nulls as N
from config import ALPHA, MARGIN

# ---------------------------------------------------------------------------
# levels
# ---------------------------------------------------------------------------
LEVEL_SPLIT_YEAR = 1987   # S1 = even years <= 1986, S2 = even years >= 1988


def entity_cross_section(sub):
    """One observation per entity: its circular-mean mantissa across the panel.

    WHY THIS LEVEL EXISTS.  The static-mixture mechanism is a statement about
    the distribution of COUNTRIES' sizes: fit a law across entities, periodize
    it, and read off the digit law.  Pooling country-years instead mixes two
    things the mechanism does not claim: the cross-sectional spread across
    countries, and the within-country drift over 74 years, which averages each
    country's mantissa toward the middle and dilutes whatever cross-sectional
    harmonic structure the mixture produced.  A panel with a real mixture
    fingerprint at the entity level would therefore show a SMALLER |c_1|
    when pooled, not a larger one -- the pooled test is the wrong test for
    this mechanism.

    The circular mean is the correct within-entity average for a quantity
    that lives on a circle: a linear mean of mantissas near 0.95 and 0.05
    averages to 0.5, which is nowhere near either value.  The concentration
    |mean exp(2 pi i m)| is reported too, because it says how much of the
    entity's identity survives the average.

    Returns a frame with one row per entity, carrying the entity's
    representative mantissa in ``mantissa`` (so every downstream function that
    works on a level frame works unchanged) plus the raw concentration.
    """
    import pandas as pd

    rows = []
    for code, g in sub.groupby("code", sort=True):
        m = g.mantissa.to_numpy(dtype=float)
        z = np.mean(np.exp(2j * np.pi * m))
        mc = float((np.angle(z) / (2 * np.pi)) % 1.0)
        rows.append({
            "code": code,
            "entity": g.entity.iloc[0] if "entity" in g.columns else code,
            "year": int(g.year.max()),
            "population": float(10.0 ** (np.log10(g.population.to_numpy(dtype=float)).mean())),
            "mantissa": mc,
            "d1": int(min(9, max(1, np.floor(10.0 ** mc + 1e-12)))),
            "concentration": float(abs(z)),
            "n_years": int(len(g)),
            "fold": "even",
        })
    out = pd.DataFrame(rows)
    out.attrs["level_note"] = (
        "one row per entity; 'mantissa' is the circular mean of the entity's even-year "
        "mantissas and 'population' the geometric mean of its even-year populations")
    return out


def make_levels(df) -> dict:
    """The three even-year observation sets.  No odd-year row ever enters.

    The reservation of the odd fold is asserted here, at the only place the
    observation sets are built, so a leak cannot reach a statistic.
    """
    ev = df[df.fold == "even"]
    s1 = ev[ev.year <= LEVEL_SPLIT_YEAR - 1]
    s2 = ev[ev.year >= LEVEL_SPLIT_YEAR + 1]
    odd_years = set(int(v) for v in df.loc[df.fold == "odd", "year"].unique())
    for name, part in (("P", ev), ("S1", s1), ("S2", s2)):
        leaked = set(int(v) for v in part.year.unique()) & odd_years
        if leaked:
            raise AssertionError(f"level {name} contains odd years {sorted(leaked)[:5]}: "
                                 f"the reserved fold must never enter a screen statistic")
    if len(s1) == 0 or len(s2) == 0:
        raise AssertionError(f"split-half levels are empty: S1={len(s1)}, S2={len(s2)}; "
                             f"the panel is too short for the two-level design")
    crossed = {"X": entity_cross_section(ev)}
    return {"P": ev, "S1": s1, "S2": s2, "X": crossed["X"]}


def summarise_observed(sub, n_eff: float | None = None) -> dict:
    """Stage 4.1 - the full observed battery on one observation set."""
    return I.companion_battery_from_values(sub.population.to_numpy(dtype=float), n_eff=n_eff)


# ---------------------------------------------------------------------------
# posterior-predictive p-values from a stored sample
# ---------------------------------------------------------------------------
def tails(samples: np.ndarray, observed: float, higher_is_signal: bool = True) -> dict:
    """One- and two-sided tail probabilities from a replicate sample."""
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return {"p_upper": float("nan"), "p_lower": float("nan"), "q_two_sided": float("nan"),
                "n_reps": 0}
    p_up = float((np.sum(s >= observed) + 1.0) / (s.size + 1.0))
    p_lo = float((np.sum(s <= observed) + 1.0) / (s.size + 1.0))
    return {"p_upper": p_up, "p_lower": p_lo,
            "p_detection": p_up if higher_is_signal else p_lo,
            "q_two_sided": float(min(1.0, 2.0 * min(p_up, p_lo))),
            "n_reps": int(s.size)}


# ---------------------------------------------------------------------------
# candidate A - static mixture harmonic fingerprint
# ---------------------------------------------------------------------------
class _BenfordLaw:
    """Exact Benford dressed as a fitted law, for the i.i.d. null helper.

    ``null_law_iid`` takes anything with a ``sampler``; exact Benford is
    exactly that law, so this avoids a second copy of the sampling code.
    """
    name = "benford"
    family = "benford"
    params: dict = {}

    @staticmethod
    def sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        d1 = rng.choice(np.arange(1, 10), size=size, p=I.BENFORD)
        lo = np.log10(d1)
        hi = np.log10(d1 + 1)
        return 10.0 ** (np.floor(rng.uniform(2, 7, size)) + lo
                        + rng.uniform(0, 1, size) * (hi - lo))


_BENFORD_LAW = _BenfordLaw()


def _benford_law() -> _BenfordLaw:
    return _BENFORD_LAW


def score_candidate_A(sub, best_fit, budget: dict, rng: np.random.Generator,
                      reps: int, want_digits: bool = False,
                      one_row_per_entity: bool = False) -> dict:
    """T_A = |c_1| on the exact mantissas.

    Detection null: exact-Benford draws with the same entity structure (what
    pure sampling noise at this panel's structure would produce).
    Adequacy null: draws from the candidate's own fitted law, same structure.

    ``one_row_per_entity`` selects the cross-sectional level, where every
    entity contributes exactly one row: there the entity-cluster null IS the
    i.i.d. law null, because an entity cannot contribute more than one value.
    Getting this wrong in the other direction (i.i.d. at the pooled level)
    is exactly the error that manufactures significance, so the two cases are
    kept apart explicitly rather than inferred.
    """
    m = sub.mantissa.to_numpy(dtype=float)
    struct = N.make_struct(sub.code.to_numpy(), m)
    t_obs = float(abs(I.c_n(m, 1)))

    if one_row_per_entity:
        det = N.null_law_iid(_benford_law(), struct, rng, reps, want_digits=want_digits)
        adq = N.null_law_iid(best_fit, struct, rng, reps, want_digits=want_digits)
    else:
        det = N.null_benford_cluster(struct, rng, reps, want_digits=want_digits)
        adq = N.null_law_cluster(best_fit, struct, rng, reps, want_digits=want_digits)

    pred_amp = float(budget.get("predicted_amp", budget.get("c1_amp", float("nan"))))
    pred_phase = float(budget.get("predicted_phase", budget.get("c1_phase", float("nan"))))
    obs_phase = float(np.angle(I.c_n(m, 1)))
    phase_gap = float(abs((obs_phase - pred_phase + np.pi) % (2 * np.pi) - np.pi)) if np.isfinite(pred_phase) else float("nan")
    ratio = float(t_obs / pred_amp) if pred_amp and np.isfinite(pred_amp) and pred_amp > 0 else float("inf")

    det_t = tails(det["amp"], t_obs)
    adq_t = tails(adq["amp"], t_obs)
    conds = {
        "p_below_margin": bool(det_t["p_detection"] < MARGIN),
        "q_at_least_margin": bool(adq_t["q_two_sided"] >= MARGIN),
        "amplitude_ratio_in_0.5_2.0": bool(np.isfinite(ratio) and 0.5 <= ratio <= 2.0),
        "phase_gap_below_0.5_rad": bool(np.isfinite(phase_gap) and phase_gap < 0.5),
    }
    # the sharpest diagnostic this design produces: how far the mechanism's
    # OWN fit falls short of the observed amplitude, in orders of magnitude.
    under = (float(np.log10(t_obs / pred_amp))
             if pred_amp and np.isfinite(pred_amp) and pred_amp > 0 else float("inf"))
    return {
        "candidate": "A", "statistic": "|c_1| on exact mantissas",
        "observed": t_obs, "predicted": pred_amp,
        "n_entities": int(struct["n_entities"]), "n_rows": int(struct["n"]),
        "one_row_per_entity": bool(one_row_per_entity),
        "observed_phase": obs_phase, "predicted_phase": pred_phase,
        "phase_gap_rad": phase_gap, "amplitude_ratio_obs_over_pred": ratio,
        "orders_of_magnitude_underpredicted": under,
        "underprediction_statement": (
            f"the BIC-selected law's own fit predicts |c1| = {pred_amp:.3e} while the panel shows "
            f"{t_obs:.4f}: the static-mixture mechanism under-predicts the observed harmonic "
            f"amplitude by {under:.2f} orders of magnitude, so it cannot be the explanation"
            if np.isfinite(under) and under > 1.0 else
            f"the BIC-selected law predicts |c1| = {pred_amp:.3e} against an observed {t_obs:.4f} "
            f"({under:.2f} orders of magnitude), within the amplitude-ratio window"),
        "testable": True,
        "detection_null": det_t, "adequacy_null": adq_t,
        "p": det_t["p_detection"], "q": adq_t["q_two_sided"],
        "support_conditions": conds,
        "support_conditions_met": bool(all(conds.values())),
        "detection_envelope": N.summarise_envelope(det["amp"], t_obs),
        "adequacy_envelope": N.summarise_envelope(adq["amp"], t_obs),
        "detection_sign_probabilities": N.sign_probabilities(det["digit_vectors"]),
        "adequacy_sign_probabilities": N.sign_probabilities(adq["digit_vectors"]),
        "falsifier_triggered": bool(det_t["p_detection"] >= MARGIN),
        "falsifier_statement": ("|c_1| lies inside the Benford-only entity-cluster envelope, "
                                "i.e. the residuals are consistent with pure sampling noise"
                                if det_t["p_detection"] >= MARGIN else
                                "|c_1| exceeds the Benford-only entity-cluster envelope"),
    }


# ---------------------------------------------------------------------------
# candidate B - temporal desynchronization
# ---------------------------------------------------------------------------
def per_year_c1(sub) -> dict:
    """Stage 4.2 - per-year c_1 and the per-year fitted lognormal sigma."""
    years, amps, phases, sigmas, ns = [], [], [], [], []
    for yr, grp in sub.groupby("year"):
        x = grp.population.to_numpy(dtype=float)
        m = I.mantissa(x)
        c = I.c_n(m, 1)
        years.append(int(yr)); amps.append(float(abs(c))); phases.append(float(np.angle(c)))
        sigmas.append(float(np.std(np.log10(x), ddof=0)))
        ns.append(int(x.size))
    return {"years": years, "amp": amps, "phase": phases,
            "lognormal_sigma": sigmas, "n": ns}


def predicted_per_year_amp(py: dict) -> np.ndarray:
    """|c_1| the fitted-lognormal mechanism predicts in each year.

    For a single lognormal on log10 scale, |c_1| = exp(-2 pi^2 sigma_t^2), so
    the year-to-year variation of the fitted sigma_t is the ONLY channel by
    which this mechanism can produce a per-year amplitude pattern.
    """
    s = np.asarray(py["lognormal_sigma"], dtype=float)
    return np.exp(-2.0 * np.pi ** 2 * s * s)


def score_candidate_B(sub, py: dict, rng: np.random.Generator, reps: int) -> dict:
    """T_B1 = correlation of observed and predicted per-year |c_1|.

    The null is the exact circular year-shift distribution: shifting the
    predicted series preserves both series' autocorrelation and destroys only
    the year-specific association.
    """
    obs = np.asarray(py["amp"], dtype=float)
    pred = predicted_per_year_amp(py)
    years = np.asarray(py["years"])
    sig = np.asarray(py["lognormal_sigma"], dtype=float)

    # a degenerate predicted series carries no information: report 'not
    # testable' rather than a correlation of numerical dust.
    #
    # The test is on the RELATIVE spread, not the absolute one.  For a single
    # lognormal |c_1(t)| = exp(-2 pi^2 sigma_t^2) is about 1e-13 for any
    # sigma_t near 1.2, so the absolute ptp is dust while the RELATIVE spread
    # is order 1 and perfectly meaningful.  The first version compared the
    # absolute ptp against the mean and therefore declared the mechanism
    # untestable on a series that carries a clean 0.71 correlation.
    rel_spread = (float(np.ptp(pred) / max(float(np.mean(pred)), 1e-300))
                  if pred.size and np.mean(pred) > 0 else 0.0)
    testable = bool(pred.size > 2 and rel_spread > 1e-6 and np.std(pred) > 0)
    r_obs = float(np.corrcoef(obs, pred)[0, 1]) if testable else float("nan")

    shift = (N.shift_year_null(obs, pred, extra_reps=max(reps, 200), rng=rng)
             if testable else {"shift_statistics": [], "n_shifts": 0,
                               "surrogate_statistics": [], "min_attainable_p": float("nan")})
    if testable:
        shifts = np.asarray(shift["shift_statistics"], dtype=float)
        p_b1_shift = float((np.sum(shifts >= r_obs) + 1.0) / (shifts.size + 1.0))
        sur = np.asarray(shift.get("surrogate_statistics", []), dtype=float)
        # the surrogate p is the primary when available: the discrete shift
        # set floors at 1/T, which on a 37-year fold is already 0.027 and
        # cannot fall below the 0.05 margin by much whatever the data say.
        p_b1 = (float((np.sum(sur >= r_obs) + 1.0) / (sur.size + 1.0))
                if sur.size >= 100 else p_b1_shift)
    else:
        p_b1 = float("nan")
        p_b1_shift = float("nan")

    # T_B2: epoch contrast in the observed per-year amplitude.
    early = obs[years <= 1989]
    late = obs[years >= 1990]
    epochs_available = bool(early.size >= 2 and late.size >= 2)
    t_b2 = float(np.log(np.mean(late) / np.mean(early))) if epochs_available else float("nan")

    # null for T_B2: resample entities within each epoch.  A split-half level
    # that lies entirely on one side of the epoch boundary has no contrast to
    # test, and that is reported rather than crashed on.
    if epochs_available:
        ev_e = sub[sub.year <= 1989]
        ev_l = sub[sub.year >= 1990]
        e_struct = N.make_struct(ev_e.code.to_numpy(), ev_e.mantissa.to_numpy(dtype=float))
        l_struct = N.make_struct(ev_l.code.to_numpy(), ev_l.mantissa.to_numpy(dtype=float))
        half = max(reps // 2, 200)
        nb_e = N.null_entity_resample(e_struct, rng, half)
        nb_l = N.null_entity_resample(l_struct, rng, half)
        n_min = min(nb_e["amp"].size, nb_l["amp"].size)
        t_b2_null = np.log(nb_l["amp"][:n_min] / nb_e["amp"][:n_min])
        b2 = tails(t_b2_null, t_b2)
    else:
        t_b2_null = np.array([])
        b2 = {"p_upper": float("nan"), "p_lower": float("nan"),
              "q_two_sided": float("nan"), "n_reps": 0}

    # the epoch contrast the mechanism PREDICTS, from fitted sigma_t
    sig_early = sig[years <= 1989]
    sig_late = sig[years >= 1990]
    pred_b2 = (float(np.log(np.mean(np.exp(-2 * np.pi ** 2 * sig_late ** 2))
                           / np.mean(np.exp(-2 * np.pi ** 2 * sig_early ** 2))))
               if epochs_available else float("nan"))
    epoch_trend_present = bool(epochs_available and np.isfinite(pred_b2) and abs(pred_b2) > 1e-6)
    sign_agree = bool(epochs_available and np.isfinite(pred_b2) and np.isfinite(t_b2)
                      and np.sign(pred_b2) == np.sign(t_b2))

    # AMPLITUDE-SCALE CHECK.  A correlation test can pass while the mechanism's
    # predicted amplitude is wrong by orders of magnitude, and that asymmetry
    # is the whole story for this candidate: exp(-2 pi^2 sigma_t^2) is about
    # 1e-13 for sigma_t near 1.2, so the fitted-log-normal mechanism predicts
    # essentially ZERO harmonic amplitude in every year while the panel shows
    # ~0.07.  It may therefore explain the year-to-year SHAPE of |c1(t)| and
    # still be incapable of producing |c1| at all.  Both readings are
    # reported; only the conjunction counts as support.
    obs_level = float(np.mean(obs)) if obs.size else float("nan")
    pred_level = float(np.mean(pred)) if pred.size else float("nan")
    level_ratio = (float(pred_level / obs_level)
                   if obs_level and np.isfinite(obs_level) and obs_level > 0 else float("nan"))
    amp_scale_ok = bool(np.isfinite(level_ratio) and 0.5 <= level_ratio <= 2.0)

    conds = {
        "testable": testable and epoch_trend_present,
        "p_b1_below_margin": bool(np.isfinite(p_b1) and p_b1 < MARGIN and r_obs > 0),
        "sign_agreement_T_B2": sign_agree,
        "predicted_amplitude_matches_observed_scale": amp_scale_ok,
    }
    if not epochs_available:
        status_note = ("not testable at this level: the level lies entirely on one side of the "
                       "1989/1990 epoch boundary, so there is no epoch contrast to test")
    elif not testable:
        status_note = ("not testable: the fitted per-year lognormal sigma_t is constant to "
                       "numerical precision, so this mechanism predicts no per-year amplitude pattern")
    elif not epoch_trend_present:
        status_note = ("not testable: the fitted sigma_t shows no epoch trend, so T_B2 has no "
                       "pre-registered sign to agree with")
    else:
        status_note = "testable"
    n_shifts = int(shift.get("n_shifts", 0))
    return {
        "candidate": "B", "statistic": "T_B1 = corr(|c1(t)|_obs, |c1(t)|_pred); T_B2 = epoch log-ratio",
        "observed": r_obs, "predicted": pred_b2,
        "t_b1_observed_r": r_obs, "t_b1_predicted_relative_spread": rel_spread,
        "t_b1_p_shift_null_exact": p_b1_shift, "t_b1_p_surrogate": p_b1,
        "t_b1_p_null_used": ("phase-randomised surrogate" if p_b1 != p_b1_shift
                             else "exact circular shift"),
        "t_b1_n_shifts": n_shifts,
        "t_b1_min_attainable_p": float(shift.get("min_attainable_p", float("nan"))),
        "t_b1_p_is_discrete": True,
        "t_b2_observed": t_b2, "t_b2_predicted": pred_b2,
        "t_b2_null_envelope": N.summarise_envelope(t_b2_null, t_b2),
        "p": p_b1, "q": b2["q_two_sided"],
        "testable": bool(testable),
        "epoch_testable": bool(epochs_available and epoch_trend_present),
        "full_support_testable_at_this_level": bool(testable and epoch_trend_present),
        "predicted_amp_scale": float(np.max(pred)) if testable else None,
        "observed_amp_mean": float(np.mean(obs)),
        "predicted_amp_mean": pred_level,
        "predicted_over_observed_amplitude": level_ratio,
        "amplitude_scale_consistent": amp_scale_ok,
        "amplitude_scale_statement": (
            f"the mechanism's per-year prediction has mean |c1| = {pred_level:.3e} against an "
            f"observed mean of {obs_level:.4f} (ratio {level_ratio:.2e}): it can reproduce the "
            f"year-to-year SHAPE of |c1(t)| but its own fit cannot produce the observed amplitude, "
            f"so the correlation is a shape-only agreement"
            if not amp_scale_ok else
            f"predicted and observed per-year amplitudes agree in scale (ratio {level_ratio:.3f})"),
        "support_conditions": conds, "support_conditions_met": bool(all(conds.values())),
        "status_note": status_note,
        "falsifier_triggered": bool(not (testable and epoch_trend_present)
                                    or (np.isfinite(b2["q_two_sided"]) and b2["q_two_sided"] >= MARGIN)),
        "falsifier_statement": ("the epoch contrast lies inside the within-epoch entity envelope, "
                                "or the mechanism has no epoch prediction to test"
                                if not (np.isfinite(b2["q_two_sided"]) and b2["q_two_sided"] < MARGIN)
                                else "the epoch contrast lies outside the within-epoch entity envelope"),
        "per_year": {"years": py["years"], "amp": py["amp"], "sigma": py["lognormal_sigma"],
                     "pred_amp": [float(v) for v in pred]},
    }


# ---------------------------------------------------------------------------
# candidate C - rounding artifact
# ---------------------------------------------------------------------------
def rounding_audit(sub) -> dict:
    """C0 - the direct evidence on the printed precision of the source values."""
    sig = sub.sig_digits.to_numpy(dtype=int)
    x = sub.population.to_numpy(dtype=float)
    n = x.size
    divisible = {}
    for k in range(1, 7):
        frac = float(np.mean(np.mod(x, 10.0 ** k) == 0))
        divisible[f"divisible_by_10^{k}"] = {"fraction": frac, "smooth_law_expectation": 10.0 ** (-k),
                                             "excess": frac - 10.0 ** (-k)}
    return {
        "median_significant_digits": float(np.median(sig)),
        "mean_significant_digits": float(np.mean(sig)),
        "sig_digit_distribution": {str(k): int(v) for k, v in
                                   zip(*np.unique(sig, return_counts=True))},
        "fraction_non_integer": float(np.mean(~sub.is_integer_value.to_numpy(dtype=bool))),
        "all_integer_values": bool(np.all(sub.is_integer_value.to_numpy(dtype=bool))),
        "divisibility": divisible,
        "grain_present": bool(np.median(sig) <= 6 or any(v["excess"] > 0.05
                                                         for v in divisible.values())),
        "n": int(n),
    }


def grain_imprint_clustered(fit, struct, rng: np.random.Generator, g: int,
                            reps: int) -> tuple[np.ndarray, float]:
    """Mean grain imprint vector w^(g) under the fitted law, entity-clustered.

    Returns ``(w, sd_of_imprint_norm)``.  The imprint is the only feature of
    rounding that a smooth law cannot produce, and it is estimated under the
    SAME entity-clustered null used for inference: estimating it i.i.d. while
    the null is clustered would put the projection's scale off by the design
    factor.

    STREAMED and VECTORISED.  The first version materialised a (reps x n)
    mantissa array -- 175 MB at the pooled n -- and then looped over rows in
    Python.  This version draws in CHUNK-sized blocks and reduces each block
    with one vectorised digit-count, so memory stays bounded and the cost is
    proportional to reps x n flops rather than reps x n Python iterations.
    """
    wsum = np.zeros(9)
    norms = []
    done = 0
    for s0, s1, mm in N.iter_cluster_blocks(fit, struct, rng, reps):
        p_s = N._digit_vectors_block(mm)
        xr = 10.0 ** N.round_significant(10.0 ** mm, g)
        p_r = N._digit_vectors_block(np.mod(np.log10(np.clip(xr, 1e-300, None)), 1.0))
        imp = p_r - p_s
        wsum += imp.sum(axis=0)
        norms.append(np.linalg.norm(imp, axis=1))
        done += s1 - s0
        del mm, p_s, p_r, imp, xr
    w = wsum / max(done, 1)
    alln = np.concatenate(norms) if norms else np.array([])
    return w, float(np.std(alln)) if alln.size else 0.0


def score_candidate_C(sub, best_fit, rng: np.random.Generator, reps: int,
                      audit: dict, imprint_reps: int | None = None) -> dict:
    """T_C = projection of the observed digit residuals onto the grain imprint.

    UNITS: both the residual vector r_d and the imprint vector w_d are on the
    probability scale (p_hat minus Benford).  Reporting multiplies by 100
    where a percentage point is wanted.

    TWO CALIBRATION REQUIREMENTS, both learned from a failed V1 check:
      * the imprint and the null must come from INDEPENDENT draws, otherwise
        the projection is compared against the very draws that generated its
        own direction and the p-value is degenerate;
      * the null must be entity-clustered, like every other null here, or the
        sampling variance is understated by the design factor -- which is
        exactly what made the first version reject 82.5% of exact-Benford
        panels at the 5% level.

    POWER LIMITATION, stated rather than discovered by the reader: rounding to
    g significant figures can only move a leading digit when a value lies
    within about 10^-(g-1) of a digit-bin boundary, so the imprint norm falls
    geometrically in g and at g >= 7 it is exactly zero on this panel.
    """
    x = sub.population.to_numpy(dtype=float)
    m = sub.mantissa.to_numpy(dtype=float)
    struct = N.make_struct(sub.code.to_numpy(), m)
    ireps = imprint_reps or max(reps // 4, 200)

    r = I.digit_probabilities_from_mantissa(m) - I.BENFORD
    r = r - r.mean()                      # residuals are constrained to sum to zero

    imprints, imprint_norms = {}, {}
    for g in (3, 4, 5, 6, 7, 8):
        w, _sd = grain_imprint_clustered(best_fit, struct, rng, g, ireps)
        w = w - w.mean()
        imprints[g] = w
        imprint_norms[g] = float(np.linalg.norm(w))

    g_star = int(np.clip(round(audit["median_significant_digits"]), 3, 8))
    w = imprints[g_star]
    nw = float(np.linalg.norm(w))
    out = {
        "candidate": "C",
        "statistic": "projection of digit residuals onto the grain imprint vector",
        "grain_used": g_star,
        "audit": audit,
        "imprint_norms": {str(k): v for k, v in imprint_norms.items()},
        "imprints": {str(k): [float(v) for v in vv] for k, vv in imprints.items()},
        "observed_residual_norm": float(np.linalg.norm(r)),
        "observed_residual_pp": [float(100 * v) for v in r],
    }
    if nw < 1e-9:
        out.update({
            "observed": float("nan"), "predicted": 0.0, "p": 1.0, "q": float("nan"),
            "testable": False, "null_envelope": {},
            "status_note": (f"rounding to {g_star} significant figures has an imprint norm of "
                            f"{nw:.3e} on the leading-digit probabilities, against an observed "
                            f"residual norm of {np.linalg.norm(r):.3e}: this channel is "
                            "arithmetically incapable of producing the observed digit pattern"),
            "support_conditions": {"grain_present": bool(audit["grain_present"]),
                                   "imprint_nonzero": False, "p_below_margin": False},
            "support_conditions_met": False,
            "falsifier_triggered": True,
            "falsifier_statement": ("the predicted imprint vanishes at the precision the source "
                                    "actually prints, so the rounding channel is excluded by "
                                    "construction rather than by a test"),
        })
        return out

    # the null: INDEPENDENT entity-clustered draws from the same fitted law,
    # projected onto the same fixed direction w/||w||.  Streamed, so the
    # (reps x n) mantissa array is never materialised.
    u = w / nw
    nreps = max(reps, 400)
    proj = np.empty(nreps)
    for s0, s1, mm in N.iter_cluster_blocks(best_fit, struct, rng, nreps):
        ri = N._digit_vectors_block(mm) - I.BENFORD
        ri = ri - ri.mean(axis=1, keepdims=True)
        proj[s0:s1] = ri @ u
        del mm, ri
    t_obs = float(np.dot(r, u))
    t = tails(proj, t_obs)
    conds = {"grain_present": bool(audit["grain_present"]),
             "p_below_margin": bool(t["p_detection"] < MARGIN)}
    out.update({
        "observed": t_obs, "predicted": 0.0, "testable": True,
        "detection_null": t, "p": t["p_detection"], "q": t["q_two_sided"],
        "null_envelope": N.summarise_envelope(proj, t_obs),
        "support_conditions": conds, "support_conditions_met": bool(all(conds.values())),
        "falsifier_triggered": bool(not all(conds.values())),
        "falsifier_statement": ("T_C inside the no-grain envelope, or the audit shows the source "
                                "prints more digits than the grain hypothesis needs"
                                if not all(conds.values()) else
                                "T_C lies outside the no-grain envelope and the audit shows real grain"),
    })
    return out


# ---------------------------------------------------------------------------
# candidate D - phase-only rotation law
# ---------------------------------------------------------------------------
def rotation_family_mad(sub, grid=None) -> dict:
    """MAD of the first-digit law under every admissible unit rescaling.

    Multiplying every value by c does not change the harmonic amplitude
    (|c_1(c x)| = |c_1(x)| exactly), but it DOES rotate the phase and it
    changes the leading-digit frequencies, hence the MAD.  If the conformity
    verdict swings with c while the harmonic amplitude cannot, then the
    verdict is a property of the unit, not of the data.
    """
    x = sub.population.to_numpy(dtype=float)
    m = sub.mantissa.to_numpy(dtype=float)
    grid = np.arange(1.0, 10.0, 0.1) if grid is None else grid
    mads, kui, amps = [], [], []
    for c in grid:
        mm = np.mod(m + math.log10(float(c)), 1.0)
        p = I.digit_probabilities_from_mantissa(mm)
        mads.append(float(np.mean(np.abs(p - I.BENFORD))))
        kui.append(I.kuiper_v(mm)["V"])
        amps.append(float(abs(I.c_n(mm, 1))))
    mads = np.array(mads); kui = np.array(kui); amps = np.array(amps)
    return {"grid": [float(v) for v in grid], "mad": [float(v) for v in mads],
            "kuiper": [float(v) for v in kui], "amp": [float(v) for v in amps],
            "mad_at_1": float(mads[0]), "mad_min": float(mads.min()), "mad_max": float(mads.max()),
            "mad_range": float(mads.max() - mads.min()),
            "kuiper_range": float(kui.max() - kui.min()),
            "amp_range": float(amps.max() - amps.min()),
            "amp_max_abs_dev_from_amp_at_1": float(np.max(np.abs(amps - amps[0]))),
            "best_case": float(grid[int(np.argmin(mads))]),
            "worst_case": float(grid[int(np.argmax(mads))])}


def score_candidate_D(sub, m: np.ndarray, ident_checks: dict, reps: int,
                      rng: np.random.Generator | None = None,
                      benford_ref: np.ndarray | None = None) -> dict:
    """T_D1 = digit-resolution validity gate; T_D2 = rotation-family range.

    T_D2 is a genuine statistical claim and gets a matched null: the same MAD
    range computed on panels that ARE exact Benford with the same entity
    structure.  Under exact Benford every rescaling is still exact Benford, so
    that null range is pure sampling noise; the claim is that the observed
    range is much larger.  Putting D on a p-value scale keeps the four-way
    ranking scale-free rather than exempting one candidate from it.
    """
    p_hat = I.digit_probabilities_from_mantissa(m)
    c1_exact = I.c_n(m, 1)
    c1_digit = I.c_n_digit_vector(p_hat, 1)
    c1_digit_mid = I.c_n_digit_midpoint_only(p_hat, 1)
    t_d1 = float(abs(c1_digit - c1_exact) / max(abs(c1_exact), 1e-12))
    t_d1_mid = float(abs(c1_digit_mid - c1_exact) / max(abs(c1_exact), 1e-12))

    rot = rotation_family_mad(sub)
    d0_pass = bool(ident_checks.get("rotation", {}).get("passed", False)
                   and ident_checks.get("mirror", {}).get("passed", False))

    # matched null for the MAD range: exact-Benford panels, same entity structure
    null_ranges = np.array([])
    if rng is not None and benford_ref is not None and benford_ref.size:
        null_ranges = _benford_mad_range_null(benford_ref, reps, rng)
    p_d2 = (float((np.sum(null_ranges >= rot["mad_range"]) + 1.0) / (null_ranges.size + 1.0))
            if null_ranges.size else float("nan"))
    q_d2 = (float(min(1.0, 2.0 * min((np.sum(null_ranges >= rot["mad_range"]) + 1.0) / (null_ranges.size + 1.0),
                                     (np.sum(null_ranges <= rot["mad_range"]) + 1.0) / (null_ranges.size + 1.0))))
            if null_ranges.size else float("nan"))

    conds = {
        "instrument_identity_passed": d0_pass,
        "digit_gate_passed": bool(t_d1 < 0.5),
        "mad_range_at_least_half_mad_at_1": bool(rot["mad_range"] >= 0.5 * rot["mad_at_1"]),
        "amplitude_invariant": bool(rot["amp_max_abs_dev_from_amp_at_1"] < 1e-10),
        "mad_range_exceeds_benford_sampling_range": bool(null_ranges.size and p_d2 < MARGIN),
    }
    return {
        "candidate": "D", "statistic": "T_D1 = digit-resolution gate; T_D2 = rotation-family MAD range",
        "observed": rot["mad_range"], "predicted": 0.0,
        "t_d1_digit_transfer_error": t_d1, "t_d1_midpoint_only_error": t_d1_mid,
        "c1_exact": {"re": float(c1_exact.real), "im": float(c1_exact.imag), "amp": float(abs(c1_exact))},
        "c1_digit_resolution": {"re": float(c1_digit.real), "im": float(c1_digit.imag),
                                "amp": float(abs(c1_digit))},
        "rotation_family": rot,
        "benford_sampling_null_mad_range": N.summarise_envelope(null_ranges, rot["mad_range"]),
        "p": p_d2, "q": q_d2, "statistic_for_p": "MAD range over the rotation family",
        "support_conditions": conds, "support_conditions_met": bool(all(conds.values())),
        "verdict_note": ("the digit-level reading is valid" if t_d1 < 0.5 else
                         "the digit-level reading is void: exact mantissa c_1 is primary"),
        "falsifier_triggered": bool(t_d1 >= 0.5 or rot["mad_range"] < 0.5 * rot["mad_at_1"]),
        "falsifier_statement": ("the rotation family leaves MAD essentially unchanged, so the "
                                "unit-artifact claim is unsupported"
                                if rot["mad_range"] < 0.5 * rot["mad_at_1"] else
                                "MAD ranges widely over the rotation family while |c_1| is exactly "
                                "invariant, so a conformity verdict is a property of the unit"),
    }


def _benford_mad_range_null(m: np.ndarray, reps: int, rng: np.random.Generator) -> np.ndarray:
    """MAD range over the rotation family on exact-Benford resamples.

    Each replicate draws a fresh Benford mantissa per entity (so the entity
    clustering is preserved) and recomputes the same range statistic.
    """
    m = np.asarray(m, dtype=float)
    n = m.size
    grid = np.arange(1.0, 10.0, 0.1)
    out = np.empty(reps)
    base = N.draw_benford(rng, n)
    for r in range(reps):
        mm = base if r == 0 else N.draw_benford(rng, n)
        p0 = I.digit_probabilities_from_mantissa(mm)
        mads = []
        for c in grid:
            pc = I.digit_probabilities_from_mantissa(np.mod(mm + math.log10(float(c)), 1.0))
            mads.append(float(np.mean(np.abs(pc - I.BENFORD))))
        out[r] = max(mads) - min(mads)
    return out


# ---------------------------------------------------------------------------
# ranking, multiplicity, verdict
# ---------------------------------------------------------------------------
def holm_adjust(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjustment over the candidate p-values."""
    items = [(k, v) for k, v in pvals.items() if v is not None and np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, v) in enumerate(items):
        running = max(running, min(1.0, (m - i) * v))
        adj[k] = running
    return adj


def rank_and_verdict(per_level: dict, budget: dict, viability: dict) -> dict:
    """Stage 7 - the pre-registered survivor rule, ranking and status labels."""
    p_P = {c: per_level["P"][c].get("p") for c in "ABCD"}
    p_S1 = {c: per_level["S1"][c].get("p") for c in "ABCD"}
    p_S2 = {c: per_level["S2"][c].get("p") for c in "ABCD"}
    q_P = {c: per_level["P"][c].get("q") for c in "ABCD"}

    cands = [c for c in "ABCD"
             if per_level["P"][c].get("p") is not None and np.isfinite(per_level["P"][c]["p"])
             and per_level["P"][c].get("testable", True)]
    order = sorted(cands, key=lambda c: per_level["P"][c]["p"])
    holm = holm_adjust({c: p_P[c] for c in order})
    margins = []
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        margins.append({"first": a, "second": b,
                        "log10_p_margin": float(np.log10(p_P[b]) - np.log10(p_P[a]))})

    status = {}
    for c in "ABCD":
        cell = per_level["P"][c]
        testable = bool(cell.get("testable", True))
        passes_P = testable and p_P[c] is not None and np.isfinite(p_P[c]) and p_P[c] < ALPHA
        passes_S = (testable and all(x is not None and np.isfinite(x) and x < ALPHA
                                     for x in (p_S1[c], p_S2[c])))
        q_ok = q_P[c] is None or not np.isfinite(q_P[c]) or q_P[c] >= MARGIN
        if passes_P and passes_S and q_ok and cell.get("support_conditions_met"):
            status[c] = "survivor"
        elif not testable:
            status[c] = "falsified by its own fit (not testable)"
        elif passes_P or passes_S:
            status[c] = "deferred"
        else:
            status[c] = "eliminated"

    survivors = [c for c in order if status[c] == "survivor"]
    winner = survivors[0] if survivors else None
    # underpowered adjustment (pre-registered in V3): a candidate whose only
    # obstacle is power is reported UNDERPOWERED, not eliminated.
    md = viability.get("md_amp_80pct_power", float("inf"))
    if winner is None and viability.get("max_component_amp", 0.0) < md:
        status["A"] = "underpowered" if status["A"] == "eliminated" else status["A"]

    # the cross-sectional level, for candidate A only: the level at which a
    # distribution-of-countries mechanism actually makes its prediction
    x_cell = per_level.get("X", {}).get("A")
    cross = None
    if x_cell is not None:
        cross = {
            "level": "X (one row per entity: circular-mean mantissa across even years)",
            "n_entities": int(x_cell.get("n_entities", 0)) or None,
            "observed_amp": x_cell["observed"],
            "predicted_amp": x_cell["predicted"],
            "p_value": x_cell["p"], "q_value": x_cell["q"],
            "amplitude_ratio_obs_over_pred": x_cell.get("amplitude_ratio_obs_over_pred"),
            "support_conditions_met": x_cell["support_conditions_met"],
            "reading": (
                "the mixture mechanism predicts a cross-sectional |c1| at the pooled fit's own "
                "value; the entity cross-section is where that prediction is sharpest, because "
                "pooling country-years averages away the between-country spread the mechanism is "
                "about"),
        }

    return {
        "levels": {"P": "pooled even fold", "S1": f"even years <= {LEVEL_SPLIT_YEAR - 1}",
                   "S2": f"even years >= {LEVEL_SPLIT_YEAR + 1}",
                   "X": "entity cross-section (one row per country)"},
        "p_values": {"P": p_P, "S1": p_S1, "S2": p_S2},
        "q_values_P": q_P,
        "ranking_at_P": order,
        "margins": margins,
        "holm_adjusted_at_P": holm,
        "status": status,
        "winner": winner,
        "cross_section_A": cross,
        "survivor_rule": (
            "c survives iff p_c < 0.05 at level P AND p_c < 0.05 in both S1 and S2 AND "
            "q_c >= 0.05 where defined AND c has the smallest p_c among those meeting the above"),
        "verdict": _verdict_text(winner, status, per_level, viability),
    }


def _verdict_text(winner, status, per_level, viability) -> str:
    if winner:
        return (f"Candidate {winner} ({_cname(winner)}) survives all pre-registered levels: it has the "
                f"smallest matched-null p-value at the pooled even fold and clears the margin in both "
                f"split halves, with its own prediction not rejected.")
    elim = [c for c in "ABCD" if status[c] == "eliminated"]
    under = [c for c in "ABCD" if status[c] == "underpowered"]
    if len(elim) == 4 or (len(elim) + len(under) == 4 and not winner):
        obs = per_level["P"]["A"]["observed"]
        n_eff = viability.get("n_eff", float("nan"))
        md80 = viability.get("md_amp_80pct_power", float("nan"))
        max_comp = viability.get("max_component_amp", float("nan"))
        return (
            "No candidate survives. The panel's distance from exact Benford, measured in harmonic "
            f"coordinates, is |c1| = {obs:.4f}. After the entity-cluster correction the effective "
            f"sample size is n_eff = {n_eff:.0f} and the 80%-power minimum detectable amplitude is "
            f"{md80:.4f}. The fitted size law's largest single-component contribution is "
            f"{max_comp:.4f}, "
            + ("below that floor, so even the mechanism that best fits the data could not be "
               "detected at this panel's effective sample size: the honest statement is a bound, "
               "not an elimination."
               if max_comp < md80 else
               "above that floor, so the mechanism is detectable in principle and its failure is "
               "a genuine negative rather than a power limitation.")
            + f" Verdict: {status.get('A', 'eliminated')} for the static-mixture mechanism.")
    return ("Candidates " + ", ".join(c for c in "ABCD" if status[c] != "eliminated")
            + " clear at least one level but none clears all pre-registered levels; the result is "
              "deferred rather than a survivor.")


def _cname(c: str) -> str:
    from config import CANDIDATES
    return CANDIDATES[c]


# ---------------------------------------------------------------------------
# pre-registration
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_pre_registration(path: Path, *, panel_path: Path, screen_path: Path,
                           extra_text: str = "") -> dict:
    """Write the pre-registration BEFORE any statistic on the real panel."""
    import datetime as _dt
    now = _dt.datetime.now().astimezone().isoformat()
    panel_sha = sha256_file(panel_path)
    screen_sha = sha256_file(screen_path)
    text = f"""# Pre-registration

Written at wall-clock {now}, BEFORE any statistic was computed on the real panel.

## Hashes

- data/panel.csv  sha256 = `{panel_sha}`
- src/screen.py   sha256 = `{screen_sha}`

## The four candidate mechanisms (verbatim, fixed before seeing data)

A. **Static mixture harmonic fingerprint.** The country population distribution is a
   mixture of lognormals of which at least one component is narrow enough to survive
   periodization in the mantissa domain. Its fingerprint is a non-zero first complex
   Fourier coefficient c_1 at the phase and amplitude its own fit predicts.

B. **Temporal desynchronization.** Different countries' size distributions drift
   out of phase over the window, so per-year amplitude |c_1(t)| tracks per-year
   distribution width sigma_t and shows an epoch contrast.

C. **Rounding artifact.** The published figures are printed at a finite number of
   significant digits, and rounding leaves a detectable imprint on the leading-digit
   frequencies beyond what a smooth law produces.

D. **Phase-only rotation law.** The apparent departure from Benford is a property of
   the chosen unit, not of the data: rescaling every value by c rotates the mantissa
   distribution and so changes the leading-digit frequencies (hence the MAD verdict),
   while leaving the harmonic amplitude |c_1| exactly invariant.

## The two scoring levels

- **P** - pooled even fold (every even year in the panel).
- **S** - split-half stability: S1 (even years <= 1986) AND S2 (even years >= 1988)
  must both clear the margin.

The odd-year fold is RESERVED for the evaluation artifact and enters no statistic in
this artifact. Consequently 'pooled' cannot mean 1950-2025; it means the pooled even
fold. Full-panel numbers are produced only after the verdict is written, and are
labelled post-selection descriptive.

## The two scores and their direction

- `p_c` = P(T_rep >= T_obs | candidate c's matched detection null). SMALL favours
  detection of the mechanism's signal.
- `q_c` = two-sided posterior-predictive adequacy against the null implied by the
  candidate's OWN quantitative prediction: q_c = min(1, 2*min(P(T_rep<=T_obs),
  P(T_rep>=T_obs))). LARGE (>= 0.05) means the candidate's own prediction is not
  rejected.

**Resolution of an ambiguity in the plan.** The direction's phrase "posterior
probability that the observed statistic lies outside that candidate's implied
envelope" does not fix whether small or large favours a candidate. The falsifiers
attached to each candidate in the direction DO fix it, and this artifact resolves it
as above before any data are read. The adequacy score is deliberately two-sided: a
strictly one-sided P(T_rep <= T_obs) can never reject a candidate whose prediction is
zero, which would let a law predicting no anomaly be declared adequate against a panel
that shows one.

## Survivor rule (pre-registered)

c survives iff p_c < 0.05 at level P AND p_c < 0.05 in both S1 and S2 AND q_c >= 0.05
where defined AND c has the smallest p_c among those meeting the above.

Status labels: survivor / underpowered / deferred (passes one level) / eliminated
(fails both).

## Constants

SEED = 20261119; ALPHA = 0.05; MARGIN = 0.05; REPS_POOLED = 10000; REPS_HALF = 4000;
KMAX = 12; levels split at 1987.

## Pre-registered power consequence

If the power curve's minimum detectable amplitude at 80% power exceeds the largest
component contribution |v_j| in the real harmonic budget, candidate A is reported as
UNDERPOWERED rather than eliminated - an honest power statement instead of a false
negative.

## Ordered tests, each gated on the previous

identity tests -> interval coverage -> synthetic calibration V1 -> synthetic truth V2
-> power curve V3 -> the real even-fold screen -> post-run audits.
{extra_text}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    logger.info(f"pre-registration written: {path}")
    return {"written_at": now, "panel_sha256": panel_sha, "screen_sha256": screen_sha}
