"""Stage 5 - matched nulls.

Every candidate is scored against a null that reproduces the structure the
candidate itself claims (or claims to be irrelevant).  No statistic is ever
judged against exact Benford alone, because the panel is a census of
countries with strong within-entity dependence: the same country appears in
~37 even years with a nearly unchanged mantissa, so the nominal n of ~7,400
is nowhere near the effective sample size and any i.i.d. envelope would
manufacture spectacular significance.

Memory discipline: a replicate yields ONE complex coefficient (or ONE 9-vector),
so storage is REPS x a few floats.  Never an (REPS x n) array.
"""

from __future__ import annotations

import math

import numpy as np
from loguru import logger

from instrument import BENFORD, mantissa

CHUNK = 500
#: largest number of (replicate x row) mantissa cells allowed at once.  A full
#: REPS_POOLED x 8732 array is 700 MB; this caps the block at ~35 MB.
MAX_MANTISSA_BLOCK = 4_400_000


# ---------------------------------------------------------------------------
# entity structure of the panel
# ---------------------------------------------------------------------------
def entity_structure(codes: np.ndarray, mantissas: np.ndarray) -> dict:
    """Per-entity year counts and per-entity mean complex contribution z_i.

    c_1 = sum_i n_i z_i / sum_i n_i with z_i = mean over entity i's rows of
    exp(2 pi i m).  This is the object the cluster nulls resample.
    """
    codes = np.asarray(codes)
    m = np.asarray(mantissas, dtype=float)
    uniq, inv, counts = np.unique(codes, return_inverse=True, return_counts=True)
    e = np.exp(2j * np.pi * m)
    sums = np.zeros(uniq.size, dtype=complex)
    np.add.at(sums, inv, e)
    z = sums / counts
    return {"codes": uniq, "inverse": inv, "counts": counts,
            "z": z, "n": int(m.size), "n_entities": int(uniq.size),
            "mean_count": float(counts.mean())}


def cluster_contributions(struct: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three scalar series a cluster resample needs: re, im, |z|."""
    return struct["z"].real, struct["z"].imag, np.abs(struct["z"])


def make_struct(codes: np.ndarray, mantissas: np.ndarray,
                calibrate: bool = True) -> dict:
    """The one constructor for an entity structure, with its null calibrated.

    Every cluster null in this artifact is built from a struct, and every one
    of them needs the same within-entity drift step.  Creating the struct in
    one place means a null cannot silently fall back to the plan's fixed 0.02
    step while another uses the calibrated one.
    """
    m = np.asarray(mantissas, dtype=float)
    s = entity_structure(codes, m)
    s["mantissas"] = m
    return attach_walk_sd(s) if calibrate else s


# ---------------------------------------------------------------------------
# design effect (Stage 4.4)
# ---------------------------------------------------------------------------
def design_effect(struct: dict) -> dict:
    """The Stage 4.4 power statement, computed from the observed panel.

    One-way ANOVA decomposition of the entity contributions: for the complex
    contributions e_i = exp(2 pi i m_i) of the N rows grouped into entities,
    the intraclass correlation of the real and imaginary parts is
    rho = between / (between + within) with

        between = (1/(k-1)) * sum_i n_i (z_i - zbar)^2
        within  = (1/(N-k)) * sum_i sum_{j in i} (e_ij - z_i)^2

    (and analogously for the modulus).  The design effect is
    deff = 1 + (mbar - 1) * rho with mbar the mean entity year count, and the
    effective sample size is n_eff = N / deff.
    """
    m = struct.get("mantissas")
    inv, counts = struct["inverse"], struct["counts"]
    if m is None:
        raise ValueError("entity_structure must carry 'mantissas' for design_effect")
    e = np.exp(2j * np.pi * np.asarray(m, dtype=float))
    an_re = _anova_rho(e.real, inv, counts)
    an_im = _anova_rho(e.imag, inv, counts)
    an_amp = _anova_rho(np.abs(e), inv, counts)
    rho_hat = float(np.mean([an_re["rho"], an_im["rho"]]))
    rho_hat = float(np.clip(rho_hat, 0.0, 0.995))
    n = int(struct["n"])
    mbar = float(struct["mean_count"])
    deff = 1.0 + (mbar - 1.0) * rho_hat
    n_eff = n / deff
    noise_floor = math.sqrt(math.pi / (4.0 * n_eff))
    sd = math.sqrt(1.0 / (2.0 * n_eff))
    thr_2sigma = 2.0 * sd
    md_80 = thr_2sigma + 0.84 * sd
    return {
        "n_rows": n, "n_entities": int(counts.size), "mean_years_per_entity": mbar,
        "rho_real": an_re["rho"], "rho_imag": an_im["rho"], "rho_abs": an_amp["rho"],
        "rho_hat": rho_hat, "design_effect": deff, "n_eff": n_eff,
        "noise_floor_E_abs_c1": noise_floor, "sd_abs_c1": sd,
        "md_amp_2sigma": thr_2sigma, "md_amp_80pct_power": md_80,
        "statement": ("Every naive i.i.d. p-value on this panel is to be ignored. "
                      f"Clustering inflates the variance by deff = {deff:.1f} "
                      f"(rho = {rho_hat:.3f}, {mbar:.1f} years per entity), so the effective "
                      f"sample size is {n_eff:.0f} rather than {n}. The Rayleigh noise floor is "
                      f"E|c1| = {noise_floor:.4f}, the 2-sigma detection threshold is "
                      f"{thr_2sigma:.4f}, and the 80%-power minimum detectable amplitude is "
                      f"{md_80:.4f}."),
    }


def _anova_rho(values: np.ndarray, inv: np.ndarray, counts: np.ndarray) -> dict:
    """Intraclass correlation for one scalar series grouped by entity."""
    k = counts.size
    n = values.size
    sums = np.zeros(k)
    np.add.at(sums, inv, values)
    means = sums / counts
    gm = values.mean()
    between = float(np.sum(counts * (means - gm) ** 2) / max(k - 1, 1))
    sq = np.zeros(k)
    np.add.at(sq, inv, (values - means[inv]) ** 2)
    within = float(np.sum(sq) / max(n - k, 1))
    total = between + within
    rho = float(between / total) if total > 0 else 0.0
    return {"between": between, "within": within, "rho": rho}


# ---------------------------------------------------------------------------
# null generators - each yields one replicate statistic at a time
# ---------------------------------------------------------------------------
def _mantissa_from_digit(d1: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform-within-bin mantissas consistent with a leading-digit draw."""
    lo = np.log10(d1)
    hi = np.log10(d1 + 1)
    return lo + rng.uniform(0.0, 1.0, d1.size) * (hi - lo)


def draw_benford(rng: np.random.Generator, n: int) -> np.ndarray:
    """i.i.d. mantissas from exact Benford."""
    d1 = rng.choice(np.arange(1, 10), size=n, p=BENFORD)
    return _mantissa_from_digit(d1, rng)


def null_benford_cluster(struct: dict, rng: np.random.Generator, reps: int,
                         want_digits: bool = False) -> dict:
    """Entity-cluster Benford null (candidate A's detection null).

    Draw ONE mantissa per entity from exact Benford, give entity i its
    observed number of years, and keep that mantissa fixed across the years
    (a country's mantissa barely moves from year to year, which is exactly
    the dependence the null must preserve).  A small within-entity jitter of
    sd 0.01 in mantissa units prevents a spuriously zero within variance.
    """
    counts = struct["counts"]
    k = counts.size
    n = int(struct["n"])
    amps = np.empty(reps)
    res = np.empty((reps, 9)) if want_digits else None
    for start in range(0, reps, CHUNK):
        stop = min(start + CHUNK, reps)
        b = stop - start
        base = draw_benford(rng, k * b).reshape(b, k)
        jitter = rng.normal(0.0, 0.01, size=(b, n))
        mm = np.repeat(base, counts, axis=1) + jitter
        mm = np.mod(mm, 1.0)
        e = np.exp(2j * np.pi * mm)
        amps[start:stop] = np.abs(e.mean(axis=1))
        if want_digits:
            d = np.clip(np.floor(10.0 ** mm + 1e-12).astype(int), 1, 9)
            for i in range(b):
                res[start + i] = np.bincount(d[i], minlength=10)[1:10] / n
    return {"amp": amps, "digit_vectors": res}


def null_law_iid(fit, struct: dict, rng: np.random.Generator, reps: int,
                 want_digits: bool = False) -> dict:
    """i.i.d. draws from a fitted law at the observed n."""
    n = int(struct["n"])
    amps = np.empty(reps)
    res = np.empty((reps, 9)) if want_digits else None
    for start in range(0, reps, CHUNK):
        stop = min(start + CHUNK, reps)
        b = stop - start
        x = fit.sampler(rng, n * b).reshape(b, n)
        mm = mantissa(np.clip(x, 1e-300, None))
        amps[start:stop] = np.abs(np.exp(2j * np.pi * mm).mean(axis=1))
        if want_digits:
            d = np.clip(np.floor(10.0 ** mm + 1e-12).astype(int), 1, 9)
            for i in range(b):
                res[start + i] = np.bincount(d[i], minlength=10)[1:10] / n
    return {"amp": amps, "digit_vectors": res}


def _digit_vectors_block(mm: np.ndarray) -> np.ndarray:
    """Nine-cell digit vectors for a block of mantissa rows, vectorised.

    ``mm`` is (b, n); the return is (b, 9).  Doing this with np.add.at on a
    flattened (row, digit) index pair replaces a per-row Python ``bincount``
    loop, which at 10,000 replicates x 8,732 rows was the dominant cost of
    the whole screen.
    """
    b, n = mm.shape
    d = np.clip(np.floor(10.0 ** mm + 1e-12).astype(np.int64), 1, 9) - 1
    res = np.zeros((b, 9))
    rows = np.repeat(np.arange(b), n)
    np.add.at(res, (rows, d.ravel()), 1.0)
    return res / float(n)


def _entity_walk_shape(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row position inside its entity, and the entity-start offsets.

    Returns ``(pos, start_cum_index)`` where ``pos[j]`` is 1..n_i for row j of
    its entity, and ``start_cum_index[i]`` is the flattened index of entity
    i's FIRST row (so the walk can be reset to zero at each entity).
    """
    n = int(counts.sum())
    pos = np.empty(n, dtype=float)
    start = np.empty(counts.size, dtype=np.int64)
    o = 0
    for i, c in enumerate(counts):
        start[i] = o
        pos[o:o + c] = np.arange(1, int(c) + 1)
        o += int(c)
    return pos, start


_WALK_CACHE: dict[float, dict] = {}


def calibrate_walk_sd(struct: dict, target_deff: float | None = None,
                      grid: tuple[float, ...] = (0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.28),
                      n_blocks: int = 4, seed: int = 12345) -> dict:
    """Choose the within-entity walk step so the NULL reproduces this panel.

    WHY THIS IS NOT AN ARBITRARY CONSTANT.  The plan prescribes a within-entity
    log10 random walk of sd 0.02 per year.  Measured on the panel, countries'
    log10 populations move by a within-entity sd of about 0.18 over the window,
    nine times the prescribed step, so a null built at 0.02 keeps every entity
    nearly frozen at its base value and OVER-clusters: its c_1 envelope would
    be wider than the data's own, and the detection test would be needlessly
    conservative -- a flattering-looking result obtained with a wrong nuisance
    parameter.  The step is therefore chosen by matching the design effect
    measured on the panel, which is the quantity that actually sets the width
    of the c_1 envelope.

    The result is cached by target design effect, because every null on the
    same observation set wants the same answer and the search costs a handful
    of synthetic n-row panels.
    """
    counts = struct["counts"]
    obs_deff = design_effect(struct)["design_effect"]
    target = float(target_deff if target_deff is not None else obs_deff)
    key = round(target, 4)
    if key in _WALK_CACHE:
        return dict(_WALK_CACHE[key], cached=True)

    pos, start = _entity_walk_shape(counts)
    n = int(struct["n"])
    k = counts.size
    rng = np.random.default_rng(seed)

    def deff_for(sd: float) -> float:
        steps = rng.normal(0.0, sd, size=(n_blocks, n))
        cum = np.cumsum(steps, axis=1)
        prior = np.zeros((n_blocks, k))
        has = start > 0
        prior[:, has] = cum[:, start[has] - 1]
        mm = np.mod((cum - np.repeat(prior, counts, axis=1)) / np.sqrt(pos)[None, :], 1.0)
        vals = []
        for i in range(n_blocks):
            s = dict(struct)
            s["mantissas"] = mm[i]
            vals.append(design_effect(s)["design_effect"])
        return float(np.mean(vals))

    rows = []
    best_sd, best_gap = grid[0], float("inf")
    for sd in grid:
        d = deff_for(sd)
        rows.append({"walk_sd": float(sd), "design_effect": d})
        gap = abs(d - target)
        if gap < best_gap:
            best_sd, best_gap = float(sd), gap
    out = {"walk_sd": best_sd, "observed_design_effect": float(obs_deff),
           "target_design_effect": target, "achieved_gap": float(best_gap),
           "grid": rows, "n_blocks": n_blocks,
           "note": ("the null's within-entity drift is set so the NULL reproduces the panel's "
                    "own measured design effect; the plan's fixed 0.02 would over-cluster")}
    _WALK_CACHE[key] = out
    return dict(out, cached=False)


def _cluster_law_block(fit, counts: np.ndarray, rng: np.random.Generator,
                       b: int, pos: np.ndarray | None = None,
                       start: np.ndarray | None = None,
                       walk_sd: float = 0.02) -> np.ndarray:
    """``b`` entity-cluster mantissa rows from a fitted law, shape (b, n).

    One value per entity from the fitted law, then entity i receives its
    observed n_i years through a mean-reverting log10 random walk.  This is
    the conservative primary null the panel's dependence structure demands;
    an i.i.d. draw at the nominal n understates the sampling variance by the
    measured design factor and would make every residual look significant.

    The walk is genuinely PER ENTITY: it is reset to zero at each entity's
    first row and damped by 1/sqrt(t) with t the within-entity year index.
    The first version accumulated one walk across the whole flattened panel,
    which coupled different countries' mantissas and so overstated the
    within-entity variance - the one quantity the design effect is built on.

    Fully vectorised over the block.
    """
    if pos is None or start is None:
        pos, start = _entity_walk_shape(counts)
    k = counts.size
    n = int(counts.sum())
    x0 = fit.sampler(rng, k * b).reshape(b, k)
    y0 = np.log10(np.clip(x0, 1e-300, None))
    steps = rng.normal(0.0, walk_sd, size=(b, n))
    cum = np.cumsum(steps, axis=1)
    # subtract each entity's walk value at the row before its first row
    prior = np.zeros((b, k))
    has_prior = start > 0
    prior[:, has_prior] = cum[:, start[has_prior] - 1]
    walk = (cum - np.repeat(prior, counts, axis=1)) / np.sqrt(pos)[None, :]
    yy = np.repeat(y0, counts, axis=1) + walk
    return np.mod(yy, 1.0)


def attach_walk_sd(struct: dict, walk_sd: float | None = None) -> dict:
    """Set the cluster null's within-entity step on the struct.

    When ``walk_sd`` is None the step is calibrated to the panel's own
    measured design effect (see ``calibrate_walk_sd``), so every null built
    from this struct reproduces the data's dependence rather than a guessed
    one.  The calibration record is stored on the struct for the report.
    """
    if walk_sd is None:
        cal = calibrate_walk_sd(struct)
        struct["walk_sd_calibration"] = cal
        struct["walk_sd"] = cal["walk_sd"]
    else:
        struct["walk_sd"] = float(walk_sd)
    return struct


def null_law_cluster_mantissas(fit, struct: dict, rng: np.random.Generator,
                               reps: int) -> np.ndarray:
    """Entity-cluster mantissa draws from a fitted law, one row per replicate.

    MEMORY DISCIPLINE: a (reps x n) float64 array is 700 MB at the pooled
    replicate count, which does not fit the artifact's RAM budget alongside
    everything else.  Callers that only need a reduction should use
    ``null_law_cluster`` (streamed); this function exists only for the small
    diagnostic uses, and refuses to build an array larger than
    ``MAX_MANTISSA_BLOCK`` replicates at once so that a silent 700 MB
    allocation cannot reappear.
    """
    counts = struct["counts"]
    n = int(struct["n"])
    if reps * n > MAX_MANTISSA_BLOCK:
        raise ValueError(
            f"refusing to materialise a ({reps} x {n}) mantissa array "
            f"({reps * n * 8 / 1e6:.0f} MB); use the streamed null_law_cluster instead")
    out = np.empty((reps, n))
    pos, start = _entity_walk_shape(counts)
    wsd = float(struct.get("walk_sd", 0.02))
    for s0 in range(0, reps, CHUNK):
        s1 = min(s0 + CHUNK, reps)
        out[s0:s1] = _cluster_law_block(fit, counts, rng, s1 - s0, pos, start, wsd)
    return out


def iter_cluster_blocks(fit, struct: dict, rng: np.random.Generator, reps: int):
    """Yield (start, stop, mantissa_block) for entity-clustered law draws.

    The single place that knows the null's construction, so the several
    statistics that need it (amplitude, digit vector, grain imprint) all see
    exactly the same law and the same dependence structure.  Blocks are
    CHUNK replicates at a time, so peak memory is CHUNK x n floats.
    """
    counts = struct["counts"]
    pos, start = _entity_walk_shape(counts)
    wsd = float(struct.get("walk_sd", 0.02))
    for s0 in range(0, reps, CHUNK):
        s1 = min(s0 + CHUNK, reps)
        yield s0, s1, _cluster_law_block(fit, counts, rng, s1 - s0, pos, start, wsd)


def null_law_cluster(fit, struct: dict, rng: np.random.Generator, reps: int,
                     want_digits: bool = False) -> dict:
    """Conservative primary null: entity-cluster draws from a fitted law.

    Streamed in blocks of ``CHUNK`` replicates, so peak memory is
    CHUNK x n floats for the mantissas (35 MB at the pooled n, not the
    700 MB a full-replicate array would need) plus one scalar - and, when
    asked, one 9-vector - per replicate.

    The within-entity drift step is read from ``struct['walk_sd']``, which
    ``attach_walk_sd`` sets to the panel's own measured design effect.
    """
    amps = np.empty(reps)
    res = np.empty((reps, 9)) if want_digits else None
    for s0, s1, mm in iter_cluster_blocks(fit, struct, rng, reps):
        amps[s0:s1] = np.abs(np.exp(2j * np.pi * mm).mean(axis=1))
        if want_digits:
            res[s0:s1] = _digit_vectors_block(mm)
        del mm
    return {"amp": amps, "digit_vectors": res}


def null_entity_resample(struct: dict, rng: np.random.Generator, reps: int,
                         want_digits: bool = False) -> dict:
    """Nonparametric entity bootstrap: resample entities, keep all their years."""
    counts = struct["counts"]
    k = counts.size
    n = int(struct["n"])
    m = np.asarray(struct["mantissas"], dtype=float)
    inv = struct["inverse"]
    amps = np.empty(reps)
    res = np.empty((reps, 9)) if want_digits else None
    for r in range(reps):
        pick = rng.integers(0, k, size=k)
        mm = np.concatenate([m[inv == p] for p in pick])
        e = np.exp(2j * np.pi * mm)
        amps[r] = abs(e.mean())
        if want_digits:
            d = np.clip(np.floor(10.0 ** mm + 1e-12).astype(int), 1, 9)
            res[r] = np.bincount(d, minlength=10)[1:10] / mm.size
    return {"amp": amps, "digit_vectors": res}


def null_rounding_grain(fit, struct: dict, rng: np.random.Generator, reps: int,
                        grain_digits: int) -> dict:
    """Grain-imprint null: simulate the fitted law, round to g significant
    figures, and record the imprint vector w_d = p_d^rounded - p_d^smooth."""
    n = int(struct["n"])
    imprints = np.empty((reps, 9))
    smooth = np.empty((reps, 9))
    for start in range(0, reps, CHUNK):
        stop = min(start + CHUNK, reps)
        b = stop - start
        x = fit.sampler(rng, n * b).reshape(b, n)
        for i in range(b):
            xi = x[i]
            p_s = np.bincount(np.clip(np.floor(10.0 ** mantissa(xi) + 1e-12).astype(int), 1, 9),
                              minlength=10)[1:10] / n
            xr = round_significant(xi, grain_digits)
            p_r = np.bincount(np.clip(np.floor(10.0 ** mantissa(xr) + 1e-12).astype(int), 1, 9),
                              minlength=10)[1:10] / n
            imprints[start + i] = p_r - p_s
            smooth[start + i] = p_s
    return {"imprint": imprints, "smooth": smooth}


def round_significant(x: np.ndarray, g: int) -> np.ndarray:
    """Round positive values to g significant decimal digits."""
    x = np.asarray(x, dtype=float)
    e = np.floor(np.log10(np.clip(x, 1e-300, None)))
    scale = 10.0 ** (g - 1 - e)
    return np.round(x * scale) / scale


# ---------------------------------------------------------------------------
# envelope summarisation
# ---------------------------------------------------------------------------
def summarise_envelope(samples: np.ndarray, observed: float | None = None) -> dict:
    """Percentiles, sd, a 100-bin histogram and the one-sided tail p."""
    s = np.asarray(samples, dtype=float).ravel()
    s = s[np.isfinite(s)]
    if s.size == 0:
        return {"n_reps": 0, "percentiles": {}, "mean": None, "sd": None,
                "min": None, "max": None, "hist": {"edges": [], "counts": []},
                "note": "no replicates: this statistic is undefined at this level or for this null"}
    out = {
        "n_reps": int(s.size),
        "percentiles": {str(q): float(np.percentile(s, q)) for q in (2.5, 5, 50, 95, 97.5)},
        "mean": float(s.mean()), "sd": float(s.std(ddof=1)) if s.size > 1 else 0.0,
        "min": float(s.min()), "max": float(s.max()),
        "hist": histogram_bins(s, 100),
    }
    if observed is not None and np.isfinite(observed):
        out["observed"] = float(observed)
        out["p_upper"] = float((np.sum(s >= observed) + 1) / (s.size + 1))
        out["p_lower"] = float((np.sum(s <= observed) + 1) / (s.size + 1))
        out["p_two_sided"] = float(min(1.0, 2.0 * min(out["p_upper"], out["p_lower"])))
        out["observed_in_envelope"] = bool(np.percentile(s, 2.5) <= observed <= np.percentile(s, 97.5))
    return out


def histogram_bins(s: np.ndarray, nbins: int) -> dict:
    """A reproducible 100-bin histogram, stored as (edges, counts)."""
    counts, edges = np.histogram(s, bins=nbins)
    return {"edges": [float(v) for v in edges], "counts": [int(v) for v in counts]}


def sign_probabilities(digit_vectors: np.ndarray) -> dict:
    """P(r_d > 0) per digit under a null - the sign-vector null."""
    if digit_vectors is None or digit_vectors.size == 0:
        return {}
    resid = digit_vectors - BENFORD[None, :]
    p_pos = (resid > 0).mean(axis=0)
    return {str(d + 1): float(p_pos[d]) for d in range(9)}


def shift_year_null(obs_series: np.ndarray, pred_series: np.ndarray,
                    extra_reps: int = 0, rng: np.random.Generator | None = None) -> dict:
    """Exact circular-shift null for the observed-vs-predicted correlation.

    Shifting the predicted series against the observed one preserves both
    series' autocorrelation and destroys only the year-specific association,
    so it is the sharpest available null for candidate B.  With T years the
    test is inherently discrete: the smallest attainable one-sided p is
    1/(T-1+1), i.e. 0.027 at T = 37.  That floor is REPORTED rather than
    hidden.

    ``extra_reps`` optionally adds a phase-randomised surrogate: the predicted
    series is Fourier-transformed, its phases are randomised, and it is
    transformed back.  This preserves the predicted series' full power
    spectrum exactly while destroying the year alignment, giving a continuous
    null the discrete shift set cannot.  Both are returned separately so the
    reader can see whether the discrete floor was binding.
    """
    obs = np.asarray(obs_series, dtype=float)
    pred = np.asarray(pred_series, dtype=float)
    T = obs.size
    stats = np.empty(T - 1)
    for s in range(1, T):
        stats[s - 1] = np.corrcoef(obs, np.roll(pred, s))[0, 1]
    out = {"shift_statistics": [float(v) for v in stats],
           "n_shifts": T - 1,
           "min_attainable_p": float(1.0 / T)}

    if extra_reps and rng is not None and np.std(obs) > 0 and np.std(pred) > 0:
        # phase-randomised surrogate of the PREDICTED series
        pred_c = pred - pred.mean()
        F = np.fft.rfft(pred_c)
        mag = np.abs(F)
        sur = []
        for _ in range(extra_reps):
            ph = rng.uniform(0, 2 * np.pi, mag.size)
            ph[0] = 0.0
            if T % 2 == 0:
                ph[-1] = 0.0
            s = np.fft.irfft(mag * np.exp(1j * ph), n=T)
            if np.std(s) > 0:
                sur.append(float(np.corrcoef(obs, s)[0, 1]))
        out["surrogate_statistics"] = sur
        out["n_surrogates"] = len(sur)
    return out

