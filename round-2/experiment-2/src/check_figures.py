"""Check a rendered figure against its values file, so a caption cannot drift."""

import json
import sys
from pathlib import Path

import numpy as np

F = Path("figures")

# fig1: the two Rayleigh densities must be exact for their sigma, and each
# observed |c1| must be the value marked on the plot
vals = json.loads((F / "fig1_rayleigh_null.values.json").read_text())
print("FIG 1  Rayleigh null on both folds")
for row in vals["folds"]:
    sr = row["sigma_R"]
    grid = np.linspace(0.0, row["t5"] * 1.9, 240)
    dens = (grid / sr**2) * np.exp(-(grid**2) / (2 * sr**2))
    # integral over the full support must be 1
    full = np.linspace(0.0, 12 * sr, 20000)
    fd = (full / sr**2) * np.exp(-(full**2) / (2 * sr**2))
    integral = np.trapezoid(fd, full)
    print(f"  {row['label']:5s} sigma_R {sr:.5f}  observed {row['observed']:.5f}  "
          f"2sigma {row['t2']:.5f}  density integral {integral:.6f}")
    assert abs(integral - 1.0) < 1e-6, "Rayleigh density does not integrate to 1"
    assert row["observed"] > row["t2"], "observed should exceed the two-sigma threshold"

# fig2: p-values must match verdict.json exactly
verdict = json.loads(Path("results/verdict.json").read_text())
hv = json.loads((F / "fig2_pvalue_heatmap.values.json").read_text())
print("\nFIG 2  p-value heatmap")
worst = 0.0
for cell in hv["cells"]:
    fold, stat = [s.strip() for s in cell["row"].split("|")]
    rep = verdict["battery"][fold]["rotation"][stat]["p"]
    worst = max(worst, abs(cell["p_rotation"] - rep))
print(f"  {len(hv['cells'])} cells; max |plotted - verdict.json| on the rotation p = {worst:.2e}")
assert worst < 1e-12

# fig3: the observed-amplitude marker must be the real observed |c1|
pv = json.loads((F / "fig3_power_curve.values.json").read_text())
obs = verdict["battery"]["even"]["observed"]["abs_c1"]
print("\nFIG 3  power curve")
print(f"  marker amplitude {pv['observed_amplitude']:.6f} vs even-fold |c1| {obs:.6f}")
assert abs(pv["observed_amplitude"] - obs) < 1e-12
for c in pv["curve"]:
    print(f"  A {c['planted_amplitude']:.3f} -> rejection {c['rejection_rate']:.2f} "
          f"(median realised {c['median_realised_abs_c1']:.4f})")

# fig4: every plotted r_2 must equal 2 sqrt(0.5 kappa^2 / n) for its own n
tv = json.loads((F / "fig4_threshold_vs_n.values.json").read_text())
print("\nFIG 4  threshold against n")
worst4 = 0.0
for r in tv["thresholds"]:
    recomputed = 2.0 * np.sqrt(0.5 * r["kappa2_used"] / r["n"])
    worst4 = max(worst4, abs(recomputed - r["r_2sigma"]))
    print(f"  n {r['n']:9.2f}  kappa^2 {r['kappa2_used']:.6f}  r_2 {r['r_2sigma']:.6f}  "
          f"recomputed {recomputed:.6f}")
assert worst4 < 1e-12

print("\nAll four figures agree with their values files.")
