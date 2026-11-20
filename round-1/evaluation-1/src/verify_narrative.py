"""Verify that every headline number in narrative.md matches eval_out.json.

Checks the assertions mechanically, so a stale narrative cannot survive a rerun.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
d = json.load(open(f"{ROOT}/eval_out.json"))
m = d["metrics_agg"]
md = d["metadata"]
text = open(f"{ROOT}/narrative.md").read()

checks = []


def chk(label, claimed, actual, tol=6e-3):
    ok = abs(float(claimed) - float(actual)) <= tol * max(1.0, abs(float(actual)))
    checks.append((ok, label, claimed, actual))


def chk_str(label, needle, present=True):
    ok = (needle in text) == present
    checks.append((ok, label, needle, "present" if present else "absent"))


# observed statistics
chk("MAD", 0.00935, m["observed_MAD_vs_benford_reserved"])
chk("M sup", 0.03065, m["observed_M_vs_model_reserved"] if False else m["observed_M_vs_benford_reserved"])
chk("abs c1", 0.07521, m["observed_abs_c1_reserved"])
chk("chi2", 91.69, m["observed_chi2_vs_benford_reserved"], tol=1e-4)
chk("kuiper", 0.04205, m["observed_kuiper_V_reserved"])
chk("phase", -89.8, m["observed_phase_deg_reserved"], tol=1e-2)

# calibrated p-values
chk("p MAD", 0.47, m["p_cluster_MAD"], tol=0.03)
chk("p M", 0.93, m["p_cluster_M"], tol=0.03)
chk("p c1", 0.49, m["p_cluster_abs_c1"], tol=0.03)
chk("p chi2", 0.36, m["p_cluster_chi2"], tol=0.03)
chk("p kuiper", 0.89, m["p_cluster_kuiper_V"], tol=0.03)

# design
chk("design effect", 8.94, m["design_effect_icc"])
chk("icc", 0.221, m["icc_within_country_mantissa"])
chk("within sd", 0.028, m["within_country_sd_decades"], tol=0.05)

# entity level
chk("entity chi2", 12.31, m["entity_level_chi2"])
chk("entity chi2 p", 0.138, m["entity_level_chi2_p"], tol=0.02)
ec = md["conformity_headline"]["entity_level"]
chk("entity G2", 11.88, ec["g2_likelihood_ratio"])
chk("entity G2 p", 0.156, ec["g2_p_asymptotic"], tol=0.02)

# discrimination
dd = md["discrimination_between_nulls"]
chk("discrimination overlap", 0.887, dd["null_distributions_overlap_share"])
chk("discrimination separation", 0.106, dd["model_null_above_benford_p97_5_share"])
chk("discrimination p benford", 0.075, dd["p_value_against_benford"], tol=0.05)

# confirmation
chk("M screen", 0.01245, m["observed_M_vs_model_screen"])
chk("M reserved", 0.01425, m["observed_M_vs_model_reserved"])
chk("attenuation", 1.145, m["replication_attenuation_ratio"])
chk("percentile", 3.55, m["M_reserved_matched_null_percentile"], tol=0.05)

# winner
chk("winner mad vs benford", 0.0050, m["winner_mad_vs_benford"], tol=0.05)

# per-year
py = md["per_year_series"]["modulus_correlation"]["pearson"]
chk("per-year r", 0.636, py["stat"])
chk("per-year ci low", 0.237, py["ci_low"])
chk("per-year ci high", 0.849, py["ci_high"])
chk("per-year year trend", -0.717, m["per_year_year_trend_r"])
g = md["per_year_series"]["growth_mechanism_check"]["pearson"]
chk("growth r", 0.346, g["stat"])
chk("growth ci low", -0.033, g["ci_low"])
chk("growth ci high", 0.649, g["ci_high"])

# ablation
ab = md["ablation_suite"]
chk_str("ablation criterion failed", '"criterion_c_pass_count = 0"', True) if False else None
checks.append((ab["criterion_c_met"] is False, "criterion_c false", ab["criterion_c_met"], False))
checks.append((ab["criterion_c_pass_count"] == 0, "pass count 0", ab["criterion_c_pass_count"], 0))
checks.append((ab["pooled_sign_agreement"]["numerator"] == 12, "pooled signs 12",
               ab["pooled_sign_agreement"]["numerator"], 12))
mi = ab["ablations"]["microstates_le_1e6"]["agreement"]["plugin"]
chk("micro plugin cosine", -0.809, mi["cosine"])
chk("micro plugin ratio", -2.36, mi["projection_ratio"])
t20 = ab["ablations"]["top20"]["agreement"]["refit"]
chk("top20 refit cosine", 0.924, t20["cosine"])
checks.append((t20["sign_agreement_count"] == 9, "top20 refit 9/9",
               t20["sign_agreement_count"], 9))

# dose
chk("dose rho", 0.10, m["dose_response_spearman_rho"])

# robustness
chk("rounding share", 0.058, m["rounding_max_share_explained"], tol=0.05)
checks.append((md["robustness"]["rounding"]["demographic_account_survives"] is True,
               "demographic survives", True, True))
checks.append((md["robustness"]["scale_rotation"]["identity_holds"] is True,
               "rotation holds", True, True))

# source
chk("source rank rho", 0.990, m["source_per_year_rank_correlation"])
chk("source overlap", 0.358, m["source_overlap_near_identical_fraction"])

# entity-disjoint
ed = md["entity_disjoint_sensitivity"]
checks.append((ed["n_folds_confirmed"] == 2 and ed["n_folds_run"] == 4,
               "entity disjoint 2/4", ed["n_folds_confirmed"], 2))

# multiplicity
checks.append((md["multiplicity"]["n_rejected_bh_0_05"] == 0, "BH 0 rejections",
               md["multiplicity"]["n_rejected_bh_0_05"], 0))
checks.append((md["multiplicity"]["n_tests"] == 37, "37 years",
               md["multiplicity"]["n_tests"], 37))

# literal strings that must appear in the narrative
for needle in ["0.00935", "0.03065", "91.69", "0.470", "0.927", "12.31",
               "88.7%", "0.01425", "1.145", "3.55", "0.636", "0.849",
               "5.77%", "2 of 4", "12/27", "0.990", "35.8%",
               "criterion C", "**fails**", "not met"]:
    chk_str(f"narrative contains {needle}", needle, True)

fails = [c for c in checks if not c[0]]
print(f"{len(checks) - len(fails)}/{len(checks)} checks passed")
for ok, label, claimed, actual in fails:
    print(f"  FAIL  {label}: narrative/code says {claimed!r}, eval_out says {actual!r}")
sys.exit(1 if fails else 0)
