"""Print the substance of the last run, for review."""

import json
from pathlib import Path

R = Path("results")

p = json.loads((R / "provenance.json").read_text())
print("PROVENANCE")
for r in p["results"]:
    print(f"  {r['recorded']['label']}: {r['status']} {[h['candidate_id'] for h in r['hits']]}")
print(f"  canonical: {p['canonical']['canonical_definition_id']}")
print(f"  recorded vector exact hits: {p['recorded_vector_exact_hits']}")

c = json.loads((R / "canonical.json").read_text())
print("\nCANONICAL HARMONIC")
print(f"  k selected {c['fit']['selected_k']}  BIC {c['fit']['bic_by_k']}")
h = c["fit"]["harmonics"]
print(f"  A_canon {h['A_canon']:.6f}  A_nocancel {h['A_nocancel']:.6f} "
      f"ratio {c['resolution']['cancellation_ratio']:.4f}")
print(f"  verdict: {c['resolution']['verdict']}")

s = json.loads((R / "stability.json").read_text())
d = s["disjoint_folds"]
print("\nDISJOINT FOLDS")
for r in d["fold_rows"]:
    print(f"  fold {r['fold']}: n_c {r['n_countries']} |c1| {r['abs_c1']:.5f} "
          f"ci [{r['rayleigh_ci_low']:.5f},{r['rayleigh_ci_high']:.5f}] "
          f"mda80 {r['mda80']:.5f} p {r['p_rotation']:.4f} covers {r['covers_pooled']}")
print(f"  Cochran Q {d['cochran_q']:.4f} df {d['cochran_q_df']} p {d['cochran_q_p']:.4f}")
print(f"  all cover pooled: {d['all_courses_cover_pooled']}")

print("\nPOWER BOUND")
for r in s["power_bound"]["rows"]:
    print(f"  A {r['budget_A']:.4f}: need n_c {r['required_n_countries']:.1f} "
          f"vs {r['available_n_countries']}, shortfall {r['shortfall_factor']:.2f}")

st = s["stability"]
print(f"\nSTABILITY pooled |c1| {st['pooled_abs_c1']:.5f} "
      f"boot [{st['bootstrap_ci_low']:.5f},{st['bootstrap_ci_high']:.5f}] "
      f"jack [{st['jackknife_range_low']:.5f},{st['jackknife_range_high']:.5f}]")

a = json.loads((R / "audit.json").read_text())
print(f"\nAUDIT mismatches {len(a['verify_mismatches'])}  wall {a['wall_seconds']}s")
print(f"  fresh-seed p {a['fresh_seed_p']:.5f} vs {a['main_p']:.5f} delta "
      f"{a['fresh_seed_delta']:.2e} tol {a['fresh_seed_tolerance']:.2e}")

v = json.loads((R / "verdict.json").read_text())
print("\nVERDICTS")
for k, vv in v["verdicts"].items():
    print(f"  {k}: {vv['branch_taken']}  |c1| {vv['abs_c1']:.5f} "
          f"2sig {vv['threshold_2sigma']:.5f}  all5 {vv['all_consistent']}")
