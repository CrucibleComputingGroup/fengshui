#!/usr/bin/env python3
"""
Estimate the impact of excluding PIM from the off-critical-path builder, by
re-evaluating the committed N=8 pools with a monkeypatch (no source change).
Compares current (off-CP PIM allowed, area-undercounted) vs fixed (PIM excluded
off-CP) for the cost-aware metrics EC and EDPc. Energy/EDP are unaffected (cost
not applied off-path) but shown for completeness.
"""
import os, sys, glob, json, math

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.normpath(os.path.join(THIS_DIR, ".."))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

ARCHGYM = os.path.normpath(os.path.join(SCRIPTS, "..", "archgym_results"))
DB = os.path.normpath(os.path.join(SCRIPTS, "..", "unified_database.csv"))

import cal_perf_phy_net as cpp
from chiplet_dataclass import ChipletConfig
from baseline import _build_virtual_nets
from chiplet_sel import run_single_optimization
from utility_functions import calculate_average_opt_value

POOL_DIRS = {("energy", False): "v6_energy_chain", ("energy", True): "v6_energy_cost_chain",
             ("edp", False): "v6_edp_chain", ("edp", True): "v6_edp_cost_chain"}


def load_pool(objective, cost_aware, n=8):
    f = sorted(glob.glob(os.path.join(ARCHGYM, POOL_DIRS[(objective, cost_aware)],
               "chain_summary_*.json")), key=os.path.getmtime)[-1]
    idents = json.load(open(f))[f"n{n}"]["best_chiplets"]
    return [ChipletConfig.from_identifier(s) for s in idents]


def geomean(vals):
    vals = [v for v in vals if v and math.isfinite(v) and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")


# --- monkeypatch: drop PIM from the off-CP builder's chiplet set ---
_orig_offcp = cpp._build_off_cp_functions
def _no_pim_offcp(off_cp_info, chiplet_group, chiplets_data, *a, **kw):
    keep = [i for i, c in enumerate(chiplet_group) if getattr(c, "arch_target", "") != "PIM"]
    return _orig_offcp(off_cp_info, [chiplet_group[i] for i in keep],
                       [chiplets_data[i] for i in keep], *a, **kw)


def evaluate(vnets, objective, cost_aware):
    pool = load_pool(objective, cost_aware)
    _, res = run_single_optimization(virtual_nets=vnets, chiplet_group=pool, objective=objective,
                                     results_file=DB, cost_aware=cost_aware, use_sequential=True,
                                     n_workers=8, use_dag_cp=True, v_het_batch=True)
    _, per = calculate_average_opt_value(res, objective)
    return {n: r["min_value"] for n, r in per.items()}


def main():
    vnets = _build_virtual_nets(database_file=DB)
    METRICS = [("Energy", "energy", False), ("Energy x Cost", "energy", True),
               ("EDP", "edp", False), ("EDP x Cost", "edp", True)]
    print(f"\n{'metric':14s} {'geomean now':>13s} {'geomean fixed':>14s} {'ratio':>7s}   worst per-net delta")
    for label, obj, ca in METRICS:
        cpp._build_off_cp_functions = _orig_offcp
        cur = evaluate(vnets, obj, ca)
        cpp._build_off_cp_functions = _no_pim_offcp
        fix = evaluate(vnets, obj, ca)
        cpp._build_off_cp_functions = _orig_offcp
        gnow, gfix = geomean(cur.values()), geomean(fix.values())
        deltas = sorted(((fix[n] / cur[n] - 1) * 100, n) for n in cur if cur[n] and math.isfinite(cur[n]))
        worst = deltas[-1] if deltas else (0, "")
        changed = [(d, n) for d, n in deltas if abs(d) > 0.1]
        print(f"{label:14s} {gnow:13.4e} {gfix:14.4e} {gfix/gnow:7.3f}   "
              f"{worst[1]} +{worst[0]:.1f}%  ({len(changed)} nets changed)")
        for d, n in changed[-4:]:
            print(f"               · {n}: {cur[n]:.3e} -> {fix[n]:.3e}  ({d:+.1f}%)")


if __name__ == "__main__":
    main()
