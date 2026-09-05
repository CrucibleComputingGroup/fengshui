#!/usr/bin/env python3
"""
Measure the end-to-end impact of the off-CP PIM fix (area=PIM_DIE_AREA_MM2 + GDDR7
leakage/PHY static power + activation operand transport) on the 4 committed N=8 pools
over the 20 Fig-11 nets.  Source-agnostic: it just evaluates whatever cal_perf_phy_net
currently is, and dumps geomeans + the 3 PIM-affected decode nets to JSON, so a
git-stash A/B (original vs fixed) gives a clean, run-order-matched delta.

Usage:  python arch_impl/measure_pim_fix.py <out.json>
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
PIM_NETS = ["llama3.1_8b_decode_kv1024_b1", "llama3.1_70b_decode_kv1024_b1",
            "qwen3_235b_a22b_decode_kv1024_b1"]


def load_pool(objective, cost_aware, n=8):
    f = sorted(glob.glob(os.path.join(ARCHGYM, POOL_DIRS[(objective, cost_aware)],
               "chain_summary_*.json")), key=os.path.getmtime)[-1]
    idents = json.load(open(f))[f"n{n}"]["best_chiplets"]
    return [ChipletConfig.from_identifier(s) for s in idents]


def geomean(vals):
    vals = [v for v in vals if v and math.isfinite(v) and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")


def evaluate(vnets, objective, cost_aware):
    pool = load_pool(objective, cost_aware)
    _, res = run_single_optimization(virtual_nets=vnets, chiplet_group=pool, objective=objective,
                                     results_file=DB, cost_aware=cost_aware, use_sequential=True,
                                     n_workers=8, use_dag_cp=True, v_het_batch=True)
    _, per = calculate_average_opt_value(res, objective)
    return {n: r["min_value"] for n, r in per.items()}


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "measure_pim_fix.json"
    vnets = _build_virtual_nets(database_file=DB)
    METRICS = [("Energy", "energy", False), ("EnergyxCost", "energy", True),
               ("EDP", "edp", False), ("EDPxCost", "edp", True)]
    result = {}
    print(f"\n{'metric':14s} {'geomean':>13s}   per-PIM-net values")
    for label, obj, ca in METRICS:
        per = evaluate(vnets, obj, ca)
        g = geomean(per.values())
        # tolerant match: result keys may carry/omit a batch suffix
        def _find(stub):
            return next((per[k] for k in per if k.startswith(stub)), None)
        pim = {n: _find(n.rsplit("_b1", 1)[0]) for n in PIM_NETS}
        result[label] = {"geomean": g, "pim_nets": pim, "n_nets": len(per)}
        pim_str = "  ".join(f"{n.split('_decode')[0]}={per.get(n, float('nan')):.3e}" for n in PIM_NETS)
        print(f"{label:14s} {g:13.5e}   {pim_str}")
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
