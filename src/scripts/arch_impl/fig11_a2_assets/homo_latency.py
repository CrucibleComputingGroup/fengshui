"""Regenerate homo_latency.json — the best homogeneous (Gemini-style) design + its per-network latency used by generate_paper_fig10.py
(Figure 9) for the cost-bearing panels (Energy x $ and EDP x $).

Runs the co-design optimizer over the shipped Timeloop database, so it needs
src/unified_database.csv in place. Paths are resolved relative to this file, so
it runs from any working directory:

    python3 src/scripts/arch_impl/fig11_a2_assets/homo_latency.py

Env overrides: FENGSHUI_DB (database path), FENGSHUI_A2_OUT (output directory).
"""
import sys, os, copy, math, json

HERE = os.path.dirname(os.path.abspath(__file__))          # .../arch_impl/fig11_a2_assets
ARCH = os.path.dirname(HERE)                                # .../scripts/arch_impl
SCRIPTS = os.path.dirname(ARCH)                             # .../scripts
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, ARCH)
# the framework resolves several inputs (network_analysis.csv, ../unified_database.csv)
# relative to the scripts directory
os.chdir(SCRIPTS)

DB = os.environ.get('FENGSHUI_DB', os.path.join(SCRIPTS, '..', 'unified_database.csv'))
OUT_DIR = os.environ.get('FENGSHUI_A2_OUT', HERE)
OUT = os.path.join(OUT_DIR, 'homo_latency.json')

from chiplet_dataclass import generate_all_chiplet_configs
from baseline import _build_virtual_nets
from chiplet_sel import run_single_optimization
from utility_functions import calculate_average_opt_value
import competing_baselines as cb

vnets=_build_virtual_nets(database_file=DB)
cand=generate_all_chiplet_configs(arch_targets=["eyeriss_like","simba_like","gemmini_like"])
print(f"{len(vnets)} nets, {len(cand)} gemini candidates", flush=True)

def eval_lat(pool, objective, cost_aware):
    _, results = run_single_optimization(virtual_nets=vnets, chiplet_group=pool,
        objective=objective, results_file=DB, cost_aware=cost_aware,
        use_sequential=True, n_workers=8, use_dag_cp=True, v_het_batch=False)
    avg,_=calculate_average_opt_value(results, objective)
    val={n:r["min_value"] for n,r in results.items()}
    lat={n:r["best_latency"] for n,r in results.items()}
    return avg, val, lat

out={}
for label,obj,cost in [("energy_cost","energy",True),("edp_cost","edp",True)]:
    best=None
    with cb.homogenize_memory("GDDR7"):
        for i,d0 in enumerate(cand):
            d=copy.deepcopy(d0); d.dram_type="GDDR7"
            avg,val,lat=eval_lat([d],obj,cost)
            if math.isfinite(avg) and (best is None or avg<best[0]):
                best=(avg,d.get_identifier(),val,lat)
    out[label]={"design":best[1],"value":best[2],"latency":best[3]}
    print(f"[{label}] best={best[1]} geomean={best[0]:.4e}", flush=True)
os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(out, f)
print(f"saved {OUT}", flush=True)
