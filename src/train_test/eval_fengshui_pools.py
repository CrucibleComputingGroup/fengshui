#!/usr/bin/env python3
"""
Evaluate the paper's Fengshui 8-chiplet pools (from archgym_results/fengshui/v6_*_chain)
on the TEST set of each train/test split (random / decode_test / moe_test), with the
pool held FIXED (only inner per-workload mapping re-optimised) — same as cross_eval.

For every (split, config) it produces `fengshui_cross` = geomean of the chosen
objective over that split's test workloads under the Fengshui pool, plus per-net
values. Output: train_test/fengshui_cross.json

Inputs (portable, resolved relative to this file / env):
  $SPLITS_DIR/<split>.json                 each split.json (has the test list);
                                           defaults to ./splits (stage the split.json
                                           files produced by run_train_test.py here)
  ../archgym_results/ae_<obj>_chain/*.csv  the 4 shipped AE saeo_isaeo_chain CSVs (n=8 pool)
"""
import os, sys, json, glob, csv, math

SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
sys.path.insert(0, SCRIPTS)
os.chdir(SCRIPTS)

import multiprocessing as mp
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

from global_parameter import NET_DIR
from network_dataclass import VirtualNetwork
from cal_perf_phy_net import preload_database, CSV_CACHE
from utility_functions import calculate_average_opt_value
from chiplet_sel import run_single_optimization
from chiplet_dataclass import ChipletConfig
import run_archgym_chiplet as rac
rac._USE_DAG_CP = True
rac._V_HET_BATCH = True

DB = os.environ.get("FENGSHUI_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "unified_database.csv"))
SPLITS_DIR = os.environ.get("SPLITS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits"))
MOZ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "archgym_results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fengshui_cross.json")

# (objective, cost_aware, config short name, fengshui v6 dir)
CONFIGS = [
    ("energy", False, "energy_nocost", "ae_energy_chain"),
    ("energy", True,  "energy_cost",   "ae_energy_cost_chain"),
    ("edp",    False, "edp_nocost",    "ae_edp_chain"),
    ("edp",    True,  "edp_cost",      "ae_edp_cost_chain"),
]
SPLIT_NAMES = ["random_split", "decode_test", "moe_test"]


def parse_uname(u):
    """'qwen3_30b_a3b_decode_kv1024_b1_seq1' -> (name, batch, seq)."""
    rest, seq = u.rsplit("_seq", 1)
    name, b = rest.rsplit("_b", 1)
    return name, int(b), int(seq)


def main():
    # 1. test workload specs per split
    split_tests = {}
    for s in SPLIT_NAMES:
        d = json.load(open(f"{SPLITS_DIR}/{s}.json"))
        split_tests[s] = [parse_uname(u) for u in d["test"]]
        print(f"{s}: {len(split_tests[s])} test nets")

    needed = set(name for specs in split_tests.values() for (name, b, seq) in specs)
    print(f"\nPreloading DB for {len(needed)} networks...")
    preload_database(DB, needed_nets=needed)
    df = CSV_CACHE.get(DB)
    db_layers = {net: set(g["layer_name"].unique())
                 for net, g in df.groupby("net", observed=True)}

    vcache = {}
    def get_vnet(name, b, seq):
        key = (name, b, seq)
        if key not in vcache:
            vn = VirtualNetwork(name, batch_size=b, sequence_length=seq)
            try:
                vn.load_from_dir(os.path.join(NET_DIR, name),
                                 db_layers=db_layers.get(name))
            except Exception as e:
                print(f"  [warn] load {name}: {e}")
            vcache[key] = vn if len(vn.layers) > 0 else None
        return vcache[key]

    result = {}
    for obj, cost, cfg, mozdir in CONFIGS:
        mcsv = sorted(glob.glob(f"{MOZ_DIR}/{mozdir}/saeo_isaeo_chain_*.csv"))[-1]
        pool = ChipletConfig.from_csv_for_n_chiplets(8, mcsv)
        archs = [c.arch_target for c in pool]
        print(f"\n=== {cfg}: Fengshui pool ({len(pool)} chiplets) {archs} ===")
        for s in SPLIT_NAMES:
            nets = [get_vnet(*spec) for spec in split_tests[s]]
            nets = [n for n in nets if n is not None]
            _, res = run_single_optimization(
                virtual_nets=nets, chiplet_group=pool, objective=obj,
                results_file=DB, cost_aware=cost, use_sequential=True,
                n_workers=8, use_dag_cp=True, v_het_batch=True)
            val, _ = calculate_average_opt_value(res, obj)
            per_net = {k: v["min_value"] for k, v in res.items()}
            result.setdefault(s, {})[cfg] = {"fengshui_cross": val, "per_net": per_net}
            print(f"  {s:12} fengshui_cross = {val:.4e}  ({len(nets)} nets)")

    json.dump(result, open(OUT, "w"), indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
