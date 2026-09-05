#!/usr/bin/env python3
"""Correctness verification for the ablation toggles.

Runs ONE decode network (memory-bound, exercises PIM + memory choice) under each
configuration and prints the WINNING solution's internals, proving each toggle does
exactly what it claims:
  - buffer_config (per-boundary DRAM type)  -> proves homo-HBM / homo-GDDR7
  - binary_string (fusion grouping)         -> proves no-fusion
  - chiplet identifiers used in the solution -> proves PIM used/absent
  - tp field in each group's config id      -> proves TP=1
"""
import os, sys, re
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS, ".."))

import ablation_study as A
from baseline import _build_virtual_nets
from chiplet_sel import run_single_optimization
from cal_perf_phy_net import preload_database

DB = A.DEFAULT_DB
NET = "llama3.1_8b_decode_kv1024"   # memory-bound decode: should use PIM in Full

# Build just this one network
import global_parameter as gp
from network_dataclass import VirtualNetwork
import pandas as pd
_db = pd.read_csv(DB)
db_layers = set(_db[_db['net'] == NET]['layer_name'].unique())
vn = VirtualNetwork(NET, batch_size=1, sequence_length=1)
vn.load_from_dir(os.path.join(gp.NET_DIR, NET), db_layers=db_layers)
preload_database(DB, needed_nets={NET})
vnets = [vn]

pool, pool_file, idents = A.load_pool("energy", False)
print(f"Pool ({len(pool)} chiplets) from {pool_file}:")
for c in pool:
    print("   ", c.get_identifier())
print(f"\nVerifying on network: {vn.get_unique_name()}  ({len(vn.layers)} layers)\n")


def parse_funcs(best_config):
    """Return list of (chiplet_arch, tp) from the winning group configs."""
    out = []
    if not best_config or "functions" not in best_config:
        return out
    for g in best_config["functions"]:
        cid = getattr(g, "id", "")
        # id = "{arch}@glb..@pe_x..@pe_y..@{tp}@{mapper}@{bonding}@{buffers}"
        m = re.match(r"^([A-Za-z0-9_]+)@glb\d+@pe_x_scale\d+@pe_y_scale\d+@(\d+)@", cid)
        if m:
            out.append((m.group(1), int(m.group(2))))
        else:
            out.append((cid, None))
    return out


for label, key in [("Full Fengshui", "full"),
                   ("w/o Near-bank PIM", "no_pim"),
                   ("w/o Heterogeneous Memory (homo-HBM)", "homo_hbm"),
                   ("homo GDDR7 (PIM kept)", "homo_gddr7"),
                   ("w/o Tensor Parallelism", "no_tp"),
                   ("w/o Layer Fusion", "no_fusion")]:
    eff_pool = [c for c in pool if c.arch_target != "PIM"] if key == "no_pim" else pool
    with A.toggle(key):
        # sanity: show patched module state
        state = f"ga.dram={A.ga.dram_options} cpp.tp={A.cpp.tp_degrees}"
        _, results = run_single_optimization(
            virtual_nets=vnets, chiplet_group=eff_pool, objective="energy",
            results_file=DB, cost_aware=False, use_sequential=True,
            n_workers=1, use_dag_cp=True, v_het_batch=True)
    r = results[vn.get_unique_name()]
    gene = r["best_gene"]
    funcs = parse_funcs(r["best_config"])
    archs = sorted(set(a for a, _ in funcs))
    tps = sorted(set(t for _, t in funcs if t is not None))
    bc = gene.get("buffer_config", [])
    drams = sorted(set(bc))
    print(f"### {label}   [{state}]")
    print(f"    energy            = {r['min_value']:.4e}")
    print(f"    binary_string     = {gene.get('binary_string')}   (all-1s => no fusion)")
    print(f"    buffer DRAM types = {drams}")
    print(f"    chiplet archs used= {archs}")
    print(f"    tp degrees used   = {tps}")
    print(f"    PIM in solution?  = {'PIM' in archs}")
    print()
