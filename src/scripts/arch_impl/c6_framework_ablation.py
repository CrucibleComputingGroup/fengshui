#!/usr/bin/env python3
"""
C6 authoritative isolation (framework, full energy model incl. fusion/het-batch/operating-point).

Isolates *dataflow* specialization at the network level by taking the real co-designed N=8
energy pool and HOMOGENIZING the compute dataflow: every compute chiplet is forced to one
dataflow (keeping its glb/pe size, PIM, switch, and memory). All variants run through the same
DAG-CP + het-batch evaluation.

  full        : real N=8 pool (heterogeneous dataflows)            [het memory]
  full_gddr7  : real N=8 pool, memory homogenized to GDDR7         [isolates from memory]
  df=OS/WS/RS : all compute chiplets -> one dataflow, GDDR7        [size+PIM+switch kept]
  single      : Gemini-style single design OS@glb1@pe2x3, GDDR7    [paper homo baseline]

dataflow specialization = (best df-homogenized pool  -  full_gddr7) / best df-homogenized pool
"""
import os, sys, copy, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chiplet_dataclass import ChipletConfig, create_switch_chiplet
from network_dataclass import VirtualNetwork
from global_parameter import NET_DIR
from cal_perf_phy_net import preload_database
from utility_functions import calculate_average_opt_value
import competing_baselines as cb

ENERGY_POOL = [
    "gemmini_like@glb1@pe_x_scale1@pe_y_scale3", "eyeriss_like@glb1@pe_x_scale4@pe_y_scale2",
    "gemmini_like@glb1@pe_x_scale2@pe_y_scale4", "simba_like@glb4@pe_x_scale3@pe_y_scale4",
    "PIM@glb1@pe_x_scale1@pe_y_scale1", "simba_like@glb4@pe_x_scale3@pe_y_scale3",
    "switch_8port@glb1@pe_x_scale3@pe_y_scale3", "simba_like@glb1@pe_x_scale4@pe_y_scale3",
]
FIXED = ("PIM", "switch_8port")

def build_nets(specs, database):
    import pandas as pd
    vns = [VirtualNetwork(n, batch_size=b, sequence_length=s) for (n, b, s) in specs]
    db = pd.read_csv(database)
    layers = {net: set(db[db.net == net].layer_name.unique()) for net in db.net.unique()}
    out = []
    for vn in vns:
        vn.load_from_dir(os.path.join(NET_DIR, vn.network_name),
                         db_layers=layers.get(vn.network_name))
        if len(vn.layers): out.append(vn)
    preload_database(database, needed_nets={vn.network_name for vn in out})
    return out

def pool_from_idents(idents):
    return [ChipletConfig.from_identifier(s) for s in idents]

def homogenize_dataflow(pool, arch):
    out = []
    for c in pool:
        if getattr(c, "arch_target", "") in FIXED:
            out.append(copy.deepcopy(c)); continue
        out.append(ChipletConfig(arch, c.global_buffer_size_scale, c.pe_x_scale, c.pe_y_scale,
                                 dram_type=getattr(c, "dram_type", "GDDR7")))
    return out

def evalp(vns, pool, database, gddr7):
    if gddr7:
        with cb.homogenize_memory("GDDR7"):
            avg, per = cb.eval_pool(vns, pool, "energy", False, True, 1, database)
    else:
        avg, per = cb.eval_pool(vns, pool, "energy", False, True, 1, database)
    return per

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--database", default="../unified_database.csv")
    a = ap.parse_args()
    specs = [("qwen3_30b_a3b_prefill_s1024", 1, 1024),
             ("qwen3_30b_a3b_prefill_s1024", 8, 1024),
             ("mobilenet_v3_small", 1, 1), ("mobilenet_v3_small", 8, 1),
             ("replknet31b", 1, 1), ("replknet31b", 8, 1)]
    vns = build_nets(specs, a.database)
    full = pool_from_idents(ENERGY_POOL)
    variants = {
        "full(het-mem)": (full, False),
        "full(GDDR7)":   (full, True),
        "df=OS(GDDR7)":  (homogenize_dataflow(full, "gemmini_like"), True),
        "df=WS(GDDR7)":  (homogenize_dataflow(full, "simba_like"), True),
        "df=RS(GDDR7)":  (homogenize_dataflow(full, "eyeriss_like"), True),
        "single OS@glb1pe2x3(GDDR7)":
            ([ChipletConfig("gemmini_like", 1, 2, 3, dram_type="GDDR7")], True),
    }
    res = {}
    for name, (pool, gddr7) in variants.items():
        print(f"\n>>> {name}: {len(pool)} chiplets", flush=True)
        res[name] = evalp(vns, pool, a.database, gddr7)
    names = [vn.get_unique_name() for vn in vns]
    print("\n" + "=" * 120)
    hdr = f"{'variant':>30s} | " + " | ".join(f"{n.split('_seq')[0][-18:]:>18s}" for n in names)
    print(hdr); print("-" * len(hdr))
    for name in variants:
        print(f"{name:>30s} | " + " | ".join(f"{res[name].get(n, float('nan')):18.4e}" for n in names))
    print("\n=== dataflow specialization (full_gddr7 vs best single-dataflow pool, GDDR7) ===")
    for n in names:
        fg = res["full(GDDR7)"][n]
        dfh = min(res[f"df={a}(GDDR7)"][n] for a in ("OS", "WS", "RS"))
        sgl = res["single OS@glb1pe2x3(GDDR7)"][n]
        bestdf = min((("OS", res["df=OS(GDDR7)"][n]), ("WS", res["df=WS(GDDR7)"][n]),
                      ("RS", res["df=RS(GDDR7)"][n])), key=lambda x: x[1])
        print(f"  {n:>42s}: full={fg:.3e}  best-1df({bestdf[0]})={dfh:.3e}  "
              f"-> dataflow {100*(1-fg/dfh):5.1f}%  | vs single-design {100*(1-fg/sgl):5.1f}%")

if __name__ == "__main__":
    main()
