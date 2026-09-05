#!/usr/bin/env python3
"""
C6 rebuttal: op-by-op dataflow-specialization walkthrough.

Backs the paper claim that prefill/CNN energy reductions (\prefillEnergyRed%,
\cnnEnergyRed%) come from *dataflow-level specialization* by showing, per operator,
which compute dataflow (RS=eyeriss / OS=gemmini / WS=simba) minimizes energy, and
that the per-op winners differ -> a single homogeneous dataflow is suboptimal.

Isolation of the dataflow axis (everything else held to the homogeneous point):
  * memory fixed to GDDR7 (het-memory is a *separate* lever; the C3 ablation shows
    it is only ~1.04x on energy, so it barely touches the prefill/CNN energy story)
  * tp_degree = 1 (tensor parallelism is energy-neutral, ~1.02x in the C3 ablation)
  * fused_layer_type = 'single' (each op evaluated standalone; fusion is a separate
    lever, the CNN no-fusion ablation)
Per-op energy uses the framework's notion (dynamic + static_power*latency at the
op's own min-energy operating point); we verify the winner is identical under
dynamic-only energy.

Reads the pre-filtered slice scripts/arch_impl/c6_data/targets.csv (built by the
one-pass awk extraction over unified_database.csv).
"""
import os, sys, json
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "c6_data", "targets.csv")
COLS = ("net,layer_name,batch_size,sequence_length,mapper_idx,fused_layer_type,"
        "tp_degree,arch_target,glb_scale,pe_x_scale,pe_y_scale,dram_i,dram_o,"
        "latency,static_power,dynamic_energy,area,utilization,i_access,w_access,"
        "o_access").split(",")

# compute dataflows (exclude simple_vector / PIM — not GEMM-dataflow choices)
DF = {"eyeriss_like": "RS", "gemmini_like": "OS", "simba_like": "WS"}
DF_LONG = {"eyeriss_like": "RS (Eyeriss)", "gemmini_like": "OS (Gemmini)",
           "simba_like": "WS (Simba)"}

# friendly operator labels
OP_LABEL = {
    "layer0_q_proj": "Q proj", "layer0_k_proj": "K proj", "layer0_v_proj": "V proj",
    "layer0_o_proj": "O proj", "layer0_attn_qk": "Attn QK^T", "layer0_attn_v": "Attn ·V",
    "layer0_softmax": "Softmax", "expert_gate_proj": "Expert gate/up", "expert_down_proj": "Expert down",
    "router": "MoE router", "lm_head": "LM head",
}

def op_energy_by_dataflow(sub):
    """For one (net, layer) slice already filtered to GDDR7/tp1/single, return
    {arch_target: (min_total_energy, min_dyn_energy, best_row)} per dataflow."""
    out = {}
    for arch in DF:
        s = sub[sub.arch_target == arch]
        if s.empty:
            continue
        total = s.dynamic_energy + s.static_power * s.latency
        i_tot = total.idxmin()
        i_dyn = s.dynamic_energy.idxmin()
        out[arch] = {
            "E_total": float(total.loc[i_tot]),
            "E_dyn":   float(s.dynamic_energy.loc[i_dyn]),
            "row":     s.loc[i_tot],
        }
    return out

def analyze(df, net, batch, metric="E_total"):
    q = df[(df.net == net) & (df.batch_size == batch) &
           (df.tp_degree == 1) & (df.fused_layer_type == "single") &
           (df.dram_i == "GDDR7") & (df.dram_o == "GDDR7")].copy()
    if q.empty:
        return None
    layers = sorted(q.layer_name.unique())
    rows = []
    het = 0.0           # sum of per-op best dataflow
    homo_tot = {a: 0.0 for a in DF}   # sum forcing all ops onto dataflow a
    vector_e = 0.0      # softmax / non-dataflow ops (run on shared vector unit both sides)
    for L in layers:
        sub = q[q.layer_name == L]
        archs = set(sub.arch_target.unique())
        de = op_energy_by_dataflow(sub)
        if not de:  # op only exists on simple_vector / PIM (e.g. softmax)
            sv = sub[sub.arch_target == "simple_vector"]
            if not sv.empty:
                e = float((sv.dynamic_energy + sv.static_power * sv.latency).min())
                vector_e += e
                rows.append({"op": L, "kind": "vector", "winner": "vector",
                             "E": e, "by_df": {}})
            continue
        by = {a: de[a][metric] for a in de}
        winner = min(by, key=by.get)
        het += by[winner]
        for a in DF:
            # if a dataflow is missing for this op, fall back to the op's max (penalize)
            homo_tot[a] += by.get(a, max(by.values()))
        best_row = de[winner]["row"]
        rows.append({"op": L, "kind": "gemm", "winner": winner,
                     "E": by[winner], "by_df": by,
                     "cfg": f"{DF[winner]}@glb{int(best_row.glb_scale)}"
                            f"@pe{int(best_row.pe_x_scale)}x{int(best_row.pe_y_scale)}"})
    # homogeneous baseline = best single dataflow (each op still picks best size within it)
    homo_best_arch = min(homo_tot, key=homo_tot.get)
    homo_best = homo_tot[homo_best_arch]
    # net-level totals INCLUDING shared vector ops (same on both sides)
    het_net = het + vector_e
    homo_net = homo_best + vector_e
    red_gemm = 100.0 * (homo_best - het) / homo_best
    red_net  = 100.0 * (homo_net - het_net) / homo_net
    return {
        "net": net, "batch": batch, "rows": rows,
        "het_gemm": het, "homo_tot": homo_tot,
        "homo_best_arch": homo_best_arch, "homo_best": homo_best,
        "vector_e": vector_e,
        "het_net": het_net, "homo_net": homo_net,
        "red_gemm": red_gemm, "red_net": red_net,
    }

def fmt(x):
    return f"{x:.3e}"

def main():
    df = pd.read_csv(DATA, names=COLS)
    nets = [
        ("qwen3_30b_a3b_prefill_s1024", 1),
        ("qwen3_30b_a3b_decode_kv1024", 1),
        ("llama3.1_8b_prefill_s1024", 1),
        ("mobilenet_v3_small", 1),
        ("replknet31b", 1),
    ]
    summary = []
    for net, b in nets:
        r = analyze(df, net, b)
        if r is None:
            print(f"\n### {net} b{b}: NO DATA"); continue
        print("\n" + "=" * 100)
        print(f"### {net}  (batch={b})   [GDDR7, tp=1, fused=single, energy=dyn+static*lat]")
        print("=" * 100)
        hdr = f"{'operator':>16s} | {'winner':>11s} | " + " | ".join(f"{DF[a]:>10s}" for a in DF) + " | penalty(homo)"
        print(hdr)
        print("-" * len(hdr))
        for row in r["rows"]:
            if row["kind"] == "vector":
                print(f"{OP_LABEL.get(row['op'],row['op']):>16s} | {'vector':>11s} | "
                      + " | ".join(f"{'—':>10s}" for _ in DF) + f" |  (shared, E={fmt(row['E'])})")
                continue
            by = row["by_df"]
            cells = []
            for a in DF:
                v = by.get(a, float('nan'))
                mark = "*" if a == row["winner"] else " "
                cells.append(f"{fmt(v)}{mark}")
            # penalty = homo_best_arch energy / winner energy for this op
            pen = by.get(r["homo_best_arch"], max(by.values())) / by[row["winner"]]
            print(f"{OP_LABEL.get(row['op'],row['op']):>16s} | {DF_LONG[row['winner']]:>11s} | "
                  + " | ".join(f"{c:>10s}" for c in cells) + f" | {pen:5.2f}x")
        print("-" * len(hdr))
        print(f"  homogeneous best single dataflow = {DF_LONG[r['homo_best_arch']]}")
        print(f"  GEMM-only:  het={fmt(r['het_gemm'])}  homo={fmt(r['homo_best'])}  -> reduction {r['red_gemm']:.1f}%")
        print(f"  net-level (incl shared vector ops): het={fmt(r['het_net'])}  homo={fmt(r['homo_net'])}  -> reduction {r['red_net']:.1f}%")
        print(f"  homo totals by dataflow: " + ", ".join(f"{DF[a]}={fmt(r['homo_tot'][a])}" for a in DF))
        summary.append(r)

    print("\n" + "#" * 100)
    print("SUMMARY: energy reduction from dataflow specialization (het per-op df vs best single df)")
    print("#" * 100)
    print(f"{'net':>34s} | {'GEMM-only':>10s} | {'net-level':>10s} | {'homo df':>10s}")
    for r in summary:
        print(f"{r['net']:>34s} | {r['red_gemm']:9.1f}% | {r['red_net']:9.1f}% | {DF[r['homo_best_arch']]:>10s}")

if __name__ == "__main__":
    main()
