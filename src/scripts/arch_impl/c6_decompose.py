#!/usr/bin/env python3
"""
C6 decomposition: against a FIXED homogeneous single design (one arch+glb+pe for all
ops, the paper's homo-ASIC point), how much of the per-op energy reduction comes from
  (a) dataflow specialization  : per-op free choice of arch, size fixed at homo's
  (b) size specialization      : per-op free choice of glb/pe, arch fixed at homo's
  (c) joint                    : per-op free choice of arch+glb+pe
All at GDDR7, tp=1, fused=single (memory + TP + fusion are separate levers).

energy per op = dynamic + static_power*latency at that config (framework notion).
"""
import os, pandas as pd, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
COLS=("net,layer_name,batch_size,sequence_length,mapper_idx,fused_layer_type,tp_degree,"
      "arch_target,glb_scale,pe_x_scale,pe_y_scale,dram_i,dram_o,latency,static_power,"
      "dynamic_energy,area,utilization,i_access,w_access,o_access").split(",")
DF={"eyeriss_like":"RS","gemmini_like":"OS","simba_like":"WS"}

def load():
    return pd.read_csv(os.path.join(HERE,"c6_data","targets.csv"),names=COLS)

def Etot(s):  # energy column for a slice
    return s.dynamic_energy + s.static_power*s.latency

def net_decompose(df, net, batch):
    q=df[(df.net==net)&(df.batch_size==batch)&(df.tp_degree==1)&
         (df.fused_layer_type=="single")&(df.dram_i=="GDDR7")&(df.dram_o=="GDDR7")&
         (df.arch_target.isin(DF))].copy()
    if q.empty: return None
    q["E"]=Etot(q)
    ops=sorted(q.layer_name.unique())
    # design key = (arch,glb,pe_x,pe_y); energy of op on a given design = min over mapper (unique)
    # build per-op dict: design -> E
    opE={}
    designs=set()
    for L in ops:
        s=q[q.layer_name==L]
        d={}
        for _,r in s.iterrows():
            key=(r.arch_target,int(r.glb_scale),int(r.pe_x_scale),int(r.pe_y_scale))
            e=float(r.E)
            if key not in d or e<d[key]: d[key]=e
            designs.add(key)
        opE[L]=d
    # homo: pick single design minimizing total over ops that SUPPORT it (require support by all ops)
    common=[k for k in designs if all(k in opE[L] for L in ops)]
    def total(design): return sum(opE[L][design] for L in ops)
    homo_design=min(common,key=total); homo=total(homo_design)
    homo_arch,homo_glb,homo_px,homo_py=homo_design
    # joint het
    het=sum(min(opE[L].values()) for L in ops)
    # df-only: arch free, size fixed to homo's (glb,pe). fall back: if op lacks that size on an arch, use op-min on that size across archs available
    def e_size_fixed(L):
        cands=[opE[L][(a,homo_glb,homo_px,homo_py)] for a in DF if (a,homo_glb,homo_px,homo_py) in opE[L]]
        return min(cands) if cands else min(opE[L].values())
    df_only=sum(e_size_fixed(L) for L in ops)
    # size-only: glb/pe free, arch fixed to homo's arch
    def e_arch_fixed(L):
        cands=[opE[L][k] for k in opE[L] if k[0]==homo_arch]
        return min(cands) if cands else min(opE[L].values())
    size_only=sum(e_arch_fixed(L) for L in ops)
    return {
        "net":net,"batch":batch,"ops":ops,"opE":opE,
        "homo_design":homo_design,"homo":homo,"het":het,
        "df_only":df_only,"size_only":size_only,
        "r_joint":100*(1-het/homo),"r_df":100*(1-df_only/homo),"r_size":100*(1-size_only/homo),
    }

def main():
    df=load()
    targets=[("qwen3_30b_a3b_prefill_s1024",1),("qwen3_30b_a3b_prefill_s1024",8),
             ("llama3.1_8b_prefill_s1024",1),
             ("mobilenet_v3_small",1),("mobilenet_v3_small",8),
             ("replknet31b",1),("replknet31b",8),
             ("qwen3_30b_a3b_decode_kv1024",1)]
    print(f"{'net':>34s} b | {'homo design':>22s} | {'df-only':>8s} {'size-only':>9s} {'joint':>7s}")
    print("-"*92)
    rows=[]
    for net,b in targets:
        r=net_decompose(df,net,b)
        if r is None: print(f"{net} b{b}: no data"); continue
        a,g,px,py=r["homo_design"]
        hd=f"{DF[a]}@glb{g}@pe{px}x{py}"
        print(f"{net:>34s} {b} | {hd:>22s} | {r['r_df']:7.1f}% {r['r_size']:8.1f}% {r['r_joint']:6.1f}%")
        rows.append(r)
    print("\nInterpretation: 'joint' = per-op pick arch+glb+pe vs one fixed homo design (per-net).")
    print("df-only isolates dataflow; size-only isolates PE-array/buffer dimensioning.")
    # detailed per-op for qwen prefill b1
    r=net_decompose(df,"qwen3_30b_a3b_prefill_s1024",1)
    print("\n=== Qwen3-30B MoE prefill b1: per-op assigned design (joint het) vs homo design ===")
    a,g,px,py=r["homo_design"]; homo_design=r["homo_design"]
    print(f"homo (single) design = {DF[a]}@glb{g}@pe{px}x{py}")
    print(f"{'operator':>16s} | {'het design (arch@size)':>24s} | {'E_het':>10s} {'E_homo':>10s} | {'op_red':>7s}")
    for L in r["ops"]:
        d=r["opE"][L]
        kbest=min(d,key=d.get)
        eb=d[kbest]; eh=d.get(homo_design,max(d.values()))
        ka=f"{DF[kbest[0]]}@glb{kbest[1]}@pe{kbest[2]}x{kbest[3]}"
        print(f"{L:>16s} | {ka:>24s} | {eb:10.3e} {eh:10.3e} | {100*(1-eb/eh):6.1f}%")

if __name__=="__main__":
    main()
