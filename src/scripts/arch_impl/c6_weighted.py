#!/usr/bin/env python3
"""
C6 weighted op-level decomposition for Qwen3-30B-A3B prefill.
Weights each operator by its true instance count in the full network (NETWORK.yaml):
  per decoder layer (x48): q,k,v,attn_qk,softmax,attn_v,o,router = 1 each;
                           expert_gate_proj = 128 experts x 2 (gate+up); expert_down = 128.
  global: lm_head x1.
DB energy is per-instance (verified: expert_down ops = 64*768*2048 = 1 expert x 64 tokens).

Compares (all GDDR7, tp=1, fused=single; energy = dynamic + static*latency):
  homo  : single best (arch,glb,pe) design for the whole net
  joint : per-op best (arch,glb,pe)              -> operator-level specialization
  df    : per-op best arch, size fixed to homo's -> dataflow alone
  size  : per-op best glb/pe, arch fixed to homo -> PE-array/buffer sizing alone
Softmax/vector ops run on simple_vector in BOTH (common term; included in net total).
"""
import os, pandas as pd, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
COLS=("net,layer_name,batch_size,sequence_length,mapper_idx,fused_layer_type,tp_degree,"
      "arch_target,glb_scale,pe_x_scale,pe_y_scale,dram_i,dram_o,latency,static_power,"
      "dynamic_energy,area,utilization,i_access,w_access,o_access").split(",")
DF={"eyeriss_like":"RS","gemmini_like":"OS","simba_like":"WS"}
NLAYER=48
WEIGHT={  # full-network instance count
 "layer0_q_proj":NLAYER,"layer0_k_proj":NLAYER,"layer0_v_proj":NLAYER,
 "layer0_attn_qk":NLAYER,"layer0_attn_v":NLAYER,"layer0_o_proj":NLAYER,"router":NLAYER,
 "expert_gate_proj":NLAYER*128*2,"expert_down_proj":NLAYER*128,"lm_head":1,
}
VECTOR_W={"layer0_softmax":NLAYER}
OPLABEL={"layer0_q_proj":"Q proj","layer0_k_proj":"K proj","layer0_v_proj":"V proj",
 "layer0_o_proj":"O proj","layer0_attn_qk":"Attn QK^T","layer0_attn_v":"Attn AV",
 "router":"MoE router","expert_gate_proj":"Expert gate/up","expert_down_proj":"Expert down",
 "lm_head":"LM head","layer0_softmax":"Softmax"}

def main():
    df=pd.read_csv(os.path.join(HERE,"c6_data","targets.csv"),names=COLS)
    net,batch="qwen3_30b_a3b_prefill_s1024",1
    q=df[(df.net==net)&(df.batch_size==batch)&(df.tp_degree==1)&(df.fused_layer_type=="single")&
         (df.dram_i=="GDDR7")&(df.dram_o=="GDDR7")].copy()
    q["E"]=q.dynamic_energy+q.static_power*q.latency
    gemm=q[q.arch_target.isin(DF)]
    # per-op per-design energy
    opE={}; designs=set()
    for L in WEIGHT:
        s=gemm[gemm.layer_name==L]; d={}
        for _,r in s.iterrows():
            k=(r.arch_target,int(r.glb_scale),int(r.pe_x_scale),int(r.pe_y_scale))
            if k not in d or r.E<d[k]: d[k]=float(r.E)
            designs.add(k)
        opE[L]=d
    common=[k for k in designs if all(k in opE[L] for L in WEIGHT)]
    pernet=min(common,key=lambda k:sum(WEIGHT[L]*opE[L][k] for L in WEIGHT))
    GEMINI=("gemmini_like",1,2,3)   # actual energy Gemini-style all-net design (competing_full_summary)
    import sys
    homo = GEMINI if (GEMINI in common) else pernet
    print(f"[per-net best single design = {DF[pernet[0]]}@glb{pernet[1]}@pe{pernet[2]}x{pernet[3]}]")
    print(f"[Gemini all-net design (energy) = {DF[GEMINI[0]]}@glb{GEMINI[1]}@pe{GEMINI[2]}x{GEMINI[3]}; "
          f"in common-set={GEMINI in common}]")
    ha,hg,hpx,hpy=homo
    # vector term (common to both sides)
    vec=0.0
    for L,w in VECTOR_W.items():
        sv=q[(q.layer_name==L)&(q.arch_target=="simple_vector")]
        if not sv.empty: vec+=w*float((sv.dynamic_energy+sv.static_power*sv.latency).min())
    def E_joint(L): return min(opE[L].values())
    def E_df(L):
        c=[opE[L][(a,hg,hpx,hpy)] for a in DF if (a,hg,hpx,hpy) in opE[L]]
        return min(c) if c else min(opE[L].values())
    def E_size(L):
        c=[opE[L][k] for k in opE[L] if k[0]==ha]
        return min(c) if c else min(opE[L].values())
    def E_homo(L): return opE[L][homo]
    tot=lambda f: sum(WEIGHT[L]*f(L) for L in WEIGHT)+vec
    Th,Tj,Tdf,Tsz=tot(E_homo),tot(E_joint),tot(E_df),tot(E_size)
    print(f"Qwen3-30B-A3B prefill b1 | homo single design = {DF[ha]}@glb{hg}@pe{hpx}x{hpy}")
    print(f"  vector(softmax) common term = {vec:.4e}  ({100*vec/Th:.1f}% of homo net)")
    print(f"  net energy (arb units, GDDR7/tp1/single):  homo={Th:.4e}")
    print(f"  reduction vs homo:  joint(arch+size)={100*(1-Tj/Th):5.1f}%   "
          f"dataflow-only={100*(1-Tdf/Th):5.1f}%   size-only={100*(1-Tsz/Th):5.1f}%")
    # weighted contribution + per-op assignment
    print(f"\n  {'operator':>14s} | {'wt':>6s} | {'homo E':>10s} | {'het design':>16s} | {'het E':>10s} | {'op_red':>6s} | {'%net':>5s}")
    contrib=[]
    for L in WEIGHT:
        eh=WEIGHT[L]*E_homo(L); ej=WEIGHT[L]*E_joint(L)
        kbest=min(opE[L],key=opE[L].get)
        contrib.append((L,eh,ej,kbest))
    for L,eh,ej,kbest in sorted(contrib,key=lambda x:-x[1]):
        ka=f"{DF[kbest[0]]}@glb{kbest[1]}@pe{kbest[2]}x{kbest[3]}"
        opred=100*(1-ej/eh) if eh>0 else 0
        print(f"  {OPLABEL[L]:>14s} | {WEIGHT[L]:6d} | {eh:10.3e} | {ka:>16s} | {ej:10.3e} | {opred:5.1f}% | {100*eh/Th:4.1f}%")
    print(f"  {'Softmax(vec)':>14s} | {NLAYER:6d} | {vec:10.3e} | {'simple_vector':>16s} | {vec:10.3e} |   0.0% | {100*vec/Th:4.1f}%")

if __name__=="__main__": main()
