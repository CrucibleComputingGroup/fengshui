#!/usr/bin/env python3
"""
C6 Q1 table: for each Qwen3-30B-A3B MoE prefill operator, which of the ACTUAL N=8
energy-pool chiplets does it map to? (framework per-group rule = min energy.)
GDDR7, tp=1, fused=single; energy = dynamic + static*latency.
PIM/switch are not compute targets for these ops; vector ops -> simple_vector.
"""
import os, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__))
COLS=("net,layer_name,batch_size,sequence_length,mapper_idx,fused_layer_type,tp_degree,"
      "arch_target,glb_scale,pe_x_scale,pe_y_scale,dram_i,dram_o,latency,static_power,"
      "dynamic_energy,area,utilization,i_access,w_access,o_access").split(",")
DF={"eyeriss_like":"RS","gemmini_like":"OS","simba_like":"WS"}
# 6 compute chiplets of the v7 energy N=8 pool (corrected DB, 2026-06-15): (arch,glb,pe_x,pe_y)
# (excludes PIM + switch_8port, which are not compute targets for these ops)
POOL=[("gemmini_like",1,2,4),("gemmini_like",1,1,1),("gemmini_like",1,4,2),
      ("simba_like",1,3,4),("simba_like",1,4,3),("eyeriss_like",1,4,2)]
ORDER=["layer0_q_proj","layer0_k_proj","layer0_v_proj","layer0_attn_qk","layer0_softmax",
       "layer0_attn_v","layer0_o_proj","router","expert_gate_proj","expert_down_proj","lm_head"]
LAB={"layer0_q_proj":"Q proj","layer0_k_proj":"K proj","layer0_v_proj":"V proj",
     "layer0_attn_qk":"Attn QK^T","layer0_softmax":"Softmax","layer0_attn_v":"Attn AV",
     "layer0_o_proj":"O proj","router":"MoE router","expert_gate_proj":"Expert gate/up",
     "expert_down_proj":"Expert down","lm_head":"LM head"}
ROLE={"layer0_q_proj":"attn proj (GQA Q)","layer0_k_proj":"attn proj (GQA K)",
      "layer0_v_proj":"attn proj (GQA V)","layer0_attn_qk":"attention score","layer0_softmax":"normalization",
      "layer0_attn_v":"attention context","layer0_o_proj":"attn out proj","router":"MoE gate",
      "expert_gate_proj":"MoE FFN (SwiGLU)","expert_down_proj":"MoE FFN down","lm_head":"vocab proj"}

def main():
    df=pd.read_csv(os.path.join(HERE,"c6_data","targets.csv"),names=COLS)
    q=df[(df.net=="qwen3_30b_a3b_prefill_s1024")&(df.batch_size==1)&(df.tp_degree==1)&
         (df.fused_layer_type=="single")&(df.dram_i=="GDDR7")&(df.dram_o=="GDDR7")].copy()
    q["E"]=q.dynamic_energy+q.static_power*q.latency
    print(f"{'operator':>14s} | {'role':>18s} | {'assigned pool chiplet':>22s} | {'2nd-best df':>11s} | {'2nd penalty':>11s}")
    print("-"*92)
    for L in ORDER:
        s=q[q.layer_name==L]
        if L=="layer0_softmax":
            print(f"{LAB[L]:>14s} | {ROLE[L]:>18s} | {'vector unit':>22s} | {'—':>11s} | {'—':>11s}")
            continue
        # per pool chiplet min energy
        best=None; perdf={}
        for (a,g,px,py) in POOL:
            sub=s[(s.arch_target==a)&(s.glb_scale==g)&(s.pe_x_scale==px)&(s.pe_y_scale==py)]
            if sub.empty: continue
            e=float(sub.E.min())
            if best is None or e<best[1]: best=((a,g,px,py),e)
            perdf.setdefault(a,1e99); perdf[a]=min(perdf[a],e)
        a,g,px,py=best[0]
        chip=f"{DF[a]} (gemmini/eyeriss/simba)".split()[0]+f"@glb{g}@pe{px}x{py}"
        chip=f"{DF[a]}@glb{g}@pe{px}x{py}"
        # 2nd best dataflow penalty
        others=sorted([(DF[k],v) for k,v in perdf.items() if k!=a],key=lambda x:x[1])
        if others:
            second=others[0]; pen=f"{second[1]/best[1]:.2f}x"
            print(f"{LAB[L]:>14s} | {ROLE[L]:>18s} | {chip:>22s} | {second[0]:>11s} | {pen:>11s}")
        else:
            print(f"{LAB[L]:>14s} | {ROLE[L]:>18s} | {chip:>22s} | {'—':>11s} | {'—':>11s}")

if __name__=="__main__": main()
