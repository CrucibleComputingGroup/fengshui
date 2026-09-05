#!/usr/bin/env python3
"""Fill the ONE gap: simba_like has no valid Timeloop mapping for the
qwen3_30b_a3b decode_kv1024 projections at batch 64 (B=64,N=1 — deterministic
empty mapspace; eyeriss/gemmini map it fine). simba_like is in the PERF pool,
so a MISSING row would make cal_perf silently skip the layer (0 cost) and the
GA could wrongly select simba. Instead we emit simba b64 rows marked INFEASIBLE
(latency/energy = inf), so the GA correctly rejects simba there — mirroring how
cal_perf marks PIM+non-GDDR7 infeasible.

Built by copying simba_like decode_kv1024 batch-16 rows (full structure: all
glb/pe/tp/fused/dram combos) and relabeling batch->64 with inf cost.

Run from chiplet_timeloop/timeloop_experiments/ (env: mozart).
"""
import argparse
import csv

DB = "unified_database_backup.csv"
OUT = "new_rows_decode_simba_inf.csv"
NET = "qwen3_30b_a3b_decode_kv1024"
OPS = {"layer0_q_proj", "layer0_k_proj", "layer0_v_proj", "layer0_o_proj",
       "lm_head", "router"}
LLAMA_OPS = {"layer0_q_proj", "layer0_k_proj", "layer0_v_proj", "layer0_o_proj",
             "layer0_gate_proj", "layer0_up_proj", "layer0_down_proj"}
SRC_BATCH, NEW_BATCH = "16", "64"

_ap = argparse.ArgumentParser()
_ap.add_argument("--net", default=NET)
_ap.add_argument("--ops", nargs="+", default=None,
                 help="op names; 'llama' = the 7 llama projections")
_ap.add_argument("--out", default=OUT)
_args = _ap.parse_args()
NET = _args.net
OPS = LLAMA_OPS if _args.ops == ["llama"] else (set(_args.ops) if _args.ops else OPS)
OUT = _args.out

C_NET, C_LAYER, C_BATCH, C_ARCH = 0, 1, 2, 7
C_LAT, C_DYN, C_UTIL = 13, 15, 17   # latency, dynamic_energy, utilization


def main():
    written = 0
    per_op = {}
    with open(DB, newline="") as fin, open(OUT, "w", newline="") as fout:
        r = csv.reader(fin)
        w = csv.writer(fout)
        header = next(r)
        w.writerow(header)
        assert header[C_LAT] == "latency" and header[C_DYN] == "dynamic_energy" \
            and header[C_ARCH] == "arch_target", header
        for row in r:
            if (row[C_NET] == NET and row[C_ARCH] == "simba_like"
                    and row[C_BATCH] == SRC_BATCH and row[C_LAYER] in OPS):
                out = list(row)
                out[C_BATCH] = NEW_BATCH
                out[C_LAT] = "inf"
                out[C_DYN] = "inf"
                out[C_UTIL] = "0"
                w.writerow(out)
                written += 1
                per_op[row[C_LAYER]] = per_op.get(row[C_LAYER], 0) + 1
    print(f"Wrote {written} simba b64 INF rows -> {OUT}")
    for k in sorted(per_op):
        print(f"  {k:<16} {per_op[k]}")


if __name__ == "__main__":
    main()
