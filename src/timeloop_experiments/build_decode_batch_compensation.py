#!/usr/bin/env python3
"""Stage B (stage-2): parse the NEW decode_kv1024 batch 32/64 projection
Timeloop outputs and emit DB-schema rows to a SMALL CSV (new_rows_decode.csv).

Same pipeline as build_unified_database / build_vit_compensation:
  discover (scoped) -> process_mapping_results -> gen_multi_fusion ->
  post_process_mapping_results -> analytical BW throttling (16 DRAM combos).

Scoped strictly to (net=qwen3_30b_a3b_decode_kv1024, batch in {32,64},
op in the 6 literal-batch projections) so NOTHING existing is touched.
A separate assembly step concatenates backup + prefill-copy + this file.

Run INSIDE Docker my_timeloop_env_v2 after run_decode_batch32_64.py:
  cd /workspace/chiplet_timeloop/timeloop_experiments
  python3 build_decode_batch_compensation.py --dry-run
  python3 build_decode_batch_compensation.py
"""
import argparse
import csv
import os
import re
import sys
import time
from collections import defaultdict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _c in [os.path.join(_THIS_DIR, "..", "scripts"), os.path.join(_THIS_DIR, "..")]:
    if os.path.isfile(os.path.join(_c, "utility_functions.py")) and _c not in sys.path:
        sys.path.insert(0, _c)

from parse_stats import process_mapping_results, post_process_mapping_results, gen_multi_fusion
from postprocess_bw import apply_bw_throttling_analytical, TARGET_DRAMS

NET = "qwen3_30b_a3b_decode_kv1024"
BATCHES = {"32", "64"}
OPS = {"layer0_q_proj", "layer0_k_proj", "layer0_v_proj", "layer0_o_proj",
       "lm_head", "router"}
LLAMA_OPS = {"layer0_q_proj", "layer0_k_proj", "layer0_v_proj", "layer0_o_proj",
             "layer0_gate_proj", "layer0_up_proj", "layer0_down_proj"}
VALID_ARCHS = ("eyeriss_like", "simba_like", "gemmini_like")
DECODE_SEQ = 1   # existing decode rows use sequence_length=1


def discover(outputs_dir, target_net=NET, ops=OPS):
    """Discover ONLY the new decode 32/64 projection configs."""
    configs = []
    net_dir = os.path.join(outputs_dir, target_net)
    if not os.path.isdir(net_dir):
        return configs
    for root, dirs, files in os.walk(net_dir):
        if "timeloop-mapper.stats.txt" not in files:
            continue
        stats_path = os.path.join(root, "timeloop-mapper.stats.txt")
        if os.stat(stats_path).st_size == 0:
            continue
        rel = os.path.relpath(root, outputs_dir)
        parts = rel.split(os.sep)
        if len(parts) < 10:
            continue
        net, layer, batch, seq = parts[0], parts[1], parts[2], parts[3]
        mapper, fused, otc, tp = parts[4], parts[5], parts[6], parts[7]
        arch_config, dram = parts[8], parts[9]
        if net != target_net or batch not in BATCHES or layer not in ops:
            continue
        m = re.match(r"arch=(\w+)@glb_scale=(\d+)@pe_x_scale=(\d+)@pe_y_scale=(\d+)",
                     arch_config)
        if not m:
            continue
        arch, glb, px, py = m.group(1), m.group(2), m.group(3), m.group(4)
        if arch not in VALID_ARCHS:
            continue
        dram_parts = dram.split("@")
        if len(dram_parts) != 2:
            continue
        config_id = "%s@glb%s@pe_x_scale%s@pe_y_scale%s@%s@%s" % (
            arch, glb, px, py, dram_parts[0], dram_parts[1])
        problem_id = "%s@%s@%s@%s@%s@%s@%s@%s" % (
            net, layer, batch, seq, mapper, fused, otc, tp)
        configs.append((config_id, problem_id))
    return configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", default=os.path.join(_THIS_DIR, "outputs"))
    ap.add_argument("--backup-csv", default=os.path.join(_THIS_DIR, "unified_database_backup.csv"),
                    help="only used to read the exact DB header/fieldnames")
    ap.add_argument("--out", default=os.path.join(_THIS_DIR, "new_rows_decode.csv"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--net", default=NET, help="decode net to discover")
    ap.add_argument("--ops", nargs="+", default=None,
                    help="op names; 'llama' = the 7 llama projections")
    args = ap.parse_args()
    ops = LLAMA_OPS if args.ops == ["llama"] else (set(args.ops) if args.ops else OPS)

    print("Discovering NEW decode 32/64 projection configs in %s ..." % args.outputs_dir)
    t0 = time.time()
    configs = discover(args.outputs_dir, target_net=args.net, ops=ops)
    print("Found %d configs (%.1fs)" % (len(configs), time.time() - t0))
    lc = defaultdict(int)
    for _, pid in configs:
        p = pid.split("@")
        lc[(p[1], p[2])] += 1
    for (layer, batch), c in sorted(lc.items()):
        print("  %-16s b=%-3s %d configs" % (layer, batch, c))
    if not configs:
        print("No configs found — did run_decode_batch32_64.py finish?")
        return
    if args.dry_run:
        print("Dry run: %d configs would be parsed." % len(configs))
        return

    print("\nParsing + fusion variants + post-process ...")
    all_results = []
    fail = 0
    for i, (cid, pid) in enumerate(configs):
        if (i + 1) % 2000 == 0:
            print("  %d/%d (%d rows, %d fail)" % (i + 1, len(configs), len(all_results), fail))
        result = process_mapping_results(cid, pid, args.outputs_dir)
        if result is None or result.get("cycles") == float("inf"):
            fail += 1
            continue
        result["sequence_length"] = DECODE_SEQ
        result["is_duplicate"] = False
        result["master_layer"] = ""
        variants = [result] + gen_multi_fusion(result)
        for res in variants:
            res["is_duplicate"] = False
            res["master_layer"] = ""
            post = post_process_mapping_results(res)
            if post:
                all_results.append(post)
    print("Parsed: %d rows, %d failed configs" % (len(all_results), fail))
    if not all_results:
        print("ERROR: nothing parsed.")
        return

    print("\nDRAM BW expansion: %s (%d combos/row)" % (
        " x ".join(TARGET_DRAMS), len(TARGET_DRAMS) ** 2))
    expanded = []
    for row in all_results:
        for di in TARGET_DRAMS:
            for do in TARGET_DRAMS:
                nr = row.copy()
                nr["dram_i"], nr["dram_o"] = di, do
                r = apply_bw_throttling_analytical(row, di, do)
                if r:
                    nr["latency"] = r["latency"]
                    nr["dynamic_energy"] = r["dynamic_energy"]
                    nr["utilization"] = r["utilization"]
                expanded.append(nr)
    print("Expanded to %d rows" % len(expanded))

    with open(args.backup_csv) as f:
        fieldnames = csv.DictReader(f).fieldnames
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(expanded)
    print("Wrote %d rows -> %s" % (len(expanded), args.out))

    # sanity summary
    by = defaultdict(int)
    for r in expanded:
        by[(r["net"], r["layer_name"], r["batch_size"])] += 1
    print("\nNew decode rows by (net, layer, batch):")
    for k in sorted(by):
        print("  %s / %-16s b=%-3s %d" % (k[0], k[1], k[2], by[k]))


if __name__ == "__main__":
    main()
