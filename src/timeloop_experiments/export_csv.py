#!/usr/bin/env python3
"""
Export sweep results to CSV in the format expected by cal_perf_phy_net.py.

Uses the existing parse_stats.py functions (process_mapping_results,
post_process_mapping_results, gen_multi_fusion) to parse Timeloop outputs
and generate the full CSV with all columns needed for downstream chiplet
optimization.

Handles:
  - Cross-network deduplication expansion: if prefill_s512/batch=4 was the
    master run for prefill_s2048/batch=1 (same N_eff=2048), this script
    emits rows for BOTH with correct net/batch_size/sequence_length.
  - Proper sequence_length: prefill_s{X} → seq=X, decode_kv{X} → seq=1
  - Attention ops: not deduped (different dims per seq/kv_len), batch=1 only
  - DRAM expansion: expands each row to all dram_i × dram_o combinations
    with analytical BW throttling (energy + latency adjustment).

Usage:
  python3 export_csv.py --outputs-dir /path/to/outputs --output-csv llama_qwen_all_dram.csv
  python3 export_csv.py --outputs-dir /path/to/outputs --output-csv db.csv --drams LPDDR5
"""
import argparse
import csv
import math
import os
import re
import sys
import yaml
from collections import defaultdict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in [os.path.join(_THIS_DIR, "..", "scripts"), os.path.join(_THIS_DIR, "..")]:
    if os.path.isfile(os.path.join(_candidate, "utility_functions.py")) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from parse_stats import process_mapping_results, post_process_mapping_results, gen_multi_fusion
from postprocess_bw import apply_bw_throttling_analytical, TARGET_DRAMS
from global_parameter import (
    LLAMA_PROJECTION_OPS, LLAMA_ATTENTION_OPS, LLAMA_BATCH_AGNOSTIC_OPS,
    QWEN_PROJECTION_OPS,
    batch_configs, prefill_seq_lens, decode_kv_lens, ARCH_BASE_PE,
    arch_targets, pe_scales, glb_scales, tp_degrees, num_mapping_per_arch
)


def parse_seq_len(net_name):
    """Extract sequence_length from network name.
    prefill_s2048 → 2048, decode_kv512 → 1 (decode processes 1 token)
    """
    m = re.search(r"prefill_s(\d+)", net_name)
    if m:
        return int(m.group(1))
    if "decode" in net_name:
        return 1
    return 1


def discover_configs(outputs_dir):
    """Walk the outputs directory and discover all (config_id, problem_id) pairs."""
    configs = []

    for root, dirs, files in os.walk(outputs_dir):
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

        m = re.match(r"arch=(\w+)@glb_scale=(\d+)@pe_x_scale=(\d+)@pe_y_scale=(\d+)", arch_config)
        if not m:
            continue
        arch, glb, px, py = m.group(1), m.group(2), m.group(3), m.group(4)
        dram_parts = dram.split("@")
        if len(dram_parts) != 2:
            continue

        config_id = "%s@glb%s@pe_x_scale%s@pe_y_scale%s@%s@%s" % (
            arch, glb, px, py, dram_parts[0], dram_parts[1])
        problem_id = "%s@%s@%s@%s@%s@%s@%s@%s" % (
            net, layer, batch, seq, mapper, fused, otc, tp)

        configs.append((config_id, problem_id))

    return configs


def build_dedup_map(workload_dir, models):
    """Build the dedup mapping: for each (model, op), map N_eff → (master_net, master_batch).
    Then for each (net, batch) combo that was deduped, record the master.

    Returns: list of dicts with {op, dup_net, dup_batch, dup_seq, master_net, master_batch}
    """
    batches = batch_configs(transformer=True)
    dedup_entries = []

    for model in models:
        proj_ops = QWEN_PROJECTION_OPS if model.startswith("qwen") else LLAMA_PROJECTION_OPS
        for op in proj_ops:
            seen_n_eff = {}  # n_eff → (net, batch, seq, yaml_path)

            all_combos = []
            # Prefill
            for seq in prefill_seq_lens:
                net = "%s_prefill_s%d" % (model, seq)
                for prefix in ["layer0_%s" % op, op]:
                    path = os.path.join(workload_dir, net, prefix + ".yaml")
                    if os.path.exists(path):
                        with open(path) as f:
                            dims = yaml.safe_load(f)["problem"]["instance"]
                        base_N = dims.get("N", 1)
                        for b in batches:
                            all_combos.append((net, b, seq, base_N * b))
                        break
            # Decode
            for kv in decode_kv_lens:
                net = "%s_decode_kv%d" % (model, kv)
                for prefix in ["layer0_%s" % op, op]:
                    path = os.path.join(workload_dir, net, prefix + ".yaml")
                    if os.path.exists(path):
                        with open(path) as f:
                            dims = yaml.safe_load(f)["problem"]["instance"]
                        base_N = dims.get("N", 1)
                        for b in batches:
                            all_combos.append((net, b, 1, base_N * b))
                        break

            for net, batch, seq, n_eff in all_combos:
                if n_eff not in seen_n_eff:
                    seen_n_eff[n_eff] = (net, batch, seq)
                else:
                    master_net, master_batch, master_seq = seen_n_eff[n_eff]
                    dedup_entries.append({
                        "op": op,
                        "dup_net": net,
                        "dup_batch": batch,
                        "dup_seq": seq,
                        "master_net": master_net,
                        "master_batch": master_batch,
                        "master_seq": master_seq,
                    })

    return dedup_entries


def main():
    parser = argparse.ArgumentParser(description="Export sweep results to database CSV")
    parser.add_argument("--output-base-dir", default=_THIS_DIR,
                        help="Base dir (outputs_dir defaults to {base}/outputs)")
    parser.add_argument("--outputs-dir", default=None,
                        help="Direct path to outputs directory (overrides --output-base-dir)")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--workload-dir", default=None,
                        help="Workload directory (default: ../workloads)")
    parser.add_argument("--models", default="llama3.1_8b,llama3.1_70b,qwen3_30b_a3b,qwen3_235b_a22b")
    parser.add_argument("--drams", default=",".join(TARGET_DRAMS),
                        help="Comma-separated target DRAM types for BW expansion "
                             "(default: LPDDR5,DDR5,GDDR7,HBM3). "
                             "Pass a single type (e.g. --drams LPDDR5) to skip expansion.")
    args = parser.parse_args()

    workload_dir = args.workload_dir or os.path.normpath(os.path.join(_THIS_DIR, "..", "workloads"))
    models = args.models.split(",")
    outputs_dir = args.outputs_dir or os.path.join(args.output_base_dir, "outputs")

    # Step 1: Discover and parse all Timeloop outputs
    print("Discovering configs in %s..." % outputs_dir)
    configs = discover_configs(outputs_dir)
    print("Found %d configs" % len(configs))

    all_results = []
    success = 0
    fail = 0

    for i, (config_id, problem_id) in enumerate(configs):
        if (i + 1) % 1000 == 0:
            print("  %d/%d processed (%d ok, %d fail)" % (i + 1, len(configs), success, fail))

        result = process_mapping_results(config_id, problem_id, outputs_dir)
        if result is None or result.get("cycles") == float("inf"):
            fail += 1
            continue

        # Fix sequence_length from network name
        result["sequence_length"] = parse_seq_len(result["net"])

        result["is_duplicate"] = False
        result["master_layer"] = ""

        # Generate fusion variants
        results_to_add = [result]
        fused_variants = gen_multi_fusion(result)
        results_to_add.extend(fused_variants)

        # Post-process each variant
        for res in results_to_add:
            res["is_duplicate"] = False
            res["master_layer"] = ""
            post = post_process_mapping_results(res)
            if post:
                all_results.append(post)
                success += 1

    print("Processed %d configs: %d successful rows, %d failed" % (len(configs), len(all_results), fail))

    # Step 2: Build dedup map and expand
    print("\nBuilding dedup map...")
    dedup_entries = build_dedup_map(workload_dir, models)
    print("Found %d dedup entries to expand" % len(dedup_entries))

    dup_count = 0
    for dinfo in dedup_entries:
        # Find all master results matching this op + master_net + master_batch
        op_name = dinfo["op"]
        # layer_name in the CSV may have layer0_ prefix or not
        master_results = [
            r for r in all_results
            if r["net"] == dinfo["master_net"]
            and r["batch_size"] == dinfo["master_batch"]
            and (r["layer_name"] == "layer0_%s" % op_name or r["layer_name"] == op_name)
        ]
        for mr in master_results:
            dup = mr.copy()
            dup["net"] = dinfo["dup_net"]
            dup["batch_size"] = dinfo["dup_batch"]
            dup["sequence_length"] = dinfo["dup_seq"]
            all_results.append(dup)
            dup_count += 1

    print("Added %d dedup copies" % dup_count)

    # Step 3: Expand DRAM combos with analytical BW throttling
    target_drams = [d.strip() for d in args.drams.split(",")]
    need_expansion = len(target_drams) > 1 or target_drams[0] != all_results[0].get("dram_i", "LPDDR5")

    if need_expansion:
        print("\nExpanding DRAM combos: %s (%d combos per row)" % (
            " x ".join(target_drams), len(target_drams) ** 2))
        expanded = []
        for row in all_results:
            for di in target_drams:
                for do in target_drams:
                    new_row = row.copy()
                    new_row["dram_i"] = di
                    new_row["dram_o"] = do
                    result = apply_bw_throttling_analytical(row, di, do)
                    if result:
                        new_row["latency"] = result["latency"]
                        new_row["dynamic_energy"] = result["dynamic_energy"]
                        new_row["utilization"] = result["utilization"]
                    expanded.append(new_row)
        all_results = expanded
        print("Expanded to %d rows" % len(all_results))

    # Step 4: Write CSV
    if all_results:
        fieldnames = list(all_results[0].keys())
        for remove_key in ["is_duplicate", "master_layer"]:
            if remove_key in fieldnames:
                fieldnames.remove(remove_key)

        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)

        print("\nWrote %d rows to %s" % (len(all_results), args.output_csv))

        # Summary
        nets = set(r["net"] for r in all_results)
        dram_combos = set((r.get("dram_i", "?"), r.get("dram_o", "?")) for r in all_results)
        print("\nNetworks: %d, DRAM combos: %d" % (len(nets), len(dram_combos)))
        for net in sorted(nets):
            layers = set(r["layer_name"] for r in all_results if r["net"] == net)
            batches = set(r["batch_size"] for r in all_results if r["net"] == net)
            n = sum(1 for r in all_results if r["net"] == net)
            print("  %-45s %d layers, batches=%s, %d rows" % (net, len(layers), sorted(batches), n))


if __name__ == "__main__":
    main()
