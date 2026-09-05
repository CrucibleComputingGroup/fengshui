#!/usr/bin/env python3
"""
Build a unified CSV database from ALL Timeloop outputs (CNN + LLM).

Models included:
  - CNN: mobilenet_v3_small, replknet31b
  - LLM: llama3.1_8b, llama3.1_70b, qwen3_30b_a3b, qwen3_235b_a22b

Steps:
  1. Discover all Timeloop stats files under outputs/
  2. Parse each via process_mapping_results + post_process_mapping_results
  3. Generate fusion variants (start/middle/end)
  4. Cross-network dedup expansion for LLM projection ops
  5. Analytical BW throttling → expand to all DRAM combos
  6. Write unified CSV

Usage:
  python3 build_unified_database.py --output-csv unified_database.csv
  python3 build_unified_database.py --output-csv unified_database.csv --drams LPDDR5  # skip DRAM expansion
  python3 build_unified_database.py --dry-run  # just count configs
"""
import argparse
import csv
import math
import os
import re
import sys
import time
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

# ============================================================
# CNN model definitions (from run_cnn_sweep.py)
# ============================================================
CNN_MODELS = ["mobilenet_v3_small", "replknet31b"]

# ============================================================
# LLM model definitions
# ============================================================
LLM_MODELS = ["llama3.1_8b", "llama3.1_70b", "qwen3_30b_a3b", "qwen3_235b_a22b"]


def parse_seq_len(net_name):
    """Extract sequence_length from network name.
    prefill_s2048 -> 2048, decode_kv512 -> 1 (decode processes 1 token).
    CNN models -> 1.
    """
    m = re.search(r"prefill_s(\d+)", net_name)
    if m:
        return int(m.group(1))
    if "decode" in net_name:
        return 1
    return 1


def is_llm_network(net_name):
    """Check if a network name belongs to an LLM model."""
    for model in LLM_MODELS:
        if net_name.startswith(model):
            return True
    return False


def is_cnn_network(net_name):
    """Check if a network name belongs to a CNN model."""
    return net_name in CNN_MODELS


def discover_configs(outputs_dir):
    """Walk the outputs directory and discover all (config_id, problem_id) pairs.

    Handles both LLM structure (10 levels) and CNN structure (10 levels, same format).
    """
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

        # Filter: 3 main accelerators + simple_vector (for softmax)
        if arch not in ("eyeriss_like", "simba_like", "gemmini_like", "simple_vector"):
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


def build_dedup_map(workload_dir, models):
    """Build dedup mapping for LLM projection ops.

    For each (model, op), map N_eff -> (master_net, master_batch).
    Returns list of dicts describing duplicate entries to expand.
    """
    batches = batch_configs(transformer=True)
    dedup_entries = []

    for model in models:
        proj_ops = QWEN_PROJECTION_OPS if model.startswith("qwen") else LLAMA_PROJECTION_OPS
        for op in proj_ops:
            seen_n_eff = {}

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
    parser = argparse.ArgumentParser(
        description="Build unified CSV database from all Timeloop outputs")
    parser.add_argument("--output-base-dir", default=_THIS_DIR,
                        help="Base dir (outputs_dir defaults to {base}/outputs)")
    parser.add_argument("--outputs-dir", default=None,
                        help="Direct path to outputs directory")
    parser.add_argument("--output-csv", default="unified_database.csv",
                        help="Output CSV path (default: unified_database.csv)")
    parser.add_argument("--workload-dir", default=None,
                        help="Workload directory (default: ../workloads)")
    parser.add_argument("--drams", default=",".join(TARGET_DRAMS),
                        help="Comma-separated target DRAM types for BW expansion "
                             "(default: LPDDR5,DDR5,GDDR7,HBM3). "
                             "Pass a single type to skip expansion.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just discover and count configs; don't parse")
    args = parser.parse_args()

    workload_dir = args.workload_dir or os.path.normpath(os.path.join(_THIS_DIR, "..", "workloads"))
    outputs_dir = args.outputs_dir or os.path.join(args.output_base_dir, "outputs")

    # ============================================================
    # Step 1: Discover all Timeloop outputs
    # ============================================================
    print("=" * 70)
    print("UNIFIED DATABASE BUILDER")
    print("=" * 70)
    print("\nDiscovering configs in %s..." % outputs_dir)
    t0 = time.time()
    configs = discover_configs(outputs_dir)
    print("Found %d configs (%.1fs)" % (len(configs), time.time() - t0))

    # Summarize by model
    net_counts = defaultdict(int)
    for _, pid in configs:
        net = pid.split("@")[0]
        # Group by base model
        for model in CNN_MODELS + LLM_MODELS:
            if net.startswith(model) or net == model:
                net_counts[model] += 1
                break
        else:
            net_counts[net] += 1
    print("\nConfigs per model:")
    for model in sorted(net_counts):
        print("  %-35s %d" % (model, net_counts[model]))

    if args.dry_run:
        return

    # ============================================================
    # Step 2: Parse all Timeloop outputs
    # ============================================================
    print("\nParsing Timeloop outputs...")
    t1 = time.time()
    all_results = []
    success = 0
    fail = 0

    for i, (config_id, problem_id) in enumerate(configs):
        if (i + 1) % 2000 == 0:
            print("  %d/%d processed (%d ok, %d fail)" % (
                i + 1, len(configs), success, fail))

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

    print("Parsed %d configs in %.1fs: %d successful rows, %d failed" % (
        len(configs), time.time() - t1, len(all_results), fail))

    if not all_results:
        print("ERROR: No results parsed. Check outputs directory.")
        return

    # ============================================================
    # Step 3: Dedup expansion (LLM projection ops only)
    # ============================================================
    print("\nBuilding dedup map for LLM models...")
    dedup_entries = build_dedup_map(workload_dir, LLM_MODELS)
    print("Found %d dedup entries to expand" % len(dedup_entries))

    dup_count = 0
    for dinfo in dedup_entries:
        op_name = dinfo["op"]
        master_results = [
            r for r in all_results
            if r["net"] == dinfo["master_net"]
            and r["batch_size"] == dinfo["master_batch"]
            and (r["layer_name"] == "layer0_%s" % op_name
                 or r["layer_name"] == op_name)
        ]
        for mr in master_results:
            dup = mr.copy()
            dup["net"] = dinfo["dup_net"]
            dup["batch_size"] = dinfo["dup_batch"]
            dup["sequence_length"] = dinfo["dup_seq"]
            all_results.append(dup)
            dup_count += 1

    print("Added %d dedup copies → %d total rows" % (dup_count, len(all_results)))

    # ============================================================
    # Step 4: DRAM BW expansion
    # ============================================================
    target_drams = [d.strip() for d in args.drams.split(",")]

    if len(target_drams) > 1:
        print("\nExpanding DRAM combos: %s (%d combos per row)" % (
            " x ".join(target_drams), len(target_drams) ** 2))
        t2 = time.time()
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
        print("Expanded to %d rows (%.1fs)" % (len(all_results), time.time() - t2))

    # ============================================================
    # Step 5: Write CSV
    # ============================================================
    fieldnames = list(all_results[0].keys())
    for remove_key in ["is_duplicate", "master_layer"]:
        if remove_key in fieldnames:
            fieldnames.remove(remove_key)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print("DONE — wrote %d rows to %s (%.1fs total)" % (
        len(all_results), args.output_csv, total_time))
    print("=" * 70)

    # ============================================================
    # Summary
    # ============================================================
    nets = sorted(set(r["net"] for r in all_results))
    archs = sorted(set(r["arch_target"] for r in all_results))
    dram_combos = sorted(set((r.get("dram_i", "?"), r.get("dram_o", "?"))
                             for r in all_results))
    batches = sorted(set(r["batch_size"] for r in all_results))
    tps = sorted(set(r["tp_degree"] for r in all_results))

    print("\nSummary:")
    print("  Networks:    %d" % len(nets))
    print("  Archs:       %s" % archs)
    print("  Batches:     %s" % batches)
    print("  TP degrees:  %s" % tps)
    print("  DRAM combos: %d" % len(dram_combos))
    print("  Total rows:  %d" % len(all_results))

    print("\nPer-network breakdown:")
    for net in nets:
        net_rows = [r for r in all_results if r["net"] == net]
        layers = set(r["layer_name"] for r in net_rows)
        net_batches = sorted(set(r["batch_size"] for r in net_rows))
        net_archs = sorted(set(r["arch_target"] for r in net_rows))
        print("  %-45s %2d layers, batches=%s, archs=%s, %d rows" % (
            net, len(layers), net_batches, net_archs, len(net_rows)))


if __name__ == "__main__":
    main()
