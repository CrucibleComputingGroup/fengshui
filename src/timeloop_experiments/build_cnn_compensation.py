#!/usr/bin/env python3
"""
Process NEW CNN compensation layers and append to unified_database_backup.csv
to create unified_database.csv.

Uses the same pipeline as build_unified_database.py:
  1. Discover new CNN Timeloop outputs (not in backup)
  2. Parse via process_mapping_results + post_process_mapping_results
  3. Generate fusion variants (start/middle/end)
  4. Analytical BW throttling → expand to all DRAM combos
  5. Append to backup → write unified_database.csv
"""
import csv
import os
import re
import sys
import time
from collections import defaultdict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in [os.path.join(_THIS_DIR, "..", "scripts"), os.path.join(_THIS_DIR, "..")]:
    if os.path.isfile(os.path.join(_candidate, "utility_functions.py")) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from parse_stats import process_mapping_results, post_process_mapping_results, gen_multi_fusion
from postprocess_bw import apply_bw_throttling_analytical, TARGET_DRAMS


def load_existing_cnn_layers(backup_csv):
    """Load (net, layer_name) pairs for CNN models already in backup."""
    existing = set()
    with open(backup_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            net = row["net"]
            if net in ("mobilenet_v3_small", "replknet31b"):
                existing.add((net, row["layer_name"]))
    return existing


def discover_new_cnn_configs(outputs_dir, existing_layers):
    """Discover CNN configs whose (net, layer) is NOT in existing_layers."""
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

        net = parts[0]
        if net not in ("mobilenet_v3_small", "replknet31b"):
            continue

        layer = parts[1]
        if (net, layer) in existing_layers:
            continue

        batch, seq = parts[2], parts[3]
        mapper, fused, otc, tp = parts[4], parts[5], parts[6], parts[7]
        arch_config, dram = parts[8], parts[9]

        m = re.match(r"arch=(\w+)@glb_scale=(\d+)@pe_x_scale=(\d+)@pe_y_scale=(\d+)", arch_config)
        if not m:
            continue
        arch, glb, px, py = m.group(1), m.group(2), m.group(3), m.group(4)

        if arch not in ("eyeriss_like", "simba_like", "gemmini_like"):
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
    outputs_dir = os.path.join(_THIS_DIR, "outputs")
    backup_csv = os.path.join(_THIS_DIR, "unified_database_backup.csv")
    output_csv = os.path.join(_THIS_DIR, "unified_database.csv")
    target_drams = TARGET_DRAMS  # LPDDR5, DDR5, GDDR7, HBM3

    print("=" * 70)
    print("CNN COMPENSATION → UNIFIED DATABASE")
    print("=" * 70)

    # Step 1: Load existing layers from backup
    print("\nLoading existing CNN layers from backup...")
    existing_layers = load_existing_cnn_layers(backup_csv)
    print("  Existing CNN layers: %d" % len(existing_layers))
    for net, layer in sorted(existing_layers):
        print("    %s / %s" % (net, layer))

    # Step 2: Discover new CNN configs
    print("\nDiscovering new CNN configs in %s..." % outputs_dir)
    t0 = time.time()
    configs = discover_new_cnn_configs(outputs_dir, existing_layers)
    print("Found %d new CNN configs (%.1fs)" % (len(configs), time.time() - t0))

    if not configs:
        print("No new CNN layers to process. Copying backup as-is.")
        import shutil
        shutil.copy2(backup_csv, output_csv)
        return

    # Summary
    layer_counts = defaultdict(int)
    for _, pid in configs:
        parts = pid.split("@")
        layer_counts[(parts[0], parts[1])] += 1
    print("\nNew layers to process:")
    for (net, layer), count in sorted(layer_counts.items()):
        print("  %s / %-55s %d configs" % (net, layer, count))

    # Step 3: Parse all new configs
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

        # CNN: sequence_length = 1
        result["sequence_length"] = 1
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

    # Step 4: No dedup needed for CNN

    # Step 5: DRAM BW expansion
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
    print("Expanded to %d new rows (%.1fs)" % (len(all_results), time.time() - t2))

    # Step 6: Read backup header, write unified CSV
    print("\nReading backup CSV header...")
    with open(backup_csv) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

    print("Writing unified database: backup + %d new rows..." % len(all_results))
    t3 = time.time()

    backup_rows = 0
    with open(output_csv, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        # Copy backup rows
        with open(backup_csv) as in_f:
            reader = csv.DictReader(in_f)
            for row in reader:
                writer.writerow(row)
                backup_rows += 1

        # Append new rows
        writer.writerows(all_results)

    total_rows = backup_rows + len(all_results)
    print("Wrote %d total rows (%d backup + %d new) to %s (%.1fs)" % (
        total_rows, backup_rows, len(all_results), output_csv, time.time() - t3))

    # Summary
    print("\n" + "=" * 70)
    print("DONE — %s" % output_csv)
    print("=" * 70)

    new_nets = sorted(set(r["net"] for r in all_results))
    new_layers = sorted(set((r["net"], r["layer_name"]) for r in all_results))
    print("\nNew layers added:")
    for net, layer in new_layers:
        layer_rows = sum(1 for r in all_results if r["net"] == net and r["layer_name"] == layer)
        print("  %s / %-55s %d rows" % (net, layer, layer_rows))
    print("\nTotal new: %d rows across %d layers" % (len(all_results), len(new_layers)))


if __name__ == "__main__":
    main()
