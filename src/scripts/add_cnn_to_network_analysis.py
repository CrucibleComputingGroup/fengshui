#!/usr/bin/env python3
"""
Add all CNN layers to network_analysis.csv.

Generates network_analysis entries for all mobilenet_v3_small (13 layers) and
replknet31b (12 layers) with batch_sizes=[1,4,8,16] (matching the unified database).

Usage:
  python3 add_cnn_to_network_analysis.py
  python3 add_cnn_to_network_analysis.py --output network_analysis.csv  # overwrite in-place
"""
import os
import sys
import csv
import yaml
import copy
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layer_size_analysis import (
    calculate_sizes,
    adjust_cnn_parameters,
    generate_all_fused_results,
)

WORKLOAD_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'workloads'))

# All CNN layers matching run_cnn_sweep.py and the unified database
CNN_LAYERS = {
    "mobilenet_v3_small": [
        # --- features.2 (block 2) ---
        "layer6_features_2_block_0_0",
        "layer7_features_2_block_1_0",
        "layer8_features_2_block_2_0",
        # --- features.5 (block 5) ---
        "layer17_features_5_block_0_0",
        "layer18_features_5_block_1_0",
        "layer19_features_5_block_2_fc1",
        "layer20_features_5_block_2_fc2",
        "layer21_features_5_block_3_0",
        # --- features.10 (block 10) ---
        "layer42_features_10_block_0_0",
        "layer43_features_10_block_1_0",
        "layer44_features_10_block_2_fc1",
        "layer45_features_10_block_2_fc2",
        "layer46_features_10_block_3_0",
    ],
    "replknet31b": [
        # --- stages_0, block 0 ---
        "layer1_stages_0_blocks_0_pw1_conv",
        "layer2_stages_0_blocks_0_large_kernel_lkb_origin_conv",
        "layer3_stages_0_blocks_0_large_kernel_small_conv_conv",
        "layer4_stages_0_blocks_0_pw2_conv",
        # --- stages_0, block 1 ---
        "layer5_stages_0_blocks_1_pw1_conv",
        "layer6_stages_0_blocks_1_pw2_conv",
        # --- stages_1, block 0 ---
        "layer13_stages_1_blocks_0_pw1_conv",
        "layer14_stages_1_blocks_0_large_kernel_lkb_origin_conv",
        "layer15_stages_1_blocks_0_large_kernel_small_conv_conv",
        "layer16_stages_1_blocks_0_pw2_conv",
        # --- stages_1, block 1 ---
        "layer17_stages_1_blocks_1_pw1_conv",
        "layer18_stages_1_blocks_1_pw2_conv",
    ],
}

BATCH_SIZES = [1, 4, 8, 16]
BITS_PER_WORD = 16  # BF16


def generate_cnn_entries():
    """Generate network_analysis rows for all CNN layers."""
    results = []

    for net_name, layers in CNN_LAYERS.items():
        for layer_name in layers:
            yaml_path = os.path.join(WORKLOAD_DIR, net_name, layer_name + ".yaml")
            if not os.path.exists(yaml_path):
                print("WARNING: %s not found, skipping" % yaml_path)
                continue

            with open(yaml_path) as f:
                data = yaml.safe_load(f)

            if 'problem' not in data:
                print("WARNING: no 'problem' in %s, skipping" % yaml_path)
                continue

            for batch_size in BATCH_SIZES:
                # CNN: tp=1 only, sequence_length=1
                adjusted_data = adjust_cnn_parameters(
                    data, tp=1, batch_size=batch_size, layer_name=layer_name)
                result = calculate_sizes(adjusted_data, BITS_PER_WORD, layer_name)
                entries = generate_all_fused_results(
                    net_name, layer_name, result,
                    tp=1, batch_size=batch_size, sequence_length=1)
                results.extend(entries)

            print("  %s / %s: %d entries (4 batches × 4 fused types)" % (
                net_name, layer_name, len(BATCH_SIZES) * 4))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Add CNN representative layers to network_analysis.csv")
    parser.add_argument("--input", default="network_analysis.csv",
                        help="Input network_analysis.csv (default: network_analysis.csv)")
    parser.add_argument("--output", default=None,
                        help="Output CSV (default: overwrite input)")
    args = parser.parse_args()

    output_path = args.output or args.input

    # Read existing CSV
    existing_rows = []
    fieldnames = ['net_name', 'layer_name', 'fused_layer_type',
                  'in_mem', 'weight_mem', 'out_mem', 'operations',
                  'tp', 'batch_size', 'sequence_length']

    if os.path.exists(args.input):
        with open(args.input) as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            if reader.fieldnames:
                fieldnames = reader.fieldnames
        print("Read %d existing rows from %s" % (len(existing_rows), args.input))
    else:
        print("No existing file found, creating new")

    # Remove old CNN entries (if any) to avoid duplicates
    cnn_nets = set(CNN_LAYERS.keys())
    old_cnn_count = sum(1 for r in existing_rows if r['net_name'] in cnn_nets)
    filtered_rows = [r for r in existing_rows if r['net_name'] not in cnn_nets]
    if old_cnn_count > 0:
        print("Removed %d old CNN entries" % old_cnn_count)

    # Generate new CNN entries
    print("\nGenerating CNN entries...")
    new_entries = generate_cnn_entries()
    print("\nGenerated %d new CNN entries" % len(new_entries))

    # Merge
    all_rows = filtered_rows + new_entries
    print("Total: %d rows" % len(all_rows))

    # Write
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)

    print("Wrote %s" % output_path)

    # Verify
    import pandas as pd
    df = pd.read_csv(output_path)
    for net in sorted(cnn_nets):
        net_df = df[df['net_name'] == net]
        layers = sorted(net_df['layer_name'].unique())
        batches = sorted(net_df['batch_size'].unique())
        print("  %s: %d rows, %d layers, batches=%s" % (
            net, len(net_df), len(layers), batches))


if __name__ == "__main__":
    main()
