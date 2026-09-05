#!/usr/bin/env python3
"""Extend network_analysis.csv to batches {16,32,64} for the LLM nets.

Root cause: layer_size_analysis.py hardcodes batch_sizes=[1,4,8], so
cal_mem_req_for_fusion_group's dict lookup silently misses at batch>=16
(mem_dict.get -> continue) => fusion-group memory caps = 0 => the SRAM
fusion check never fails and DRAM provisioning is undersized for every
batch>=16 evaluation on transformer nets.

This wrapper reuses layer_size_analysis's own per-layer-class logic
(attention/softmax | projection/gemm | generic transformer) for the new
batches and APPENDS rows, deduping against existing keys.

Self-check: --validate regenerates batch-4 rows and diffs them against the
existing CSV rows (must match exactly) before any append is trusted.

Usage (from chiplet_timeloop/scripts/):
  python3 extend_network_analysis_batches.py --validate     # golden self-check only
  python3 extend_network_analysis_batches.py                # backup + append b16/32/64
"""
import argparse
import csv
import os
import shutil
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import global_parameter
import utility_functions
from layer_size_analysis import (
    adjust_attention_parameters, adjust_projection_parameters,
    calculate_sizes, generate_all_fused_results, _get_seq_len_from_instance,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "network_analysis.csv")
WORKLOADS = os.path.join(os.path.dirname(HERE), "workloads")

TARGET_NETS = [f"qwen3_30b_a3b_{p}" for p in
               ["prefill_s512", "prefill_s1024", "prefill_s2048", "prefill_s4096",
                "decode_kv512", "decode_kv1024", "decode_kv2048", "decode_kv4096"]] + \
              [f"llama3.1_8b_{p}" for p in
               ["prefill_s512", "prefill_s1024", "prefill_s2048", "prefill_s4096",
                "decode_kv512", "decode_kv1024", "decode_kv2048", "decode_kv4096"]]
NEW_BATCHES = [16, 32, 64]
FIELDS = ['net_name', 'layer_name', 'fused_layer_type', 'in_mem', 'weight_mem',
          'out_mem', 'operations', 'tp', 'batch_size', 'sequence_length']


def gen_rows_for_net(net_name, batch_sizes, bits_per_word=8):
    """Replicates layer_size_analysis.analyze_network's three transformer
    branches for the given batches (CNN branch irrelevant for these nets)."""
    network_dir = os.path.join(WORKLOADS, net_name)
    results = []
    layer_files = [f for f in sorted(os.listdir(network_dir)) if f.endswith('.yaml')]
    for filename in layer_files:
        path = os.path.join(network_dir, filename)
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            if 'problem' not in data:
                continue
            layer_name = os.path.splitext(filename)[0]
            instance = data['problem']['instance']
            default_seq_len = _get_seq_len_from_instance(instance, net_name=net_name)
            if utility_functions.is_attention_layers(layer_name) or \
                    utility_functions.is_softmax_layers(layer_name):
                for tp in global_parameter.tp_degrees:
                    for b in batch_sizes:
                        adj = adjust_attention_parameters(data, tp=tp, sequence_length=None,
                                                          layer_name=layer_name)
                        res = calculate_sizes(adj, bits_per_word, layer_name)
                        results.extend(generate_all_fused_results(
                            net_name, layer_name, res, tp, batch_size=b,
                            sequence_length=default_seq_len))
            elif utility_functions.is_projection_layers(layer_name) or \
                    utility_functions.is_gemm_layer(layer_name):
                for tp in global_parameter.tp_degrees:
                    for b in batch_sizes:
                        adj = adjust_projection_parameters(data, tp=tp, batch_size=b,
                                                           sequence_length=None,
                                                           layer_name=layer_name)
                        res = calculate_sizes(adj, bits_per_word, layer_name)
                        results.extend(generate_all_fused_results(
                            net_name, layer_name, res, tp, b, default_seq_len))
            else:
                for tp in global_parameter.tp_degrees:
                    for b in batch_sizes:
                        res = calculate_sizes(data, bits_per_word, layer_name)
                        results.extend(generate_all_fused_results(
                            net_name, layer_name, res, tp, b, default_seq_len))
        except Exception as e:
            print(f"  [WARN] {path}: {e}")
    return results


def key(row):
    return (row['net_name'], row['layer_name'], row['fused_layer_type'],
            str(row['tp']), str(row['batch_size']), str(row['sequence_length']))


def load_existing():
    rows = {}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            rows[key(r)] = r
    return rows


def class_key(row, batch="4"):
    """Key with the batch slot pinned — identifies a (net,layer,fused,tp,seq) class."""
    return (row['net_name'], row['layer_name'], row['fused_layer_type'],
            str(row['tp']), batch, str(row['sequence_length']))


def validate(existing):
    """Regenerate batch-4 rows; every generated row whose CLASS exists in the
    CSV must match the existing row exactly on the value columns. Generated
    rows of classes absent at b4 (e.g. legacy softmax sub-ops the original
    generator never emitted) are excluded from the append, so they don't
    block validation — they are counted as 'skipped_class'."""
    bad = matched = skipped_class = 0
    for net in TARGET_NETS:
        for row in gen_rows_for_net(net, [4]):
            ex = existing.get(key(row))
            if ex is None:
                skipped_class += 1
                continue
            for col in ('in_mem', 'weight_mem', 'out_mem', 'operations'):
                if abs(float(ex[col]) - float(row[col])) > 1e-12 * max(1.0, abs(float(ex[col]))):
                    bad += 1
                    if bad <= 5:
                        print(f"  [DIFF] {key(row)} {col}: existing {ex[col]} vs regen {row[col]}")
                    break
            else:
                matched += 1
    print(f"validate(b4): matched={matched} diffs={bad} skipped_class={skipped_class}")
    return bad == 0 and matched > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="self-check against existing b4 rows; no write")
    args = ap.parse_args()

    existing = load_existing()
    print(f"existing rows: {len(existing)}")
    print("Running b4 golden self-check...")
    if not validate(existing):
        print("VALIDATION FAILED — not writing anything.")
        sys.exit(1)
    if args.validate:
        print("VALIDATION PASS (no write requested).")
        return

    backup = CSV_PATH + ".bak_pre_batch_ext"
    if not os.path.exists(backup):
        shutil.copyfile(CSV_PATH, backup)
        print(f"backed up -> {backup}")

    new_rows, dup, no_class = [], 0, 0
    for net in TARGET_NETS:
        rows = gen_rows_for_net(net, NEW_BATCHES)
        for row in rows:
            if key(row) in existing:
                dup += 1
                continue
            if class_key(row) not in existing:
                no_class += 1   # class never existed at b4 — don't introduce it
                continue
            new_rows.append(row)
    print(f"(skipped {no_class} rows of classes absent at b4 — legacy convention preserved)")
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writerows(new_rows)
    print(f"appended {len(new_rows)} rows (skipped {dup} already-present) -> {CSV_PATH}")

    per_b = {}
    for r in new_rows:
        per_b[str(r['batch_size'])] = per_b.get(str(r['batch_size']), 0) + 1
    print("new rows by batch:", per_b)


if __name__ == "__main__":
    main()
