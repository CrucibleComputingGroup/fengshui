#!/usr/bin/env python3
"""
Post-process infinite-BW sweep results to derive actual DRAM-constrained metrics.

Supports two input formats:
  1. Database format (from export_csv.py): columns include dynamic_energy, latency,
     i_access, w_access, o_access.  BW throttling is computed analytically.
  2. Raw sweep format (from run_sweep.py): columns include energy_uj, cycles.
     BW throttling uses stats files from the outputs directory.

Expands each input row to 4x4=16 rows (one per dram_i x dram_o combination).

Usage:
  # Database format (no stats files needed):
  python3 postprocess_bw.py --input llama_qwen_database.csv --output llama_qwen_all_dram.csv

  # Raw sweep format (needs stats files):
  python3 postprocess_bw.py --input llama_full_sweep.csv --output llama_all_dram.csv \
      --outputs-dir /path/to/outputs
"""
import argparse
import csv
import math
import os
import re
import sys
from collections import Counter

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in [os.path.join(_THIS_DIR, "..", "scripts"), os.path.join(_THIS_DIR, "..")]:
    if os.path.isfile(os.path.join(_candidate, "global_parameter.py")) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from global_parameter import dram_type_bandwidth_width_dict, word_size, cycle_time

# DRAM configs to generate
TARGET_DRAMS = ['LPDDR5', 'DDR5', 'GDDR7', 'HBM3']


# ============================================================
# Analytical BW throttling for database format
# ============================================================

def apply_bw_throttling_analytical(row, target_dram_i, target_dram_o):
    """Apply BW throttling analytically from database CSV columns.

    Uses i_access, w_access, o_access (scalar counts) and the source DRAM
    type (from dram_i/dram_o columns) to compute new latency and energy.

    Database format units:
      - latency: seconds
      - dynamic_energy: Joules
      - utilization: percentage (0-100)
      - i_access, w_access, o_access: scalar counts

    Returns dict with updated fields, or None on error.
    """
    try:
        i_access = float(row.get('i_access', 0))
        w_access = float(row.get('w_access', 0))
        o_access = float(row.get('o_access', 0))
        original_latency = float(row.get('latency', 0))
        dynamic_energy = float(row.get('dynamic_energy', 0))
        utilization = float(row.get('utilization', 0))
        tp_degree = int(float(row.get('tp_degree', 1)))
        source_dram_i = row.get('dram_i', 'LPDDR5')
        source_dram_o = row.get('dram_o', 'LPDDR5')
    except (ValueError, TypeError):
        return None

    if original_latency <= 0:
        return None

    original_cycles = original_latency / cycle_time

    # --- Energy recalculation ---
    # Subtract source DRAM energy, add target DRAM energy
    source_e_i = dram_type_bandwidth_width_dict[source_dram_i]['final_e']  # pJ/bit
    source_e_o = dram_type_bandwidth_width_dict[source_dram_o]['final_e']
    target_e_i = dram_type_bandwidth_width_dict[target_dram_i]['final_e']
    target_e_o = dram_type_bandwidth_width_dict[target_dram_o]['final_e']

    # DRAM_I serves both inputs and weights
    source_I_energy = (i_access + w_access) * source_e_i * word_size * 1e-12  # Joules
    source_O_energy = o_access * source_e_o * word_size * 1e-12
    target_I_energy = (i_access + w_access) * target_e_i * word_size * 1e-12
    target_O_energy = o_access * target_e_o * word_size * 1e-12

    compute_energy = dynamic_energy - source_I_energy - source_O_energy
    new_dynamic_energy = compute_energy + target_I_energy + target_O_energy

    # --- BW throttling (latency) ---
    target_bw_I = dram_type_bandwidth_width_dict[target_dram_i]['bandwidth'] * 8 / word_size / tp_degree
    target_bw_O = dram_type_bandwidth_width_dict[target_dram_o]['bandwidth'] * 8 / word_size / tp_degree

    dram_I_total = i_access + w_access
    dram_I_cycles = math.ceil(dram_I_total / target_bw_I) if target_bw_I > 0 and dram_I_total > 0 else 0
    dram_O_cycles = math.ceil(o_access / target_bw_O) if target_bw_O > 0 and o_access > 0 else 0

    new_cycles = max(original_cycles, dram_I_cycles, dram_O_cycles)
    new_latency = new_cycles * cycle_time

    # --- Utilization ---
    # Preserve ideal_cycles = original_cycles * (utilization / 100)
    if new_cycles > 0 and utilization > 0:
        new_utilization = utilization * (original_cycles / new_cycles)
    else:
        new_utilization = utilization

    bw_throttled = 1 if new_cycles > original_cycles else 0

    return {
        'latency': new_latency,
        'dynamic_energy': new_dynamic_energy,
        'utilization': round(new_utilization, 6),
        'bw_throttled': bw_throttled,
    }


# ============================================================
# Stats-file-based BW throttling for raw sweep format
# ============================================================

SOURCE_DRAM = 'LPDDR5'
SOURCE_PJ_PER_BIT = dram_type_bandwidth_width_dict[SOURCE_DRAM]['timeloop_e']


def parse_dram_stats(stats_path):
    """Parse DRAM_I and DRAM_O stats from a Timeloop stats file."""
    if not os.path.exists(stats_path):
        return None

    with open(stats_path) as f:
        content = f.read()

    result = {
        'dram_I_accesses': 0, 'dram_I_cycles': 0, 'dram_I_energy_pj': 0, 'dram_I_bw': 0,
        'dram_O_accesses': 0, 'dram_O_cycles': 0, 'dram_O_energy_pj': 0, 'dram_O_bw': 0,
        'total_energy_pj': 0,
    }

    for line in content.split('\n'):
        ls = line.strip()
        if ls.startswith('Energy:'):
            val = ls.split(':')[1].strip()
            if 'uJ' in val:
                result['total_energy_pj'] = float(val.replace('uJ', '').strip()) * 1e6
            elif 'mJ' in val:
                result['total_energy_pj'] = float(val.replace('mJ', '').strip()) * 1e9
            elif 'pJ' in val:
                result['total_energy_pj'] = float(val.replace('pJ', '').strip())

    lines = content.split('\n')
    parsed_drams = set()
    i = 0
    while i < len(lines):
        ls = lines[i].strip()
        if ls.startswith('Networks') or ls.startswith('Operational Intensity'):
            break
        if ls in ('=== DRAM_I ===', '=== DRAM_O ===') and ls not in parsed_drams:
            parsed_drams.add(ls)
            dram_key = 'dram_I' if 'DRAM_I' in ls else 'dram_O'
            j = i + 1
            in_stats = False
            level_cycles = 0
            level_energy = 0
            level_accesses = 0
            while j < len(lines):
                dl = lines[j].strip()
                if dl.startswith('Level ') or dl.startswith('Networks') or dl.startswith('Operational'):
                    break
                if dl.startswith('STATS'):
                    in_stats = True
                elif in_stats:
                    if dl.startswith('Cycles') and ':' in dl and 'throttling' not in dl:
                        try:
                            level_cycles = int(dl.split(':')[1].strip())
                        except (ValueError, IndexError):
                            pass
                    elif dl.startswith('Scalar reads (per-instance)'):
                        try:
                            level_accesses += int(dl.split(':')[1].strip())
                        except (ValueError, IndexError):
                            pass
                    elif dl.startswith('Scalar updates (per-instance)'):
                        try:
                            level_accesses += int(dl.split(':')[1].strip())
                        except (ValueError, IndexError):
                            pass
                    elif dl.startswith('Scalar fills (per-instance)'):
                        try:
                            level_accesses += int(dl.split(':')[1].strip())
                        except (ValueError, IndexError):
                            pass
                    elif dl.startswith('Energy (total)'):
                        try:
                            val = dl.split(':')[1].strip().replace('pJ', '').strip()
                            level_energy += float(val)
                        except (ValueError, IndexError):
                            pass
                j += 1
            result[f'{dram_key}_cycles'] = level_cycles
            result[f'{dram_key}_energy_pj'] = level_energy
            result[f'{dram_key}_accesses'] = level_accesses
            if level_cycles > 0:
                result[f'{dram_key}_bw'] = level_accesses / level_cycles
            i = j
            continue
        i += 1
    return result


def find_stats_file(outputs_dir, row):
    """Reconstruct the stats file path from a CSV row."""
    net = row.get('workload') or row.get('net', '')
    op = row.get('operator') or row.get('layer_name', '')
    batch = str(row['batch_size'])
    tp = str(row['tp_degree'])
    arch = row.get('arch') or row.get('arch_target', '')
    pe_x = row.get('pe_x_scale', '')
    pe_y = row.get('pe_y_scale', '')
    if not pe_x or not pe_y:
        ps = row.get('pe_scale', '1x1')
        parts = ps.split('x')
        pe_x = pe_x or parts[0]
        pe_y = pe_y or (parts[1] if len(parts) > 1 else parts[0])
    glb = row.get('glb_scale', '1')
    dram_I = row.get('dram_I') or row.get('dram_i', 'LPDDR5')
    dram_O = row.get('dram_O') or row.get('dram_o', 'LPDDR5')

    arch_dir = "arch=%s@glb_scale=%s@pe_x_scale=%s@pe_y_scale=%s" % (arch, glb, pe_x, pe_y)
    dram_dir = "%s@%s" % (dram_I, dram_O)

    for op_prefix in ["layer0_%s" % op, op]:
        base = os.path.join(outputs_dir, net, op_prefix, batch, "1", "0", "single")
        if not os.path.isdir(base):
            continue
        try:
            for otc in os.listdir(base):
                tp_path = os.path.join(base, otc, tp, arch_dir, dram_dir,
                                       "timeloop-mapper.stats.txt")
                if os.path.exists(tp_path):
                    return tp_path
        except OSError:
            continue
    return None


def apply_bw_throttling_from_stats(dram_stats, dram_i_type, dram_o_type, tp_degree, original_cycles):
    """Apply BW throttling using parsed stats file data (raw sweep format)."""
    if dram_stats is None:
        return original_cycles, 0, 0, 0

    bw_I_config = dram_type_bandwidth_width_dict[dram_i_type]
    bw_O_config = dram_type_bandwidth_width_dict[dram_o_type]
    target_bw_I = bw_I_config['bandwidth'] * 8 / word_size / tp_degree
    target_bw_O = bw_O_config['bandwidth'] * 8 / word_size / tp_degree

    dram_I_bw = dram_stats['dram_I_bw']
    dram_O_bw = dram_stats['dram_O_bw']
    dram_I_cycles = dram_stats['dram_I_cycles']
    dram_O_cycles = dram_stats['dram_O_cycles']

    if dram_I_bw > 0 and dram_I_bw > target_bw_I:
        actual_I_cycles = math.ceil(dram_I_cycles * dram_I_bw / target_bw_I)
    else:
        actual_I_cycles = dram_I_cycles

    if dram_O_bw > 0 and dram_O_bw > target_bw_O:
        actual_O_cycles = math.ceil(dram_O_cycles * dram_O_bw / target_bw_O)
    else:
        actual_O_cycles = dram_O_cycles

    new_cycles = max(original_cycles, actual_I_cycles, actual_O_cycles)

    old_dram_I_energy = dram_stats['dram_I_energy_pj']
    old_dram_O_energy = dram_stats['dram_O_energy_pj']
    new_dram_I_energy = old_dram_I_energy * bw_I_config['timeloop_e'] / SOURCE_PJ_PER_BIT
    new_dram_O_energy = old_dram_O_energy * bw_O_config['timeloop_e'] / SOURCE_PJ_PER_BIT

    compute_energy = dram_stats['total_energy_pj'] - old_dram_I_energy - old_dram_O_energy
    new_total_energy = compute_energy + new_dram_I_energy + new_dram_O_energy

    return new_cycles, new_total_energy / 1e6, (new_dram_I_energy + new_dram_O_energy) / 1e6, compute_energy / 1e6


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Post-process infinite-BW results for actual DRAM configs")
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument("--output", required=True, help="Output CSV with all DRAM configs")
    parser.add_argument("--outputs-dir", default=None,
                        help="Direct path to outputs directory (for raw sweep format)")
    parser.add_argument("--output-base-dir", default=_THIS_DIR,
                        help="Base dir (outputs-dir defaults to {base}/outputs)")
    parser.add_argument("--drams", default=",".join(TARGET_DRAMS),
                        help="Comma-separated target DRAM types")
    args = parser.parse_args()

    target_drams = args.drams.split(",")
    outputs_dir = args.outputs_dir or os.path.join(args.output_base_dir, "outputs")

    # Read input CSV
    with open(args.input) as f:
        reader = csv.DictReader(f)
        input_rows = list(reader)
        fieldnames_in = reader.fieldnames
    print(f"Read {len(input_rows)} rows from {args.input}")

    # Detect format: database has 'dynamic_energy', raw sweep has 'energy_uj'
    is_database_format = 'dynamic_energy' in fieldnames_in

    if is_database_format:
        print("Detected database format — using analytical BW throttling")
        _process_database_format(input_rows, fieldnames_in, target_drams, args.output)
    else:
        print("Detected raw sweep format — using stats-file BW throttling")
        _process_sweep_format(input_rows, fieldnames_in, target_drams, outputs_dir, args.output)


def _process_database_format(input_rows, fieldnames_in, target_drams, output_path):
    """Process database format CSV with analytical BW throttling."""
    output_rows = []

    for row in input_rows:
        for di in target_drams:
            for do in target_drams:
                new_row = dict(row)
                new_row['dram_i'] = di
                new_row['dram_o'] = do

                result = apply_bw_throttling_analytical(row, di, do)
                if result:
                    new_row['latency'] = result['latency']
                    new_row['dynamic_energy'] = result['dynamic_energy']
                    new_row['utilization'] = result['utilization']

                output_rows.append(new_row)

    # Write output — same columns as input (dram_i/dram_o now have target values)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_in, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {output_path}")
    _print_summary(output_rows)


def _process_sweep_format(input_rows, fieldnames_in, target_drams, outputs_dir, output_path):
    """Process raw sweep format CSV with stats-file BW throttling."""
    # Parse DRAM stats
    stats_cache = {}
    missing = 0
    for row in input_rows:
        stats_path = find_stats_file(outputs_dir, row)
        if stats_path and stats_path not in stats_cache:
            stats_cache[stats_path] = parse_dram_stats(stats_path)
        if stats_path is None:
            missing += 1
    print(f"Parsed {len(stats_cache)} unique stats files ({missing} missing)")

    output_rows = []
    extra_fields = ['dram_energy_uj', 'compute_energy_uj',
                    'bw_throttled', 'original_cycles', 'original_utilization']
    fieldnames_out = list(fieldnames_in) + extra_fields

    for row in input_rows:
        stats_path = find_stats_file(outputs_dir, row)
        dram_stats = stats_cache.get(stats_path) if stats_path else None

        if row.get('cycles'):
            try:
                original_cycles = int(float(row['cycles']))
            except (ValueError, OverflowError):
                original_cycles = 0
        elif row.get('latency_s'):
            try:
                original_cycles = int(float(row['latency_s']) / cycle_time)
            except (ValueError, OverflowError):
                original_cycles = 0
        else:
            original_cycles = 0

        original_util = float(row.get('utilization', 0))
        # Raw sweep utilization is 0-1 (already normalized by run_sweep.py)
        tp_degree = int(float(row.get('tp_degree', 1)))
        ideal_cycles = original_cycles * original_util if original_util > 0 else 0

        for di in target_drams:
            for do in target_drams:
                new_row = dict(row)
                for k in ('dram_i', 'dram_o', 'dram_I', 'dram_O'):
                    if k in new_row:
                        new_row[k] = di if k.endswith(('i', 'I')) else do
                new_row['original_cycles'] = original_cycles
                new_row['original_utilization'] = original_util

                if dram_stats:
                    new_cycles, new_energy, dram_e, compute_e = \
                        apply_bw_throttling_from_stats(
                            dram_stats, di, do, tp_degree, original_cycles)
                    new_util = ideal_cycles / new_cycles if new_cycles > 0 else 0
                    new_row['cycles'] = new_cycles
                    new_row['utilization'] = round(new_util, 6)
                    new_row['energy_uj'] = round(new_energy, 2)
                    new_row['latency_s'] = new_cycles * cycle_time
                    new_row['dram_energy_uj'] = round(dram_e, 2)
                    new_row['compute_energy_uj'] = round(compute_e, 2)
                    new_row['bw_throttled'] = 1 if new_cycles > original_cycles else 0
                else:
                    new_row['dram_energy_uj'] = 0
                    new_row['compute_energy_uj'] = 0
                    new_row['bw_throttled'] = -1

                output_rows.append(new_row)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_out, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {output_path}")
    _print_summary(output_rows)


def _print_summary(output_rows):
    """Print DRAM pair summary."""
    pair_counts = Counter()
    pair_throttled = Counter()
    for r in output_rows:
        di = r.get('dram_i') or r.get('dram_I', '?')
        do = r.get('dram_o') or r.get('dram_O', '?')
        pair = (di, do)
        pair_counts[pair] += 1
        if r.get('bw_throttled') == 1:
            pair_throttled[pair] += 1
    print("  %-10s %-10s %8s %8s" % ("dram_i", "dram_o", "rows", "throttled"))
    for pair in sorted(pair_counts.keys()):
        print("  %-10s %-10s %8d %8d" % (
            pair[0], pair[1], pair_counts[pair], pair_throttled[pair]))


if __name__ == "__main__":
    main()
