#!/usr/bin/env python3
"""
run_remap_cross_eval_v2.py — Enhanced cross-evaluate LLaMA 70B prefill → decode remap.

Improvements over v1:
  - Pickle caching for unified_database (4.6GB CSV → ~200MB pickle, 10x faster load)
  - Extended network combinations: all prefill lengths × all decode KV lengths
  - Per-network EDP/energy breakdown at each n_chiplets
  - Detailed analysis CSV output

Usage (inside docker or host with pandas):
    python3 run_remap_cross_eval_v2.py \
        --csv archgym_results/v6_energy_sa/incremental_chiplet_sweep_energy_20260407_032034.csv \
        --database unified_database.csv \
        --output-dir remap_results_70b
"""

import os
import sys
import ast
import json
import csv
import pickle
import hashlib
import argparse
import time
from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CHIPLET_TL_DIR = os.path.normpath(os.path.join(THIS_DIR, '..'))
PR_DIR = os.path.join(CHIPLET_TL_DIR, 'P&R')
for p in [THIS_DIR, CHIPLET_TL_DIR, PR_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from network_dataclass import VirtualNetwork
from global_parameter import NET_DIR, dram_type_bandwidth_width_dict
from cal_perf_phy_net import create_cp_spec, calculate_inter_chiplet_communication
from generate_pnr_config import (
    parse_config_id, dag_binary_to_operator_groups,
    load_virtual_network,
)


# ---------------------------------------------------------------------------
# Pickle caching for unified_database
# ---------------------------------------------------------------------------

def _get_cache_path(db_path):
    """Return pickle cache path for given database CSV."""
    cache_dir = os.path.join(os.path.dirname(db_path), '.db_cache')
    os.makedirs(cache_dir, exist_ok=True)
    # Use file mtime + size as cache key (faster than hashing 4.6GB)
    stat = os.stat(db_path)
    cache_key = f"{os.path.basename(db_path)}_{stat.st_size}_{int(stat.st_mtime)}"
    return os.path.join(cache_dir, f"70b_index_{cache_key}.pkl")


def _build_db_index(db):
    """Pre-index database by (net, layer, batch, seq, arch, glb, pe_x, pe_y)
    for O(1) lookups instead of O(N) pandas filters."""
    index = {}
    for _, row in db.iterrows():
        key = (row['net'], row['layer_name'], int(row['batch_size']),
               int(row['sequence_length']), row['arch_target'],
               int(row['glb_scale']), int(row['pe_x_scale']), int(row['pe_y_scale']))
        e = row['dynamic_energy']
        if key not in index or e < index[key]['dynamic_energy_J']:
            index[key] = {
                'latency_s': float(row['latency']),
                'dynamic_energy_J': float(row['dynamic_energy']),
                'static_power_W': float(row['static_power']),
                'area_um2': float(row['area']),
                'utilization': float(row.get('utilization', 0)),
                'i_access': int(row.get('i_access', 0)) if not pd.isna(row.get('i_access', 0)) else 0,
                'w_access': int(row.get('w_access', 0)) if not pd.isna(row.get('w_access', 0)) else 0,
                'o_access': int(row.get('o_access', 0)) if not pd.isna(row.get('o_access', 0)) else 0,
                'dram_i': str(row.get('dram_i', '')),
                'dram_o': str(row.get('dram_o', '')),
            }
    return index


def load_70b_database(db_path):
    """Load unified_database filtered to 70B networks, with pickle caching."""
    cache_path = _get_cache_path(db_path)

    if os.path.exists(cache_path):
        print(f"  Loading cached index from {cache_path}...")
        t0 = time.time()
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        dt = time.time() - t0
        print(f"  Loaded {len(cached['db_index'])} index keys, "
              f"{len(cached['db_layers_per_net'])} nets in {dt:.1f}s")
        return cached['db_index'], cached['db_layers_per_net']

    print(f"  Reading database: {db_path} (filtering to 70b only)...")
    t0 = time.time()
    chunks = []
    total_rows = 0
    for chunk in pd.read_csv(db_path, chunksize=500000):
        total_rows += len(chunk)
        filtered = chunk[chunk['net'].str.contains('70b', na=False)]
        if not filtered.empty:
            chunks.append(filtered)
        if total_rows % 5000000 == 0:
            print(f"    Processed {total_rows/1e6:.1f}M rows...")

    db = pd.concat(chunks, ignore_index=True)
    dt_read = time.time() - t0
    print(f"  Loaded {len(db)} rows for 70B networks in {dt_read:.1f}s")

    db_layers_per_net = {
        net: set(sub['layer_name'].unique())
        for net, sub in db.groupby('net')
    }
    print(f"  Found nets: {sorted(db_layers_per_net.keys())}")

    print(f"  Building O(1) lookup index...")
    t1 = time.time()
    db_index = _build_db_index(db)
    dt_index = time.time() - t1
    print(f"  Index size: {len(db_index)} unique keys in {dt_index:.1f}s")

    # Save to pickle cache
    print(f"  Saving cache to {cache_path}...")
    t2 = time.time()
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'db_index': db_index,
            'db_layers_per_net': db_layers_per_net,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    dt_cache = time.time() - t2
    print(f"  Cache saved in {dt_cache:.1f}s")

    del db  # free memory
    return db_index, db_layers_per_net


def query_operator_stats_indexed(db_index, net_name, layer_name, batch_size, seq_len,
                                  arch, glb, pe_x, pe_y, tp=1, mapper=0):
    """Fast O(1) lookup from pre-indexed database."""
    key = (net_name, layer_name, int(batch_size), int(seq_len), arch,
           int(glb), int(pe_x), int(pe_y))
    return db_index.get(key)


# ---------------------------------------------------------------------------
# Remap evaluation (same as v1)
# ---------------------------------------------------------------------------

def evaluate_remap(
    source_config_ids,
    source_gene,
    target_net_name,
    target_batch_size,
    target_seq_len,
    db_index,
    db_layers_per_net,
):
    """Evaluate a target network on a source network's chiplet topology."""
    target_db_layers = db_layers_per_net.get(target_net_name)
    if not target_db_layers:
        return None

    vn = load_virtual_network(
        target_net_name, target_batch_size, target_seq_len, target_db_layers)
    if not vn.layers:
        return None

    parsed_stages = [parse_config_id(cid) for cid in source_config_ids]
    binary_string = source_gene.get('binary_string', '')
    buffer_config = source_gene.get('buffer_config', [])

    cp_groups, off_cp_ops = dag_binary_to_operator_groups(vn, binary_string)

    n_cp_stages = len(cp_groups)
    n_offcp = len(off_cp_ops)

    stage_specs = []
    for stage_idx, parsed in enumerate(parsed_stages):
        arch = parsed.get('arch')
        glb = parsed.get('glb_scale', 1) or 1
        pe_x = parsed.get('pe_x', 1) or 1
        pe_y = parsed.get('pe_y', 1) or 1
        tp = parsed.get('tp', 1) or 1
        mapper = parsed.get('mapper', 0) or 0

        sub_configs = parsed.get('sub_configs', [])
        if parsed.get('is_vg') and sub_configs:
            sc = sub_configs[0]
            arch = sc.get('arch', arch)
            glb = sc.get('glb_scale', glb)
            pe_x = sc.get('pe_x', pe_x)
            pe_y = sc.get('pe_y', pe_y)

        stage_specs.append({
            'parsed': parsed,
            'arch': arch, 'glb': glb, 'pe_x': pe_x, 'pe_y': pe_y,
            'tp': tp, 'mapper': mapper,
            'sub_configs': sub_configs,
            'is_vg': parsed.get('is_vg', False),
            'is_offcp': parsed.get('is_offcp', False),
        })

    total_latency = 0.0
    total_energy = 0.0
    total_ops = 0
    unassigned_count = 0
    stage_details = []

    for stage_idx in range(len(source_config_ids)):
        if stage_idx < n_cp_stages:
            group_info = cp_groups[stage_idx]
            group_ops = group_info['operators']
            is_parallel = group_info.get('is_parallel', False)
        elif stage_idx - n_cp_stages < n_offcp:
            off_info = off_cp_ops[stage_idx - n_cp_stages]
            group_ops = [off_info['operator']]
            is_parallel = False
        else:
            group_ops = []
            is_parallel = False

        if not group_ops:
            continue

        spec = stage_specs[stage_idx] if stage_idx < len(stage_specs) else stage_specs[-1]
        sub_configs = spec['sub_configs']

        stage_lat = 0.0
        stage_eng = 0.0
        op_details = []

        for op_idx, op_name in enumerate(group_ops):
            if sub_configs and op_idx < len(sub_configs):
                sc = sub_configs[op_idx]
                a = sc.get('arch', spec['arch'])
                g = sc.get('glb_scale', spec['glb'])
                px = sc.get('pe_x', spec['pe_x'])
                py = sc.get('pe_y', spec['pe_y'])
            else:
                a, g, px, py = spec['arch'], spec['glb'], spec['pe_x'], spec['pe_y']

            stats = query_operator_stats_indexed(
                db_index, target_net_name, op_name,
                target_batch_size, target_seq_len,
                a, g, px, py, spec['tp'], spec['mapper'])

            if stats is None:
                unassigned_count += 1
                total_ops += 1
                continue

            if is_parallel:
                stage_lat = max(stage_lat, stats['latency_s'])
            else:
                stage_lat += stats['latency_s']
            stage_eng += stats['dynamic_energy_J']
            total_ops += 1

            op_details.append({
                'name': op_name,
                'latency_s': stats['latency_s'],
                'energy_J': stats['dynamic_energy_J'],
            })

        total_latency += stage_lat
        total_energy += stage_eng
        stage_details.append({
            'stage_idx': stage_idx,
            'arch': spec['arch'],
            'n_ops': len(group_ops),
            'latency_s': stage_lat,
            'energy_J': stage_eng,
            'is_parallel': is_parallel,
            'operators': op_details,
        })

    if unassigned_count > 0 and unassigned_count == total_ops:
        return {
            'feasible': False,
            'n_assigned': total_ops - unassigned_count,
            'n_total': total_ops,
        }

    comm_energy = 0.0
    for i in range(min(n_cp_stages - 1, len(source_config_ids) - 1)):
        if i >= len(stage_details) or i + 1 >= len(stage_details):
            continue
        src_stage = stage_details[i]
        dst_stage = stage_details[i + 1]
        if not src_stage['operators'] or not dst_stage['operators']:
            continue
        src_spec = stage_specs[src_stage['stage_idx']] if src_stage['stage_idx'] < len(stage_specs) else stage_specs[-1]
        for op_info in src_stage['operators']:
            stats = query_operator_stats_indexed(
                db_index, target_net_name, op_info['name'],
                target_batch_size, target_seq_len,
                src_spec['arch'], src_spec['glb'], src_spec['pe_x'], src_spec['pe_y'],
                src_spec['tp'], src_spec['mapper'])
            if stats:
                bonding = src_spec['parsed'].get('bonding', '2.5D')
                comm_bits = stats['o_access'] * 16  # FP16
                comm_energy += calculate_inter_chiplet_communication(
                    comm_bits, bonding, num_hop=1)

    total_energy += comm_energy
    edp = total_latency * total_energy

    return {
        'feasible': True,
        'total_latency_s': total_latency,
        'total_energy_J': total_energy,
        'total_edp': edp,
        'comm_energy_J': comm_energy,
        'n_stages_used': len([s for s in stage_details if s['operators']]),
        'n_stages_total': len(source_config_ids),
        'n_ops': total_ops,
        'n_unassigned': unassigned_count,
        'stage_details': stage_details,
    }


# ---------------------------------------------------------------------------
# Network definitions
# ---------------------------------------------------------------------------

# All LLaMA 70B variants: (csv_key_prefix, net_name, batch_size, seq_len)
LLAMA_70B_NETWORKS = {
    # Prefill variants
    'prefill_s512_b1': ('llama3.1_70b_prefill_s512_b1_seq512',
                        'llama3.1_70b_prefill_s512', 1, 512),
    'prefill_s1024_b1': ('llama3.1_70b_prefill_s1024_b1_seq1024',
                         'llama3.1_70b_prefill_s1024', 1, 1024),
    'prefill_s2048_b1': ('llama3.1_70b_prefill_s2048_b1_seq2048',
                         'llama3.1_70b_prefill_s2048', 1, 2048),
    'prefill_s4096_b1': ('llama3.1_70b_prefill_s4096_b1_seq4096',
                         'llama3.1_70b_prefill_s4096', 1, 4096),
    # Decode variants
    'decode_kv512_b1': ('llama3.1_70b_decode_kv512_b1_seq1',
                        'llama3.1_70b_decode_kv512', 1, 1),
    'decode_kv1024_b1': ('llama3.1_70b_decode_kv1024_b1_seq1',
                         'llama3.1_70b_decode_kv1024', 1, 1),
    'decode_kv2048_b1': ('llama3.1_70b_decode_kv2048_b1_seq1',
                         'llama3.1_70b_decode_kv2048', 1, 1),
    'decode_kv4096_b1': ('llama3.1_70b_decode_kv4096_b1_seq1',
                         'llama3.1_70b_decode_kv4096', 1, 1),
}

LLAMA_70B_NETWORKS_B8 = {
    'prefill_s1024_b8': ('llama3.1_70b_prefill_s1024_b8_seq1024',
                         'llama3.1_70b_prefill_s1024', 8, 1024),
    'decode_kv1024_b8': ('llama3.1_70b_decode_kv1024_b8_seq1',
                         'llama3.1_70b_decode_kv1024', 8, 1),
}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_cross_eval(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Load database with caching
    print(f"{'='*70}")
    print(f"Loading unified database...")
    print(f"{'='*70}")
    db_index, db_layers_per_net = load_70b_database(args.database)

    print(f"\nLoading CSV: {args.csv}")
    sweep_df = pd.read_csv(args.csv)
    n_chiplets_list = sorted(sweep_df['n_chiplets'].unique())
    print(f"  n_chiplets range: {n_chiplets_list}")

    # Determine available source configs from CSV
    available_sources = {}
    all_nets = {**LLAMA_70B_NETWORKS, **LLAMA_70B_NETWORKS_B8}
    for key, (csv_key, net_name, bs, sl) in all_nets.items():
        config_col = f'{csv_key}_config'
        gene_col = f'{csv_key}_gene'
        if config_col in sweep_df.columns and gene_col in sweep_df.columns:
            available_sources[key] = (csv_key, net_name, bs, sl)

    print(f"\nAvailable source networks (have optimized configs in CSV):")
    for key in sorted(available_sources):
        print(f"  {key}: {available_sources[key][0]}")

    # Target networks: all 70B variants available in unified_database
    target_networks = {}
    for key, (csv_key, net_name, bs, sl) in all_nets.items():
        if net_name in db_layers_per_net:
            target_networks[key] = (csv_key, net_name, bs, sl)

    print(f"\nTarget networks (available in unified_database):")
    for key in sorted(target_networks):
        print(f"  {key}: {target_networks[key][1]}")

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    eval_count = 0
    total_evals = len(n_chiplets_list) * len(available_sources) * len(target_networks)
    print(f"\nTotal evaluations planned: {total_evals}")

    for n_chip in n_chiplets_list:
        print(f"\n{'='*70}")
        print(f"n_chiplets = {n_chip}")
        print(f"{'='*70}")

        row = sweep_df[sweep_df['n_chiplets'] == n_chip].iloc[0]

        for src_key, (src_csv_key, src_net, src_bs, src_sl) in sorted(available_sources.items()):
            config_col = f'{src_csv_key}_config'
            gene_col = f'{src_csv_key}_gene'

            config_str = row.get(config_col)
            gene_str = row.get(gene_col)

            if pd.isna(config_str) or pd.isna(gene_str):
                print(f"  [SKIP] {src_key}: no config at n_chiplets={n_chip}")
                continue

            config_ids = ast.literal_eval(config_str)
            gene = ast.literal_eval(gene_str)

            src_energy_col = f'{src_csv_key}_min_energy'
            src_lat_col = f'{src_csv_key}_latency'
            src_own_energy = float(row.get(src_energy_col, 0)) if not pd.isna(row.get(src_energy_col, np.nan)) else None
            src_own_latency = float(row.get(src_lat_col, 0)) if not pd.isna(row.get(src_lat_col, np.nan)) else None

            print(f"\n  Source: {src_key} ({len(config_ids)} stages)")
            if src_own_energy:
                print(f"    Own energy: {src_own_energy:.6e} J, latency: {src_own_latency:.6e} s")

            for tgt_key, (tgt_csv_key, tgt_net, tgt_bs, tgt_sl) in sorted(target_networks.items()):
                eval_count += 1

                # Get target's own optimal from CSV
                tgt_energy_col = f'{tgt_csv_key}_min_energy'
                tgt_lat_col = f'{tgt_csv_key}_latency'
                tgt_own_energy = float(row.get(tgt_energy_col, 0)) if tgt_energy_col in row and not pd.isna(row.get(tgt_energy_col, np.nan)) else None
                tgt_own_latency = float(row.get(tgt_lat_col, 0)) if tgt_lat_col in row and not pd.isna(row.get(tgt_lat_col, np.nan)) else None

                result = evaluate_remap(
                    config_ids, gene,
                    tgt_net, tgt_bs, tgt_sl,
                    db_index, db_layers_per_net,
                )

                rec = {
                    'n_chiplets': n_chip,
                    'source': src_key,
                    'source_net': src_net,
                    'source_type': 'prefill' if 'prefill' in src_key else 'decode',
                    'target': tgt_key,
                    'target_net': tgt_net,
                    'target_type': 'prefill' if 'prefill' in tgt_key else 'decode',
                    'source_own_energy': src_own_energy,
                    'source_own_latency': src_own_latency,
                    'target_own_energy': tgt_own_energy,
                    'target_own_latency': tgt_own_latency,
                }

                if result is None:
                    rec.update({'feasible': False, 'remap_energy': None,
                                'remap_latency': None, 'remap_edp': None,
                                'comm_energy': None, 'n_stages_used': None,
                                'unassigned': None})
                    print(f"    → {tgt_key}: INFEASIBLE (no ops) [{eval_count}/{total_evals}]")
                elif not result.get('feasible', True):
                    n_unassigned = result.get('n_total', 0) - result.get('n_assigned', 0)
                    rec.update({'feasible': False, 'remap_energy': None,
                                'remap_latency': None, 'remap_edp': None,
                                'comm_energy': None, 'n_stages_used': None,
                                'unassigned': n_unassigned})
                    print(f"    → {tgt_key}: INFEASIBLE ({n_unassigned} unassigned) [{eval_count}/{total_evals}]")
                else:
                    rec.update({
                        'feasible': True,
                        'remap_energy': result['total_energy_J'],
                        'remap_latency': result['total_latency_s'],
                        'remap_edp': result['total_edp'],
                        'comm_energy': result.get('comm_energy_J', 0),
                        'n_stages_used': result['n_stages_used'],
                        'unassigned': result.get('n_unassigned', 0),
                    })
                    if tgt_own_energy and tgt_own_energy > 0:
                        rec['energy_overhead_pct'] = (result['total_energy_J'] - tgt_own_energy) / tgt_own_energy * 100
                    if tgt_own_latency and tgt_own_latency > 0:
                        rec['latency_overhead_pct'] = (result['total_latency_s'] - tgt_own_latency) / tgt_own_latency * 100

                    overhead_str = ""
                    if 'energy_overhead_pct' in rec:
                        overhead_str = f", E_oh: {rec['energy_overhead_pct']:+.1f}%"
                    print(f"    → {tgt_key}: E={result['total_energy_J']:.4e}, "
                          f"L={result['total_latency_s']:.4e}, "
                          f"EDP={result['total_edp']:.4e}{overhead_str} [{eval_count}/{total_evals}]")

                all_results.append(rec)

    # Save results
    out_csv = os.path.join(args.output_dir, f'remap_cross_eval_v2_{timestamp}.csv')
    df_out = pd.DataFrame(all_results)
    df_out.to_csv(out_csv, index=False)
    print(f"\n\nResults saved to {out_csv}")
    print(f"Total evaluations: {len(all_results)}")

    # Generate analysis
    print_analysis(df_out, args.output_dir, timestamp)

    return out_csv


def print_analysis(df, output_dir, timestamp):
    """Detailed analysis of prefill→decode remap impact."""
    feasible = df[df['feasible'] == True].copy()
    if feasible.empty:
        print("\nNo feasible remap results found!")
        return

    prefill_sources = sorted([s for s in feasible['source'].unique() if 'prefill' in s])
    decode_targets = sorted([t for t in feasible['target'].unique() if 'decode' in t])

    # ===== Table 1: Prefill → Decode Energy =====
    if prefill_sources and decode_targets:
        print(f"\n{'='*90}")
        print("ANALYSIS 1: Prefill config → Decode workload ENERGY (J)")
        print(f"{'='*90}")

        for n_chip in sorted(feasible['n_chiplets'].unique()):
            subset = feasible[feasible['n_chiplets'] == n_chip]
            print(f"\n  n_chiplets = {n_chip}")
            col_hdr = "Source / Target"
            header = f"  {col_hdr:<25s}" + "".join(
                f"{t:>20s}" for t in decode_targets)
            print(header)
            print("  " + "-" * (25 + 20 * len(decode_targets)))

            for src in prefill_sources:
                row_str = f"  {src:<25s}"
                for tgt in decode_targets:
                    match = subset[(subset['source'] == src) & (subset['target'] == tgt)]
                    if not match.empty and pd.notna(match.iloc[0]['remap_energy']):
                        val = match.iloc[0]['remap_energy']
                        row_str += f"{val:>20.4e}"
                    else:
                        na_str = "N/A"
                        row_str += f"{na_str:>20s}"
                print(row_str)

            # Own optimal row
            own_label = "[OWN OPTIMAL]"
            row_str = f"  {own_label:<25s}"
            for tgt in decode_targets:
                match = subset[subset['target'] == tgt]
                if not match.empty:
                    val = match.iloc[0].get('target_own_energy')
                    if val and not pd.isna(val):
                        row_str += f"{val:>20.4e}"
                    else:
                        row_str += f"{'N/A':>20s}"
                else:
                    row_str += f"{'N/A':>20s}"
            print(row_str)

    # ===== Table 2: Prefill → Decode EDP =====
    if prefill_sources and decode_targets:
        print(f"\n{'='*90}")
        print("ANALYSIS 2: Prefill config → Decode workload EDP")
        print(f"{'='*90}")

        for n_chip in sorted(feasible['n_chiplets'].unique()):
            subset = feasible[feasible['n_chiplets'] == n_chip]
            print(f"\n  n_chiplets = {n_chip}")
            col_hdr = "Source / Target"
            header = f"  {col_hdr:<25s}" + "".join(
                f"{t:>20s}" for t in decode_targets)
            print(header)
            print("  " + "-" * (25 + 20 * len(decode_targets)))

            for src in prefill_sources:
                row_str = f"  {src:<25s}"
                for tgt in decode_targets:
                    match = subset[(subset['source'] == src) & (subset['target'] == tgt)]
                    if not match.empty and pd.notna(match.iloc[0]['remap_edp']):
                        val = match.iloc[0]['remap_edp']
                        row_str += f"{val:>20.4e}"
                    else:
                        na_str = "N/A"
                        row_str += f"{na_str:>20s}"
                print(row_str)

    # ===== Table 3: Energy overhead % =====
    if prefill_sources and decode_targets:
        print(f"\n{'='*90}")
        print("ANALYSIS 3: Energy Overhead % (remap vs target's own optimal)")
        print(f"{'='*90}")

        for n_chip in sorted(feasible['n_chiplets'].unique()):
            subset = feasible[feasible['n_chiplets'] == n_chip]
            print(f"\n  n_chiplets = {n_chip}")
            col_hdr = "Source / Target"
            header = f"  {col_hdr:<25s}" + "".join(
                f"{t:>20s}" for t in decode_targets)
            print(header)
            print("  " + "-" * (25 + 20 * len(decode_targets)))

            for src in prefill_sources:
                row_str = f"  {src:<25s}"
                for tgt in decode_targets:
                    match = subset[(subset['source'] == src) & (subset['target'] == tgt)]
                    if not match.empty and 'energy_overhead_pct' in match.columns:
                        val = match.iloc[0].get('energy_overhead_pct')
                        if pd.notna(val):
                            row_str += f"{val:>19.1f}%"
                        else:
                            na_str = "N/A"
                            row_str += f"{na_str:>20s}"
                    else:
                        na_str = "N/A"
                        row_str += f"{na_str:>20s}"
                print(row_str)

    # ===== Table 4: Per-network summary across n_chiplets =====
    print(f"\n{'='*90}")
    print("ANALYSIS 4: Best remap energy per (source, target) across all n_chiplets")
    print(f"{'='*90}")

    for src in prefill_sources:
        for tgt in decode_targets:
            subset = feasible[(feasible['source'] == src) & (feasible['target'] == tgt)]
            if subset.empty:
                continue
            best_idx = subset['remap_energy'].idxmin()
            best = subset.loc[best_idx]
            overhead_str = ""
            if 'energy_overhead_pct' in best and pd.notna(best.get('energy_overhead_pct')):
                overhead_str = f", overhead: {best['energy_overhead_pct']:+.1f}%"
            print(f"  {src:<25s} → {tgt:<20s}: "
                  f"best E={best['remap_energy']:.4e} at n_chiplets={int(best['n_chiplets'])}"
                  f"{overhead_str}")

    # ===== Save summary JSON =====
    summary = {
        'timestamp': timestamp,
        'n_feasible': len(feasible),
        'n_total': len(df),
        'prefill_sources': prefill_sources,
        'decode_targets': decode_targets,
    }

    per_nchip = {}
    for n_chip in sorted(feasible['n_chiplets'].unique()):
        subset = feasible[feasible['n_chiplets'] == n_chip]
        per_nchip[int(n_chip)] = {
            'n_evals': len(subset),
            'avg_energy': float(subset['remap_energy'].mean()),
            'min_energy': float(subset['remap_energy'].min()),
            'max_energy': float(subset['remap_energy'].max()),
        }

        # Per source-target breakdown
        for src in prefill_sources:
            for tgt in decode_targets:
                match = subset[(subset['source'] == src) & (subset['target'] == tgt)]
                if not match.empty and pd.notna(match.iloc[0]['remap_energy']):
                    key_name = f"{src}_to_{tgt}"
                    per_nchip[int(n_chip)][key_name] = {
                        'energy': float(match.iloc[0]['remap_energy']),
                        'latency': float(match.iloc[0]['remap_latency']),
                        'edp': float(match.iloc[0]['remap_edp']),
                    }

    summary['per_n_chiplets'] = per_nchip

    summary_path = os.path.join(output_dir, f'remap_summary_v2_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced LLaMA 70B prefill → decode remap cross-evaluation')
    parser.add_argument('--csv', required=True,
                        help='Path to incremental_chiplet_sweep_energy CSV')
    parser.add_argument('--database', default='unified_database.csv',
                        help='Path to unified_database.csv')
    parser.add_argument('--output-dir', default='remap_results_70b',
                        help='Output directory')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force rebuild of database cache')
    args = parser.parse_args()

    if args.no_cache:
        cache_path = _get_cache_path(args.database)
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"Removed cache: {cache_path}")

    run_cross_eval(args)


if __name__ == '__main__':
    main()
