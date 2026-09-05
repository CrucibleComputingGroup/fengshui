#!/usr/bin/env python3
"""
reoptimize_pnr_config.py — Read chiplet pool from CSV, re-run GA optimization
with the current cal_perf_phy_net, and generate fresh PnR configs.

Unlike generate_pnr_config.py which reads gene/config directly from CSV,
this script only reads the chiplet pool composition from CSV and then
re-searches for optimal chiplet assignments using the updated evaluation.

Usage:
    python3 reoptimize_pnr_config.py \
        --csv archgym_results/edp_saeo/incremental_chiplet_sweep_edp_*.csv \
        --database timeloop_experiments/unified_database.csv \
        --n-chiplets 8 \
        --objective edp \
        --output-dir pnr_configs_edp_reopt/
"""

import os
import sys
import json
import math
import argparse
import time
import pandas as pd
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
_CHIPLET_TL_DIR = os.path.normpath(os.path.join(THIS_DIR, '..'))
if _CHIPLET_TL_DIR not in sys.path:
    sys.path.insert(0, _CHIPLET_TL_DIR)
_EXP_DIR = os.path.normpath(os.path.join(THIS_DIR, '..', 'timeloop_experiments'))
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

from chiplet_dataclass import ChipletConfig
from network_dataclass import VirtualNetwork
from global_parameter import NET_DIR, dram_type_bandwidth_width_dict
from cal_perf_phy_net import preload_database, create_cp_spec
from chiplet_sel import run_single_optimization
from utility_functions import config_to_ids
from compute_area import get_chiplet_area_mm2
from generate_pnr_config import (
    parse_config_id, _unified_phy_mm2, _unified_ctrl_mm2, _build_aggregate,
    DRAM_SPECS, _DRAM_MODULE_SPECS, _PHY_SPECS,
    get_chiplet_identifier, query_operator_stats,
    dag_binary_to_operator_groups, load_virtual_network,
)


# ---------------------------------------------------------------------------
# Read chiplet pool from CSV → ChipletConfig objects
# ---------------------------------------------------------------------------

def read_chiplet_pool_from_csv(csv_path, n_chiplets):
    """Read chiplet pool composition from incremental sweep CSV.

    Returns:
        List[ChipletConfig]: chiplet pool as ChipletConfig objects
        List[dict]: chiplet pool metadata dicts (for PnR JSON)
    """
    df = pd.read_csv(csv_path)
    row = df[df['n_chiplets'] == n_chiplets]
    if row.empty:
        raise ValueError(f"No row with n_chiplets={n_chiplets} in {csv_path}")
    row = row.iloc[0]

    chiplet_configs = []
    pool_metadata = []

    for i in range(1, n_chiplets + 1):
        arch = row.get(f'chiplet_{i}_arch')
        if pd.isna(arch):
            continue
        glb = int(row.get(f'chiplet_{i}_glb_scale', 1))
        pe_x = int(row.get(f'chiplet_{i}_pe_x_scale', 1))
        pe_y = int(row.get(f'chiplet_{i}_pe_y_scale', 1))

        chiplet_configs.append(ChipletConfig(
            arch_target=arch,
            global_buffer_size_scale=glb,
            pe_x_scale=pe_x,
            pe_y_scale=pe_y,
        ))

        # Also build metadata dict for PnR JSON output
        meta = {
            'index': i,
            'identifier': get_chiplet_identifier(arch, glb, pe_x, pe_y),
            'architecture': arch,
            'glb_scale': glb,
            'pe_x_scale': pe_x,
            'pe_y_scale': pe_y,
            'glb_capacity_KB': glb * 108,
            'pe_array_dim': f"{pe_x * 64} x {pe_y * 64}",
            'pe_count': pe_x * pe_y * 64 * 64 if arch not in ('PIM', 'switch_8port') else 0,
        }

        if arch == 'PIM':
            meta['type'] = 'processing_in_memory'
            meta['memory_tech'] = 'GDDR7'
            meta['die_area_mm2'] = 224.0
        elif arch == 'switch_8port':
            meta['type'] = 'switch'
            meta['num_ports'] = 8
        else:
            meta['type'] = 'compute'
            dataflow_map = {
                'eyeriss_like': 'row_stationary',
                'simba_like': 'weight_stationary',
                'gemmini_like': 'output_stationary',
            }
            meta['dataflow'] = dataflow_map.get(arch, arch)
            area_bd = get_chiplet_area_mm2(arch, glb_scale=glb, pe_x_scale=pe_x, pe_y_scale=pe_y)
            phy = _unified_phy_mm2('2.5D')
            ctrl = _unified_ctrl_mm2()
            meta['area_breakdown_mm2'] = {
                'mac_area': area_bd['mac_area_mm2'],
                'buffer_area': area_bd['buffer_area_mm2'],
                'glb_area': area_bd['glb_area_mm2'],
                'memory_area': area_bd['buffer_area_mm2'] + area_bd['glb_area_mm2'],
                'vector_area': area_bd['vector_area_mm2'],
                'total_core_area': area_bd['total_area_mm2'],
                'phy_area': phy,
                'ctrl_area': ctrl,
                'total_die_area': area_bd['total_area_mm2'] + phy + ctrl,
            }

        pool_metadata.append(meta)

    return chiplet_configs, pool_metadata


# ---------------------------------------------------------------------------
# Build PnR config from fresh optimization results
# ---------------------------------------------------------------------------

def build_pnr_from_opt_result(
    net_key, net_name, batch_size, seq_len,
    best_gene, best_config, min_value,
    db, db_layers_per_net,
):
    """Build PnR config dict from optimization result (same output format
    as generate_pnr_config.generate_pnr_for_network)."""

    binary_string = best_gene['binary_string']
    buffer_config = best_gene['buffer_config']
    config_ids = config_to_ids(best_config)
    energy = min_value
    latency = best_config.get('latency', None)

    # Load virtual network
    db_layers = db_layers_per_net.get(net_name, None)
    vn = load_virtual_network(net_name, batch_size, seq_len, db_layers)

    # Reconstruct fusion groups using DAG-aware logic
    cp_groups, off_cp_ops = dag_binary_to_operator_groups(vn, binary_string)

    n_cp_stages = len(cp_groups)
    n_offcp = len(off_cp_ops)
    print(f"  Network: {net_name}, layers={len(vn.layers)}, "
          f"cp_stages={n_cp_stages}, off_cp={n_offcp}, config_ids={len(config_ids)}")

    from collections import defaultdict
    stages = []
    chiplet_usage = defaultdict(list)

    for stage_idx in range(len(config_ids)):
        cid = config_ids[stage_idx]
        parsed = parse_config_id(cid)

        # Determine operators
        offcp_feeds_from = None
        offcp_feeds_to = None
        if stage_idx < n_cp_stages:
            group_info = cp_groups[stage_idx]
            group_ops = group_info['operators']
            is_parallel = group_info['is_parallel']
            is_offcp = False
        elif stage_idx - n_cp_stages < n_offcp:
            off_info = off_cp_ops[stage_idx - n_cp_stages]
            group_ops = [off_info['operator']]
            is_parallel = False
            is_offcp = True
            offcp_feeds_from = off_info.get('start_pos', 0)
            offcp_feeds_to = off_info.get('end_pos', 0)
        else:
            group_ops = []
            is_parallel = False
            is_offcp = parsed.get('is_offcp', False)

        # Compute correct DRAM boundaries from buffer_config
        if stage_idx < n_cp_stages:
            buf_start = group_info['buf_start']
            buf_end = group_info['buf_end']
        elif is_offcp:
            buf_start = off_info.get('start_pos', 0)
            buf_end = off_info.get('end_pos', 0)
        else:
            buf_start = stage_idx if stage_idx < len(buffer_config) else len(buffer_config) - 1
            buf_end = buf_start + 1

        stage_dram_i = buffer_config[buf_start] if buf_start < len(buffer_config) else 'HBM3'
        stage_dram_o = buffer_config[buf_end] if buf_end < len(buffer_config) else 'HBM3'
        stage_dram = stage_dram_i  # primary DRAM type for the stage

        chiplet_id = get_chiplet_identifier(
            parsed.get('arch', 'unknown'),
            parsed.get('glb_scale', 1),
            parsed.get('pe_x', 1),
            parsed.get('pe_y', 1),
        )
        chiplet_usage[chiplet_id].append(stage_idx)

        # Query per-operator stats
        operators = []
        stage_latency = 0.0
        stage_energy = 0.0
        stage_area = 0.0
        stage_power = 0.0
        stage_total_data = 0

        for layer_name in group_ops:
            stats = query_operator_stats(
                db, net_name, layer_name, batch_size, seq_len,
                parsed.get('arch'), parsed.get('glb_scale', 1),
                parsed.get('pe_x', 1), parsed.get('pe_y', 1),
                parsed.get('tp', 1), parsed.get('mapper', 0),
            )
            op_info = {'name': layer_name}
            if stats:
                op_info.update({
                    'latency_s': stats['latency_s'],
                    'dynamic_energy_J': stats['dynamic_energy_J'],
                    'static_power_W': stats['static_power_W'],
                    'area_um2': stats['area_um2'],
                    'utilization': stats['utilization'],
                    'input_accesses': stats['i_access'],
                    'weight_accesses': stats['w_access'],
                    'output_accesses': stats['o_access'],
                    'dram_i': stage_dram_i,
                    'dram_o': stage_dram_o,
                })
                if is_parallel:
                    stage_latency = max(stage_latency, stats['latency_s'])
                else:
                    stage_latency += stats['latency_s']
                stage_energy += stats['dynamic_energy_J']
                stage_area = max(stage_area, stats['area_um2'])
                stage_power = max(stage_power, stats['static_power_W'])
                stage_total_data += stats['i_access'] + stats['w_access'] + stats['o_access']
            else:
                op_info['note'] = 'not found in database'
            operators.append(op_info)

        stage_info = {
            'stage_index': stage_idx,
            'fusion_group_operators': group_ops,
            'num_operators': len(group_ops),
            'is_parallel': is_parallel,
            'is_off_critical_path': is_offcp,
            'buf_start': buf_start,
            'buf_end': buf_end,
            'offcp_feeds_from_stage': offcp_feeds_from if is_offcp else None,
            'offcp_feeds_to_stage': offcp_feeds_to if is_offcp else None,
            'is_virtual_group': parsed.get('is_vg', False),
            'assigned_chiplet': {
                'identifier': chiplet_id,
                'architecture': parsed.get('arch'),
                'glb_scale': parsed.get('glb_scale'),
                'pe_x_scale': parsed.get('pe_x'),
                'pe_y_scale': parsed.get('pe_y'),
                'glb_capacity_KB': (parsed.get('glb_scale', 1) or 1) * 108,
                'pe_array_dim': f"{(parsed.get('pe_x', 1) or 1)*64} x "
                                f"{(parsed.get('pe_y', 1) or 1)*64}",
            },
            'tensor_parallelism': parsed.get('tp', 1),
            'mapper_index': parsed.get('mapper', 0),
            'bonding': parsed.get('bonding', '2.5D'),
            'dram_type': stage_dram,
            'dram_spec': DRAM_SPECS.get(stage_dram, {}),
            'aggregate': _build_aggregate(
                stage_latency, stage_energy, stage_power, stage_area,
                stage_total_data, parsed, stage_dram,
            ),
            'operators': operators,
        }
        stages.append(stage_info)

    # Inter-stage communication
    cp_stages = [s for s in stages if not s['is_off_critical_path']]
    communications = []
    for i in range(len(cp_stages) - 1):
        src = cp_stages[i]
        dst = cp_stages[i + 1]
        src_chiplet = src['assigned_chiplet']['identifier']
        dst_chiplet = dst['assigned_chiplet']['identifier']

        comm_volume = 0
        if src['operators']:
            last_op = src['operators'][-1]
            comm_volume = last_op.get('output_accesses', 0)

        is_inter_chiplet = (src_chiplet != dst_chiplet)
        comm_info = {
            'from_stage': src['stage_index'],
            'to_stage': dst['stage_index'],
            'from_chiplet': src_chiplet,
            'to_chiplet': dst_chiplet,
            'is_inter_chiplet': is_inter_chiplet,
            'data_volume_accesses': comm_volume,
            'data_volume_bytes': comm_volume * 2,
            'src_dram': src['dram_type'],
            'dst_dram': dst['dram_type'],
            'src_bonding': src['bonding'],
            'dst_bonding': dst['bonding'],
        }
        if is_inter_chiplet:
            comm_info['inter_chiplet_energy_pJ_per_bit'] = 1.3
        communications.append(comm_info)

    # Off-CP communication
    for s in stages:
        if s['is_off_critical_path'] and s['operators']:
            op = s['operators'][0]
            communications.append({
                'from_stage': 'off_cp',
                'to_stage': s['stage_index'],
                'from_chiplet': s['assigned_chiplet']['identifier'],
                'to_chiplet': s['assigned_chiplet']['identifier'],
                'is_inter_chiplet': False,
                'data_volume_accesses': op.get('output_accesses', 0),
                'data_volume_bytes': op.get('output_accesses', 0) * 2,
                'note': 'off-critical-path operator',
            })

    # Chiplet connectivity graph
    from collections import defaultdict
    connectivity = defaultdict(lambda: {'data_volume_bytes': 0, 'num_transfers': 0})
    for comm in communications:
        if comm.get('is_inter_chiplet', False):
            key = (comm['from_chiplet'], comm['to_chiplet'])
            connectivity[key]['data_volume_bytes'] += comm['data_volume_bytes']
            connectivity[key]['num_transfers'] += 1

    connectivity_list = []
    for (src, dst), info in connectivity.items():
        connectivity_list.append({
            'src_chiplet': src,
            'dst_chiplet': dst,
            'total_data_volume_bytes': info['data_volume_bytes'],
            'num_transfers': info['num_transfers'],
        })

    return {
        'network': net_name,
        'batch_size': batch_size,
        'sequence_length': seq_len,
        'csv_key': net_key,
        'total_energy_J': float(energy) if energy else None,
        'total_latency_s': float(latency) if latency else None,
        'gene': {'binary_string': binary_string, 'buffer_config': buffer_config},
        'num_stages': len(stages),
        'num_cp_stages': n_cp_stages,
        'num_offcp_stages': n_offcp,
        'stages': stages,
        'inter_stage_communication': communications,
        'chiplet_connectivity': connectivity_list,
        'chiplet_utilization': {
            cid: {'stages_assigned': idxs, 'num_stages': len(idxs)}
            for cid, idxs in chiplet_usage.items()
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Re-optimize PnR configs: read chiplet pool from CSV, '
                    're-run GA search with current cal_perf_phy_net')
    parser.add_argument('--csv', required=True,
                        help='Incremental sweep CSV (for chiplet pool composition)')
    parser.add_argument('--database', default='timeloop_experiments/unified_database.csv',
                        help='Performance database CSV')
    parser.add_argument('--n-chiplets', type=int, default=8,
                        help='Chiplet pool size to extract from CSV')
    parser.add_argument('--objective', default='edp', choices=['energy', 'edp'],
                        help='Optimization objective')
    parser.add_argument('--output-dir', default='pnr_configs_reopt',
                        help='Output directory for PnR JSON files')
    parser.add_argument('--pop-size', type=int, default=10,
                        help='GA population size (used in non-DAG mode)')
    parser.add_argument('--generations', type=int, default=20,
                        help='GA generations (used in non-DAG mode)')
    parser.add_argument('--dag', action='store_true', default=True,
                        help='Use DAG-aware critical-path GA (default: True)')
    parser.add_argument('--no-dag', action='store_false', dest='dag',
                        help='Disable DAG-aware critical-path GA')
    parser.add_argument('--include-cnn', action='store_true', default=False,
                        help='Include CNN workloads (replknet31b, mobilenet_v3_small)')
    args = parser.parse_args()

    # 1. Read chiplet pool from CSV
    print(f"Reading chiplet pool from: {args.csv} (n={args.n_chiplets})")
    chiplet_configs, pool_metadata = read_chiplet_pool_from_csv(args.csv, args.n_chiplets)
    print(f"  Pool size: {len(chiplet_configs)} chiplets")
    for c in chiplet_configs:
        print(f"    {c.get_identifier()}")

    # 2. Load target networks
    targets = [
        ('llama3.1_8b_prefill_s1024', 1, 1024),
        ('llama3.1_8b_decode_kv1024', 1, 1),
        ('qwen3_30b_a3b_prefill_s1024', 1, 1024),
        ('qwen3_30b_a3b_decode_kv1024', 1, 1),
        ('qwen3_235b_a22b_prefill_s1024', 1, 1024),
        ('qwen3_235b_a22b_decode_kv1024', 1, 1),
    ]
    if args.include_cnn:
        targets.append(('replknet31b', 1, 1))

    print(f"\nLoading performance database: {args.database}")
    db_full = pd.read_csv(args.database)
    db_layers_per_net = {
        net: set(db_full[db_full['net'] == net]['layer_name'].unique())
        for net in db_full['net'].unique()
    }

    virtual_nets = []
    for net_name, bs, seq in targets:
        vn = VirtualNetwork(net_name, batch_size=bs, sequence_length=seq)
        db_layers = db_layers_per_net.get(net_name, None)
        try:
            vn.load_from_dir(os.path.join(NET_DIR, net_name), db_layers=db_layers)
            if len(vn.layers) > 0:
                virtual_nets.append(vn)
                print(f"  Loaded {vn.get_unique_name()}: {len(vn.layers)} layers")
            else:
                print(f"  WARNING: {net_name} has 0 layers, skipping")
        except Exception as e:
            print(f"  WARNING: Failed to load {net_name}: {e}")

    if not virtual_nets:
        print("ERROR: No networks loaded successfully")
        sys.exit(1)

    # 3. Preload database for fast evaluation
    needed_nets = set(vn.network_name for vn in virtual_nets)
    preload_database(args.database, needed_nets=needed_nets)

    # 4. Run optimization with the fixed chiplet pool
    print(f"\n{'='*60}")
    print(f"Running GA optimization (objective={args.objective}, "
          f"pop={args.pop_size}, gen={args.generations}, dag={args.dag})")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    _, opt_results = run_single_optimization(
        virtual_nets=virtual_nets,
        chiplet_group=chiplet_configs,
        objective=args.objective,
        results_file=args.database,
        use_sequential=True,
        n_workers=min(8, len(virtual_nets)),
        use_dag_cp=args.dag,
        v_het_batch=True,
    )
    dt = time.perf_counter() - t0
    print(f"\nOptimization completed in {dt:.1f}s")

    # 5. Build PnR configs from optimization results
    os.makedirs(args.output_dir, exist_ok=True)

    dram_modules = {}
    for dname, dspec in _DRAM_MODULE_SPECS.items():
        dram_modules[dname] = {
            'module_capacity_GB': dspec['module_cap_GB'],
            'stack_layers': dspec['stack_layers'],
            'area_per_GB_mm2': dspec['area_per_GB'],
            'footprint_mm2': dspec['footprint_mm2'],
            'footprint_dimensions_mm': dspec['footprint_dimensions_mm'],
            'total_silicon_mm2': dspec['total_silicon_mm2'],
            'phy_area_mm2': _PHY_SPECS[dname]['phy_mm2'],
            'ctrl_area_mm2': _PHY_SPECS[dname]['ctrl_mm2'],
        }

    all_configs = {
        'metadata': {
            'source_csv': args.csv,
            'database': args.database,
            'n_chiplets': args.n_chiplets,
            'objective': args.objective,
            'reoptimized': True,
            'ga_pop_size': args.pop_size,
            'ga_generations': args.generations,
            'dag_mode': args.dag,
            'inter_chiplet_energy_pJ_per_bit': 1.3,
            'technology_node_nm': 14,
            'clock_GHz': 1.0,
            'dram_module_specs': dram_modules,
        },
        'chiplet_pool': pool_metadata,
        'networks': {},
    }

    for vn in virtual_nets:
        uname = vn.get_unique_name()
        net_name = vn.network_name
        bs = vn.batch_size
        seq = vn.sequence_length

        if uname not in opt_results:
            print(f"\n  WARNING: No optimization result for {uname}, skipping")
            continue

        nr = opt_results[uname]
        best_gene = nr['best_gene']
        best_config = nr['best_config']
        min_value = nr['min_value']

        print(f"\n--- Building PnR config for {uname} ---")
        print(f"  Objective value: {min_value:.4e}")
        print(f"  Latency: {nr['best_latency']:.4e} s")

        config = build_pnr_from_opt_result(
            uname, net_name, bs, seq,
            best_gene, best_config, min_value,
            db_full, db_layers_per_net,
        )
        if config:
            all_configs['networks'][uname] = config
            out_file = os.path.join(args.output_dir, f'pnr_{uname}.json')
            with open(out_file, 'w') as f:
                json.dump({
                    'metadata': all_configs['metadata'],
                    'chiplet_pool': pool_metadata,
                    'network': config,
                }, f, indent=2, default=str)
            print(f"  Saved: {out_file}")

    combined_file = os.path.join(args.output_dir, 'pnr_all_networks.json')
    with open(combined_file, 'w') as f:
        json.dump(all_configs, f, indent=2, default=str)
    print(f"\nSaved combined config: {combined_file}")

    # Summary
    print("\n" + "=" * 70)
    print("RE-OPTIMIZED PnR CONFIG SUMMARY")
    print("=" * 70)
    for net_key, cfg in all_configs['networks'].items():
        print(f"\n{net_key}:")
        print(f"  Stages: {cfg['num_cp_stages']} CP + {cfg['num_offcp_stages']} off-CP")
        if cfg['total_energy_J']:
            print(f"  Total {args.objective}: {cfg['total_energy_J']:.4e}")
        if cfg['total_latency_s']:
            print(f"  Total latency: {cfg['total_latency_s']:.4e} s")
        inter = sum(1 for c in cfg['inter_stage_communication']
                    if c.get('is_inter_chiplet', False))
        total = len(cfg['inter_stage_communication'])
        print(f"  Inter-chiplet transfers: {inter}/{total}")
        chiplets_used = {s['assigned_chiplet']['identifier'] for s in cfg['stages']}
        print(f"  Unique chiplets used: {len(chiplets_used)}")
        for s in cfg['stages']:
            tag = "OFF-CP" if s['is_off_critical_path'] else "CP"
            par = " (parallel)" if s['is_parallel'] else ""
            print(f"    Stage {s['stage_index']} [{tag}{par}]: "
                  f"{s['fusion_group_operators']} → "
                  f"{s['assigned_chiplet']['architecture']} "
                  f"(TP={s['tensor_parallelism']}, {s['bonding']}, "
                  f"DRAM={s['dram_type']})")


if __name__ == '__main__':
    main()
