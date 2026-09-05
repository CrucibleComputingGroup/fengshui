#!/usr/bin/env python3
"""
generate_pnr_config.py — Generate PnR (Place-and-Route) configurations from
chiplet pool optimization results.

For each target network, extracts:
  - Chiplet pool specification (arch, area, power, PE dims, GLB, DRAM)
  - Per-stage (fusion group) operator assignment, chiplet, latency, energy, area
  - Inter-stage communication data volume
  - DRAM types, bonding technique, tensor parallelism per stage
  - Chiplet connectivity graph

Usage:
    python3 generate_pnr_config.py \
        --csv archgym_results/energy_saeo/incremental_chiplet_sweep_energy_*.csv \
        --database unified_database.csv \
        --n-chiplets 8 \
        --output-dir pnr_configs/
"""

import os
import sys
import ast
import re
import json
import math
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
_CHIPLET_TL_DIR = os.path.normpath(os.path.join(THIS_DIR, '..'))
if _CHIPLET_TL_DIR not in sys.path:
    sys.path.insert(0, _CHIPLET_TL_DIR)
_EXP_DIR = os.path.normpath(os.path.join(THIS_DIR, '..', 'timeloop_experiments'))
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

from network_dataclass import VirtualNetwork
from global_parameter import (
    NET_DIR, dram_type_bandwidth_width_dict,
)
from cal_perf_phy_net import create_cp_spec
from compute_area import get_chiplet_area_mm2


# ---------------------------------------------------------------------------
# DRAM specs for PnR
# ---------------------------------------------------------------------------
DRAM_SPECS = {}
for dram_name, info in dram_type_bandwidth_width_dict.items():
    DRAM_SPECS[dram_name] = {
        'bandwidth_GBps': info['bandwidth'],
        'bus_width_bits': info['width'],
        'energy_pJ_per_bit': info.get('final_e', info.get('timeloop_e', 0)),
    }


# PHY area specs per DRAM type (mm2), from get_cost.py MEM_SPECS
_PHY_SPECS = {
    'LPDDR5': {'phy_mm2': 7.5, 'ctrl_mm2': 0.07},
    'DDR5':   {'phy_mm2': 7.5, 'ctrl_mm2': 0.07},
    'GDDR7':  {'phy_mm2': 8.0, 'ctrl_mm2': 0.07},
    'HBM3':   {'phy_mm2': 19.28, 'ctrl_mm2': 1.00},
}


def _unified_phy_mm2(bonding='2.5D'):
    """Unified PHY area. LPDDR5/DDR5 share one PHY; GDDR7 and HBM3 each have their own."""
    # Shared PHY groups: DDR5/LPDDR5 share one PHY
    phy_groups = {
        'DDR_shared': 7.5,   # DDR5 / LPDDR5 shared PHY
        'GDDR7': 8.0,
        'HBM3': 19.28,
    }
    total = 0
    for p in phy_groups.values():
        total += p if bonding == '2.5D' else p * 2.2
    return total


def _unified_ctrl_mm2():
    """Unified controller area. DDR5/LPDDR5 share one controller."""
    # 3 controllers: DDR_shared + GDDR7 + HBM3
    return 0.07 + 0.07 + 1.00


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# DRAM module specs from get_cost.py MEM_SPECS
# module_cap_GB: capacity of one module (die or 3D stack)
# area_per_GB: mm² per GB (per-layer die area density)
# stack_layers: number of 3D-stacked DRAM layers (1 for 2D packages)
# footprint_mm2: interposer/package footprint (= per-layer die area)
# total_silicon_mm2: total silicon across all stacked layers
import math as _math
_DRAM_MODULE_SPECS = {
    'LPDDR5': {'module_cap_GB': 4,  'area_per_GB': 29.3, 'stack_layers': 1},
    'DDR5':   {'module_cap_GB': 8,  'area_per_GB': 18.0, 'stack_layers': 1},
    'GDDR7':  {'module_cap_GB': 2,  'area_per_GB': 84.0, 'stack_layers': 1},
    'HBM3':   {'module_cap_GB': 24, 'area_per_GB': 62.5, 'stack_layers': 12},
}
for _spec in _DRAM_MODULE_SPECS.values():
    _total_si = _spec['module_cap_GB'] * _spec['area_per_GB']
    _spec['total_silicon_mm2'] = _total_si
    _cap_per_layer = _spec['module_cap_GB'] / _spec['stack_layers']
    _footprint = _cap_per_layer * _spec['area_per_GB']
    _spec['footprint_mm2'] = _footprint
    _side = _math.sqrt(_footprint)
    _spec['footprint_dimensions_mm'] = f"{_side:.1f} x {_side:.1f}"


def _build_aggregate(stage_latency, stage_energy, stage_power, stage_area,
                     stage_total_data, parsed, stage_dram):
    """Build aggregate dict with area breakdown including memory unit area."""
    arch = parsed.get('arch')
    glb_scale = parsed.get('glb_scale', 1) or 1
    pe_x = parsed.get('pe_x', 1) or 1
    pe_y = parsed.get('pe_y', 1) or 1
    bonding = parsed.get('bonding', '2.5D')

    core_area_mm2 = stage_area / 1e6 if stage_area > 0 else 0
    phy_area = _unified_phy_mm2(bonding)
    ctrl_area = _unified_ctrl_mm2()

    agg = {
        'latency_s': stage_latency,
        'dynamic_energy_J': stage_energy,
        'static_power_W': stage_power,
        'core_area_um2': stage_area,
        'core_area_mm2': core_area_mm2,
        'unified_phy_area_mm2': phy_area,
        'unified_ctrl_area_mm2': ctrl_area,
        'total_area_mm2': core_area_mm2 + phy_area + ctrl_area,
        'total_data_accesses': stage_total_data,
    }

    # Add memory area breakdown for compute chiplets
    if arch in ('eyeriss_like', 'gemmini_like', 'simba_like'):
        area_bd = get_chiplet_area_mm2(
            arch, glb_scale=glb_scale, pe_x_scale=pe_x, pe_y_scale=pe_y)
        agg['area_breakdown_mm2'] = {
            'mac_area': area_bd['mac_area_mm2'],
            'buffer_area': area_bd['buffer_area_mm2'],
            'glb_area': area_bd['glb_area_mm2'],
            'memory_area': area_bd['buffer_area_mm2'] + area_bd['glb_area_mm2'],
            'vector_area': area_bd['vector_area_mm2'],
            'total_core_area': area_bd['total_area_mm2'],
            'phy_area': phy_area,
            'ctrl_area': ctrl_area,
            'total_die_area': area_bd['total_area_mm2'] + phy_area + ctrl_area,
        }
    elif arch == 'PIM':
        # PIM: entire die is memory (GDDR7-based, 224 mm²)
        agg['area_breakdown_mm2'] = {
            'mac_area': 0.0,
            'buffer_area': 0.0,
            'glb_area': 0.0,
            'memory_area': 224.0,
            'vector_area': 0.0,
            'total_core_area': 224.0,
            'phy_area': 0.0,
            'ctrl_area': 0.0,
            'total_die_area': 224.0,
        }

    # DRAM module specs (actual off-chip memory chip area and dimensions)
    dram_mod = _DRAM_MODULE_SPECS.get(stage_dram)
    if dram_mod:
        agg['dram_module'] = {
            'type': stage_dram,
            'module_capacity_GB': dram_mod['module_cap_GB'],
            'stack_layers': dram_mod['stack_layers'],
            'area_per_GB_mm2': dram_mod['area_per_GB'],
            'footprint_mm2': dram_mod['footprint_mm2'],
            'footprint_dimensions_mm': dram_mod['footprint_dimensions_mm'],
            'total_silicon_mm2': dram_mod['total_silicon_mm2'],
        }

    return agg


def parse_config_id(config_id: str) -> dict:
    """Parse a config_id string into structured fields."""
    result = {'raw': config_id, 'is_vg': False, 'is_offcp': False, 'sub_configs': []}

    if config_id.startswith('VG('):
        result['is_vg'] = True
        inner = config_id[3:-1]
        sub_ids = inner.split('||')
        result['sub_configs'] = [parse_config_id(s) for s in sub_ids]
        if result['sub_configs']:
            rep = result['sub_configs'][0]
            for k in ['arch', 'glb_scale', 'pe_x', 'pe_y', 'tp', 'mapper', 'bonding']:
                result[k] = rep.get(k)
        result['dram_str'] = None
        return result

    if config_id.endswith('@offcp'):
        result['is_offcp'] = True
        config_id = config_id[:-6]

    parts = config_id.split('@')
    if len(parts) >= 7:
        result['arch'] = parts[0]
        result['glb_scale'] = int(parts[1].replace('glb', ''))
        result['pe_x'] = int(parts[2].replace('pe_x_scale', ''))
        result['pe_y'] = int(parts[3].replace('pe_y_scale', ''))
        result['tp'] = int(parts[4])
        result['mapper'] = int(parts[5])
        result['bonding'] = parts[6]
        result['dram_str'] = parts[7] if len(parts) > 7 else None
    else:
        result['arch'] = parts[0] if len(parts) > 0 else None
        result['glb_scale'] = int(parts[1].replace('glb', '')) if len(parts) > 1 else None
        result['pe_x'] = int(parts[2].replace('pe_x_scale', '')) if len(parts) > 2 else None
        result['pe_y'] = int(parts[3].replace('pe_y_scale', '')) if len(parts) > 3 else None
        result['tp'] = int(parts[4]) if len(parts) > 4 else 1
        result['mapper'] = int(parts[5]) if len(parts) > 5 else 0
        result['bonding'] = parts[6] if len(parts) > 6 else '2.5D'
        result['dram_str'] = None

    return result


def load_virtual_network(net_name, batch_size, seq_len, db_layers):
    """Load a VirtualNetwork with layers filtered by database."""
    vn = VirtualNetwork(net_name, batch_size=batch_size, sequence_length=seq_len)
    net_dir = os.path.join(NET_DIR, net_name)
    if os.path.isdir(net_dir):
        vn.load_from_dir(net_dir, db_layers=db_layers)
    return vn


def get_chiplet_identifier(arch, glb, pe_x, pe_y):
    return f"{arch}@glb{glb}@pe_x{pe_x}@pe_y{pe_y}"


def safe_int(val, default=0):
    """Convert to int, handling NaN/None."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return int(val)


def query_operator_stats(db, net_name, layer_name, batch_size, seq_len,
                         arch, glb, pe_x, pe_y, tp=1, mapper=0):
    """Query database for per-operator statistics."""
    mask = (
        (db['net'] == net_name) &
        (db['layer_name'] == layer_name) &
        (db['batch_size'] == batch_size) &
        (db['sequence_length'] == seq_len) &
        (db['arch_target'] == arch) &
        (db['glb_scale'] == glb) &
        (db['pe_x_scale'] == pe_x) &
        (db['pe_y_scale'] == pe_y) &
        (db['tp_degree'] == tp) &
        (db['mapper_idx'] == mapper)
    )
    rows = db[mask]
    if rows.empty:
        # Try without mapper/TP constraint
        mask_relaxed = (
            (db['net'] == net_name) &
            (db['layer_name'] == layer_name) &
            (db['batch_size'] == batch_size) &
            (db['sequence_length'] == seq_len) &
            (db['arch_target'] == arch) &
            (db['glb_scale'] == glb) &
            (db['pe_x_scale'] == pe_x) &
            (db['pe_y_scale'] == pe_y)
        )
        rows = db[mask_relaxed]
    if rows.empty:
        # Last resort: match only net + layer + arch (different PE/GLB)
        mask_loose = (
            (db['net'] == net_name) &
            (db['layer_name'] == layer_name) &
            (db['batch_size'] == batch_size) &
            (db['sequence_length'] == seq_len) &
            (db['arch_target'] == arch)
        )
        rows = db[mask_loose]
    if rows.empty:
        return None
    best = rows.loc[rows['dynamic_energy'].idxmin()]
    return {
        'latency_s': float(best['latency']),
        'dynamic_energy_J': float(best['dynamic_energy']),
        'static_power_W': float(best['static_power']),
        'area_um2': float(best['area']),
        'utilization': float(best.get('utilization', 0)),
        'i_access': safe_int(best.get('i_access', 0)),
        'w_access': safe_int(best.get('w_access', 0)),
        'o_access': safe_int(best.get('o_access', 0)),
        'dram_i': str(best.get('dram_i', '')),
        'dram_o': str(best.get('dram_o', '')),
    }


def dag_binary_to_operator_groups(vn, binary_string):
    """Use CriticalPathSpec to map binary_string to operator groups.

    For transformers, binary_string indexes CriticalPath positions (each
    containing 1+ parallel operators). For CNNs, it indexes layers directly.

    Returns:
        cp_groups: list of lists of operator names (critical path)
        off_cp_ops: list of off-critical-path operator dicts
    """
    cp_spec = create_cp_spec(vn)
    binary = binary_string

    # Group positions by binary_string
    position_groups = []
    current = []
    for i, bit in enumerate(binary):
        if i >= len(cp_spec.positions):
            break
        if bit == '1' and i > 0 and current:
            position_groups.append(current)
            current = [i]
        elif bit == '1' and i == 0:
            current = [i]
        else:
            current.append(i)
    if current:
        position_groups.append(current)

    layer_names = {l.name for l in vn.layers}

    # Build fusion groups from positions
    cp_groups = []
    for pos_indices in position_groups:
        ops = []
        is_parallel = (len(pos_indices) == 1 and
                       cp_spec.positions[pos_indices[0]].parallel)
        for pi in pos_indices:
            pos = cp_spec.positions[pi]
            for op in pos.ops:
                if op in layer_names:
                    ops.append(op)
        cp_groups.append({
            'operators': ops,
            'is_parallel': is_parallel,
            'buf_start': pos_indices[0],
            'buf_end': pos_indices[-1] + 1,
        })

    # Off-critical-path operators
    off_cp_ops = []
    for off in cp_spec.off_cp_ops:
        if off.op_name in layer_names:
            off_cp_ops.append({
                'operator': off.op_name,
                'start_pos': off.start_pos,
                'end_pos': off.end_pos,
                'slack_stages': off.slack_stages,
            })

    return cp_groups, off_cp_ops


# ---------------------------------------------------------------------------
# Main config generation
# ---------------------------------------------------------------------------

def generate_pnr_for_network(
    net_key, net_name, batch_size, seq_len,
    csv_row, db, db_layers_per_net,
):
    """Generate PnR config for one network."""

    gene_str = csv_row.get(f'{net_key}_gene')
    config_str = csv_row.get(f'{net_key}_config')
    # Auto-detect objective column (min_energy or min_edp)
    energy = csv_row.get(f'{net_key}_min_energy', None)
    if energy is None or (isinstance(energy, float) and pd.isna(energy)):
        energy = csv_row.get(f'{net_key}_min_edp', None)
    latency = csv_row.get(f'{net_key}_latency', None)

    if pd.isna(gene_str) or pd.isna(config_str):
        print(f"  WARNING: No data for {net_key}, skipping")
        return None

    gene = ast.literal_eval(gene_str)
    config_ids = ast.literal_eval(config_str)
    binary_string = gene['binary_string']
    buffer_config = gene['buffer_config']

    # Load virtual network
    db_layers = db_layers_per_net.get(net_name, None)
    vn = load_virtual_network(net_name, batch_size, seq_len, db_layers)

    # Use DAG-aware fusion group reconstruction
    cp_groups, off_cp_ops = dag_binary_to_operator_groups(vn, binary_string)

    # The config_ids list = cp_groups + off_cp entries
    n_cp_stages = len(cp_groups)
    n_offcp = len(off_cp_ops)
    print(f"  Network: {net_name}, layers={len(vn.layers)}, "
          f"cp_stages={n_cp_stages}, off_cp={n_offcp}, config_ids={len(config_ids)}")

    # Build stages: first n_cp_stages from cp_groups, remaining from off_cp
    stages = []
    chiplet_usage = defaultdict(list)

    for stage_idx in range(len(config_ids)):
        cid = config_ids[stage_idx]
        parsed = parse_config_id(cid)

        # Determine operators for this stage
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

        # Per-operator chiplet info from VG sub_configs (if different chiplets)
        sub_configs = parsed.get('sub_configs', [])

        # Query per-operator stats
        operators = []
        stage_latency = 0.0
        stage_energy = 0.0
        stage_area = 0.0
        stage_power = 0.0
        stage_total_data = 0

        for op_idx, layer_name in enumerate(group_ops):
            # Use per-operator chiplet if this is a VG with sub_configs
            op_parsed = sub_configs[op_idx] if op_idx < len(sub_configs) else parsed
            stats = query_operator_stats(
                db, net_name, layer_name, batch_size, seq_len,
                op_parsed.get('arch'), op_parsed.get('glb_scale', 1),
                op_parsed.get('pe_x', 1), op_parsed.get('pe_y', 1),
                op_parsed.get('tp', 1), op_parsed.get('mapper', 0),
            )
            op_info = {'name': layer_name}
            # Add per-operator chiplet assignment if VG with different chiplets
            if sub_configs and op_idx < len(sub_configs):
                sc = sub_configs[op_idx]
                op_chiplet_id = get_chiplet_identifier(
                    sc.get('arch', 'unknown'), sc.get('glb_scale', 1),
                    sc.get('pe_x', 1), sc.get('pe_y', 1))
                op_info['assigned_chiplet'] = {
                    'identifier': op_chiplet_id,
                    'architecture': sc.get('arch'),
                    'glb_scale': sc.get('glb_scale'),
                    'pe_x_scale': sc.get('pe_x'),
                    'pe_y_scale': sc.get('pe_y'),
                }
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

    # Inter-stage communication (CP stages only — sequential pipeline)
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
            'data_volume_bytes': comm_volume * 2,  # FP16
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
        'total_energy_J': float(energy) if energy and not pd.isna(energy) else None,
        'total_latency_s': float(latency) if latency and not pd.isna(latency) else None,
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


def extract_chiplet_pool(csv_row, n_chiplets):
    """Extract chiplet pool specification from CSV row."""
    pool = []
    for i in range(1, n_chiplets + 1):
        arch = csv_row.get(f'chiplet_{i}_arch')
        if pd.isna(arch):
            continue
        glb = int(csv_row.get(f'chiplet_{i}_glb_scale', 1))
        pe_x = int(csv_row.get(f'chiplet_{i}_pe_x_scale', 1))
        pe_y = int(csv_row.get(f'chiplet_{i}_pe_y_scale', 1))

        chiplet = {
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
            chiplet['type'] = 'processing_in_memory'
            chiplet['memory_tech'] = 'GDDR7'
            chiplet['die_area_mm2'] = 224.0
            chiplet['note'] = 'Near-bank compute, no separate PE array'
        elif arch == 'switch_8port':
            chiplet['type'] = 'switch'
            chiplet['num_ports'] = 8
            chiplet['note'] = 'MoE expert-parallelism router, no compute'
        else:
            chiplet['type'] = 'compute'
            dataflow_map = {
                'eyeriss_like': 'row_stationary',
                'simba_like': 'weight_stationary',
                'gemmini_like': 'output_stationary',
            }
            chiplet['dataflow'] = dataflow_map.get(arch, arch)

            # Add area breakdown from analytical model
            area_breakdown = get_chiplet_area_mm2(
                arch, glb_scale=glb, pe_x_scale=pe_x, pe_y_scale=pe_y)
            phy = _unified_phy_mm2('2.5D')
            ctrl = _unified_ctrl_mm2()
            chiplet['area_breakdown_mm2'] = {
                'mac_area': area_breakdown['mac_area_mm2'],
                'buffer_area': area_breakdown['buffer_area_mm2'],
                'glb_area': area_breakdown['glb_area_mm2'],
                'memory_area': area_breakdown['buffer_area_mm2'] + area_breakdown['glb_area_mm2'],
                'vector_area': area_breakdown['vector_area_mm2'],
                'total_core_area': area_breakdown['total_area_mm2'],
                'phy_area': phy,
                'ctrl_area': ctrl,
                'total_die_area': area_breakdown['total_area_mm2'] + phy + ctrl,
            }

        pool.append(chiplet)
    return pool


def main():
    parser = argparse.ArgumentParser(description='Generate PnR configs from optimization results')
    parser.add_argument('--csv', required=True, help='Path to incremental sweep CSV')
    parser.add_argument('--database', default='unified_database.csv', help='Performance database CSV')
    parser.add_argument('--n-chiplets', type=int, default=8, help='Chiplet pool size to extract')
    parser.add_argument('--output-dir', default='pnr_configs', help='Output directory')
    args = parser.parse_args()

    print(f"Loading database: {args.database}")
    db = pd.read_csv(args.database)
    db_layers_per_net = {
        net: set(db[db['net'] == net]['layer_name'].unique())
        for net in db['net'].unique()
    }

    print(f"Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    row = df[df['n_chiplets'] == args.n_chiplets]
    if row.empty:
        print(f"ERROR: No row with n_chiplets={args.n_chiplets}")
        sys.exit(1)
    row = row.iloc[0]

    print(f"\nExtracting chiplet pool (N={args.n_chiplets})...")
    chiplet_pool = extract_chiplet_pool(row, args.n_chiplets)
    print(f"  Pool size: {len(chiplet_pool)} chiplets")
    for c in chiplet_pool:
        print(f"    [{c['index']}] {c['identifier']} ({c.get('dataflow', c.get('type', '?'))})")

    # Target networks
    targets = [
        ('replknet31b_b1_seq1', 'replknet31b', 1, 1),
        ('llama3.1_8b_prefill_s1024_b1_seq1024', 'llama3.1_8b_prefill_s1024', 1, 1024),
        ('llama3.1_8b_decode_kv1024_b1_seq1', 'llama3.1_8b_decode_kv1024', 1, 1),
        ('qwen3_30b_a3b_prefill_s1024_b1_seq1024', 'qwen3_30b_a3b_prefill_s1024', 1, 1024),
        ('qwen3_30b_a3b_decode_kv1024_b1_seq1', 'qwen3_30b_a3b_decode_kv1024', 1, 1),
        ('qwen3_235b_a22b_prefill_s1024_b1_seq1024', 'qwen3_235b_a22b_prefill_s1024', 1, 1024),
        ('qwen3_235b_a22b_decode_kv1024_b1_seq1', 'qwen3_235b_a22b_decode_kv1024', 1, 1),
    ]

    os.makedirs(args.output_dir, exist_ok=True)

    # Build DRAM module reference table for metadata
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
            'inter_chiplet_energy_pJ_per_bit': 1.3,
            'technology_node_nm': 14,
            'clock_GHz': 1.0,
            'dram_module_specs': dram_modules,
        },
        'chiplet_pool': chiplet_pool,
        'networks': {},
    }

    for net_key, net_name, bs, seq in targets:
        print(f"\n--- Generating PnR config for {net_key} ---")
        config = generate_pnr_for_network(
            net_key, net_name, bs, seq, row, db, db_layers_per_net)
        if config:
            all_configs['networks'][net_key] = config
            out_file = os.path.join(args.output_dir, f'pnr_{net_key}.json')
            with open(out_file, 'w') as f:
                json.dump({
                    'metadata': all_configs['metadata'],
                    'chiplet_pool': chiplet_pool,
                    'network': config,
                }, f, indent=2, default=str)
            print(f"  Saved: {out_file}")

    combined_file = os.path.join(args.output_dir, 'pnr_all_networks.json')
    with open(combined_file, 'w') as f:
        json.dump(all_configs, f, indent=2, default=str)
    print(f"\nSaved combined config: {combined_file}")

    # Summary
    print("\n" + "=" * 70)
    print("PnR CONFIG SUMMARY")
    print("=" * 70)
    for net_key, cfg in all_configs['networks'].items():
        print(f"\n{net_key}:")
        print(f"  Stages: {cfg['num_cp_stages']} CP + {cfg['num_offcp_stages']} off-CP")
        if cfg['total_energy_J']:
            print(f"  Total energy: {cfg['total_energy_J']:.4e} J")
        if cfg['total_latency_s']:
            print(f"  Total latency: {cfg['total_latency_s']:.4e} s")
        inter = sum(1 for c in cfg['inter_stage_communication']
                    if c.get('is_inter_chiplet', False))
        total = len(cfg['inter_stage_communication'])
        print(f"  Inter-chiplet transfers: {inter}/{total}")
        chiplets_used = {s['assigned_chiplet']['identifier'] for s in cfg['stages']}
        print(f"  Unique chiplets used: {len(chiplets_used)}")
        for s in cfg['stages']:
            tag = "OFF-CP" if s['is_off_critical_path'] else f"CP"
            par = " (parallel)" if s['is_parallel'] else ""
            print(f"    Stage {s['stage_index']} [{tag}{par}]: "
                  f"{s['fusion_group_operators']} → "
                  f"{s['assigned_chiplet']['architecture']} "
                  f"(TP={s['tensor_parallelism']}, {s['bonding']}, "
                  f"DRAM={s['dram_type']})")


if __name__ == '__main__':
    main()
