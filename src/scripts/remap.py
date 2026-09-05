#!/usr/bin/env python3
"""
remap.py -- Remap a new network onto a fixed ASIC chiplet topology.

Given:
  - A PnR config JSON (defines the physical ASIC: die instances, types, topology,
    DRAM assignments -- all derived from network 1's optimized mapping)
  - A new network (network 2) whose operators need to be mapped onto this fixed ASIC
  - A Timeloop performance database

Produces:
  - A new PnR config JSON with network 2's operators mapped onto the fixed dies

Physical model:
  - Each stage in the original PnR = one physical die instance
  - VG (virtual group / parallel) stages with per-operator chiplet assignments
    produce one die per parallel operator
  - Dies are connected in a chain: die_i <-> die_{i+1}
  - Off-CP dies connect to their feeds_from and feeds_to dies
  - Only directly connected dies can communicate (no multi-hop)
  - Each die has a fixed type (arch, glb, pe_x, pe_y) and DRAM type

Usage:
    python3 remap.py \\
        --pnr-config pnr_configs_edp/pnr_llama3.1_8b_prefill_s1024_b1_seq1024.json \\
        --new-network llama3.1_8b_decode_kv1024 \\
        --batch-size 1 --sequence-length 1 \\
        --database unified_database.csv \\
        --output remap_output.json \\
        --objective edp
"""

import os
import sys
import json
import math
import random
import argparse
import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.normpath(os.path.join(THIS_DIR, '..', 'scripts'))
CHIPLET_TL_DIR = os.path.normpath(os.path.join(THIS_DIR, '..'))
for p in [SCRIPTS_DIR, CHIPLET_TL_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from network_dataclass import VirtualNetwork
from global_parameter import NET_DIR, dram_type_bandwidth_width_dict
from cal_perf_phy_net import create_cp_spec, calculate_inter_chiplet_communication


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PhysicalDie:
    """One physical die instance on the ASIC."""
    die_id: int                     # unique id (0-indexed)
    identifier: str                 # e.g. "gemmini_like@glb1@pe_x3@pe_y3"
    architecture: str               # e.g. "gemmini_like"
    glb_scale: int
    pe_x_scale: int
    pe_y_scale: int
    dram_type: str                  # e.g. "HBM3", fixed by physical layout
    bonding: str                    # e.g. "2.5D"
    neighbors: Set[int] = field(default_factory=set)  # die_ids of direct neighbors

    @property
    def is_compute(self):
        return self.architecture not in ('switch_8port',)

    @property
    def is_pim(self):
        return self.architecture == 'PIM'


@dataclass
class AsicTopology:
    """The fixed physical ASIC: dies + adjacency."""
    dies: List[PhysicalDie]
    metadata: dict

    def __len__(self):
        return len(self.dies)

    def neighbors_of(self, die_id: int) -> Set[int]:
        return self.dies[die_id].neighbors

    def are_adjacent(self, a: int, b: int) -> bool:
        return a == b or b in self.dies[a].neighbors


# ---------------------------------------------------------------------------
# Phase 1: Extract physical topology from PnR config
# ---------------------------------------------------------------------------

def extract_topology(pnr_config: dict) -> AsicTopology:
    """Build AsicTopology from a PnR config JSON.

    Each stage = one die.  VG stages with per-operator chiplet assignments
    produce one die per parallel operator.
    Adjacency: chain along CP stages, off-CP connects to feeds_from/feeds_to.
    """
    metadata = pnr_config.get('metadata', {})
    network_data = pnr_config.get('network', pnr_config)
    stages = network_data['stages']

    dies: List[PhysicalDie] = []
    # Map from stage_index to list of die_ids (usually 1, but VG may have 2+)
    stage_to_dies: Dict[int, List[int]] = {}

    for stage in stages:
        si = stage['stage_index']
        operators = stage.get('operators', [])
        is_vg = stage.get('is_virtual_group', False)

        # Check if VG operators have different chiplet assignments
        has_per_op_chiplets = (
            is_vg and len(operators) > 1 and
            any('assigned_chiplet' in op for op in operators)
        )

        if has_per_op_chiplets:
            # Each parallel operator in VG gets its own die
            die_ids = []
            for op in operators:
                op_chiplet = op.get('assigned_chiplet', stage['assigned_chiplet'])
                die = PhysicalDie(
                    die_id=len(dies),
                    identifier=op_chiplet['identifier'],
                    architecture=op_chiplet['architecture'],
                    glb_scale=op_chiplet.get('glb_scale', 1),
                    pe_x_scale=op_chiplet.get('pe_x_scale', 1),
                    pe_y_scale=op_chiplet.get('pe_y_scale', 1),
                    dram_type=stage.get('dram_type', 'HBM3'),
                    bonding=stage.get('bonding', '2.5D'),
                )
                die_ids.append(die.die_id)
                dies.append(die)
            # VG parallel dies are adjacent to each other
            for i in range(len(die_ids)):
                for j in range(i + 1, len(die_ids)):
                    dies[die_ids[i]].neighbors.add(die_ids[j])
                    dies[die_ids[j]].neighbors.add(die_ids[i])
            stage_to_dies[si] = die_ids
        else:
            die = PhysicalDie(
                die_id=len(dies),
                identifier=stage['assigned_chiplet']['identifier'],
                architecture=stage['assigned_chiplet']['architecture'],
                glb_scale=stage['assigned_chiplet'].get('glb_scale', 1),
                pe_x_scale=stage['assigned_chiplet'].get('pe_x_scale', 1),
                pe_y_scale=stage['assigned_chiplet'].get('pe_y_scale', 1),
                dram_type=stage.get('dram_type', 'HBM3'),
                bonding=stage.get('bonding', '2.5D'),
            )
            stage_to_dies[si] = [die.die_id]
            dies.append(die)

    # Build chain adjacency for CP stages
    cp_stages = [s for s in stages if not s.get('is_off_critical_path', False)]
    cp_stages.sort(key=lambda s: s['stage_index'])

    for i in range(len(cp_stages) - 1):
        si_a = cp_stages[i]['stage_index']
        si_b = cp_stages[i + 1]['stage_index']
        # Last die(s) of stage A connect to first die(s) of stage B
        for da in stage_to_dies[si_a]:
            for db in stage_to_dies[si_b]:
                dies[da].neighbors.add(db)
                dies[db].neighbors.add(da)

    # Off-CP dies connect to their feeds_from and feeds_to stages
    for stage in stages:
        if stage.get('is_off_critical_path', False):
            si = stage['stage_index']
            feeds_from = stage.get('offcp_feeds_from_stage')
            feeds_to = stage.get('offcp_feeds_to_stage')
            for target_si in [feeds_from, feeds_to]:
                if target_si is not None and target_si in stage_to_dies:
                    for d_off in stage_to_dies[si]:
                        for d_target in stage_to_dies[target_si]:
                            dies[d_off].neighbors.add(d_target)
                            dies[d_target].neighbors.add(d_off)

    return AsicTopology(dies=dies, metadata=metadata)


# ---------------------------------------------------------------------------
# Phase 2: Build cost matrix from Timeloop database
# ---------------------------------------------------------------------------

def build_cost_matrix(
    db: pd.DataFrame,
    new_net_name: str,
    batch_size: int,
    seq_len: int,
    topology: AsicTopology,
) -> Dict[str, Dict[int, dict]]:
    """Build cost[operator_name][die_id] = {latency, energy, ...}.

    Returns dict: op_name -> die_id -> stats dict (or None if infeasible).
    """
    # Get all unique layers for this network in the database
    net_layers = set(db[db['net'] == new_net_name]['layer_name'].unique())

    cost = {}
    for layer_name in net_layers:
        cost[layer_name] = {}
        for die in topology.dies:
            if not die.is_compute:
                cost[layer_name][die.die_id] = None
                continue

            stats = _query_die_stats(
                db, new_net_name, layer_name, batch_size, seq_len, die)
            cost[layer_name][die.die_id] = stats

    return cost


def _query_die_stats(
    db: pd.DataFrame,
    net_name: str, layer_name: str,
    batch_size: int, seq_len: int,
    die: PhysicalDie,
) -> Optional[dict]:
    """Query database for operator performance on a specific die."""
    mask = (
        (db['net'] == net_name) &
        (db['layer_name'] == layer_name) &
        (db['batch_size'] == batch_size) &
        (db['sequence_length'] == seq_len) &
        (db['arch_target'] == die.architecture) &
        (db['glb_scale'] == die.glb_scale) &
        (db['pe_x_scale'] == die.pe_x_scale) &
        (db['pe_y_scale'] == die.pe_y_scale)
    )
    rows = db[mask]

    if rows.empty:
        # Relax: match arch only (different PE/GLB may still give a rough estimate)
        mask_loose = (
            (db['net'] == net_name) &
            (db['layer_name'] == layer_name) &
            (db['batch_size'] == batch_size) &
            (db['sequence_length'] == seq_len) &
            (db['arch_target'] == die.architecture)
        )
        rows = db[mask_loose]

    if rows.empty:
        return None

    best = rows.loc[rows['dynamic_energy'].idxmin()]
    return {
        'latency_s': float(best['latency']),
        'dynamic_energy_J': float(best['dynamic_energy']),
        'static_power_W': float(best['static_power']),
        'area_um2': float(best['area']) if not pd.isna(best['area']) else 0.0,
        'utilization': float(best.get('utilization', 0)),
        'i_access': int(best.get('i_access', 0)) if not pd.isna(best.get('i_access', 0)) else 0,
        'w_access': int(best.get('w_access', 0)) if not pd.isna(best.get('w_access', 0)) else 0,
        'o_access': int(best.get('o_access', 0)) if not pd.isna(best.get('o_access', 0)) else 0,
        'dram_i': die.dram_type,
        'dram_o': die.dram_type,
        'tp_degree': int(best.get('tp_degree', 1)),
        'mapper_idx': int(best.get('mapper_idx', 0)),
    }


# ---------------------------------------------------------------------------
# Phase 3: Remap search (Simulated Annealing)
# ---------------------------------------------------------------------------

@dataclass
class RemapSolution:
    """A mapping of new network operators to physical dies."""
    # op_name -> die_id
    mapping: Dict[str, int]
    # Cached cost
    total_latency: float = float('inf')
    total_energy: float = float('inf')
    total_edp: float = float('inf')


def get_new_network_ops(
    new_net_name: str, batch_size: int, seq_len: int,
    db_layers: set,
) -> Tuple[List[str], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Load new network's operator DAG.

    Returns:
        all_ops: list of operator names
        cp_edges: list of (src_op, dst_op) for critical path dependencies
        offcp_edges: list of (offcp_op, feeds_to_op) for off-CP dependencies
    """
    vn = VirtualNetwork(new_net_name, batch_size=batch_size,
                        sequence_length=seq_len)
    net_dir = os.path.join(NET_DIR, new_net_name)
    if os.path.isdir(net_dir):
        vn.load_from_dir(net_dir, db_layers=db_layers)

    cp_spec = create_cp_spec(vn)

    # All CP operators in order
    cp_ops = []
    for pos in cp_spec.positions:
        for op in pos.ops:
            cp_ops.append(op)

    # CP edges: sequential dependencies between positions
    cp_edges = []
    for i in range(len(cp_spec.positions) - 1):
        # Last op of position i -> first op of position i+1
        pos_a = cp_spec.positions[i]
        pos_b = cp_spec.positions[i + 1]
        if pos_a.ops and pos_b.ops:
            for op_a in pos_a.ops:
                for op_b in pos_b.ops:
                    cp_edges.append((op_a, op_b))

    # Parallel ops within a position have no edge between them (they are independent)

    # Off-CP edges
    offcp_ops = []
    offcp_edges = []
    for off in cp_spec.off_cp_ops:
        offcp_ops.append(off.op_name)
        # off-cp op needs input from start_pos and delivers output to end_pos
        if off.start_pos < len(cp_spec.positions):
            for op in cp_spec.positions[off.start_pos].ops:
                offcp_edges.append((op, off.op_name))
        if off.end_pos < len(cp_spec.positions):
            for op in cp_spec.positions[off.end_pos].ops:
                offcp_edges.append((off.op_name, op))

    all_ops = cp_ops + offcp_ops
    # Filter to only ops that exist in database
    all_ops = [op for op in all_ops if op in db_layers]
    cp_edges = [(a, b) for a, b in cp_edges if a in db_layers and b in db_layers]
    offcp_edges = [(a, b) for a, b in offcp_edges if a in db_layers and b in db_layers]

    return all_ops, cp_edges, offcp_edges


def check_adjacency_constraint(
    mapping: Dict[str, int],
    edges: List[Tuple[str, str]],
    topology: AsicTopology,
) -> bool:
    """Check that all dependent operator pairs are on adjacent (or same) dies."""
    for src_op, dst_op in edges:
        if src_op not in mapping or dst_op not in mapping:
            continue
        src_die = mapping[src_op]
        dst_die = mapping[dst_op]
        if not topology.are_adjacent(src_die, dst_die):
            return False
    return True


def evaluate_solution(
    mapping: Dict[str, int],
    cost_matrix: Dict[str, Dict[int, dict]],
    all_edges: List[Tuple[str, str]],
    topology: AsicTopology,
    objective: str = 'edp',
) -> Tuple[float, float, float]:
    """Evaluate a mapping. Returns (latency, energy, edp).

    Execution model: dies form a pipeline. All transformer layers flow through.
      - Per-die latency = sum of operator latencies assigned to that die
      - Pipeline throughput bottleneck = max(per-die latency)
      - total_latency = max(per-die latency)
      - total_energy = sum of all operator energies + inter-die comm

    Returns (inf, inf, inf) if infeasible.
    """
    if not check_adjacency_constraint(mapping, all_edges, topology):
        return float('inf'), float('inf'), float('inf')

    total_energy = 0.0
    die_latency: Dict[int, float] = defaultdict(float)

    for op_name, die_id in mapping.items():
        if op_name not in cost_matrix:
            return float('inf'), float('inf'), float('inf')
        stats = cost_matrix[op_name].get(die_id)
        if stats is None:
            return float('inf'), float('inf'), float('inf')

        die_latency[die_id] += stats['latency_s']
        total_energy += stats['dynamic_energy_J']

    # Inter-chiplet communication energy
    for src_op, dst_op in all_edges:
        if src_op not in mapping or dst_op not in mapping:
            continue
        src_die = mapping[src_op]
        dst_die = mapping[dst_op]
        if src_die != dst_die:
            src_stats = cost_matrix.get(src_op, {}).get(src_die)
            if src_stats:
                bonding = topology.dies[src_die].bonding
                comm_bits = (src_stats['o_access'] + src_stats['w_access']) * 16
                comm_energy = calculate_inter_chiplet_communication(
                    comm_bits, bonding, num_hop=1)
                total_energy += comm_energy

    # Pipeline bottleneck: latency = max per-die latency
    total_latency = max(die_latency.values()) if die_latency else 0.0

    edp = total_latency * total_energy
    return total_latency, total_energy, edp


def generate_initial_solution(
    all_ops: List[str],
    all_edges: List[Tuple[str, str]],
    cost_matrix: Dict[str, Dict[int, dict]],
    topology: AsicTopology,
) -> Dict[str, int]:
    """Generate a greedy initial solution.

    Assign each operator to the die that gives lowest energy,
    then fix adjacency violations by moving ops to neighbors.
    """
    mapping = {}

    for op in all_ops:
        if op not in cost_matrix:
            continue
        best_die = None
        best_energy = float('inf')
        for die_id, stats in cost_matrix[op].items():
            if stats is not None and stats['dynamic_energy_J'] < best_energy:
                best_energy = stats['dynamic_energy_J']
                best_die = die_id
        if best_die is not None:
            mapping[op] = best_die

    # Fix adjacency: for each violated edge, move dst to a neighbor of src
    for _ in range(len(all_ops) * 10):  # iterate until stable
        changed = False
        for src_op, dst_op in all_edges:
            if src_op not in mapping or dst_op not in mapping:
                continue
            src_die = mapping[src_op]
            dst_die = mapping[dst_op]
            if not topology.are_adjacent(src_die, dst_die):
                # Try to move dst_op to src_die or one of its neighbors
                candidates = [src_die] + list(topology.neighbors_of(src_die))
                best_c = None
                best_e = float('inf')
                for c in candidates:
                    stats = cost_matrix.get(dst_op, {}).get(c)
                    if stats is not None and stats['dynamic_energy_J'] < best_e:
                        best_e = stats['dynamic_energy_J']
                        best_c = c
                if best_c is not None:
                    mapping[dst_op] = best_c
                    changed = True
        if not changed:
            break

    return mapping


def simulated_annealing(
    all_ops: List[str],
    all_edges: List[Tuple[str, str]],
    cost_matrix: Dict[str, Dict[int, dict]],
    topology: AsicTopology,
    objective: str = 'edp',
    max_iter: int = 50000,
    t_init: float = 1.0,
    t_min: float = 1e-6,
    cooling: float = 0.9995,
    seed: int = 42,
) -> RemapSolution:
    """Simulated annealing search for optimal operator-to-die mapping."""
    rng = random.Random(seed)
    n_dies = len(topology)

    # Initial solution
    current_mapping = generate_initial_solution(
        all_ops, all_edges, cost_matrix, topology)

    # Ops that have at least one feasible die
    movable_ops = [op for op in all_ops
                   if op in cost_matrix and
                   any(v is not None for v in cost_matrix[op].values())]

    if not movable_ops:
        return RemapSolution(mapping=current_mapping)

    def obj_val(lat, eng, edp):
        if objective == 'energy':
            return eng
        elif objective == 'latency':
            return lat
        return edp

    current_lat, current_eng, current_edp = evaluate_solution(
        current_mapping, cost_matrix, all_edges, topology, objective)
    current_cost = obj_val(current_lat, current_eng, current_edp)

    best_mapping = dict(current_mapping)
    best_cost = current_cost
    best_lat, best_eng, best_edp = current_lat, current_eng, current_edp

    temp = t_init

    for iteration in range(max_iter):
        # Pick a random operator and move it to a random feasible die
        op = rng.choice(movable_ops)
        old_die = current_mapping.get(op)

        # Candidate dies: prefer neighbors of current die + current die's neighbors' neighbors
        # But also allow any die with some probability for exploration
        feasible_dies = [d for d in range(n_dies)
                         if cost_matrix.get(op, {}).get(d) is not None]
        if not feasible_dies:
            continue

        new_die = rng.choice(feasible_dies)
        if new_die == old_die:
            continue

        # Apply move
        current_mapping[op] = new_die
        new_lat, new_eng, new_edp = evaluate_solution(
            current_mapping, cost_matrix, all_edges, topology, objective)
        new_cost = obj_val(new_lat, new_eng, new_edp)

        # Accept or reject
        delta = new_cost - current_cost
        if delta < 0 or (temp > 0 and rng.random() < math.exp(-delta / (temp + 1e-30))):
            current_cost = new_cost
            current_lat, current_eng, current_edp = new_lat, new_eng, new_edp
            if new_cost < best_cost:
                best_mapping = dict(current_mapping)
                best_cost = new_cost
                best_lat, best_eng, best_edp = new_lat, new_eng, new_edp
        else:
            # Revert
            current_mapping[op] = old_die

        temp *= cooling

        if temp < t_min:
            break

    return RemapSolution(
        mapping=best_mapping,
        total_latency=best_lat,
        total_energy=best_eng,
        total_edp=best_edp,
    )


# ---------------------------------------------------------------------------
# Phase 4: Generate output PnR config JSON
# ---------------------------------------------------------------------------

DRAM_SPECS = {}
for dram_name, info in dram_type_bandwidth_width_dict.items():
    DRAM_SPECS[dram_name] = {
        'bandwidth_GBps': info['bandwidth'],
        'bus_width_bits': info['width'],
        'energy_pJ_per_bit': info.get('final_e', info.get('timeloop_e', 0)),
    }


def generate_remap_pnr_config(
    solution: RemapSolution,
    topology: AsicTopology,
    cost_matrix: Dict[str, Dict[int, dict]],
    all_ops: List[str],
    cp_edges: List[Tuple[str, str]],
    offcp_edges: List[Tuple[str, str]],
    new_net_name: str,
    batch_size: int,
    seq_len: int,
    original_pnr: dict,
) -> dict:
    """Generate a PnR config JSON for the remapped network."""

    # Reconstruct chiplet_pool from original
    chiplet_pool = original_pnr.get('chiplet_pool', [])

    # Group operators by die
    die_to_ops: Dict[int, List[str]] = defaultdict(list)
    for op in all_ops:
        if op in solution.mapping:
            die_to_ops[solution.mapping[op]].append(op)

    # Build stages: one stage per die that has operators assigned
    stages = []
    active_dies = sorted(die_to_ops.keys())

    for stage_idx, die_id in enumerate(active_dies):
        die = topology.dies[die_id]
        ops = die_to_ops[die_id]

        # Query per-operator stats
        operators = []
        stage_latency = 0.0
        stage_energy = 0.0
        stage_power = 0.0
        stage_area = 0.0
        stage_total_data = 0

        for op_name in ops:
            stats = cost_matrix.get(op_name, {}).get(die_id)
            op_info = {'name': op_name}
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
                    'dram_i': die.dram_type,
                    'dram_o': die.dram_type,
                })
                stage_latency += stats['latency_s']
                stage_energy += stats['dynamic_energy_J']
                stage_area = max(stage_area, stats['area_um2'])
                stage_power = max(stage_power, stats['static_power_W'])
                stage_total_data += (stats['i_access'] + stats['w_access']
                                     + stats['o_access'])
            else:
                op_info['note'] = 'not found in database'
            operators.append(op_info)

        stage_info = {
            'stage_index': stage_idx,
            'die_id': die_id,
            'fusion_group_operators': ops,
            'num_operators': len(ops),
            'assigned_chiplet': {
                'identifier': die.identifier,
                'architecture': die.architecture,
                'glb_scale': die.glb_scale,
                'pe_x_scale': die.pe_x_scale,
                'pe_y_scale': die.pe_y_scale,
            },
            'dram_type': die.dram_type,
            'dram_spec': DRAM_SPECS.get(die.dram_type, {}),
            'bonding': die.bonding,
            'aggregate': {
                'latency_s': stage_latency,
                'dynamic_energy_J': stage_energy,
                'static_power_W': stage_power,
                'total_data_accesses': stage_total_data,
            },
            'operators': operators,
        }
        stages.append(stage_info)

    # Inter-stage communication
    die_to_stage_idx = {die_id: i for i, die_id in enumerate(active_dies)}
    all_edges = cp_edges + offcp_edges
    communications = []
    for src_op, dst_op in all_edges:
        if src_op not in solution.mapping or dst_op not in solution.mapping:
            continue
        src_die = solution.mapping[src_op]
        dst_die = solution.mapping[dst_op]
        if src_die not in die_to_stage_idx or dst_die not in die_to_stage_idx:
            continue

        src_stats = cost_matrix.get(src_op, {}).get(src_die)
        data_vol = src_stats['o_access'] * 2 if src_stats else 0  # FP16

        is_inter = (src_die != dst_die)
        comm_info = {
            'from_stage': die_to_stage_idx[src_die],
            'to_stage': die_to_stage_idx[dst_die],
            'from_die_id': src_die,
            'to_die_id': dst_die,
            'from_chiplet': topology.dies[src_die].identifier,
            'to_chiplet': topology.dies[dst_die].identifier,
            'is_inter_chiplet': is_inter,
            'data_volume_bytes': data_vol,
            'src_dram': topology.dies[src_die].dram_type,
            'dst_dram': topology.dies[dst_die].dram_type,
        }
        if is_inter:
            comm_info['inter_chiplet_energy_pJ_per_bit'] = 1.3
        communications.append(comm_info)

    # Chiplet utilization
    utilization = defaultdict(lambda: {'die_ids': [], 'stages_assigned': []})
    for stage_idx, die_id in enumerate(active_dies):
        die = topology.dies[die_id]
        utilization[die.identifier]['die_ids'].append(die_id)
        utilization[die.identifier]['stages_assigned'].append(stage_idx)

    # Die adjacency (physical topology)
    adjacency_list = []
    seen = set()
    for die in topology.dies:
        for nb in sorted(die.neighbors):
            edge = (min(die.die_id, nb), max(die.die_id, nb))
            if edge not in seen:
                seen.add(edge)
                adjacency_list.append({
                    'die_a': edge[0],
                    'die_b': edge[1],
                    'chiplet_a': topology.dies[edge[0]].identifier,
                    'chiplet_b': topology.dies[edge[1]].identifier,
                })

    return {
        'metadata': {
            **topology.metadata,
            'remap': True,
            'original_network': original_pnr.get('network', {}).get('network', 'unknown'),
            'remapped_network': new_net_name,
        },
        'chiplet_pool': chiplet_pool,
        'physical_topology': {
            'num_dies': len(topology),
            'dies': [
                {
                    'die_id': d.die_id,
                    'identifier': d.identifier,
                    'architecture': d.architecture,
                    'glb_scale': d.glb_scale,
                    'pe_x_scale': d.pe_x_scale,
                    'pe_y_scale': d.pe_y_scale,
                    'dram_type': d.dram_type,
                    'bonding': d.bonding,
                    'neighbors': sorted(d.neighbors),
                }
                for d in topology.dies
            ],
            'adjacency': adjacency_list,
        },
        'network': {
            'network': new_net_name,
            'batch_size': batch_size,
            'sequence_length': seq_len,
            'total_energy_J': solution.total_energy,
            'total_latency_s': solution.total_latency,
            'total_edp': solution.total_edp,
            'num_stages': len(stages),
            'stages': stages,
            'inter_stage_communication': communications,
            'chiplet_utilization': {
                k: dict(v) for k, v in utilization.items()
            },
        },
    }


# ---------------------------------------------------------------------------
# Database loading
# ---------------------------------------------------------------------------

def _load_db_filtered(database_path: str, net_name: str,
                      chunk_size: int = 500_000) -> pd.DataFrame:
    """Load only rows matching `net_name` from a large CSV, using chunked reading."""
    chunks = []
    for chunk in pd.read_csv(database_path, chunksize=chunk_size):
        filtered = chunk[chunk['net'] == net_name]
        if not filtered.empty:
            chunks.append(filtered)
    if chunks:
        return pd.concat(chunks, ignore_index=True)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def remap(
    pnr_config_path: str,
    new_net_name: str,
    batch_size: int,
    seq_len: int,
    database_path: str,
    output_path: str,
    objective: str = 'edp',
    max_iter: int = 50000,
    seed: int = 42,
    verbose: bool = True,
):
    """End-to-end remap pipeline."""

    # Load PnR config
    with open(pnr_config_path) as f:
        pnr_config = json.load(f)

    # Phase 1: Extract topology
    if verbose:
        print("Phase 1: Extracting physical topology...")
    topology = extract_topology(pnr_config)
    if verbose:
        print(f"  {len(topology)} physical dies extracted")
        for d in topology.dies:
            print(f"    die_{d.die_id}: {d.identifier} "
                  f"(DRAM={d.dram_type}, neighbors={sorted(d.neighbors)})")

    # Load database -- only rows for the target network (avoids loading 4.6GB)
    if verbose:
        print(f"\n  Loading database (filtered for '{new_net_name}')...")
    db = _load_db_filtered(database_path, new_net_name)
    if verbose:
        print(f"  Loaded {len(db)} rows")

    # Get layers in database for this network
    db_layers = set(db['layer_name'].unique())
    if verbose:
        print(f"\n  New network '{new_net_name}' has {len(db_layers)} layers in DB")

    # Phase 2: Build cost matrix
    if verbose:
        print("\nPhase 2: Building cost matrix...")
    cost_matrix = build_cost_matrix(db, new_net_name, batch_size, seq_len, topology)
    if verbose:
        n_feasible = sum(
            1 for op in cost_matrix
            for d, s in cost_matrix[op].items() if s is not None)
        n_total = sum(len(v) for v in cost_matrix.values())
        print(f"  {len(cost_matrix)} operators x {len(topology)} dies "
              f"= {n_feasible}/{n_total} feasible entries")

    # Get operator DAG
    all_ops, cp_edges, offcp_edges = get_new_network_ops(
        new_net_name, batch_size, seq_len, db_layers)
    all_edges = cp_edges + offcp_edges
    if verbose:
        print(f"  {len(all_ops)} operators, "
              f"{len(cp_edges)} CP edges, {len(offcp_edges)} off-CP edges")

    # Phase 3: Search
    if verbose:
        print(f"\nPhase 3: Simulated annealing (objective={objective}, "
              f"max_iter={max_iter})...")
    solution = simulated_annealing(
        all_ops, all_edges, cost_matrix, topology,
        objective=objective, max_iter=max_iter, seed=seed)
    if verbose:
        print(f"  Best solution: latency={solution.total_latency:.6e} s, "
              f"energy={solution.total_energy:.6e} J, "
              f"EDP={solution.total_edp:.6e}")
        print("  Mapping:")
        for op, die_id in sorted(solution.mapping.items(),
                                  key=lambda x: x[1]):
            die = topology.dies[die_id]
            print(f"    {op} -> die_{die_id} ({die.identifier})")

    # Phase 4: Generate output
    if verbose:
        print(f"\nPhase 4: Generating output PnR config...")
    output = generate_remap_pnr_config(
        solution, topology, cost_matrix,
        all_ops, cp_edges, offcp_edges,
        new_net_name, batch_size, seq_len, pnr_config)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    if verbose:
        print(f"  Written to {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description='Remap a new network onto a fixed ASIC chiplet topology.')
    parser.add_argument('--pnr-config', required=True,
                        help='Path to the original PnR config JSON')
    parser.add_argument('--new-network', required=True,
                        help='Name of the new network to remap')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--sequence-length', type=int, default=1)
    parser.add_argument('--database', required=True,
                        help='Path to unified_database.csv')
    parser.add_argument('--output', required=True,
                        help='Output JSON path')
    parser.add_argument('--objective', default='edp',
                        choices=['edp', 'energy', 'latency'])
    parser.add_argument('--max-iter', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()

    remap(
        pnr_config_path=args.pnr_config,
        new_net_name=args.new_network,
        batch_size=args.batch_size,
        seq_len=args.sequence_length,
        database_path=args.database,
        output_path=args.output,
        objective=args.objective,
        max_iter=args.max_iter,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
