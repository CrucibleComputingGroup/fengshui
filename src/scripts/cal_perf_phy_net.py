import math
import csv
import yaml
from collections import defaultdict
from dataclasses import dataclass, field
from global_parameter import *
from chiplet_dataclass import *
from network_dataclass import *
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Union
import os
import functools
import time
from convex_hull import *

from utility_functions import cal_opt_val_fused, all_fillings, is_attention_layers
import utility_functions
import gqa_kv

from get_cost import calculate_die_cost, area_with_mem_overheads
import get_cost as _get_cost


def get_cost_defaults():
    """Live handle on the cost-model defaults (re-read, never cached)."""
    return _get_cost.DEFAULT_COST_PARAMS

# Add project root and timeloop_experiments/ to path
import sys as _sys
_project_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
_exp_dir = os.path.join(_project_root, 'timeloop_experiments')
for _p in (_project_root, _exp_dir):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from compute_area import get_chiplet_area_mm2

from timeloop.mem_spec import *
# 0) (optional) reduce pandas copy overhead
pd.options.mode.copy_on_write = True

# 1) cache + helpers
CSV_CACHE = {}
# Tier-2 dict caches: keyed by (csv_path, arch, glb, pe_x, pe_y, net) -> row dict
_CHIPLET_DICT_CACHE = {}   # csv_path -> {(arch,glb,px,py,net): DataFrame}
_ROW_DICT_CACHE = {}       # id(DataFrame) -> {(layer,bs,seq,mapper,tp,fuse,dram_i,dram_o): Series}

_CAT_COLS = ['net', 'layer_name', 'fused_layer_type', 'arch_target', 'dram_i', 'dram_o']
_INT16_COLS = ['batch_size', 'sequence_length', 'mapper_idx', 'tp_degree',
               'glb_scale', 'pe_x_scale', 'pe_y_scale']

# ---------------------------------------------------------------------------
# Case-study-only knob (rebuttal W6, fig:batch_sweep): BOUNDED ATTENTION REPLICATION
# ---------------------------------------------------------------------------
# Default (both None/False): legacy model — batch-agnostic attention ops are
# fully per-batch-replicated (one chiplet replica per request), so the hardware
# terms (area/cost/leakage) scale by batch_size while the attention stage
# latency stays at its batch=1 value. That over-states the hardware at large
# batch (e.g. 64 attention replicas at batch 64).
#
# When BOUNDED_ATTN_REPLICATION is True AND ATTN_REPL_BUDGET_S is a positive
# per-stage latency budget (seconds, = the in-block throughput bottleneck),
# attention is replicated only enough to keep its stage latency <= that budget:
#   R = clamp(ceil(batch * t_attn_base / budget), 1, batch)
# R drives area/cost/leakage (replacing the ×batch full-replication factor) and
# the attention stage latency becomes t_attn_base * batch / R (<= budget, so it
# never becomes the binding stage). DYNAMIC energy is left untouched: it tracks
# total work (×batch) and is independent of R. Leakage energy drops modestly
# (fewer idle replicas), so the energy figure is ~unchanged (equal-or-slightly
# lower), while area/cost (hence EC) drop to the realistic bounded count.
#
# IMPORTANT: every experiment that does NOT explicitly set these is byte-identical
# to before — the new branch is fully gated, so Fig 11 / ablation / competing are
# unaffected. Only the datacenter-LLM case study (batch_sweep_pd.py) sets them.
BOUNDED_ATTN_REPLICATION = False
ATTN_REPL_BUDGET_S = None

def _read_csv_cached(path, **kw):
    df = CSV_CACHE.get(path)
    if df is None:
        df = pd.read_csv(path, **kw)
        # Tier 1: category + int16 dtypes to reduce memory & speed up filters
        for col in _CAT_COLS:
            if col in df.columns:
                df[col] = df[col].astype('category')
        for col in _INT16_COLS:
            if col in df.columns:
                df[col] = df[col].astype('int16')
        CSV_CACHE[path] = df
    return df

def _build_chiplet_dict(path):
    """Build Tier-2 dict: (arch,glb,px,py,net) -> DataFrame subset."""
    if path in _CHIPLET_DICT_CACHE:
        return _CHIPLET_DICT_CACHE[path]
    df = _read_csv_cached(path)
    d = {}
    for key, grp in df.groupby(['arch_target', 'glb_scale', 'pe_x_scale', 'pe_y_scale', 'net'],
                                observed=True):
        d[key] = grp
    _CHIPLET_DICT_CACHE[path] = d
    return d

_ROW_DICT_FIELDS = ['dynamic_energy', 'latency', 'static_power', 'i_access', 'w_access', 'o_access']


def _get_bw_contention(net_name, layer_name):
    """Return (dram_i_divisor, dram_o_divisor) for a layer's BW contention."""
    # Strip layer0_ prefix to match BW_CONTENTION_MAP keys
    bare = layer_name.split('layer0_', 1)[-1] if 'layer0_' in layer_name else layer_name
    for net_key, layer_map in BW_CONTENTION_MAP.items():
        if net_key in net_name.lower():
            if bare in layer_map:
                return layer_map[bare]
            break
    return (1, 1)


def _apply_bw_contention(layer_latency, layer_row, dram_i, dram_o, tp, net_name, layer_name):
    """Re-throttle latency for DRAM bandwidth contention from parallel ops.

    When parallel operations share a physical DRAM, they split the available
    bandwidth.  This function recomputes the BW-limited latency with the
    reduced bandwidth, using the same roofline formula as
    postprocess_bw.apply_bw_throttling_analytical.
    """
    div_i, div_o = _get_bw_contention(net_name, layer_name)
    if div_i == 1 and div_o == 1:
        return layer_latency

    original_cycles = layer_latency / cycle_time
    bw_I = dram_type_bandwidth_width_dict[dram_i]['bandwidth'] * 8 / word_size / tp / div_i
    bw_O = dram_type_bandwidth_width_dict[dram_o]['bandwidth'] * 8 / word_size / tp / div_o

    i_total = float(layer_row['i_access']) + float(layer_row['w_access'])
    o_total = float(layer_row['o_access'])

    dram_I_cycles = math.ceil(i_total / bw_I) if bw_I > 0 and i_total > 0 else 0
    dram_O_cycles = math.ceil(o_total / bw_O) if bw_O > 0 and o_total > 0 else 0

    new_cycles = max(original_cycles, dram_I_cycles, dram_O_cycles)
    return new_cycles * cycle_time

def _build_row_dict(chiplet_df):
    """Build Tier-2 row-level dict for a chiplet DataFrame subset.
    Stores plain dicts instead of Pandas Series to avoid costly Series.__getitem__.
    GQA: the KV-cache rows of the attention ops are corrected here, once (gqa_kv)."""
    df_id = id(chiplet_df)
    if df_id in _ROW_DICT_CACHE:
        return _ROW_DICT_CACHE[df_id]
    d = {}
    # Vectorized extraction: get column arrays once, iterate by index
    cols_key = ['layer_name', 'batch_size', 'sequence_length',
                'mapper_idx', 'tp_degree', 'fused_layer_type', 'dram_i', 'dram_o']
    cols_val = _ROW_DICT_FIELDS
    cols_chiplet = ['net', 'arch_target', 'glb_scale', 'pe_x_scale', 'pe_y_scale']
    key_arrays = {c: chiplet_df[c].values for c in cols_key}
    val_arrays = {c: chiplet_df[c].values for c in cols_val}
    chiplet_arrays = {c: chiplet_df[c].values for c in cols_chiplet}
    # KV-head grouping factor of every network in this subset (raises if a network has no source)
    kv_group = {net: gqa_kv.kv_group_factor(net) for net in chiplet_df['net'].unique()}
    for net, layers in chiplet_df.groupby('net')['layer_name'].unique().items():
        gqa_kv.check_kv_ops(net, layers, kv_group[net])   # a GQA attention op outside KV_OPS raises
    n = len(chiplet_df)
    for i in range(n):
        key = (key_arrays['layer_name'][i],
               int(key_arrays['batch_size'][i]),
               int(key_arrays['sequence_length'][i]),
               int(key_arrays['mapper_idx'][i]),
               int(key_arrays['tp_degree'][i]),
               key_arrays['fused_layer_type'][i],
               key_arrays['dram_i'][i],
               key_arrays['dram_o'][i])
        row = {c: val_arrays[c][i] for c in cols_val}
        if key[0] in gqa_kv.KV_OPS:
            net = chiplet_arrays['net'][i]
            gqa_kv.correct_db_row(row, net, key[0], chiplet_arrays['arch_target'][i],
                                  chiplet_arrays['glb_scale'][i], chiplet_arrays['pe_x_scale'][i],
                                  chiplet_arrays['pe_y_scale'][i], key[4], key[6], key[7],
                                  kv_group[net])
        d[key] = row
    _ROW_DICT_CACHE[df_id] = d
    return d

# 3) bind the global you use elsewhere
net_mem_df = _read_csv_cached("network_analysis.csv")#CSV_CACHE["network_analysis.csv"]

# Tier-2 dict for net_mem_df: (net_name, layer_name, fused_layer_type, batch_size, seq_len) -> row
_NET_MEM_DICT = None
_NET_MEM_FIELDS = ['in_mem', 'weight_mem', 'out_mem', 'operations']

def _get_net_mem_dict():
    global _NET_MEM_DICT
    if _NET_MEM_DICT is None:
        _NET_MEM_DICT = {}
        # Vectorized extraction for speed
        key_cols = ['net_name', 'layer_name', 'fused_layer_type', 'batch_size', 'sequence_length']
        key_arrays = {c: net_mem_df[c].values for c in key_cols}
        val_arrays = {c: net_mem_df[c].values for c in _NET_MEM_FIELDS}
        # GQA: K / V (weight_mem of the attention ops) sized with num_key_value_heads (gqa_kv)
        val_arrays[gqa_kv.KV_CAPACITY] = gqa_kv.kv_capacity(net_mem_df).values
        tp_arr = net_mem_df['tp'].values if 'tp' in net_mem_df.columns else None
        n = len(net_mem_df)
        with_tp = {}
        for i in range(n):
            row_dict = {c: val_arrays[c][i] for c in _NET_MEM_FIELDS}
            key = (key_arrays['net_name'][i], key_arrays['layer_name'][i],
                   key_arrays['fused_layer_type'][i],
                   int(key_arrays['batch_size'][i]),
                   int(key_arrays['sequence_length'][i]))
            _NET_MEM_DICT[key] = row_dict
            if tp_arr is not None:
                key_tp = key + (int(tp_arr[i]),)
                with_tp[key_tp] = row_dict
        _NET_MEM_DICT['_with_tp'] = with_tp
    return _NET_MEM_DICT


def preload_database(csv_file: str, needed_nets=None):
    """Pre-load and pre-filter the performance database into memory.

    Call this ONCE before any optimization loop to avoid repeated CSV reads
    and expensive groupby operations. Optionally filter to only needed networks
    to reduce memory and speedup groupby dramatically.

    Args:
        csv_file: Path to the CSV database file
        needed_nets: Optional set/list of network names to keep. If None, keep all.

    Returns:
        Number of groups built
    """
    import time as _time
    t0 = _time.perf_counter()

    # Read CSV (or use cache)
    df = CSV_CACHE.get(csv_file)
    if df is None:
        df = pd.read_csv(csv_file)
        for col in _CAT_COLS:
            if col in df.columns:
                df[col] = df[col].astype('category')
        for col in _INT16_COLS:
            if col in df.columns:
                df[col] = df[col].astype('int16')

    # Filter to needed networks if specified
    if needed_nets is not None:
        needed_nets = set(needed_nets)
        df = df[df['net'].isin(needed_nets)].copy()
        # Re-categorize after filter to drop unused categories
        for col in _CAT_COLS:
            if col in df.columns and hasattr(df[col], 'cat'):
                df[col] = df[col].cat.remove_unused_categories()

    CSV_CACHE[csv_file] = df

    # Build chiplet dict (groupby)
    d = {}
    for key, grp in df.groupby(['arch_target', 'glb_scale', 'pe_x_scale', 'pe_y_scale', 'net'],
                                observed=True):
        d[key] = grp
    _CHIPLET_DICT_CACHE[csv_file] = d

    # Row dicts are built lazily by _build_row_dict() on first access per group

    dt = _time.perf_counter() - t0
    print(f'preload_database: {len(df):,} rows, {len(d)} groups, {dt:.1f}s')
    return len(d)


def get_chiplet_data(
    csv_file: str,
    arch_target: str,
    glb_scale: int,
    pe_x_scale: int,
    pe_y_scale: int,
    net_name: str
):
    """
    Get filtered performance data for a specific chiplet and network.
    Uses Tier-2 dict lookup for O(1) access.

    Args:
        csv_file: Path to the CSV file
        arch_target: Architecture target name
        glb_scale: Global buffer scale factor
        pe_x_scale: PE X dimension scale factor
        pe_y_scale: PE Y dimension scale factor
        net_name: Network name

    Returns:
        DataFrame with filtered chiplet data
    """
    # PIM has a single valid config (glb=1, pe=1x1, GDDR7).
    # Normalize any PIM request to canonical params so the algorithm
    # doesn't need to discover the exact config.
    if arch_target == 'PIM':
        glb_scale, pe_x_scale, pe_y_scale = 1, 1, 1

    chiplet_dict = _build_chiplet_dict(csv_file)
    key = (arch_target, glb_scale, pe_x_scale, pe_y_scale, net_name)
    return chiplet_dict.get(key, pd.DataFrame())

def extract_para_from_row(layer_row, name):
    # despite unnecessary, keep it in case pre-processsing is needed before using directly
    return layer_row[name]

def get_fused_layer_type(layer_idx, num_layers):
    if num_layers == 1:
        return "single"
    if layer_idx == 0:
        return "start"
    if layer_idx == num_layers - 1:
        return "end"
    return "middle"

def cal_mem_req_for_fusion_group(net_name, fusion_group, batch_size, sequence_length):
    # calculate memory requirements for a fusion group
    # on_sram:
    # true, if memory check is met, true fusion (cached intermediate data)
    # false, add dram_backup, ignore memory check

    # store sum(weights for all layers and input for the first layers)
    dram_i_cap = 0
    # store max(intermediate outputs for all layers) if on_sram
    dram_buffer_cap = 0
    # store output for the last layer
    dram_o_cap = 0

    mem_dict = _get_net_mem_dict()
    for layer_idx, layer in enumerate(fusion_group.layers):
        fused_layer_type = get_fused_layer_type(layer_idx, len(fusion_group.layers))

        key = (net_name, layer.name, fused_layer_type, batch_size, sequence_length)
        layer_row = mem_dict.get(key)
        if layer_row is None:
            continue

        i_utilized_capacity = layer_row['in_mem']
        w_utilized_capacity = layer_row['weight_mem']
        o_utilized_capacity = layer_row['out_mem']

        # Parallel ops sharing a physical DRAM: scale weight/output capacity
        # by the number of sharers (e.g. Q/K/V share dram_i → 3× weights).
        div_i, div_o = _get_bw_contention(net_name, layer.name)
        w_utilized_capacity *= div_i
        o_utilized_capacity *= div_o

        # add weight no matter fuse layer type
        dram_i_cap +=w_utilized_capacity
        # add i_utilized if first (start) layer in the fusion group
        if fused_layer_type == 'start' or fused_layer_type == 'single':
            dram_i_cap += i_utilized_capacity
        # add o_utilized if last (end) layer in the fusion group
        if fused_layer_type == 'end' or fused_layer_type == 'single':
            dram_o_cap += o_utilized_capacity
        # calculate buffer capacity max(inputs of all intermediate layers)

        if fused_layer_type != 'single' and fused_layer_type != 'start':
            dram_buffer_cap = max(dram_buffer_cap, i_utilized_capacity)
    return dram_i_cap, dram_buffer_cap, dram_o_cap

# --- Caches for cost/area functions called in inner loop ---
_DIE_COST_CACHE = {}
_AREA_OVERHEAD_CACHE = {}
_CHIPLET_AREA_CACHE = {}

def _cached_die_cost(area_mm2, bonding):
    # catch_layer selects the per-node CATCH cost/yield row, so it is a real
    # varying parameter; the cache MUST discriminate on it or a second run at a
    # different node silently returns the first run's costs.
    key = (area_mm2, bonding, get_cost_defaults().catch_layer)
    r = _DIE_COST_CACHE.get(key)
    if r is None:
        r = calculate_die_cost(area_mm2, bonding)
        _DIE_COST_CACHE[key] = r
    return r

def _cached_area_overheads(area_mm2, bonding, dram_type, all_phy=False):
    key = (area_mm2, bonding, dram_type, all_phy)
    r = _AREA_OVERHEAD_CACHE.get(key)
    if r is None:
        r = area_with_mem_overheads(area_mm2, bonding, dram_type=dram_type, all_phy=all_phy)
        _AREA_OVERHEAD_CACHE[key] = r
    return r

def _cached_chiplet_area(arch_target, pe_x_scale, pe_y_scale, glb_scale):
    key = (arch_target, pe_x_scale, pe_y_scale, glb_scale)
    r = _CHIPLET_AREA_CACHE.get(key)
    if r is None:
        r = get_chiplet_area_mm2(arch_target, pe_x_scale=pe_x_scale,
                                  pe_y_scale=pe_y_scale, glb_scale=glb_scale)
        _CHIPLET_AREA_CACHE[key] = r
    return r


def _offcp_core_mm2(chiplet):
    """Core die area (mm^2) that the off-critical-path builder prices a chiplet at.

    Single source of truth so the regression guard (verify_pim_integration.py) can
    exercise the real call-site decision rather than re-implement it.  A PIM chiplet's
    die IS the GDDR7 module (PIM_DIE_AREA_MM2), not the small PE compute core -- mirror
    the main path (cal_perf_phy_net.py:465).  Single die here: the main path's x2 is
    double-buffering for pipelined on-CP groups, which an off-CP op running in slack
    does not require.
    """
    if getattr(chiplet, 'arch_target', '') == 'PIM':
        return PIM_DIE_AREA_MM2
    return _cached_chiplet_area(
        chiplet.arch_target, chiplet.pe_x_scale,
        chiplet.pe_y_scale, chiplet.global_buffer_size_scale)["total_area_mm2"]


def calculate_actual_area(area, dram_type, bonding, all_phy=False):
    overheads = _cached_area_overheads(area, bonding, dram_type, all_phy=all_phy)
    final_area = overheads["final_area"]
    return final_area


def calculate_inter_chiplet_communication(num_bits,bonding,num_hop):
    # given number of bits and number of distance (hops) to travel
    # calculate energy based on bonding technique
    if bonding == "2D":
        # J
        return num_bits*1*num_hop*0.5*1e-12
    if bonding == "2.5D":
        return num_bits*num_hop*0.3*1e-12
    return num_bits*num_hop*0.5*1e-12


# --- Unit reconciliation for inter-chiplet communication energy ---------------
# calculate_inter_chiplet_communication() above applies an energy-per-BIT constant
# (0.3 pJ/bit for 2.5D, 0.5 pJ/bit for 2D), but Timeloop reports
# i_access / w_access / o_access as SCALAR (word) counts:
#
#   parse_stats.py    i_access = i_scalar_reads + i_scalar_fills + i_scalar_updates
#   postprocess_bw.py docstring: "i_access, w_access, o_access: scalar counts"
#
# Every other per-bit term in this codebase therefore multiplies by word_size
# before applying a pJ/bit figure -- parse_stats.py (the DRAM re-basing),
# postprocess_bw.py (apply_bw_throttling_analytical), the MoE switch model below
# (bits = batch_tokens * H * word_size), and remap.py (explicit words->bits).
#
# Two of the four call sites were handing raw scalar counts straight through,
# which under-charged those paths by a factor of word_size:
#
#   main path        (search / reproduction path)  -- scalar counts, now converted
#   off-CP, non-PIM                                -- scalar counts, now converted
#   off-CP, PIM      (_in_bits = in_mem * 1e9 * 8) -- already bits, unchanged
#   remap.py         (explicit * 16 words->bits)    -- already bits, unchanged
#
# Measured effect of this correction on the shipped n=10 energy pool, evaluated
# over the 20 virtual networks: geometric-mean energy 3.472661e-02 -> 3.482846e-02
# (+0.293%), with no change to the chiplet configuration chosen for any network.


def _accesses_to_bits(n_accesses):
    """Timeloop scalar (word) access count -> bits."""
    return n_accesses * word_size


def calculate_network_performance_with_memory_check(
    physical_network,
    csv_file: str,
    # given chiplet config (dram/arch included)
    chiplet_config,
    buffer_config,
    net_name: str,
    # batch_size and sequence_length are kind of like global parameter
    batch_size: int,
    sequence_length: int,
    chiplet_data: pd.DataFrame,
    chiplet_vector_data: Optional[pd.DataFrame],
    het_batch_candidates: Optional[Dict[str, List[int]]] = None,
) -> Dict:
    """
    Calculate the dynamic energy, static power, and latency for each fusion group in the physical network,
    on a given chiplet config
    on_sram:
    true, if memory check is met, true fusion (cached intermediate data)
    false, add dram_backup, ignore memory check
    
    Args:
        physical_network: PhysicalNetwork object containing fusion groups and layer info
        csv_file: Path to the CSV file containing chiplet performance data
        chiplet_config: ChipletConfig object representing the chiplet being used
        net_name: Network name corresponding to entries in the CSV
        
    Returns:
        Dictionary with performance metrics for each fusion group.
        Area for this chiplet
        If a specific fusion group doesn't meet memory constraints, its metrics are set to infinity.
    """
    # Get chiplet data — if no data exists for this chiplet+network combo
    # (e.g. PIM on CNN workloads), return empty results so other chiplets
    # in the pool can still handle this network.
    if chiplet_data.empty:
        return {}

    is_pim = (chiplet_config.arch_target == 'PIM')

    # Get available memory for this chiplet (in words)
    # PIM: memory is inherent — no separate GLB capacity check
    available_memory = (float('inf') if is_pim
                        else chiplet_config.global_buffer_size_scale * chiplet_config.arch_para_dict[chiplet_config.arch_target][0])

    # Dictionary to store results
    results = {}
    # length should be len(fusion_groups)+1
    fusion_group_mem_dict = {}
    # length should be len(fusion_groups)
    # used for on_sram
    fusion_group_mem_buffer_dict = {}
    #
    fusion_group_mem_spec_dict = {}

    # Memory calculation
    num_group = len(physical_network.fusion_groups)

    out_mem_prev = 0
    for group_idx, fusion_group in enumerate(physical_network.fusion_groups):
        fusion_group_mem_dict[group_idx] = 0

        in_mem, buffer_mem, out_mem = cal_mem_req_for_fusion_group(net_name, fusion_group, batch_size, sequence_length)

        fusion_group_mem_dict[group_idx] = max(out_mem_prev, in_mem)
        out_mem_prev = out_mem
        fusion_group_mem_buffer_dict[group_idx] = buffer_mem
    # last layer, add additional memory for output
    fusion_group_mem_dict[len(physical_network.fusion_groups)] = out_mem_prev

    # Memory spec for double-buffering (needed for both traditional and PIM)
    glb_layer_idx=0
    for group_idx, fusion_group in enumerate(physical_network.fusion_groups):
        num_layers_per_group = len(fusion_group.layers)
        fusion_group_mem_spec_dict[group_idx] = {}
        fusion_group_mem_spec_dict[group_idx][buffer_config[glb_layer_idx]]=get_memory_spec(fusion_group_mem_dict[group_idx], buffer_config[glb_layer_idx])
        glb_layer_idx+=num_layers_per_group

    # account for the last layer (final output)
    fusion_group_mem_spec_dict[len(physical_network.fusion_groups)] = {}
    fusion_group_mem_spec_dict[len(physical_network.fusion_groups)][buffer_config[glb_layer_idx]]=get_memory_spec(fusion_group_mem_dict[len(physical_network.fusion_groups)], buffer_config[glb_layer_idx])

    if not is_pim:
        # Pre-compute chiplet core area (invariant across groups/tp/bonding)
        # Read from CSV 'area' column (um²); fall back to Accelergy computation if unavailable
        _csv_area = chiplet_data['area'].iloc[0] if 'area' in chiplet_data.columns else float('nan')
        if not math.isnan(_csv_area) and _csv_area > 0:
            core_area_mm2 = _csv_area / 1e6  # um² → mm²
        else:
            core_area_info = _cached_chiplet_area(
                chiplet_config.arch_target,
                chiplet_config.pe_x_scale,
                chiplet_config.pe_y_scale,
                chiplet_config.global_buffer_size_scale
            )
            core_area_mm2 = core_area_info["total_area_mm2"]

        # Vector unit leakage (1D softmax unit attached to each chiplet, scales with pe_x)
        from parse_stats import add_vector_unit
        _, vector_leak_W = add_vector_unit(int(pe_x_base_size * chiplet_config.pe_x_scale))
    else:
        # PIM: die area from GDDR7 module + compute overhead (see global_parameter.py)
        core_area_mm2 = PIM_DIE_AREA_MM2
        vector_leak_W = 0.0  # PIM handles softmax natively, no separate vector unit

    # Pre-build row dicts for both chiplet data and vector data (once, not per-layer)
    row_dict_main = _build_row_dict(chiplet_data)
    # PIM handles all ops (including softmax) natively — no separate vector unit
    row_dict_vector = None
    if not is_pim:
        row_dict_vector = _build_row_dict(chiplet_vector_data) if (chiplet_vector_data is not None and not chiplet_vector_data.empty) else None

    # Pre-compute softmax layer set for this network
    _is_softmax = utility_functions.is_softmax_layers

    # Process each fusion group
    glb_layer_idx=0
    for group_idx, fusion_group in enumerate(physical_network.fusion_groups):
        # Initialize results for this group
        results[group_idx] = {}
        results[group_idx]['group_results'] = {}

        num_layers_per_group = len(fusion_group.layers)

        # Pre-compute batch_agnostic check (invariant across tp/mapper/bonding)
        contain_batch_agnostic = any(is_attention_layers(layer.name) for layer in fusion_group.layers)
        batch_size_scale = batch_size if contain_batch_agnostic else 1

        # DRAM type for this fusion group (from buffer_config)
        # PIM is GDDR-based: only activated when buffer_config assigns GDDR7
        group_dram_type = buffer_config[glb_layer_idx]

        # Pre-compute fused_layer_type, softmax, and attention flags per layer (invariant)
        layer_info = []
        for idx, layer in enumerate(fusion_group.layers):
            flt = get_fused_layer_type(idx, num_layers_per_group)
            is_sm = _is_softmax(layer.name)
            is_attn = is_attention_layers(layer.name)
            layer_info.append((layer.name, flt, is_sm, is_attn))

        for tp in tp_degrees:
            results[group_idx]['group_results'][tp] = {}
            for mapper_idx in range(num_mapping_per_arch):
                results[group_idx]['group_results'][tp][mapper_idx] = {}
                for bonding in bonding_techniques:
                    results[group_idx]['group_results'][tp][mapper_idx][bonding] = {}
                    for on_sram in on_srams:
                        group_results = {
                            "dynamic_energy": 0.0,
                            "static_power": 0.0,
                            "latency": 0.0,
                            "cost": 0.0,
                            # Input-boundary buffer components (for PIM-adjacency deduction)
                            "area_input_buf": 0.0,
                            "cost_input_buf": 0.0,
                            "leak_input_buf": 0.0,
                        }

                        layer_row = None
                        for idx, (layer_name, fused_layer_type, is_sm, is_attn) in enumerate(layer_info):
                            # PIM handles all ops natively; traditional arch uses separate vector unit for softmax
                            rd = row_dict_main if is_pim else (row_dict_vector if is_sm else row_dict_main)
                            if rd is None:
                                continue

                            # Attention ops are batch-agnostic: DB only has batch=1.
                            # PIM DB also only has batch=1; for batch>1 we look up
                            # batch=1 and scale energy/latency by batch_size below.
                            # MoE expert layers: use batch=1 because each expert
                            # only processes B*k/E tokens (routing fraction), not
                            # the full batch; scaling handled in _moe_expand_expert_groups.
                            is_expert = layer_name in MOE_EXPERT_OPS
                            lookup_batch = 1 if (is_attn or is_pim or is_expert) else batch_size
                            pim_batch_scale = batch_size if (is_pim and not is_attn and batch_size > 1) else 1

                            # PIM data has dram_i/dram_o = 'GDDR7' — only matches
                            # when buffer_config assigns GDDR7 at this boundary.
                            # PIM + non-GDDR7 is infeasible: mark as inf.
                            dram_i_here = buffer_config[idx+glb_layer_idx]
                            dram_o_here = buffer_config[idx+glb_layer_idx+1]
                            if is_pim and (dram_i_here != 'GDDR7' or dram_o_here != 'GDDR7'):
                                group_results["dynamic_energy"] = float("inf")
                                group_results["latency"] = float("inf")
                                break

                            row_key = (layer_name, lookup_batch, sequence_length,
                                       mapper_idx, tp, fused_layer_type,
                                       dram_i_here, dram_o_here)
                            io_row = rd.get(row_key)

                            layer_data = None
                            if on_sram:
                                layer_data = io_row

                            if layer_data is None:
                                continue

                            # layer_data is now a plain dict (fast key access)
                            layer_row = layer_data

                            layer_latency = layer_row['latency'] * pim_batch_scale

                            # BW contention: parallel ops sharing a physical DRAM
                            if not is_pim:
                                layer_latency = _apply_bw_contention(
                                    layer_latency, layer_row,
                                    buffer_config[idx+glb_layer_idx],
                                    buffer_config[idx+glb_layer_idx+1],
                                    tp, net_name, layer_name)

                            if is_pim:
                                # PIM: energy comes directly from DB (already accounts for in-memory compute)
                                # No inter-chiplet communication.
                                # For batch>1, scale energy linearly (pim_batch_scale).
                                dynamic_energy = layer_row['dynamic_energy'] * tp * pim_batch_scale
                            else:
                                # --- Dynamic energy: try het_batch candidates, pick lowest per-item ---
                                lookup_dram_i = buffer_config[idx+glb_layer_idx]
                                lookup_dram_o = buffer_config[idx+glb_layer_idx+1]
                                candidates = (het_batch_candidates.get(layer_name, [lookup_batch])
                                              if het_batch_candidates else [lookup_batch])
                                best_dyn_e = layer_row['dynamic_energy']

                                for cand_bs in candidates:
                                    if cand_bs == lookup_batch:
                                        continue
                                    het_key = (layer_name, cand_bs, sequence_length,
                                               mapper_idx, tp, fused_layer_type,
                                               lookup_dram_i, lookup_dram_o)
                                    het_row = rd.get(het_key)
                                    if het_row is not None:
                                        multiplier = cand_bs / lookup_batch
                                        amortized = het_row['dynamic_energy'] / multiplier
                                        if amortized < best_dyn_e:
                                            best_dyn_e = amortized

                                dynamic_energy = best_dyn_e

                                reads = layer_row['i_access']
                                writes = layer_row['o_access']

                                # DB stores per-chip energy; total = per_chip * tp
                                dynamic_energy *= tp

                                read_inter_chiplet_energy = calculate_inter_chiplet_communication(_accesses_to_bits(reads),bonding,num_hop=1)
                                write_inter_chiplet_energy = calculate_inter_chiplet_communication(_accesses_to_bits(writes),bonding,num_hop=1)

                                group_results["dynamic_energy"] += read_inter_chiplet_energy
                                group_results["dynamic_energy"] += write_inter_chiplet_energy

                            # Batch-agnostic ops (attention): chiplets are duplicated
                            # for batch parallelism, so dynamic energy scales with
                            # batch_size while latency stays unchanged.
                            if is_attn and batch_size > 1:
                                dynamic_energy *= batch_size

                            group_results["dynamic_energy"] += dynamic_energy
                            group_results["latency"] += layer_latency

                        if on_sram and not is_pim:
                            if fusion_group_mem_buffer_dict[group_idx] > available_memory:
                                group_results["dynamic_energy"] = float("inf")
                                group_results["latency"] = float("inf")

                        if layer_row is not None:
                            if is_pim:
                                # PIM: compute is embedded inside GDDR7 memory modules.
                                # Double buffering = 2 PIM dies [PIM_1, PIM_2] that
                                # alternate roles (one computes, one serves as buffer).
                                # No separate DRAM chips — everything is PIM.

                                full_area = calculate_actual_area(core_area_mm2, dram_type='GDDR7', bonding=bonding)
                                layer_acc_cost = _cached_die_cost(full_area, bonding) * tp * 2 * batch_size_scale

                                # Static power: compute set (from DB) + buffer set (idle, memory leakage only) + PHY
                                group_results["static_power"] = layer_row['static_power'] * tp * batch_size_scale
                                group_results["static_power"] += tp * mem_specs['GDDR7']['leakage_power'] * batch_size_scale
                                phy_leakage_density = 0.0001  # W/mm² at 14nm
                                phy_overheads = _cached_area_overheads(0, bonding, 'GDDR7')
                                group_results["static_power"] += (phy_overheads["phy_area"] + phy_overheads["ctrl_area"]) * phy_leakage_density * batch_size_scale

                                # Area: 2× PIM dies for double buffering
                                group_results['area'] = core_area_mm2 * tp * 2 * batch_size_scale

                                # Cost: 2× PIM die cost
                                group_results["cost"] = layer_acc_cost
                            else:
                                # Compute chiplets carry PHY for all DRAM types
                                full_area = calculate_actual_area(core_area_mm2, dram_type=group_dram_type, bonding=bonding, all_phy=True)

                                layer_acc_cost = _cached_die_cost(full_area, bonding) * tp * batch_size_scale
                                # double buffering
                                layer_buffer_cost = 2*fusion_group_mem_spec_dict[group_idx][buffer_config[glb_layer_idx]]["cost"]* batch_size_scale
                                if group_idx==num_group-1:
                                    layer_buffer_cost += 2*fusion_group_mem_spec_dict[group_idx+1][buffer_config[glb_layer_idx+num_layers_per_group]]["cost"]*batch_size_scale
                                # W

                                group_results["static_power"] = layer_row['static_power'] * tp*batch_size_scale + vector_leak_W * batch_size_scale + 2*fusion_group_mem_spec_dict[group_idx][buffer_config[glb_layer_idx]]["leakage_power"]*batch_size_scale
                                if group_idx==num_group-1:
                                    group_results["static_power"] += 2*fusion_group_mem_spec_dict[group_idx+1][buffer_config[glb_layer_idx+num_layers_per_group]]["leakage_power"]*batch_size_scale
                                # PHY + controller leakage for all DRAM interfaces on compute chiplet
                                phy_leakage_density = 0.0001  # W/mm² at 14nm (from Accelergy)
                                phy_overheads = _cached_area_overheads(0, bonding, group_dram_type, all_phy=True)
                                group_results["static_power"] += (phy_overheads["phy_area"] + phy_overheads["ctrl_area"]) * phy_leakage_density * batch_size_scale
                                # double buffering — all in mm², footprint for 3D-stacked
                                group_results['area'] = core_area_mm2 * tp * batch_size_scale + 2*fusion_group_mem_spec_dict[group_idx][buffer_config[glb_layer_idx]]["footprint_area"]*batch_size_scale
                                if group_idx==num_group-1:
                                    group_results['area'] += 2*fusion_group_mem_spec_dict[group_idx+1][buffer_config[glb_layer_idx+num_layers_per_group]]["footprint_area"]*batch_size_scale
                                group_results["cost"] = layer_acc_cost + layer_buffer_cost

                                # Track input-boundary buffer for PIM-adjacency deduction
                                group_results["area_input_buf"] = 2*fusion_group_mem_spec_dict[group_idx][buffer_config[glb_layer_idx]]["footprint_area"]*batch_size_scale
                                group_results["cost_input_buf"] = 2*fusion_group_mem_spec_dict[group_idx][buffer_config[glb_layer_idx]]["cost"]*batch_size_scale
                                group_results["leak_input_buf"] = 2*fusion_group_mem_spec_dict[group_idx][buffer_config[glb_layer_idx]]["leakage_power"]*batch_size_scale

                        # --- Bounded attention replication (case-study knob; default OFF) ---
                        # Legacy path replicates attention fully per request (batch_size_scale ==
                        # batch_size above; stage latency at ×1). When enabled, replicate only
                        # enough to keep the attention stage <= the in-block bottleneck budget:
                        # rescale the per-batch hardware terms down to the bounded replica count R
                        # and stretch the stage latency to match. Dynamic energy is untouched
                        # (R-invariant); leakage drops with the replica count.
                        if (BOUNDED_ATTN_REPLICATION and contain_batch_agnostic
                                and batch_size > 1 and ATTN_REPL_BUDGET_S
                                and layer_row is not None
                                and math.isfinite(group_results["latency"])
                                and group_results["latency"] > 0):
                            _t_base = group_results["latency"]            # ×1 (full-repl) attn stage latency
                            _R = max(1, min(batch_size,
                                            math.ceil(batch_size * _t_base / ATTN_REPL_BUDGET_S)))
                            _corr = _R / float(batch_size_scale)          # batch_size_scale == batch_size here
                            for _k in ("static_power", "cost", "area",
                                       "area_input_buf", "cost_input_buf", "leak_input_buf"):
                                group_results[_k] *= _corr
                            group_results["latency"] = _t_base * batch_size / _R

                        # Store the fusion group results
                        # batch_size, and sequence length are not included as they are sort of global parameter
                        results[group_idx]['group_results'][tp][mapper_idx][bonding][on_sram] = group_results
        glb_layer_idx += num_layers_per_group
    return results

def cal_buffer_config(chiplet_max_pes, physical_network: PhysicalNetwork, batch_size: int, sequence_length: int, max_tp: int):
    # find one buffer config that won't hurt the performance
    # used as initial population for genetic algorithm
    # assign one buffer config to one layer location (even if it's within a fusion group for consistency)
    layer_buffer_config = ["HBM3" for _ in range(len(physical_network.virtual_network.layers)+1)]

    min_buffer_config = [0 for _ in range(len(physical_network.fusion_groups)+1)]
    group_buffer_config = ["HBM3" for _ in range(len(physical_network.fusion_groups)+1)]
    fusion_groups = physical_network.fusion_groups

    mem_dict = _get_net_mem_dict()
    for group_idx, fusion_group in enumerate(fusion_groups):
        total_group_compute = 0
        total_group_iw_memory = 0
        total_group_o_memory = 0
        for layer_idx, layer in enumerate(fusion_group.layers):
            fused_layer_type = get_fused_layer_type(layer_idx, len(fusion_group.layers))

            lookup_name = getattr(physical_network.virtual_network, 'original_name', physical_network.virtual_network.network_name)
            key = (lookup_name, layer.name, fused_layer_type, batch_size, sequence_length, 1)
            layer_data = mem_dict['_with_tp'].get(key)
            if layer_data is None:
                continue
            total_group_compute += layer_data["operations"]
            total_group_iw_memory += layer_data["in_mem"]+layer_data["weight_mem"]

            total_group_o_memory += layer_data["out_mem"]

        if group_idx == 0:
            prev_buffer_req = 0
        else:
            prev_buffer_req = min_buffer_config[group_idx-1]
        if total_group_compute == 0:
            min_buffer_config[group_idx] = prev_buffer_req
        else:
            min_buffer_config[group_idx] = max(total_group_iw_memory/(total_group_compute/chiplet_max_pes*1e-9)*max_tp, prev_buffer_req)

        for dram_type in dram_options:
            # naturally small to large
            if min_buffer_config[group_idx] <= dram_type_bandwidth_width_dict[dram_type]["bandwidth"]:
                group_buffer_config[group_idx] = dram_type
                break

    glb_layer_idx = 0
    for group_idx, fusion_group in enumerate(fusion_groups):
        for layer_idx, layer in enumerate(fusion_group.layers):
            layer_buffer_config[glb_layer_idx] = group_buffer_config[group_idx]
            glb_layer_idx += 1

    # assign final output buffer type
    layer_buffer_config[-1] = group_buffer_config[-1]
    return layer_buffer_config

def cal_buffer_configs(chiplet_max_pes, physical_network: PhysicalNetwork, batch_size: int, sequence_length: int, max_tp: int):
    # STALE
    # used to calculate all possible buffer configs
    # pruning based on roofline model
    # for simplicity
    # calculate the bandwidth requirement for tp =1
    # then multiply by max_tp (2) to determine the actual buffer type to use
    buffer_configs = []
    # each element is a buffer_config
    min_buffer_config = [0 for _ in range(len(physical_network.fusion_groups)+1)]
    buffer_config_options = [["HBM3"] for _ in range(len(physical_network.fusion_groups)+1)]

    fusion_groups = physical_network.fusion_groups

    mem_dict = _get_net_mem_dict()
    for group_idx, fusion_group in enumerate(fusion_groups):
        total_group_compute = 0
        total_group_iw_memory = 0
        total_group_o_memory = 0
        for layer_idx, layer in enumerate(fusion_group.layers):
            fused_layer_type = get_fused_layer_type(layer_idx, len(fusion_group.layers))

            lookup_name = getattr(physical_network.virtual_network, 'original_name', physical_network.virtual_network.network_name)
            key = (lookup_name, layer.name, fused_layer_type, batch_size, sequence_length, 1)
            layer_data = mem_dict['_with_tp'].get(key)
            if layer_data is None:
                continue
            total_group_compute += layer_data["operations"]
            total_group_iw_memory += layer_data["in_mem"]+layer_data["weight_mem"]

            total_group_o_memory += layer_data["out_mem"]

        if group_idx == 0:
            prev_buffer_req = 0
        else:
            prev_buffer_req = min_buffer_config[group_idx-1]
        if total_group_compute == 0:
            min_buffer_config[group_idx] = prev_buffer_req
        else:
            min_buffer_config[group_idx] = max(total_group_iw_memory/(total_group_compute/chiplet_max_pes*1e-9)*max_tp, prev_buffer_req)

        for dram_type in dram_options:
            if min_buffer_config[group_idx] <= dram_type_bandwidth_width_dict[dram_type]["bandwidth"]:
                if dram_type!="HBM3":
                    buffer_config_options[group_idx].append(dram_type)

    return all_fillings(buffer_config_options)

def _moe_expand_expert_groups(group_parsed, fusion_groups, moe_config, virtual_network, chiplet_group):
    """Expand expert fusion groups with EP variants for MoE workloads.

    For each (lat, sp, de, id) tuple in an expert group, generates EP variants:
      EP=1: (lat × E,     sp,       de × E,           id@moe_ep1)  — sequential
      EP=4: (lat × E/4 + comm, sp × 4 + sw, de × E + comm_e, id@moe_ep4)
      EP=8: (lat × E/8 + comm, sp × 8 + sw, de × E + comm_e, id@moe_ep8)

    Two physical switches enable pipelined execution:
      dispatch switch (before first expert group) — routes tokens to experts
      combine  switch (after last expert group)   — gathers expert outputs

    Switch latency/energy/area are split across groups:
      - Dispatch one-way latency + energy → first expert group
      - Combine  one-way latency + energy → last expert group
      - 2× switch area/power               → first expert group (bookkeeping)

    Each expert chiplet's DRAM must store weights for all E/EP experts assigned
    to it.  The DRAM leakage in the original single-expert `sp` is replaced with
    the leakage for the correctly-sized (E/EP)-expert DRAM provisioning.

    Switch communication uses framework-consistent 0.3 pJ/bit for 2.5D.
    """
    E = moe_config['num_experts']
    k = moe_config['num_experts_per_tok']
    H = moe_config['hidden_size']
    batch_tokens = virtual_network.batch_size * virtual_network.sequence_length

    # Check if switch chiplet is in the pool
    has_switch = any(getattr(c, 'arch_target', '') == SWITCH_ARCH_TARGET
                     for c in chiplet_group)
    ep_degrees = MOE_EP_DEGREES if has_switch else [1]

    # Switch parameters (calibrated — see moe/SWITCH_CALIBRATION.md)
    switch_link_bw = 64e9  # 64 GB/s
    switch_static_density = 0.15  # W/mm²

    def switch_area(n_ports):
        total_ports = n_ports + 1
        return total_ports * 0.78 + 0.048 * (total_ports / 5) ** 2 + total_ports * 0.233 + 0.027

    def switch_oneway_latency(n_ports):
        """One-way transfer latency through a single switch."""
        if n_ports <= 1:
            return 0.0
        total_bytes = batch_tokens * H * (word_size // 8)
        return total_bytes / switch_link_bw

    def switch_dispatch_energy(n_ports, bonding='2.5D'):
        """Energy for dispatch: uplink + k multicast downlinks."""
        if n_ports <= 1:
            return 0.0
        bits = batch_tokens * H * word_size
        e_per_bit = 0.3e-12 if bonding == '2.5D' else 0.5e-12
        # Uplink (once) + k downlinks (multicast)
        return bits * (0.5e-12 + k * e_per_bit + 0.002e-12 * k)

    def switch_combine_energy(n_ports, bonding='2.5D'):
        """Energy for combine: k uplinks + reduce + downlink."""
        if n_ports <= 1:
            return 0.0
        bits = batch_tokens * H * word_size
        # k expert uplinks + in-network reduce + 1 downlink
        return bits * (0.8e-12 * (1 + k) + 0.002e-12 * k)

    # Identify expert groups and compute per-group metadata
    expert_gis = set()
    gate_gis = set()   # groups with expert_gate_proj (has parallel up_proj partner)
    for gi, fg in enumerate(fusion_groups):
        if any(l.name in MOE_EXPERT_OPS for l in fg.layers):
            expert_gis.add(gi)
        if any(l.name == 'expert_gate_proj' for l in fg.layers):
            gate_gis.add(gi)

    # Pre-compute single-expert weight memory (GB) and global layer index
    # for each expert fusion group, used for DRAM capacity scaling.
    mem_dict = _get_net_mem_dict()
    net_name = virtual_network.network_name
    bs = virtual_network.batch_size
    seq = virtual_network.sequence_length
    expert_weight_mem = {}   # gi -> single-expert weight_mem in GB
    expert_glb_idx = {}      # gi -> global layer index (for buffer_config lookup)
    glb_idx = 0
    for gi, fg in enumerate(fusion_groups):
        if gi in expert_gis:
            expert_glb_idx[gi] = glb_idx
            w_mem = 0.0
            for li, layer in enumerate(fg.layers):
                flt = get_fused_layer_type(li, len(fg.layers))
                key = (net_name, layer.name, flt, bs, seq)
                row = mem_dict.get(key)
                if row is not None:
                    w_mem += row['weight_mem']
            # expert_up_proj has identical shape to expert_gate_proj (SwiGLU);
            # double weight memory only for gate_proj groups (gate + up on same chiplet).
            # expert_down_proj has no parallel partner — use actual weight memory.
            expert_weight_mem[gi] = w_mem * 2 if gi in gate_gis else w_mem
        glb_idx += len(fg.layers)

    # Determine first/last expert groups for switch cost assignment.
    # Dispatch switch latency/energy → first expert group (gate/up),
    # combine switch latency/energy → last expert group (down),
    # 2× switch area/power → first expert group (bookkeeping only).
    sorted_expert_gis = sorted(expert_gis)
    first_expert_gi = sorted_expert_gis[0]
    last_expert_gi = sorted_expert_gis[-1]

    # Expand expert groups
    for gi in expert_gis:
        original = group_parsed.get(gi, [])
        if not original:
            continue
        single_w_mem = expert_weight_mem.get(gi, 0)
        gi_glb_idx = expert_glb_idx.get(gi, 0)

        expanded = []
        has_up_proj = gi in gate_gis
        for (lat, sp, de, cid) in original:
            # Account for expert_up_proj (identical shape, same chiplet):
            # parallel compute (same latency), energy doubled.
            # Only applies to gate_proj groups; down_proj has no parallel partner.
            if has_up_proj:
                de = de * 2

            # Skip TP > 1 for expert layers (check config_id for tp value)
            # The cid format is: chiplet_id@tp@mapper@bonding@buffer
            parts = cid.split('@')
            tp_val = 1
            for p in parts:
                if p.isdigit() and int(p) in tp_degrees:
                    tp_val = int(p)
                    break
            if tp_val > 1:
                continue  # disable TP for MoE experts

            # Extract DRAM type from buffer_config_str (last @ segment)
            buffer_str = parts[-1]
            dram_types = buffer_str.split('_')
            dram_type = dram_types[gi_glb_idx] if gi_glb_idx < len(dram_types) else 'HBM3'

            # DRAM spec for single expert (already baked into sp via
            # fusion_group_mem_spec_dict in calculate_network_performance_with_memory_check)
            single_spec = get_memory_spec(single_w_mem, dram_type)

            for ep in ep_degrees:
                experts_per_chiplet = E // ep
                scaled_w_mem = experts_per_chiplet * single_w_mem
                scaled_spec = get_memory_spec(scaled_w_mem, dram_type)

                # Delta DRAM leakage per chiplet (2× for double buffering)
                delta_leakage = 2 * (scaled_spec["leakage_power"] - single_spec["leakage_power"])

                # B*k total expert activations; each chiplet handles B*k/ep.
                B = virtual_network.batch_size
                ep_lat = lat * B * k / ep
                ep_de = de * B * k
                ep_sp = sp * ep + delta_leakage * ep

                # Dispatch switch: one-way latency + energy on first expert group
                if gi == first_expert_gi:
                    ep_lat += switch_oneway_latency(ep)
                    ep_de += switch_dispatch_energy(ep)
                # Combine switch: one-way latency + energy on last expert group
                if gi == last_expert_gi:
                    ep_lat += switch_oneway_latency(ep)
                    ep_de += switch_combine_energy(ep)
                # Two physical switches (dispatch + combine); add area/power
                # to first group only to avoid double-counting across groups.
                if gi == first_expert_gi and ep > 1:
                    ep_sp += 2 * switch_area(ep) * switch_static_density

                expanded.append((ep_lat, ep_sp, ep_de, f'{cid}@moe_ep{ep}'))
        group_parsed[gi] = expanded


def cal_perf_phy_net(chiplet_group, chiplets_data, chiplets_vector_data, physical_network,
                     res_csv_file,
                     buffer_config,
                     dag=None,
                     query_points = None,
                     verbose=True,
                     cost_aware=False,
                     moe_config=None):

    batch_size = physical_network.virtual_network.batch_size
    sequence_length = physical_network.virtual_network.sequence_length

    # --- Compute het_batch candidates from DAG structural slack ---
    het_batch_candidates = None
    if dag is not None:
        het_batch_candidates = compute_het_batch_candidates(dag, batch_size)
        if verbose:
            _print_het_batch(dag, het_batch_candidates, batch_size)

    # a list of buffer configs to check
    # current only 1 at a time
    buffer_configs=[buffer_config]

    fusion_groups = physical_network.fusion_groups
    fusiongroups_result_dict = {} # str(buffer_config) -> group_idx -> List(min_e, func) @ freq
    group_parsed_result_dict = {} # str(buffer_config) -> group_idx -> List(a,b,c, id) @ chiplet + mapper
    adaptive_query_points_dict = {} # str(buffer_config) -> list of adaptive query points


    # Side dict for PIM-adjacency deduction: (buf_cfg, group_idx, config_id) → leak
    _input_buf_leak = {}

    for buffer_config_idx, buffer_config in enumerate(buffer_configs):
        buffer_config_str = "_".join(buffer_config)

        fusiongroups_result_dict[buffer_config_str] = {}
        group_parsed_result_dict[buffer_config_str] = {}
        for group_idx,_ in enumerate(fusion_groups):
            group_parsed_result_dict[buffer_config_str][group_idx] = []
        for chiplet_idx, chiplet in enumerate(chiplet_group):
            # Skip switch chiplets (no compute capability)
            if getattr(chiplet, 'arch_target', '') == SWITCH_ARCH_TARGET:
                continue
            performance = calculate_network_performance_with_memory_check(
                physical_network=physical_network,
                csv_file=res_csv_file,
                chiplet_config=chiplet,
                buffer_config=buffer_config,
                net_name=physical_network.virtual_network.network_name,
                batch_size=batch_size,
                sequence_length=sequence_length,
                chiplet_data=chiplets_data[chiplet_idx],
                chiplet_vector_data=chiplets_vector_data[chiplet_idx],
                het_batch_candidates=het_batch_candidates,
            )

            for group_idx in performance:
                for tp in performance[group_idx]['group_results']:
                    for mapper_idx in performance[group_idx]['group_results'][tp]:
                        for bonding in performance[group_idx]['group_results'][tp][mapper_idx]:
                            for is_on_sram in performance[group_idx]['group_results'][tp][mapper_idx][bonding]: # only False for now
                                config_id = f'{chiplet.get_identifier()}@{tp}@{mapper_idx}@{bonding}@{buffer_config_str}'

                                _grp = performance[group_idx]["group_results"][tp][mapper_idx][bonding][is_on_sram]
                                if (_grp["latency"] > 0 and
                                    not math.isnan(_grp["latency"]) and
                                    not math.isinf(_grp["latency"]) and
                                    not math.isnan(_grp["static_power"]) and
                                    not math.isinf(_grp["static_power"]) and
                                    _grp["dynamic_energy"] > 0 and
                                    not math.isnan(_grp["dynamic_energy"]) and
                                    not math.isinf(_grp["dynamic_energy"])):

                                    if cost_aware:

                                        group_parsed_result_dict[buffer_config_str][group_idx].append((
                                            performance[group_idx]["group_results"][tp][mapper_idx][bonding][is_on_sram]["latency"],
                                            performance[group_idx]["group_results"][tp][mapper_idx][bonding][is_on_sram]["static_power"]*performance[group_idx]["group_results"][tp][mapper_idx][bonding][is_on_sram]["cost"],
                                            performance[group_idx]["group_results"][tp][mapper_idx][bonding][is_on_sram]["dynamic_energy"]*performance[group_idx]["group_results"][tp][mapper_idx][bonding][is_on_sram]["cost"],
                                            config_id
                                        ))
                                    else:
                                        group_parsed_result_dict[buffer_config_str][group_idx].append((
                                            performance[group_idx]["group_results"][tp][mapper_idx][bonding][is_on_sram]["latency"],
                                            performance[group_idx]["group_results"][tp][mapper_idx][bonding][is_on_sram]["static_power"],
                                            performance[group_idx]["group_results"][tp][mapper_idx][bonding][is_on_sram]["dynamic_energy"],
                                            config_id
                                        ))

                                    # Record input-buffer leakage for PIM-adjacency deduction
                                    _input_buf_leak[(buffer_config_str, group_idx, config_id)] = _grp.get("leak_input_buf", 0.0)

        # --- MoE expert parallelism expansion ---
        # For MoE workloads, expand expert fusion groups with EP variants.
        # Each expert group's (lat, sp, de) is scaled by (E/EP, EP, E) and
        # switch communication is added for EP > 1.
        if moe_config is not None:
            _moe_expand_expert_groups(
                group_parsed_result_dict[buffer_config_str],
                fusion_groups, moe_config,
                physical_network.virtual_network,
                chiplet_group)

        # Drop empty groups (e.g. softmax layers without vector data in MoE)
        if moe_config is not None:
            gp = group_parsed_result_dict[buffer_config_str]
            valid = {}
            old_to_new = {}
            for gi in sorted(gp.keys()):
                if gp[gi]:
                    old_to_new[gi] = len(valid)
                    valid[len(valid)] = gp[gi]
            group_parsed_result_dict[buffer_config_str] = valid
            # Re-index _input_buf_leak to match remapped group indices
            remapped = {}
            for (bcfg, old_gi, cid), leak in list(_input_buf_leak.items()):
                if bcfg == buffer_config_str and old_gi in old_to_new:
                    remapped[(bcfg, old_to_new[old_gi], cid)] = leak
            for k in list(_input_buf_leak.keys()):
                if k[0] == buffer_config_str:
                    del _input_buf_leak[k]
            _input_buf_leak.update(remapped)

        if query_points is not None:
            # Legacy fixed-grid path
            for group_idx in group_parsed_result_dict[buffer_config_str]:
                res = convex_hull_min_e(group_parsed_result_dict[buffer_config_str][group_idx], query_points)
                fusiongroups_result_dict[buffer_config_str][group_idx] = res
            adaptive_query_points_dict[buffer_config_str] = query_points
        else:
            # Adaptive path: only evaluate at critical breakpoints
            adaptive_qp, group_results = convex_hull_min_e_multi_group_adaptive(
                group_parsed_result_dict[buffer_config_str]
            )
            adaptive_query_points_dict[buffer_config_str] = adaptive_qp
            fusiongroups_result_dict[buffer_config_str] = group_results

    (min_e, min_e_config), (min_edp, min_edp_config) = cal_opt_val_fused(
        fusiongroups_result_dict,
        query_points_dict=adaptive_query_points_dict
    )

    # --- PIM-adjacency deduction ---
    # When PIM (group N) is followed by non-PIM (group N+1), the non-PIM
    # group's input boundary GDDR7 is the PIM die — already counted in PIM's
    # 2× area.  Deduct the double-counted buffer leakage from total energy.
    def _pim_adj_deduct(config):
        funcs = config.get('functions', [])
        lat = config.get('latency')
        if lat is None or len(funcs) < 2:
            return 0.0
        buf_cfg_str = config.get('buffer_idx', [None])[0]
        deduction = 0.0
        for i in range(len(funcs) - 1):
            prev_is_pim = funcs[i].id.startswith('PIM@')
            curr_is_pim = funcs[i + 1].id.startswith('PIM@')
            if prev_is_pim and not curr_is_pim:
                leak = _input_buf_leak.get((buf_cfg_str, i + 1, funcs[i + 1].id), 0.0)
                deduction += leak * lat
        return deduction

    for cfg in (min_e_config, min_edp_config):
        d = _pim_adj_deduct(cfg)
        if d > 0:
            cfg['min_val'] -= d

    return (min_e_config['min_val'], min_e_config),(min_edp_config['min_val'], min_edp_config)

# Example usage
if __name__ == "__main__":

    DB_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'timeloop_experiments', 'unified_database.csv')

    net_name_to_test = "replknet31b"
    network = VirtualNetwork(net_name_to_test, batch_size=1,sequence_length=1)
    network.load_from_dir(os.path.join(NET_DIR, net_name_to_test))
    gene = {'binary_string': '100111001110',
    'buffer_config': ['LPDDR5', 'LPDDR5', 'LPDDR5', 'HBM3', 'GDDR7', 'HBM3', 'HBM3', 'HBM3', 'HBM3', 'HBM3', 'LPDDR5', 'HBM3', 'HBM3']}

    physical_network = create_physical_network_from_gene(network, gene)

    # Create a chiplet configuration
    query_points = base_query_points
    arch_targets = DEFAULT_ARCH_TARGETS
    chiplet_group = generate_chiplet_group(
            n_chiplets=4,
            arch_targets=arch_targets,
            glb_scale_options=glb_scales,
            pe_scale_options=pe_scales,
            seed=43
        )
    for chiplet_config in chiplet_group:
        print(chiplet_config.get_identifier())
    chiplets_data = []
    chiplets_vector_data = []
    for chiplet_idx, chiplet_config in enumerate(chiplet_group):
        chiplets_data.append(get_chiplet_data(
            DB_CSV,
            chiplet_config.arch_target,
            chiplet_config.global_buffer_size_scale,
            chiplet_config.pe_x_scale,
            chiplet_config.pe_y_scale,
            net_name_to_test
        ))
        # PIM handles all ops natively — no separate vector unit
        if chiplet_config.arch_target == 'PIM':
            chiplets_vector_data.append(None)
        elif net_name_to_test in transformer_nets:
            chiplets_vector_data.append(get_chiplet_data(
                DB_CSV,
                arch_vec_targets[0],
                chiplet_config.global_buffer_size_scale,
                chiplet_config.pe_x_scale,
                chiplet_config.pe_y_scale,
                net_name_to_test
            ))
        else:
            chiplets_vector_data.append(None)

    import time
    start = time.time()


    (min_e,min_e_config),(min_edp, min_edp_config) = cal_perf_phy_net(chiplet_group, chiplets_data, chiplets_vector_data, physical_network,DB_CSV,cost_aware=False, buffer_config=gene['buffer_config'])
    print(f"time:{time.time()-start}")
    print(min_e, min_e_config.get('latency', 'N/A'))
    for group_config in min_e_config['functions']:
        print(group_config.id)
    print(min_edp, min_edp_config.get('latency', 'N/A'))
    for group_config in min_edp_config['functions']:
        print(group_config.id)


# ============================================================
# Backward-compatible alias
# ============================================================

cal_perf_phy_net_dag = cal_perf_phy_net  # dag=None gives linear behavior


# ============================================================
# OperatorDAG
# ============================================================

class OperatorDAG:
    """Operator DAG for computing per-op slack.

    Only needs topological order, edges, and parallel levels.
    Compute predecessors/successors traverse through non-compute nodes
    (placeholder, residual_add) transparently.
    """

    def __init__(self, dag_yaml_path: Optional[str] = None,
                 dag_dict: Optional[dict] = None):
        if dag_yaml_path is not None:
            with open(dag_yaml_path) as f:
                d = yaml.safe_load(f)['dag']
        elif dag_dict is not None:
            d = dag_dict
        else:
            raise ValueError("Provide dag_yaml_path or dag_dict")

        self.model_name: str = d['model_name']
        self.nodes: Dict[str, dict] = {nd['id']: nd for nd in d['nodes']}
        self.edges: List[dict] = d.get('edges', [])
        self.successors: Dict[str, List[str]] = defaultdict(list)
        self.predecessors: Dict[str, List[str]] = defaultdict(list)
        for ed in self.edges:
            self.successors[ed['src']].append(ed['dst'])
            self.predecessors[ed['dst']].append(ed['src'])
        self.topo_order: List[str] = d['topological_order']
        self.parallel_levels: List[dict] = d.get('parallel_levels', [])
        self.repeat_blocks: dict = d.get('repeat_blocks', {})

        _NON_COMPUTE = frozenset(
            ('placeholder', 'residual_add', 'elementwise_add',
             'elementwise_mul', 'moe_combine'))
        self._compute_nodes = {
            nid: nd for nid, nd in self.nodes.items()
            if nd.get('op_type') not in _NON_COMPUTE}

    @property
    def compute_nodes(self):
        return self._compute_nodes

    def get_compute_topo_order(self) -> List[str]:
        return [n for n in self.topo_order if n in self._compute_nodes]

    def compute_slack(self, latency_dict: Dict[str, float]) -> Dict[str, float]:
        """Per-node slack.  0 = on critical path."""
        es: Dict[str, float] = {}
        for nid in self.topo_order:
            preds = self.predecessors[nid]
            es[nid] = max((es.get(p, 0.0) + latency_dict.get(p, 0.0)
                           for p in preds), default=0.0)
        cp = max((es.get(n, 0.0) + latency_dict.get(n, 0.0)
                  for n in self.topo_order), default=0.0)
        ls: Dict[str, float] = {}
        for nid in reversed(self.topo_order):
            succs = self.successors[nid]
            lat = latency_dict.get(nid, 0.0)
            valid = [s for s in succs if s in ls]
            ls[nid] = (min(ls[s] for s in valid) - lat) if valid else cp - lat
        return {nid: max(0.0, ls.get(nid, 0.0) - es.get(nid, 0.0))
                for nid in self.topo_order}

    def critical_path_length(self, latency_dict: Dict[str, float]) -> float:
        dist: Dict[str, float] = {}
        for nid in self.topo_order:
            lat = latency_dict.get(nid, 0.0)
            preds = self.predecessors[nid]
            dist[nid] = max((dist.get(p, 0.0) for p in preds), default=0.0) + lat
        return max(dist.values()) if dist else 0.0

    @classmethod
    def create_llama_dag(cls, model_name='llama3.1_8b', batch=1, seq_len=2048,
                         hidden=4096, n_heads=32, kv_heads=8,
                         ffn_inter=14336, head_dim=128):
        B, S, H = batch, seq_len, hidden
        NH, KVH, D, I = n_heads, kv_heads, head_dim, ffn_inter
        hs = [B, S, H]
        nodes = [
            {'id': 'layer0_input', 'op_type': 'placeholder', 'yaml_file': '', 'tensor_dims': {}},
            {'id': 'layer0_q_proj', 'op_type': 'linear', 'yaml_file': 'layer0_q_proj.yaml', 'tensor_dims': {}},
            {'id': 'layer0_k_proj', 'op_type': 'linear', 'yaml_file': 'layer0_k_proj.yaml', 'tensor_dims': {}},
            {'id': 'layer0_v_proj', 'op_type': 'linear', 'yaml_file': 'layer0_k_proj.yaml', 'tensor_dims': {}},
            {'id': 'layer0_attn_qk', 'op_type': 'attn_qk', 'yaml_file': 'layer0_attn_qk.yaml', 'tensor_dims': {}},
            {'id': 'layer0_attn_v', 'op_type': 'attn_v', 'yaml_file': 'layer0_attn_v.yaml', 'tensor_dims': {}},
            {'id': 'layer0_o_proj', 'op_type': 'linear', 'yaml_file': 'layer0_o_proj.yaml', 'tensor_dims': {}},
            {'id': 'layer0_attn_residual_add', 'op_type': 'residual_add', 'yaml_file': '', 'tensor_dims': {}},
            {'id': 'layer0_gate_proj', 'op_type': 'linear', 'yaml_file': 'layer0_gate_proj.yaml', 'tensor_dims': {}},
            {'id': 'layer0_up_proj', 'op_type': 'linear', 'yaml_file': 'layer0_up_proj.yaml', 'tensor_dims': {}},
            {'id': 'layer0_down_proj', 'op_type': 'linear', 'yaml_file': 'layer0_down_proj.yaml', 'tensor_dims': {}},
            {'id': 'layer0_mlp_residual_add', 'op_type': 'residual_add', 'yaml_file': '', 'tensor_dims': {}},
        ]
        edges = [
            {'src': 'layer0_input', 'dst': 'layer0_q_proj', 'tensor_name': 'hs', 'tensor_shape': hs},
            {'src': 'layer0_input', 'dst': 'layer0_k_proj', 'tensor_name': 'hs', 'tensor_shape': hs},
            {'src': 'layer0_input', 'dst': 'layer0_v_proj', 'tensor_name': 'hs', 'tensor_shape': hs},
            {'src': 'layer0_q_proj', 'dst': 'layer0_attn_qk', 'tensor_name': 'q', 'tensor_shape': [B,NH,S,D]},
            {'src': 'layer0_k_proj', 'dst': 'layer0_attn_qk', 'tensor_name': 'k', 'tensor_shape': [B,KVH,S,D]},
            {'src': 'layer0_attn_qk', 'dst': 'layer0_attn_v', 'tensor_name': 'scores', 'tensor_shape': [B,NH,S,S]},
            {'src': 'layer0_v_proj', 'dst': 'layer0_attn_v', 'tensor_name': 'v', 'tensor_shape': [B,KVH,S,D]},
            {'src': 'layer0_attn_v', 'dst': 'layer0_o_proj', 'tensor_name': 'ao', 'tensor_shape': [B,S,NH*D]},
            {'src': 'layer0_o_proj', 'dst': 'layer0_attn_residual_add', 'tensor_name': 'op', 'tensor_shape': hs},
            {'src': 'layer0_input', 'dst': 'layer0_attn_residual_add', 'tensor_name': 'res', 'tensor_shape': hs},
            {'src': 'layer0_attn_residual_add', 'dst': 'layer0_gate_proj', 'tensor_name': 'pa', 'tensor_shape': hs},
            {'src': 'layer0_attn_residual_add', 'dst': 'layer0_up_proj', 'tensor_name': 'pa', 'tensor_shape': hs},
            {'src': 'layer0_gate_proj', 'dst': 'layer0_down_proj', 'tensor_name': 'go', 'tensor_shape': [B,S,I]},
            {'src': 'layer0_up_proj', 'dst': 'layer0_down_proj', 'tensor_name': 'uo', 'tensor_shape': [B,S,I]},
            {'src': 'layer0_down_proj', 'dst': 'layer0_mlp_residual_add', 'tensor_name': 'do', 'tensor_shape': hs},
            {'src': 'layer0_attn_residual_add', 'dst': 'layer0_mlp_residual_add', 'tensor_name': 'res', 'tensor_shape': hs},
        ]
        topo = ['layer0_input', 'layer0_q_proj', 'layer0_k_proj', 'layer0_v_proj',
                'layer0_attn_qk', 'layer0_attn_v', 'layer0_o_proj',
                'layer0_attn_residual_add', 'layer0_gate_proj', 'layer0_up_proj',
                'layer0_down_proj', 'layer0_mlp_residual_add']
        levels = [
            {'level': 0, 'nodes': ['layer0_input']},
            {'level': 1, 'nodes': ['layer0_q_proj', 'layer0_k_proj', 'layer0_v_proj']},
            {'level': 2, 'nodes': ['layer0_attn_qk']},
            {'level': 3, 'nodes': ['layer0_attn_v']},
            {'level': 4, 'nodes': ['layer0_o_proj']},
            {'level': 5, 'nodes': ['layer0_attn_residual_add']},
            {'level': 6, 'nodes': ['layer0_gate_proj', 'layer0_up_proj']},
            {'level': 7, 'nodes': ['layer0_down_proj']},
            {'level': 8, 'nodes': ['layer0_mlp_residual_add']},
        ]
        return cls(dag_dict={
            'model_name': model_name, 'num_nodes': len(nodes),
            'num_edges': len(edges), 'topological_order': topo,
            'parallel_levels': levels, 'nodes': nodes, 'edges': edges,
            'repeat_blocks': {},
        })


# ============================================================
# Structural slack and heterogeneous batch
# ============================================================

def compute_structural_slack(dag: OperatorDAG) -> Dict[str, int]:
    """Compute slack in HOP COUNT, independent of operator latency."""
    unit_lat = {}
    for nid in dag.topo_order:
        unit_lat[nid] = 1 if nid in dag._compute_nodes else 0
    raw_slack = dag.compute_slack(unit_lat)
    return {nid: int(round(raw_slack.get(nid, 0)))
            for nid in dag.get_compute_topo_order()}


def compute_het_batch_candidates(
    dag: OperatorDAG,
    base_batch: int,
    available_batches: Optional[List[int]] = None,
    max_multiplier: int = 16,
) -> Dict[str, List[int]]:
    """For each op, return candidate batch sizes from STRUCTURAL slack."""
    if available_batches is None:
        available_batches = sorted(batch_configs(transformer=True))

    slack_hops = compute_structural_slack(dag)
    result: Dict[str, List[int]] = {}
    _ba_suffixes = tuple('_' + op for op in LLAMA_BATCH_AGNOSTIC_OPS)

    for nid in dag.get_compute_topo_order():
        if any(nid.endswith(s) for s in _ba_suffixes):
            result[nid] = [base_batch]
            continue
        hops = slack_hops.get(nid, 0)
        if hops <= 0:
            result[nid] = [base_batch]
            continue
        mult = min(1 + hops, max_multiplier)
        target = base_batch * mult
        candidates = [b for b in available_batches
                      if base_batch <= b <= target]
        if not candidates:
            candidates = [base_batch]
        result[nid] = candidates

    return result


def _print_het_batch(dag, het_candidates, base_batch):
    slack_hops = compute_structural_slack(dag)
    cp_hops = sum(1 for n in dag.get_compute_topo_order()
                  if slack_hops.get(n, 0) == 0)
    print(f"\n  DAG het-batch candidates (structural slack, {cp_hops} ops on CP, base={base_batch})")
    print(f"  {'Op':<30} {'Slack':>5} {'Candidates':>20}")
    print(f"  {'-' * 30} {'-' * 5} {'-' * 20}")
    for nid in dag.get_compute_topo_order():
        s = slack_hops.get(nid, 0)
        cands = het_candidates.get(nid, [base_batch])
        cp_mark = " *CP*" if s == 0 else ""
        print(f"  {nid:<30} {s:>5} {str(cands):>20}{cp_mark}")


# ============================================================
# GA integration
# ============================================================

def evaluate_gene_dag(gene, virtual_network, chiplet_group, chiplets_data,
                      chiplets_vector_data, results_file, objective, verbose,
                      cost_aware, dag=None):
    """Drop-in for genetic_algo_opt_phy_net's evaluate_gene."""
    try:
        physical_network = create_physical_network_from_gene(virtual_network, gene)
        (min_e, min_e_config), (min_edp, min_edp_config) = cal_perf_phy_net(
            chiplet_group, chiplets_data, chiplets_vector_data,
            physical_network, res_csv_file=results_file,
            buffer_config=gene["buffer_config"],
            dag=dag, cost_aware=cost_aware, verbose=verbose)
        if objective == "energy":
            return gene, min_e, min_e_config, None
        else:
            return gene, min_edp, min_edp_config, None
    except Exception as e:
        return gene, float('inf'), None, str(e)


# ============================================================
# Critical-Path-Based DAG Gene Encoding
# ============================================================

@dataclass
class CriticalPathPosition:
    """One position on the critical path."""
    ops: List[str]
    parallel: bool = False


@dataclass
class OffCPOp:
    """An operator that is NOT on the critical path."""
    op_name: str
    start_pos: int
    end_pos: int
    slack_stages: int


@dataclass
class CriticalPathSpec:
    """Describes the critical-path structure of a network for the DAG GA."""
    positions: List[CriticalPathPosition]
    off_cp_ops: List[OffCPOp] = field(default_factory=list)
    forced_boundaries: List[int] = field(default_factory=lambda: [0])

    @property
    def gene_length(self) -> int:
        return len(self.positions)

    @property
    def buffer_config_length(self) -> int:
        return self.gene_length + 1

    def all_cp_ops(self) -> List[str]:
        ops = []
        for pos in self.positions:
            ops.extend(pos.ops)
        return ops

    def all_ops(self) -> List[str]:
        ops = self.all_cp_ops()
        for off in self.off_cp_ops:
            ops.append(off.op_name)
        return ops


# ---- Spec builders ----

_SOFTMAX_DROP_WARNED = set()


def _prune_spec_to_network(positions, off_cp, network):
    """Prune CP positions + off-CP ops to the ops actually present in the loaded network.

    LOUD GUARD: if a *softmax* op is on the critical path but absent from the loaded
    network, it would silently contribute ZERO latency/energy/cost — this is exactly
    the bug that zeroed ViT softmax (the DB carried legacy sub-op names, so the fused
    `layer0_softmax` was filtered out at load). Warn to stderr once per (net, op)
    instead of dropping it silently. Fix is a DB issue (see fuse_vit_softmax.py), not
    a code issue — this guard just makes a recurrence impossible to miss.
    """
    layer_names = {l.name for l in network.layers}
    for pos in positions:
        for op in pos.ops:
            if op not in layer_names and utility_functions.is_softmax_layers(op):
                k = (getattr(network, 'network_name', '?'), op)
                if k not in _SOFTMAX_DROP_WARNED:
                    _SOFTMAX_DROP_WARNED.add(k)
                    _sys.stderr.write(
                        f"[WARN] softmax op '{op}' is on the critical path but ABSENT "
                        f"from loaded network '{k[0]}' -> contributes ZERO cost. "
                        f"Likely a DB naming mismatch (fused 'layer0_softmax' missing). "
                        f"See fuse_vit_softmax.py / Bug.md ViT-softmax note.\n")
        pos.ops = [op for op in pos.ops if op in layer_names]
    off_cp = [o for o in off_cp if o.op_name in layer_names]
    return positions, off_cp


def create_llama_cp_spec(network=None, prefix: str = 'layer0') -> CriticalPathSpec:
    p = prefix
    positions = [
        CriticalPathPosition([f'{p}_q_proj', f'{p}_k_proj'], parallel=True),
        # DB now stores a single fused softmax row (`layer0_softmax`); the legacy
        # 4 sub-op names (_softmax_max/_sub_exp/_sum/_div) no longer exist and would
        # be filtered out, silently dropping softmax from the critical path.
        CriticalPathPosition([f'{p}_attn_qk', f'{p}_softmax'], parallel=False),
        CriticalPathPosition([f'{p}_attn_v'], parallel=False),
        CriticalPathPosition([f'{p}_o_proj'], parallel=False),
        CriticalPathPosition([f'{p}_gate_proj', f'{p}_up_proj'], parallel=True),
        CriticalPathPosition([f'{p}_down_proj'], parallel=False),
    ]
    off_cp = [OffCPOp(f'{p}_v_proj', start_pos=0, end_pos=2, slack_stages=2)]
    forced = [0, 1, 3]
    if network is not None:
        positions, off_cp = _prune_spec_to_network(positions, off_cp, network)
    return CriticalPathSpec(positions=positions, off_cp_ops=off_cp,
                            forced_boundaries=forced)


def create_qwen_moe_cp_spec(network=None, prefix: str = 'layer0') -> CriticalPathSpec:
    p = prefix
    positions = [
        CriticalPathPosition([f'{p}_q_proj', f'{p}_k_proj'], parallel=True),
        CriticalPathPosition([f'{p}_attn_qk', f'{p}_softmax'], parallel=False),
        CriticalPathPosition([f'{p}_attn_v'], parallel=False),
        CriticalPathPosition([f'{p}_o_proj'], parallel=False),
        CriticalPathPosition(['router'], parallel=False),
        CriticalPathPosition(['expert_gate_proj'], parallel=False),
        CriticalPathPosition(['expert_down_proj'], parallel=False),
    ]
    off_cp = [OffCPOp(f'{p}_v_proj', start_pos=0, end_pos=2, slack_stages=2)]
    forced = list(range(len(positions)))
    if network is not None:
        positions, off_cp = _prune_spec_to_network(positions, off_cp, network)
    return CriticalPathSpec(positions=positions, off_cp_ops=off_cp,
                            forced_boundaries=forced)


def create_vit_cp_spec(network=None, prefix: str = 'layer0') -> CriticalPathSpec:
    """ViT encoder-block critical path.

    Same structure as the llama spec (parallel q/k proj, off-CP v_proj, attention,
    o_proj, MLP) with two ViT-specific differences:
      - GELU MLP, not SwiGLU: only fc1 (gate_proj) + fc2 (down_proj); no up_proj.
      - softmax is referenced by its fused name `layer0_softmax` (same as the llama/qwen
        specs). This only resolves once the ViT DB carries a fused `layer0_softmax` row
        (added by fuse_vit_softmax.py); before that it was silently dropped — see the
        loud guard in `_prune_spec_to_network`.
    k/v/o_proj share q_proj's dimensions (standard MHA).
    """
    p = prefix
    positions = [
        CriticalPathPosition([f'{p}_q_proj', f'{p}_k_proj'], parallel=True),
        CriticalPathPosition([f'{p}_attn_qk', f'{p}_softmax'], parallel=False),
        CriticalPathPosition([f'{p}_attn_v'], parallel=False),
        CriticalPathPosition([f'{p}_o_proj'], parallel=False),
        CriticalPathPosition([f'{p}_gate_proj'], parallel=False),  # fc1 (no up_proj)
        CriticalPathPosition([f'{p}_down_proj'], parallel=False),  # fc2
    ]
    off_cp = [OffCPOp(f'{p}_v_proj', start_pos=0, end_pos=2, slack_stages=2)]
    forced = [0, 1, 3]
    if network is not None:
        positions, off_cp = _prune_spec_to_network(positions, off_cp, network)
    return CriticalPathSpec(positions=positions, off_cp_ops=off_cp,
                            forced_boundaries=forced)


def create_cnn_cp_spec(network) -> CriticalPathSpec:
    positions = [CriticalPathPosition([layer.name], parallel=False)
                 for layer in network.layers]
    forced = net_topology_dict.get(network.network_name, [0])
    return CriticalPathSpec(positions=positions, off_cp_ops=[],
                            forced_boundaries=forced)


def create_cp_spec(network) -> CriticalPathSpec:
    """Auto-detect network type and return the appropriate CriticalPathSpec."""
    name = network.network_name
    if 'qwen' in name.lower():
        return create_qwen_moe_cp_spec(network)
    if 'llama' in name.lower():
        return create_llama_cp_spec(network)
    if 'vit' in name.lower():
        return create_vit_cp_spec(network)
    if name in transformer_nets:
        return create_cnn_cp_spec(network)
    return create_cnn_cp_spec(network)


# ---- Gene → fusion groups ----

def dag_gene_to_fusion_groups(cp_spec: CriticalPathSpec, gene: dict,
                              virtual_network) -> Tuple[List, List[int], dict]:
    """Convert a DAG gene into fusion groups + off-CP group info."""
    binary = gene['binary_string']
    position_groups = []
    current_group = []
    for i, bit in enumerate(binary):
        if i < len(cp_spec.positions):
            if bit == '1' and i > 0:
                if current_group:
                    position_groups.append(current_group)
                current_group = [i]
            elif bit == '1' and i == 0:
                current_group = [i]
            else:
                current_group.append(i)
    if current_group:
        position_groups.append(current_group)

    layer_dict = {l.name: l for l in virtual_network.layers}
    cp_fusion_groups = []
    parallel_group_indices = []

    for grp_idx, pos_indices in enumerate(position_groups):
        is_single_parallel = (len(pos_indices) == 1 and
                              cp_spec.positions[pos_indices[0]].parallel)
        if is_single_parallel:
            parallel_group_indices.append(grp_idx)
        fg = FusionGroup(is_attn=False)
        for pos_idx in pos_indices:
            pos = cp_spec.positions[pos_idx]
            for op_name in pos.ops:
                if op_name in layer_dict:
                    layer = layer_dict[op_name]
                    fg.add_layer(layer)
                    if not fg.is_attn and (is_attention_layers(op_name) or
                                           utility_functions.is_projection_layers(op_name)):
                        fg.is_attn = True
        cp_fusion_groups.append(fg)

    off_cp_info = {}
    for off in cp_spec.off_cp_ops:
        if off.op_name in layer_dict:
            off_cp_info[off.op_name] = {
                'layer': layer_dict[off.op_name],
                'start_pos': off.start_pos,
                'end_pos': off.end_pos,
                'slack_stages': off.slack_stages,
            }

    return cp_fusion_groups, parallel_group_indices, off_cp_info


def _merge_parallel_configs(member_configs: List[List[Tuple]]) -> List[Tuple]:
    """Cross-product configs of parallel ops into virtual group configs."""
    import itertools
    if len(member_configs) == 1:
        return member_configs[0]
    combos = list(itertools.product(*member_configs))
    merged = []
    for combo in combos:
        lat = max(c[0] for c in combo)
        sp = sum(c[1] for c in combo)
        de = sum(c[2] for c in combo)
        cid = "VG(" + "||".join(c[3] for c in combo) + ")"
        merged.append((lat, sp, de, cid))
    return merged


def _build_off_cp_functions(off_cp_info: dict, chiplet_group, chiplets_data,
                            chiplets_vector_data, gene: dict, cp_spec: CriticalPathSpec,
                            virtual_network, res_csv_file: str,
                            cost_aware: bool,
                            v_het_batch: bool = True) -> List[Tuple]:
    """Build convex hull tuples for off-CP ops (V proj)."""
    results = []
    batch_size = virtual_network.batch_size
    seq_len = virtual_network.sequence_length
    net_name = virtual_network.network_name

    for op_name, info in off_cp_info.items():
        layer = info['layer']
        slack = info['slack_stages']
        start_pos = info['start_pos']
        end_pos = info['end_pos']

        dram_i = gene['buffer_config'][start_pos]
        dram_o = gene['buffer_config'][end_pos]

        fg = FusionGroup(is_attn=layer.is_attn)
        fg.add_layer(layer)
        pn = PhysicalNetwork(virtual_network)
        pn.add_fusion_group(fg)

        het_batch = batch_size * slack if v_het_batch else batch_size
        available_batches = sorted(batch_configs(transformer=True))
        batch_candidates = [het_batch, batch_size] if v_het_batch else [batch_size]

        for chiplet_idx, chiplet in enumerate(chiplet_group):
            if getattr(chiplet, 'arch_target', '') == 'switch_8port':
                continue
            cd = chiplets_data[chiplet_idx]
            cvd = chiplets_vector_data[chiplet_idx]
            if cd is None or (hasattr(cd, 'empty') and cd.empty):
                continue

            row_dict = _build_row_dict(cd)

            for tp in tp_degrees:
                for mapper_idx in range(num_mapping_per_arch):
                    for bonding in bonding_techniques:
                        best_dyn = None
                        best_lat = None

                        for try_batch in batch_candidates:
                            lookup_b = 1 if is_attention_layers(op_name) else try_batch
                            if lookup_b not in available_batches and lookup_b > 1:
                                lookup_b = max(b for b in available_batches if b <= lookup_b) \
                                           if any(b <= lookup_b for b in available_batches) \
                                           else available_batches[0]

                            row_key = (op_name, lookup_b, seq_len,
                                       mapper_idx, tp, "single", dram_i, dram_o)
                            row = row_dict.get(row_key)
                            if row is None:
                                continue

                            lat = row['latency']
                            dyn = row['dynamic_energy']

                            is_pim_chiplet = getattr(chiplet, 'arch_target', '') == 'PIM'
                            if is_pim_chiplet:
                                # PIM: the lumped DB energy already covers near-bank
                                # compute + resident-weight reads + the PIM die's
                                # internal GDDR7 activation R/W.  Resident weights never
                                # cross a die boundary, so charge ONLY the activation
                                # operand transport (input + output) over the
                                # inter-chiplet link -- NOT weights, and NOT a fresh DRAM
                                # access (either would double-count the lumped value).
                                # Activation volume is read from network_analysis
                                # (in_mem/out_mem), explicitly excluding weight_mem.
                                # Conservative: always charge both sides.  PIM here maps
                                # to a decode GEMV whose CP neighbours are non-PIM, so the
                                # transfer is real on both sides (this is exact); were a
                                # neighbour ever PIM, the at-most-one-side over-charge is
                                # small and in the honest (PIM-looks-worse) direction.
                                dyn *= tp
                                _m = _get_net_mem_dict().get(
                                    (net_name, op_name, "single",
                                     int(lookup_b), int(seq_len)))
                                if _m is not None:
                                    _in_bits = _m['in_mem'] * 1e9 * 8
                                    _out_bits = _m['out_mem'] * 1e9 * 8
                                    dyn += calculate_inter_chiplet_communication(_in_bits, bonding, 1)
                                    dyn += calculate_inter_chiplet_communication(_out_bits, bonding, 1)
                            else:
                                reads = row['i_access']
                                writes = row['o_access']
                                dyn *= tp

                                r_e = calculate_inter_chiplet_communication(_accesses_to_bits(reads), bonding, 1)
                                w_e = calculate_inter_chiplet_communication(_accesses_to_bits(writes), bonding, 1)
                                dyn += r_e + w_e

                            if try_batch == het_batch and try_batch > batch_size:
                                multiplier = try_batch / batch_size
                                dyn_per_item = dyn / multiplier
                                lat_actual = lat
                            else:
                                dyn_per_item = dyn
                                lat_actual = lat * slack

                            if best_dyn is None or dyn_per_item < best_dyn:
                                best_dyn = dyn_per_item
                                best_lat = lat_actual

                        if best_dyn is None:
                            continue

                        sp = 0.0
                        for try_b in [batch_size, 1]:
                            rk = (op_name, try_b, seq_len, mapper_idx, tp,
                                  "single", dram_i, dram_o)
                            r = row_dict.get(rk)
                            if r is not None:
                                sp = r['static_power'] * tp
                                break

                        is_pim_chiplet = getattr(chiplet, 'arch_target', '') == 'PIM'
                        if is_pim_chiplet and sp > 0:
                            # Mirror the main path (:641-644): the raw DB static_power is
                            # the compute set only; add GDDR7 die leakage + PHY/controller
                            # leakage that the on-CP PIM path adds.
                            sp += tp * mem_specs['GDDR7']['leakage_power']
                            _phy = _cached_area_overheads(0, bonding, 'GDDR7')
                            sp += (_phy["phy_area"] + _phy["ctrl_area"]) * 0.0001

                        core_mm2 = _offcp_core_mm2(chiplet)
                        full_area = calculate_actual_area(
                            core_mm2, dram_type=dram_i, bonding=bonding,
                            all_phy=(not is_pim_chiplet))
                        cost = _cached_die_cost(full_area, bonding) * tp

                        if cost_aware:
                            sp_val = sp * cost
                            dyn_val = best_dyn * cost
                        else:
                            sp_val = sp
                            dyn_val = best_dyn

                        buf_str = f"{dram_i}_{dram_o}"
                        cid = f'{chiplet.get_identifier()}@{tp}@{mapper_idx}@{bonding}@{buf_str}@offcp'

                        v_x1 = best_lat / slack
                        v_a = slack * sp_val
                        v_b = dyn_val

                        if (v_x1 > 0 and not math.isnan(v_x1) and not math.isinf(v_x1)
                                and v_b > 0 and not math.isnan(v_b)):
                            results.append((v_x1, v_a, v_b, cid))

    return results


# ---- Main DAG-CP evaluation ----

def cal_perf_phy_net_dag_cp(
    cp_spec: CriticalPathSpec,
    virtual_network,
    chiplet_group,
    chiplets_data,
    chiplets_vector_data,
    gene: dict,
    res_csv_file: str,
    cost_aware: bool = False,
    verbose: bool = False,
    v_het_batch: bool = True,
):
    """DAG-aware evaluation using critical-path gene encoding."""
    batch_size = virtual_network.batch_size
    seq_len = virtual_network.sequence_length
    net_name = virtual_network.network_name
    buffer_config = gene['buffer_config']

    # Step 1: Gene -> fusion groups
    cp_groups, parallel_indices, off_cp_info = dag_gene_to_fusion_groups(
        cp_spec, gene, virtual_network)

    if verbose:
        print(f"  DAG-CP: {len(cp_groups)} CP groups, "
              f"{len(parallel_indices)} parallel, "
              f"{len(off_cp_info)} off-CP ops")
        for i, fg in enumerate(cp_groups):
            marker = " [parallel]" if i in parallel_indices else ""
            ops = [l.name for l in fg.layers]
            print(f"    Group {i}: {ops}{marker}")

    # Step 2: Build per-group buffer config mapping
    group_position_ranges = []
    current_positions = []
    binary = gene['binary_string']
    for i, bit in enumerate(binary):
        if i >= len(cp_spec.positions):
            break
        if bit == '1' and i > 0:
            if current_positions:
                group_position_ranges.append(current_positions)
            current_positions = [i]
        elif bit == '1' and i == 0:
            current_positions = [i]
        else:
            current_positions.append(i)
    if current_positions:
        group_position_ranges.append(current_positions)

    buffer_config_str = "_".join(buffer_config)
    group_parsed = {}

    for grp_idx, fg in enumerate(cp_groups):
        pos_range = group_position_ranges[grp_idx]
        grp_buf_start = pos_range[0]
        grp_buf_end = pos_range[-1] + 1

        group_dram = buffer_config[grp_buf_start]
        n_layers_in_group = len(fg.layers)
        local_buf = [group_dram] * n_layers_in_group + [buffer_config[grp_buf_end]]

        is_parallel = grp_idx in parallel_indices

        if is_parallel and len(fg.layers) > 1:
            member_configs = []
            for layer in fg.layers:
                single_fg = FusionGroup(is_attn=layer.is_attn)
                single_fg.add_layer(layer)
                single_pn = PhysicalNetwork(virtual_network)
                single_pn.add_fusion_group(single_fg)
                single_buf = [group_dram, buffer_config[grp_buf_end]]

                op_configs = []
                for ci, chiplet in enumerate(chiplet_group):
                    if getattr(chiplet, 'arch_target', '') == 'switch_8port':
                        continue
                    perf = calculate_network_performance_with_memory_check(
                        physical_network=single_pn,
                        csv_file=res_csv_file,
                        chiplet_config=chiplet,
                        buffer_config=single_buf,
                        net_name=net_name,
                        batch_size=batch_size,
                        sequence_length=seq_len,
                        chiplet_data=chiplets_data[ci],
                        chiplet_vector_data=chiplets_vector_data[ci],
                    )
                    for tp in perf.get(0, {}).get('group_results', {}):
                        for mi in perf[0]['group_results'][tp]:
                            for bd in perf[0]['group_results'][tp][mi]:
                                for sram in perf[0]['group_results'][tp][mi][bd]:
                                    p = perf[0]['group_results'][tp][mi][bd][sram]
                                    if (p['latency'] > 0 and not math.isinf(p['latency'])
                                            and not math.isnan(p['latency'])
                                            and p['dynamic_energy'] > 0
                                            and not math.isnan(p['dynamic_energy'])
                                            and not math.isinf(p['dynamic_energy'])):
                                        cid = f'{chiplet.get_identifier()}@{tp}@{mi}@{bd}'
                                        if cost_aware:
                                            op_configs.append((
                                                p['latency'],
                                                p['static_power'] * p['cost'],
                                                p['dynamic_energy'] * p['cost'],
                                                cid))
                                        else:
                                            op_configs.append((
                                                p['latency'], p['static_power'],
                                                p['dynamic_energy'], cid))
                member_configs.append(op_configs)

            if all(mc for mc in member_configs):
                group_parsed[grp_idx] = _merge_parallel_configs(member_configs)
            else:
                group_parsed[grp_idx] = []
        else:
            pn = PhysicalNetwork(virtual_network)
            pn.add_fusion_group(fg)

            group_parsed[grp_idx] = []
            for ci, chiplet in enumerate(chiplet_group):
                if getattr(chiplet, 'arch_target', '') == 'switch_8port':
                    continue
                perf = calculate_network_performance_with_memory_check(
                    physical_network=pn,
                    csv_file=res_csv_file,
                    chiplet_config=chiplet,
                    buffer_config=local_buf,
                    net_name=net_name,
                    batch_size=batch_size,
                    sequence_length=seq_len,
                    chiplet_data=chiplets_data[ci],
                    chiplet_vector_data=chiplets_vector_data[ci],
                )
                for tp in perf.get(0, {}).get('group_results', {}):
                    for mi in perf[0]['group_results'][tp]:
                        for bd in perf[0]['group_results'][tp][mi]:
                            for sram in perf[0]['group_results'][tp][mi][bd]:
                                p = perf[0]['group_results'][tp][mi][bd][sram]
                                if (p['latency'] > 0 and not math.isinf(p['latency'])
                                        and not math.isnan(p['latency'])
                                        and p['dynamic_energy'] > 0
                                        and not math.isnan(p['dynamic_energy'])
                                        and not math.isinf(p['dynamic_energy'])):
                                    cid = f'{chiplet.get_identifier()}@{tp}@{mi}@{bd}@{buffer_config_str}'
                                    if cost_aware:
                                        group_parsed[grp_idx].append((
                                            p['latency'],
                                            p['static_power'] * p['cost'],
                                            p['dynamic_energy'] * p['cost'],
                                            cid))
                                    else:
                                        group_parsed[grp_idx].append((
                                            p['latency'], p['static_power'],
                                            p['dynamic_energy'], cid))

    # MoE expert parallelism expansion
    moe_config = getattr(virtual_network, 'moe_config', None)
    if moe_config is not None:
        _moe_expand_expert_groups(group_parsed, cp_groups, moe_config,
                                  virtual_network, chiplet_group)

    # Check for infeasible groups: if any CP group has no valid configs,
    # the gene is infeasible.  (Previously empty groups were silently
    # dropped, which made partial solutions look artificially cheap.)
    for gi in sorted(group_parsed.keys()):
        if not group_parsed[gi]:
            return (float('inf'), None), (float('inf'), None)

    # Re-index (MoE expansion may have changed keys)
    valid_parsed = {}
    for gi in sorted(group_parsed.keys()):
        valid_parsed[len(valid_parsed)] = group_parsed[gi]
    group_parsed = valid_parsed

    # Step 3: Convex hull
    cp_hull_results = {}
    adaptive_qp, cp_group_results = convex_hull_min_e_multi_group_adaptive(group_parsed)
    cp_hull_results[buffer_config_str] = cp_group_results
    qp_dict = {buffer_config_str: adaptive_qp}

    # Step 4: Off-CP ops
    off_cp_tuples = _build_off_cp_functions(
        off_cp_info, chiplet_group, chiplets_data, chiplets_vector_data,
        gene, cp_spec, virtual_network, res_csv_file, cost_aware,
        v_het_batch=v_het_batch)

    # Step 5: Combine
    if off_cp_tuples:
        v_hull_results = convex_hull_min_e(off_cp_tuples, adaptive_qp)
        n_cp_groups = len(cp_hull_results[buffer_config_str])
        cp_hull_results[buffer_config_str][n_cp_groups] = v_hull_results

    # Step 6: Find optimal
    (min_e, min_e_config), (min_edp, min_edp_config) = cal_opt_val_fused(
        cp_hull_results, query_points_dict=qp_dict)

    return (min_e, min_e_config), (min_edp, min_edp_config)