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
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Set, Union
import os
import functools
import time
from convex_hull import *

from utility_functions import cal_opt_val_fused, all_fillings, is_attention_layers
import utility_functions
import gqa_kv
import network_analysis_sizes
import softmax_vector

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
    postprocess_bw.apply_bw_throttling_analytical (database build pipeline, in
    the Zenodo code archive (doi:10.5281/zenodo.21524898, fengshui-code.tar.gz)).
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
    GQA: the KV-cache rows of the attention ops are corrected here, once (gqa_kv); the fused
    attention rows around softmax are then rebuilt from them (_apply_attention_rows)."""
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
    _apply_attention_rows(d, chiplet_df)
    _ROW_DICT_CACHE[df_id] = d
    return d


# --- Softmax on the vector unit ------------------------------------------------
# A non-PIM softmax runs on its chiplet's 1-D vector unit and is priced analytically
# (softmax_vector.py), not from the database's simple_vector rows (see that module). Whether the
# scores S it reads and the probabilities P it writes stay in the GLB comes from the Timeloop
# mappings behind the neighbouring attn_qk and attn_v rows: attention_residency.csv
# (tools/build_attention_residency.py), per (op, net, arch, glb, pe_x, pe_y, tp). The table is
# required: a missing file or key raises.
# [A] S -> E -> P happen in place, and attn_qk's tile of S and attn_v's tile of P line up at the
#     handoff; the two residency checks are applied independently.
ATTENTION_RESIDENCY_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "attention_residency.csv")
_ATTN_RESIDENCY = None


def _attention_residency():
    """(op, net, arch, glb, pe_x, pe_y, tp) -> True when the Timeloop mapping behind the database
    row keeps every S row (attn_qk) or P row (attn_v) it needs in the GLB."""
    global _ATTN_RESIDENCY
    if _ATTN_RESIDENCY is None:
        if not os.path.isfile(ATTENTION_RESIDENCY_CSV):
            raise FileNotFoundError(f"{ATTENTION_RESIDENCY_CSV} is required to price softmax; "
                                    f"build it with tools/build_attention_residency.py")
        df = pd.read_csv(ATTENTION_RESIDENCY_CSV)
        _ATTN_RESIDENCY = {
            (r.op, r.net, r.arch_target, int(r.glb_scale), int(r.pe_x_scale), int(r.pe_y_scale),
             int(r.tp_degree)): bool(r.fits) for r in df.itertuples()}
    return _ATTN_RESIDENCY


@functools.lru_cache(maxsize=None)
def _softmax_row_cached(sm_dims, net_name, arch, glb_scale, pe_x_scale, pe_y_scale, tp,
                        after_qk, before_v, dram_i, dram_o):
    """The analytic softmax row of one chiplet (softmax_vector.softmax_row).

    S stays in the GLB only when attn_qk runs just before softmax in its stage (after_qk) and
    attn_qk's mapping keeps the S rows in the GLB; otherwise it is read from DRAM. P stays only
    when attn_v runs just after softmax in its stage (before_v) and attn_v's mapping keeps the
    P rows in the GLB; otherwise it is written to DRAM. On the DAG-CP path attn_qk opens
    softmax's stage and a fused attn_v follows it, so this is softmax's stage position; the
    linear path's order is not topological (load_from_dir sorts by layer number only, so the
    layer0_* ops keep os.listdir's order), and there the neighbours are other ops."""
    H, Q, K, B = sm_dims
    if not all(float(v).is_integer() for v in (glb_scale, pe_x_scale, pe_y_scale)):
        raise ValueError(f"non-integer chiplet scale {(glb_scale, pe_x_scale, pe_y_scale)}")
    chip = (net_name, arch, int(glb_scale), int(pe_x_scale), int(pe_y_scale), tp)
    res = _attention_residency()
    scores_in_dram = not (after_qk and res[('attn_qk',) + chip])
    probs_to_dram = not (before_v and res[('attn_v',) + chip])
    return softmax_vector.softmax_row(
        heads=H, q_len=Q, k_len=K, batch_in_problem=B, tp=tp, pe_x_scale=pe_x_scale,
        glb_scale=glb_scale, scores_in_dram=scores_in_dram, probs_to_dram=probs_to_dram,
        dram_i=dram_i, dram_o=dram_o, dram_table=dram_type_bandwidth_width_dict)


# --- Attention rows around softmax -----------------------------------------------
# The fused attention rows of a compute chiplet are made consistent with the softmax above, once
# per row dict and after the GQA correction, so every row read below has its K / V sized with
# num_key_value_heads:
#   attn_qk 'start'/'middle': the fused rows write S into the GLB and nothing to DRAM. When the
#       mapping cannot keep the S rows in the GLB (attention_residency.csv), S spills: the sibling
#       row that writes S is used instead ('single' for 'start', 'end' for 'middle').
#   attn_v 'middle'/'end': the fusion split (parse_stats.py:404-423) zeroes the Inputs2 fields, and
#       for attn_v Inputs2 is V (i_access) while the fused operand is P (Inputs1, w_access), so these
#       rows read P from DRAM and not V. They are rebuilt from the 'single' row at the same key: the
#       V read is kept; the P read and its DRAM energy are removed when P stays in the GLB and kept
#       when it spills; the output is kept only at 'end'; the latency is re-derived with gqa_kv's
#       roofline on attn_v's compute-only cycles (attn_compute_cycles.csv). P's GLB fill stays in
#       the 'single' row's on-chip energy, which is why softmax books neither P's GLB write nor its
#       port time when P stays.
# attn_qk 'middle'/'end' rows are split the same way (no K read) and are left so: on the DAG path
# attn_qk always opens its stage.
# [A] Of the 3,072 mappings per op of llama3.1 8B/70B and qwen3 30B/235B at s1024/kv1024, 24 attn_qk
#     and 36 attn_v ones (all gemmini) fit yet touch S or P up to 1.25x / 1.125x; the rebuilt rows
#     move those extra accesses from DRAM into the GLB without charging GLB energy for them.


def _apply_attention_rows(d, chiplet_df):
    """Rewrite a compute chiplet's attn_qk 'start'/'middle' and attn_v 'middle'/'end' rows in d,
    in place (see above). PIM keeps its rows."""
    if chiplet_df.empty:
        return
    chips = chiplet_df[['net', 'arch_target', 'glb_scale', 'pe_x_scale', 'pe_y_scale']].drop_duplicates()
    if len(chips) != 1:
        raise ValueError(f"a row dict holds one chiplet of one network, got {len(chips)}")
    net, arch, glb, px, py = chips.iloc[0]
    if arch == 'PIM':
        return
    keys = [k for k in d if (k[0] == 'layer0_attn_qk' and k[5] in ('start', 'middle'))
            or (k[0] == 'layer0_attn_v' and k[5] in ('middle', 'end'))]
    if not keys:
        return
    chip = (net, arch, int(glb), int(px), int(py))
    res = _attention_residency()
    for key in keys:
        layer, bs, sq, mi, tp, ft, di, do = key
        if layer == 'layer0_attn_qk':
            if not res[('attn_qk',) + chip + (tp,)]:
                sib = 'single' if ft == 'start' else 'end'
                d[key] = dict(d[(layer, bs, sq, mi, tp, sib, di, do)])
            continue
        row, single = d[key], d[(layer, bs, sq, mi, tp, 'single', di, do)]
        C = gqa_kv.compute_cycles(net, layer, arch, glb, px, py, tp)
        old = gqa_kv.roofline_cycles(C, row['i_access'], row['w_access'], row['o_access'], di, do, tp)
        if not math.isclose(old * cycle_time, row['latency'], rel_tol=1e-9):
            raise ValueError(f"{chip} {key}: latency {row['latency']!r} is not max(C, DRAM bound) = "
                             f"{old} cycles; {gqa_kv.COMPUTE_CYCLES_CSV} does not describe this database")
        p_resident = res[('attn_v',) + chip + (tp,)]
        i_new = float(single['i_access'])
        w_new = 0.0 if p_resident else float(single['w_access'])
        o_new = float(single['o_access']) if ft == 'end' else 0.0
        e_new = (single['dynamic_energy']
                 - (float(single['w_access']) - w_new) * word_size
                 * dram_type_bandwidth_width_dict[di]['final_e'] * 1e-12
                 - (float(single['o_access']) - o_new) * word_size
                 * dram_type_bandwidth_width_dict[do]['final_e'] * 1e-12)
        d[key] = {'dynamic_energy': e_new,
                  'latency': gqa_kv.roofline_cycles(C, i_new, w_new, o_new, di, do, tp) * cycle_time,
                  'static_power': row['static_power'],
                  'i_access': i_new, 'w_access': w_new, 'o_access': o_new}


# 3) bind the global you use elsewhere
net_mem_df = _read_csv_cached("network_analysis.csv")#CSV_CACHE["network_analysis.csv"]

# Tier-2 dict for net_mem_df: (net_name, layer_name, fused_layer_type, batch_size, seq_len) -> row
_NET_MEM_DICT = None
_NET_MEM_FIELDS = ['in_mem', 'weight_mem', 'out_mem', 'operations']

def _get_net_mem_dict():
    global _NET_MEM_DICT
    if _NET_MEM_DICT is None:
        _NET_MEM_DICT = {}
        # Whole tensors (tp = 1 rows) in bf16 GB, K / V sized with num_key_value_heads, and the
        # fused softmax rows the file lacks (network_analysis_sizes)
        whole = network_analysis_sizes.with_fused_softmax(
            network_analysis_sizes.whole_tensor_sizes(net_mem_df))
        # Vectorized extraction for speed
        key_cols = ['net_name', 'layer_name', 'fused_layer_type', 'batch_size', 'sequence_length']
        key_arrays = {c: whole[c].values for c in key_cols}
        val_arrays = {c: whole[c].values for c in _NET_MEM_FIELDS}
        for i in range(len(whole)):
            key = (key_arrays['net_name'][i], key_arrays['layer_name'][i],
                   key_arrays['fused_layer_type'][i],
                   int(key_arrays['batch_size'][i]),
                   int(key_arrays['sequence_length'][i]))
            _NET_MEM_DICT[key] = {c: val_arrays[c][i] for c in _NET_MEM_FIELDS}
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

        # a layer without sizes cannot be provisioned: a missing row raises
        key = (net_name, layer.name, fused_layer_type, batch_size, sequence_length)
        layer_row = mem_dict[key]

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

    A PIM chiplet's die IS the GDDR7 module (PIM_DIE_AREA_MM2), not the small PE compute
    core -- mirror the main path (calculate_network_performance_with_memory_check).  Single
    die here: the main path's x2 is double-buffering for pipelined on-CP groups, which an
    off-CP op running in slack does not require.
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
#   (postprocess_bw.py is part of the database build pipeline, in the Zenodo code archive)
#
# Every other per-bit term therefore multiplies by word_size before applying a
# pJ/bit figure -- parse_stats.py (the DRAM re-basing), postprocess_bw.py
# (apply_bw_throttling_analytical) and the MoE switch model below
# (bits = batch_tokens * H * word_size).
#
# Two of the three call sites here were handing raw scalar counts straight through,
# which under-charged those paths by a factor of word_size:
#
#   main path        (search / reproduction path)  -- scalar counts, now converted
#   off-CP, non-PIM                                -- scalar counts, now converted
#   off-CP, PIM      (_in_bits = in_mem * 1e9 * 8) -- already bits, unchanged
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
    het_batch_candidates: Optional[Dict[str, List[int]]] = None,
    *,
    min_input_boundary_gb: float,
    charge_output_boundary: bool,
) -> Dict:
    """
    Calculate the dynamic energy, static power, and latency for each fusion group in the physical network,
    on a given chiplet config
    on_sram:
    true, if memory check is met, true fusion (cached intermediate data)
    false, add dram_backup, ignore memory check

    Stage-boundary DRAM: the boundary in front of a group is charged by that group (double
    buffered, x the group's batch replication) and holds the previous group's output and the
    group's own input and weights. A PIM group's boundary is its own pair of PIM dies. The
    boundary after the last group is charged by the last group only when charge_output_boundary.
    One call on the whole network (the linear path) charges every boundary once with
    min_input_boundary_gb=0 and charge_output_boundary=True. The DAG path prices each
    critical-path stage as a one-group network: it passes the previous stage's output size as
    min_input_boundary_gb (so the stage's input boundary is the one physical boundary between
    the two stages) and charges the output boundary on the last stage only.

    Args:
        physical_network: PhysicalNetwork object containing fusion groups and layer info
        csv_file: Path to the CSV file containing chiplet performance data
        chiplet_config: ChipletConfig object representing the chiplet being used
        net_name: Network name corresponding to entries in the CSV
        min_input_boundary_gb: lower bound (GB) on the first group's input boundary: the output
            of the stage in front of it (0 at the network input)
        charge_output_boundary: charge the boundary after the last group (the network output)

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

    out_mem_prev = min_input_boundary_gb
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

    # Pre-build the row dict (once, not per-layer)
    row_dict_main = _build_row_dict(chiplet_data)

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
        group_names = [layer.name for layer in fusion_group.layers]
        for idx, layer in enumerate(fusion_group.layers):
            flt = get_fused_layer_type(idx, num_layers_per_group)
            is_sm = _is_softmax(layer.name)
            is_attn = is_attention_layers(layer.name)
            # softmax is priced as the one fused op: a legacy sub-op (layer<N>_softmax_max/_sub_exp/
            # _sum/_div) carries the whole softmax's H/Q/K instance, so priced here it would count
            # a whole softmax per sub-op. load_from_dir does not load them next to the fused op.
            if layer.name.endswith(('_softmax_max', '_softmax_sub_exp', '_softmax_sum',
                                    '_softmax_div')):
                raise ValueError(f"legacy softmax sub-op {layer.name!r} of {net_name!r} is "
                                 f"not priced; use the fused layer<N>_softmax")
            # a non-PIM softmax is priced from its problem instance (H, Q, K, B), which
            # LayerConfig.instance does not carry
            sm_dims = sm_nbrs = None
            if is_sm and not is_pim:
                inst = layer.problem_data.get('instance', {})
                if not all(d in inst for d in ('H', 'Q', 'K', 'B')):
                    raise ValueError(f"softmax {layer.name!r} of {net_name!r} has no H/Q/K/B "
                                     f"problem instance ({sorted(inst)})")
                sm_dims = (int(inst['H']), int(inst['Q']), int(inst['K']), int(inst['B']))
                # S / P can stay in the GLB only across a handoff inside the stage: is softmax's
                # producer just before it and its consumer just after it (_softmax_row_cached)?
                sm_nbrs = (idx > 0 and group_names[idx - 1].endswith('attn_qk'),
                           idx + 1 < len(group_names) and group_names[idx + 1].endswith('attn_v'))
            layer_info.append((layer.name, flt, is_sm, is_attn, sm_dims, sm_nbrs))

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
                            # The boundary DRAM this group charges, also inside cost /
                            # static_power / area: its input boundary (a non-PIM group) and
                            # the network-output boundary (the last group, when charged)
                            "area_input_buf": 0.0,
                            "cost_input_buf": 0.0,
                            "leak_input_buf": 0.0,
                            "mods_input_buf": 0.0,    # its DRAM modules (x2, x replication)
                            "area_output_buf": 0.0,
                            "cost_output_buf": 0.0,
                            "leak_output_buf": 0.0,
                            "mods_output_buf": 0.0,
                        }

                        layer_row = None
                        compute_row = None   # the compute chiplet's own row (not an analytic softmax row)
                        for idx, (layer_name, fused_layer_type, is_sm, is_attn, sm_dims, sm_nbrs) in enumerate(layer_info):
                            # non-PIM softmax runs on the chiplet's vector unit and is priced
                            # analytically (softmax_vector); PIM runs every op from its rows
                            analytic_sm = sm_dims is not None

                            # Attention ops are batch-agnostic: DB only has batch=1.
                            # PIM DB also only has batch=1; for batch>1 we look up
                            # batch=1 and scale energy/latency by batch_size below.
                            # MoE expert layers: use batch=1. A batch-1 row is priced as one
                            # expert's GEMM on the tokens routed to it (1 in decode; S*k/E in
                            # prefill, S on a PIM row); _moe_expand_expert_groups charges the
                            # rows a layer runs.
                            is_expert = layer_name in MOE_EXPERT_OPS
                            # softmax is batch-agnostic like attention: its row is per item
                            lookup_batch = 1 if (is_attn or is_sm or is_pim or is_expert) else batch_size
                            # Per-item ops are scaled elsewhere: attention and softmax by the
                            # batch-agnostic energy scaling below (their stage is replicated per
                            # item), experts by the MoE expansion (its rows per layer,
                            # _moe_expand_expert_groups). Only the other PIM ops run the batch
                            # serially on the looked-up batch-1 row.
                            per_item = is_attn or is_sm or is_expert
                            pim_batch_scale = batch_size if (is_pim and not per_item and batch_size > 1) else 1

                            # PIM data has dram_i/dram_o = 'GDDR7' — only matches
                            # when buffer_config assigns GDDR7 at this boundary.
                            # PIM + non-GDDR7 is infeasible: mark as inf.
                            dram_i_here = buffer_config[idx+glb_layer_idx]
                            dram_o_here = buffer_config[idx+glb_layer_idx+1]
                            if is_pim and (dram_i_here != 'GDDR7' or dram_o_here != 'GDDR7'):
                                group_results["dynamic_energy"] = float("inf")
                                group_results["latency"] = float("inf")
                                break

                            if analytic_sm:
                                io_row = _softmax_row_cached(
                                    sm_dims, net_name, chiplet_config.arch_target,
                                    chiplet_config.global_buffer_size_scale,
                                    chiplet_config.pe_x_scale, chiplet_config.pe_y_scale,
                                    tp, *sm_nbrs, dram_i_here, dram_o_here)
                            else:
                                row_key = (layer_name, lookup_batch, sequence_length,
                                           mapper_idx, tp, fused_layer_type,
                                           dram_i_here, dram_o_here)
                                io_row = row_dict_main.get(row_key)

                            layer_data = None
                            if on_sram:
                                layer_data = io_row

                            if layer_data is None:
                                # no row for this layer on this chiplet: the stage cannot be
                                # priced, so the option is rejected, not priced without the layer
                                group_results["dynamic_energy"] = float("inf")
                                group_results["latency"] = float("inf")
                                layer_row = None
                                break

                            # layer_data is now a plain dict (fast key access)
                            layer_row = layer_data
                            if not analytic_sm:
                                compute_row = layer_data

                            layer_latency = layer_row['latency'] * pim_batch_scale
                            if is_sm and batch_size > 1 and batch_size_scale == 1:
                                # a softmax in a stage that is not replicated per item runs the
                                # batch serially on the one chiplet's vector unit (or PIM die)
                                layer_latency *= batch_size

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
                                # (an analytic softmax row is per item and has none)
                                lookup_dram_i = buffer_config[idx+glb_layer_idx]
                                lookup_dram_o = buffer_config[idx+glb_layer_idx+1]
                                candidates = (het_batch_candidates.get(layer_name, [lookup_batch])
                                              if (het_batch_candidates and not analytic_sm)
                                              else [lookup_batch])
                                best_dyn_e = layer_row['dynamic_energy']

                                for cand_bs in candidates:
                                    if cand_bs == lookup_batch:
                                        continue
                                    het_key = (layer_name, cand_bs, sequence_length,
                                               mapper_idx, tp, fused_layer_type,
                                               lookup_dram_i, lookup_dram_o)
                                    het_row = row_dict_main.get(het_key)
                                    if het_row is not None:
                                        multiplier = cand_bs / lookup_batch
                                        amortized = het_row['dynamic_energy'] / multiplier
                                        if amortized < best_dyn_e:
                                            best_dyn_e = amortized

                                dynamic_energy = best_dyn_e

                                reads = layer_row['i_access']
                                writes = layer_row['o_access']

                                read_inter_chiplet_energy = calculate_inter_chiplet_communication(_accesses_to_bits(reads),bonding,num_hop=1)
                                write_inter_chiplet_energy = calculate_inter_chiplet_communication(_accesses_to_bits(writes),bonding,num_hop=1)

                                # The DB's accesses are per chip, like its energy, so the
                                # crossings join the per-chip energy before the x tp here and
                                # the attention x batch below.
                                dynamic_energy += read_inter_chiplet_energy + write_inter_chiplet_energy

                                # DB stores per-chip energy; total = per_chip * tp
                                dynamic_energy *= tp

                            # Batch-agnostic ops (attention, softmax): chiplets are duplicated
                            # for batch parallelism, so dynamic energy scales with
                            # batch_size while latency stays unchanged.
                            if (is_attn or is_sm) and batch_size > 1:
                                dynamic_energy *= batch_size

                            group_results["dynamic_energy"] += dynamic_energy
                            group_results["latency"] += layer_latency

                        if on_sram and not is_pim:
                            if fusion_group_mem_buffer_dict[group_idx] > available_memory:
                                group_results["dynamic_energy"] = float("inf")
                                group_results["latency"] = float("inf")

                        if layer_row is not None and not is_pim and compute_row is None:
                            # a stage of softmax alone: the compute chiplet's own leakage has no
                            # row to come from, so the stage cannot be priced
                            group_results["dynamic_energy"] = float("inf")
                            group_results["latency"] = float("inf")
                            layer_row = None

                        if layer_row is not None:
                            # the boundary after the last group: the network output (or, on the
                            # DAG path, charged by the last critical-path stage only)
                            charge_out = charge_output_boundary and group_idx == num_group - 1
                            out_spec = fusion_group_mem_spec_dict[group_idx+1][buffer_config[glb_layer_idx+num_layers_per_group]]
                            if is_pim:
                                # PIM: compute is embedded inside GDDR7 memory modules.
                                # Double buffering = 2 PIM dies [PIM_1, PIM_2] that
                                # alternate roles (one computes, one serves as buffer).
                                # No separate DRAM chips — everything is PIM: the dies are
                                # this group's input boundary.

                                full_area = calculate_actual_area(core_area_mm2, dram_type='GDDR7', bonding=bonding)
                                # Each PIM unit is one whole CENT device: PIM_DIES_PER_UNIT separately
                                # priced 224-mm^2 dies, each with its own GDDR7 PHY + controller
                                # (full_area); see global_parameter.py
                                layer_acc_cost = _cached_die_cost(full_area, bonding) * PIM_DIES_PER_UNIT * tp * 2 * batch_size_scale

                                # Static power: compute set (whole-device standby) + buffer set (idle,
                                # memory leakage only) + PHY, charged per die. A NaN DB row stays NaN,
                                # so the set of options is unchanged.
                                _pim_db_static = layer_row['static_power']
                                _pim_unit_static = _pim_db_static if math.isnan(_pim_db_static) else CENT_DEVICE_STANDBY_W
                                group_results["static_power"] = _pim_unit_static * tp * batch_size_scale
                                group_results["static_power"] += tp * mem_specs['GDDR7']['leakage_power'] * PIM_DIES_PER_UNIT * batch_size_scale
                                phy_leakage_density = 0.0001  # W/mm² at 14nm
                                phy_overheads = _cached_area_overheads(0, bonding, 'GDDR7')
                                group_results["static_power"] += (phy_overheads["phy_area"] + phy_overheads["ctrl_area"]) * phy_leakage_density * PIM_DIES_PER_UNIT * batch_size_scale

                                # Area: 2× PIM units for double buffering
                                group_results['area'] = core_area_mm2 * PIM_DIES_PER_UNIT * tp * 2 * batch_size_scale

                                # Cost: 2× PIM die cost
                                group_results["cost"] = layer_acc_cost

                                # a PIM group that ends the network writes the output to its own
                                # (GDDR7) output boundary, double-buffered like any other
                                if charge_out:
                                    group_results["cost"] += 2*out_spec["cost"]*batch_size_scale
                                    group_results["static_power"] += 2*out_spec["leakage_power"]*batch_size_scale
                                    group_results['area'] += 2*out_spec["footprint_area"]*batch_size_scale
                            else:
                                # Compute chiplets carry PHY for all DRAM types
                                full_area = calculate_actual_area(core_area_mm2, dram_type=group_dram_type, bonding=bonding, all_phy=True)

                                layer_acc_cost = _cached_die_cost(full_area, bonding) * tp * batch_size_scale
                                # double buffering
                                layer_buffer_cost = 2*fusion_group_mem_spec_dict[group_idx][buffer_config[glb_layer_idx]]["cost"]* batch_size_scale
                                if charge_out:
                                    layer_buffer_cost += 2*out_spec["cost"]*batch_size_scale
                                # W

                                # the compute chiplet's leakage from its own row (an analytic softmax
                                # row has none; the vector unit's is vector_leak_W)
                                group_results["static_power"] = compute_row['static_power'] * tp*batch_size_scale + vector_leak_W * batch_size_scale + 2*fusion_group_mem_spec_dict[group_idx][buffer_config[glb_layer_idx]]["leakage_power"]*batch_size_scale
                                if charge_out:
                                    group_results["static_power"] += 2*out_spec["leakage_power"]*batch_size_scale
                                # PHY + controller leakage for all DRAM interfaces on compute chiplet
                                phy_leakage_density = 0.0001  # W/mm² at 14nm (from Accelergy)
                                phy_overheads = _cached_area_overheads(0, bonding, group_dram_type, all_phy=True)
                                group_results["static_power"] += (phy_overheads["phy_area"] + phy_overheads["ctrl_area"]) * phy_leakage_density * batch_size_scale
                                # double buffering — all in mm², footprint for 3D-stacked
                                group_results['area'] = core_area_mm2 * tp * batch_size_scale + 2*fusion_group_mem_spec_dict[group_idx][buffer_config[glb_layer_idx]]["footprint_area"]*batch_size_scale
                                if charge_out:
                                    group_results['area'] += 2*out_spec["footprint_area"]*batch_size_scale
                                group_results["cost"] = layer_acc_cost + layer_buffer_cost

                                # the input boundary's share of the above
                                group_results["area_input_buf"] = 2*fusion_group_mem_spec_dict[group_idx][buffer_config[glb_layer_idx]]["footprint_area"]*batch_size_scale
                                group_results["cost_input_buf"] = 2*fusion_group_mem_spec_dict[group_idx][buffer_config[glb_layer_idx]]["cost"]*batch_size_scale
                                group_results["leak_input_buf"] = 2*fusion_group_mem_spec_dict[group_idx][buffer_config[glb_layer_idx]]["leakage_power"]*batch_size_scale
                                group_results["mods_input_buf"] = 2*fusion_group_mem_spec_dict[group_idx][buffer_config[glb_layer_idx]]["num_modules"]*batch_size_scale
                            if charge_out:
                                # the output boundary's share of the above
                                group_results["area_output_buf"] = 2*out_spec["footprint_area"]*batch_size_scale
                                group_results["cost_output_buf"] = 2*out_spec["cost"]*batch_size_scale
                                group_results["leak_output_buf"] = 2*out_spec["leakage_power"]*batch_size_scale
                                group_results["mods_output_buf"] = 2*out_spec["num_modules"]*batch_size_scale

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
            key = (lookup_name, layer.name, fused_layer_type, batch_size, sequence_length)
            layer_data = mem_dict[key]   # a layer without sizes raises
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
            key = (lookup_name, layer.name, fused_layer_type, batch_size, sequence_length)
            layer_data = mem_dict[key]   # a layer without sizes raises
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

class BoundaryCharge(NamedTuple):
    """One stage-boundary DRAM an option charges (double-buffered, x the stage's replication)."""
    cost: float      # $
    leak: float      # W
    modules: float   # DRAM modules


class StageOption(NamedTuple):
    """One priced option of a critical-path stage, before any cost weighting."""
    lat: float                    # s
    sp: float                     # static power, W (includes the boundary parts below)
    de: float                     # dynamic energy, J
    cost: float                   # $ (includes the boundary parts below)
    cid: str
    in_buf: BoundaryCharge        # the input boundary it charges (none for PIM: its dies)
    out_buf: BoundaryCharge       # the network-output boundary it charges (last stage only)
    tp: int
    bonding: str


def _stage_option(p: Dict, cid: str, tp: int, bonding: str) -> StageOption:
    """A StageOption from one evaluator result (group_results[tp][mapper][bonding][on_sram])."""
    return StageOption(p['latency'], p['static_power'], p['dynamic_energy'], p['cost'], cid,
                       BoundaryCharge(p['cost_input_buf'], p['leak_input_buf'], p['mods_input_buf']),
                       BoundaryCharge(p['cost_output_buf'], p['leak_output_buf'], p['mods_output_buf']),
                       tp, bonding)


def _finalize_option(o: StageOption, cost_aware: bool) -> Tuple:
    """A StageOption -> the (latency, static, dynamic, id) the hull takes: under cost_aware the
    static and dynamic terms are weighted by the option's own cost."""
    if cost_aware:
        return (o.lat, o.sp * o.cost, o.de * o.cost, o.cid)
    return (o.lat, o.sp, o.de, o.cid)


def _moe_expand_expert_groups(stage_options, fusion_groups, moe_config, virtual_network, chiplet_group,
                              stage_dram):
    """Expand expert stages with EP variants for MoE workloads.

    stage_options: stage index -> [StageOption] (raw, not cost-weighted); an expert stage's
    options are replaced by their EP variants. stage_dram: stage index -> the DRAM type of the
    stage's input boundary, where each expert chiplet holds its experts' weights.

    An EP=ep variant of an option runs the stage on ep copies of the option's chiplet, each with
    its own double-buffered DRAM holding the weights of E/ep experts, and, for ep > 1, through two
    switches: dispatch in front of the first expert stage, the one holding expert_gate_proj
    (fanning out to its ep chiplets), combine after the last, the one holding expert_down_proj
    (gathering from its ep chiplets); the roles follow the op, not the stage index. With the
    option's raw (lat, sp, de, cost), n the expert rows a layer runs (below), (c_out, sp_out)
    its network-output boundary (non-zero on the last stage only; one DRAM behind the combine
    switch, not replicated), d = 2 x (spec(E/ep experts) - spec(1 expert)) the growth of each
    chiplet's DRAM from the one expert the stage was priced with to its E/ep experts
    (get_memory_spec, cost and leakage), and n_sw the switches the stage owns for ep > 1 (1 on
    the gate stage: dispatch; 1 on the down stage: combine; 2 when one stage holds both; else 0):
      latency = lat x n / ep                + one-way switch latency (gate and down stage)
      dynamic = de x n (x 2 for gate+up)    + dispatch energy (gate) / combine energy (down)
      static  = (sp - sp_out) x ep + sp_out + d_leakage x ep + n_sw x switch_area(ep) x 0.15 W/mm^2
      cost    = (cost - c_out) x ep + c_out + d_cost x ep + n_sw x die_cost(switch_area(ep))
    Each switch is sized (ep + 1 ports) and charged by the stage it serves, at that stage's own
    ep, as its latency and energy are, so a design whose expert stages pick different EP degrees
    pays for each switch at its own port count. cost - c_out is one expert chiplet: its die
    (compute + DRAM PHYs, or its two PIM dies) and its input boundary. EP=1 is one chiplet whose
    DRAM holds all E experts. Options with tp > 1 are dropped (no TP for experts).

    The rows: an option is priced with one DB row, looked up at batch 1 and charged as one
    expert's GEMM on the tokens routed to it. A layer routes B*S*k token-expert pairs (B items of
    S tokens; S = 1 on a decode network, whose DB rows are keyed at sequence_length 1), so
    n = B*S*k / (row tokens):
      decode: a row is one token (workload N = 1): n = B*k;
      prefill, a Timeloop row: one expert's share of an item's prompt, N = S*k/E tokens (uniform
        routing, NETWORK.yaml tokens_per_expert): n = B*E, every expert once per item;
      prefill, a PIM row: CENT runs the prompt token by token, a row is S tokens (S x the decode
        row): n = B*k.

    Switch communication uses framework-consistent 0.3 pJ/bit for 2.5D.
    """
    E = moe_config['num_experts']
    k = moe_config['num_experts_per_tok']
    H = moe_config['hidden_size']
    B = virtual_network.batch_size
    S = virtual_network.sequence_length      # 1 on a decode network
    batch_tokens = B * S

    if S > 1:
        # n = B*E counts B*S*k tokens only if a Timeloop row holds S*k/E: check the workload the
        # rows were simulated from (a prefill expert YAML has B = 1, N = S*k/E)
        for fg in fusion_groups:
            for l in fg.layers:
                inst = l.problem_data['instance'] if l.name in MOE_EXPERT_OPS else None
                if inst is not None and (inst['B'] != 1 or inst['N'] * E != S * k):
                    raise ValueError(f"{virtual_network.network_name} {l.name}: a prefill expert "
                                     f"row holds S*k/E = {S}*{k}/{E} tokens; its workload has "
                                     f"B = {inst['B']}, N = {inst['N']}")

    def expert_rows(o):
        """n, the DB rows a layer runs on option o (docstring)."""
        if S == 1 or o.cid.split('@', 1)[0] == 'PIM':
            return B * k
        return B * E

    # Check if switch chiplet is in the pool
    has_switch = any(getattr(c, 'arch_target', '') == SWITCH_ARCH_TARGET
                     for c in chiplet_group)
    ep_degrees = MOE_EP_DEGREES if has_switch else [1]

    # Switch die (14 nm), from the calibration note moe/SWITCH_CALIBRATION.md in the
    # chiplet_timeloop development repository (not included in this repository), cross-checked
    # against NVSwitch Gen 1 (Hot Chips 30, 2018). Per port (ports = ep + 1): a UCIe PHY,
    # 0.78 mm^2 (the per-signal area of get_cost.py's HBM3 PHY, 19.28 mm^2 / 1024, x 32 signals
    # x 1.3 for clock and control), and in/out SRAM buffers of 64 KiB each at 4.5 Mb/mm^2 usable
    # (TSMC 16 nm HD bit cell, IEDM 2013, x 3 for peripherals): 2 x 65536 B / 562500 B/mm^2 =
    # 0.233 mm^2 (the note's table uses 0.224, 0.125 MiB / 0.56 MB/mm^2, and so lists 5.10 and
    # 9.22 mm^2 at 4 and 8 expert ports where this code gives 5.14 and 9.30). Plus an ORION 2.0
    # (DATE 2009) 5x5 crossbar scaled to 14 nm and 512 bits, 0.048 x (ports/5)^2 mm^2, and a
    # 64-MAC reduce engine, 0.027 mm^2. Static power 0.15 W/mm^2 (NVSwitch's, derated). The die
    # cost is _cached_die_cost (get_cost.calculate_die_cost, as for every compute die).
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

    # Pre-compute single-expert weight memory (GB) for each expert fusion group,
    # used for DRAM capacity scaling.
    mem_dict = _get_net_mem_dict()
    net_name = virtual_network.network_name
    bs = virtual_network.batch_size
    seq = virtual_network.sequence_length
    expert_weight_mem = {}   # gi -> single-expert weight_mem in GB (bf16, whole tensor)
    for gi, fg in enumerate(fusion_groups):
        if gi in expert_gis:
            w_mem = 0.0
            for li, layer in enumerate(fg.layers):
                flt = get_fused_layer_type(li, len(fg.layers))
                # an expert without sizes cannot be provisioned: a missing row raises
                w_mem += mem_dict[(net_name, layer.name, flt, bs, seq)]['weight_mem']
            # expert_up_proj has identical shape to expert_gate_proj (SwiGLU);
            # double weight memory only for gate_proj groups (gate + up on same chiplet).
            # expert_down_proj has no parallel partner — use actual weight memory.
            expert_weight_mem[gi] = w_mem * 2 if gi in gate_gis else w_mem

    # The switch roles, chosen by op, not by group order: the linear path orders its groups as
    # the workload directory lists its files, which puts expert_down_proj in front of
    # expert_gate_proj for the qwen3 nets. The dispatch switch (latency, energy, area, static
    # power, die cost) goes to the group holding expert_gate_proj, sized by its ep; the combine
    # switch to the group holding expert_down_proj, sized by its ep (the same group when one
    # holds both).
    down_gis = {gi for gi, fg in enumerate(fusion_groups)
                if any(l.name == 'expert_down_proj' for l in fg.layers)}
    if len(gate_gis) != 1 or len(down_gis) != 1:
        raise ValueError(f"MoE switch roles need one expert_gate_proj group and one "
                         f"expert_down_proj group, got {sorted(gate_gis)} / {sorted(down_gis)}")
    first_expert_gi, = gate_gis      # dispatch
    last_expert_gi, = down_gis       # combine

    # Expand expert groups
    for gi in expert_gis:
        original = stage_options.get(gi, [])
        if not original:
            continue
        single_w_mem = expert_weight_mem[gi]
        # the stage's input-boundary DRAM, which each expert chiplet provisions for its experts
        dram_type = stage_dram[gi]
        # DRAM spec for one expert (the stage was priced with one expert's weights)
        single_spec = get_memory_spec(single_w_mem, dram_type)

        expanded = []
        has_up_proj = gi in gate_gis
        for o in original:
            if o.tp > 1:
                continue  # disable TP for MoE experts
            # Account for expert_up_proj (identical shape, same chiplet):
            # parallel compute (same latency), energy doubled.
            # Only applies to gate_proj groups; down_proj has no parallel partner.
            de = o.de * 2 if has_up_proj else o.de
            c_out, sp_out, _ = o.out_buf
            n = expert_rows(o)

            for ep in ep_degrees:
                experts_per_chiplet = E // ep
                scaled_spec = get_memory_spec(experts_per_chiplet * single_w_mem, dram_type)

                # A PIM option keeps its experts in its own units first. Growth over single_spec
                # (one module) credits a 2-GB unit; a CENT-device unit holds PIM_DIES_PER_UNIT x 2 GB
                # before any GDDR7 is added (global_parameter.py).
                cost_spec = leak_spec = single_spec
                if o.cid.split('@', 1)[0] == 'PIM':
                    _pim_need_gb = experts_per_chiplet * single_w_mem
                    cost_spec = leak_spec = get_memory_spec(
                        min(_pim_need_gb, PIM_DIES_PER_UNIT * PIM_GDDR7_MODULE_GB), dram_type)

                # DRAM growth per chiplet (2× for double buffering)
                delta_leakage = 2 * (scaled_spec["leakage_power"] - leak_spec["leakage_power"])
                delta_cost = 2 * (scaled_spec["cost"] - cost_spec["cost"])

                # n expert rows per layer; each of the ep chiplets runs n / ep of them
                ep_lat = o.lat * n / ep
                ep_de = de * n
                ep_sp = (o.sp - sp_out) * ep + sp_out + delta_leakage * ep
                ep_cost = (o.cost - c_out) * ep + c_out + delta_cost * ep

                # Dispatch switch: one-way latency + energy on the gate group
                if gi == first_expert_gi:
                    ep_lat += switch_oneway_latency(ep)
                    ep_de += switch_dispatch_energy(ep)
                # Combine switch: one-way latency + energy on the down group
                if gi == last_expert_gi:
                    ep_lat += switch_oneway_latency(ep)
                    ep_de += switch_combine_energy(ep)
                # Two physical switches: dispatch (gate group) and combine (down group), each
                # charged once, by the group it serves, with that group's ep ports
                n_sw = (gi == first_expert_gi) + (gi == last_expert_gi)
                if n_sw and ep > 1:
                    ep_sp += n_sw * switch_area(ep) * switch_static_density
                    ep_cost += n_sw * _cached_die_cost(switch_area(ep), o.bonding)

                c_in, l_in, m_in = o.in_buf
                delta_modules = 2 * (scaled_spec["num_modules"] - cost_spec["num_modules"])
                expanded.append(o._replace(
                    lat=ep_lat, sp=ep_sp, de=ep_de, cost=ep_cost, cid=f'{o.cid}@moe_ep{ep}',
                    in_buf=BoundaryCharge((c_in + delta_cost) * ep, (l_in + delta_leakage) * ep,
                                          (m_in + delta_modules) * ep)))
        stage_options[gi] = expanded


def cal_perf_phy_net(chiplet_group, chiplets_data, physical_network,
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
                het_batch_candidates=het_batch_candidates,
                # the whole network in one call: every boundary is charged once, by the group
                # after it, and the network output by the last group
                min_input_boundary_gb=0.0,
                charge_output_boundary=True,
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

                                    # raw: weighted by cost (cost_aware) after the MoE expansion
                                    group_parsed_result_dict[buffer_config_str][group_idx].append(
                                        _stage_option(_grp, config_id, tp, bonding))

        # --- MoE expert parallelism expansion ---
        # For MoE workloads, expand expert fusion groups with EP variants
        # (_moe_expand_expert_groups), on the raw options.
        if moe_config is not None:
            stage_dram, glb = {}, 0
            for group_idx, fg in enumerate(fusion_groups):
                stage_dram[group_idx] = buffer_config[glb]
                glb += len(fg.layers)
            _moe_expand_expert_groups(
                group_parsed_result_dict[buffer_config_str],
                fusion_groups, moe_config,
                physical_network.virtual_network,
                chiplet_group, stage_dram)

        # the (latency, static, dynamic, id) the hull takes; under cost_aware the static and
        # dynamic terms are weighted by each option's own cost
        for group_idx, options in group_parsed_result_dict[buffer_config_str].items():
            group_parsed_result_dict[buffer_config_str][group_idx] = [
                _finalize_option(o, cost_aware) for o in options]

        # A group with no priced option makes the gene infeasible, as on the DAG-CP path
        # (cal_perf_phy_net_dag_cp). The MoE path used to drop such groups (a softmax stage
        # without vector data), and the hull skips them, pricing the network without them.
        if any(not options for options in group_parsed_result_dict[buffer_config_str].values()):
            return (float('inf'), None), (float('inf'), None)

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

    # No PIM-adjacency deduction: a non-PIM group after a PIM group charges its input boundary,
    # which holds its own weights (sized in cal_mem_req_for_fusion_group), not the PIM dies (the
    # PIM group's own boundary). The deduction removed that boundary's leakage (energy only, and
    # in energy units from EDP and cost-weighted values) from the reported value.
    return (min_e, min_e_config), (min_edp, min_edp_config)

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
    for chiplet_idx, chiplet_config in enumerate(chiplet_group):
        chiplets_data.append(get_chiplet_data(
            DB_CSV,
            chiplet_config.arch_target,
            chiplet_config.global_buffer_size_scale,
            chiplet_config.pe_x_scale,
            chiplet_config.pe_y_scale,
            net_name_to_test
        ))

    import time
    start = time.time()


    (min_e,min_e_config),(min_edp, min_edp_config) = cal_perf_phy_net(chiplet_group, chiplets_data, physical_network,DB_CSV,cost_aware=False, buffer_config=gene['buffer_config'])
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
                      results_file, objective, verbose,
                      cost_aware, dag=None):
    """Drop-in for genetic_algo_opt_phy_net's evaluate_gene."""
    try:
        physical_network = create_physical_network_from_gene(virtual_network, gene)
        (min_e, min_e_config), (min_edp, min_edp_config) = cal_perf_phy_net(
            chiplet_group, chiplets_data,
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

def _prune_spec_to_network(positions, off_cp, network):
    """Prune CP positions + off-CP ops to the ops actually present in the loaded network.

    A *softmax* op on the critical path that is absent from the loaded network raises: pruned,
    it would contribute ZERO latency/energy/cost -- the bug that once zeroed ViT softmax, when
    the fused `layer0_softmax` was filtered out at load. load_from_dir always loads the fused
    softmax (the evaluator prices it analytically, not from database rows), so its absence is a
    workload defect.
    """
    layer_names = {l.name for l in network.layers}
    for pos in positions:
        for op in pos.ops:
            if op not in layer_names and utility_functions.is_softmax_layers(op):
                raise ValueError(
                    f"softmax op {op!r} is on the critical path but absent from loaded "
                    f"network {getattr(network, 'network_name', '?')!r}")
        pos.ops = [op for op in pos.ops if op in layer_names]
    off_cp = [o for o in off_cp if o.op_name in layer_names]
    return positions, off_cp


def create_llama_cp_spec(network=None, prefix: str = 'layer0') -> CriticalPathSpec:
    p = prefix
    positions = [
        CriticalPathPosition([f'{p}_q_proj', f'{p}_k_proj'], parallel=True),
        # softmax is the one fused op `layer0_softmax`; load_from_dir does not load the
        # legacy 4 sub-op names (_softmax_max/_sub_exp/_sum/_div) next to it
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
        specs); `_prune_spec_to_network` raises if the loaded network lacks it.
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


def _stage_raw_options(perf, chiplet, cid_suffix: str) -> List[StageOption]:
    """The valid options of a one-group evaluation (calculate_network_performance_with_memory_check)."""
    options = []
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
                        options.append(_stage_option(
                            p, f'{chiplet.get_identifier()}@{tp}@{mi}@{bd}{cid_suffix}', tp, bd))
    return options


_NO_BOUNDARY = BoundaryCharge(0.0, 0.0, 0.0)


def _shared_dram_source(combo: Sequence[StageOption]) -> int:
    """Index of the parallel-group member whose DRAM holds the group's shared boundaries: the
    member with the most of it (the modules of the boundaries it charges, then their $; a PIM
    member has no input DRAM, its dies are its input boundary), the first of equals. The Fengshui
    port's feasibility.shared_dram_source, which places that member's modules, uses the same key
    but also counts a member's own output-boundary DRAM on an inner stage, where this evaluator
    charges none. On the q||k and gate||up groups both pick the first non-PIM member."""
    return max(range(len(combo)), key=lambda j: (combo[j].in_buf.modules + combo[j].out_buf.modules,
                                                 combo[j].in_buf.cost + combo[j].out_buf.cost))


def _merge_parallel_configs(member_configs: List[List[StageOption]], cost_aware: bool) -> List[Tuple]:
    """Cross-product the raw options of a parallel group's members into virtual group options.

    The members share the group's boundaries: every non-PIM member is priced with the group's
    input boundary (and, on the last stage, every member with the network-output boundary). In
    each combination the member whose DRAM holds them (_shared_dram_source) keeps both charges and
    the others drop theirs from their cost and leakage. Each member is then weighted by its own
    cost (cost_aware: sum_i sp_i x c_i, sum_i de_i x c_i, the shared DRAM in the source member's
    sp and c) and the members summed.
    """
    import itertools
    if len(member_configs) == 1:
        return [_finalize_option(o, cost_aware) for o in member_configs[0]]
    merged = []
    for combo in itertools.product(*member_configs):
        src = _shared_dram_source(combo)
        members = []
        for j, o in enumerate(combo):
            if j != src:
                o = o._replace(sp=o.sp - o.in_buf.leak - o.out_buf.leak,
                               cost=o.cost - o.in_buf.cost - o.out_buf.cost,
                               in_buf=_NO_BOUNDARY, out_buf=_NO_BOUNDARY)
            members.append(_finalize_option(o, cost_aware))
        lat = max(c[0] for c in members)
        sp = sum(c[1] for c in members)
        de = sum(c[2] for c in members)
        cid = "VG(" + "||".join(c[3] for c in members) + ")"
        merged.append((lat, sp, de, cid))
    return merged


def _build_off_cp_functions(off_cp_info: dict, chiplet_group, chiplets_data,
                            gene: dict, cp_spec: CriticalPathSpec,
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

        # The het candidate runs `slack` batches in one pass and splits the energy of the row it
        # looks up over them, so it exists only where the DB holds the het batch itself. At batch
        # 1 that is batch 2, which the DB does not hold: clamped down to the batch-1 row and still
        # split two ways, the op was priced at half its energy and half its latency. An attention
        # op is looked up at batch 1, so it has no het row either.
        het_batch = batch_size * slack if v_het_batch else batch_size
        available_batches = sorted(batch_configs(transformer=True))
        if batch_size not in available_batches:
            raise ValueError(f"off-path op {op_name!r} of {net_name!r}: batch {batch_size} "
                             f"is not a DB batch {available_batches}")
        has_het = (v_het_batch and het_batch > batch_size and het_batch in available_batches
                   and not is_attention_layers(op_name))
        batch_candidates = [het_batch, batch_size] if has_het else [batch_size]

        for chiplet_idx, chiplet in enumerate(chiplet_group):
            if getattr(chiplet, 'arch_target', '') == 'switch_8port':
                continue
            cd = chiplets_data[chiplet_idx]
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
                                # These volumes are whole tensors, not per die, so they
                                # are not scaled by tp as the DB energy is. A GEMM's tp
                                # dies each hold a column slice of the weights and read
                                # the whole input (the non-PIM rows split it so: a GEMM's
                                # per-chip i_access does not shrink with tp); a head-split
                                # op's dies read only their heads. The output slices add
                                # up to out_mem. The sizes are bf16 GB (network_analysis_sizes);
                                # an op without them cannot be priced, so a missing row raises.
                                dyn *= tp
                                _m = _get_net_mem_dict()[
                                    (net_name, op_name, "single", int(lookup_b), int(seq_len))]
                                _readers = tp if utility_functions.is_gemm_layer(op_name) else 1
                                _in_bits = _m['in_mem'] * 1e9 * 8 * _readers
                                _out_bits = _m['out_mem'] * 1e9 * 8
                                dyn += calculate_inter_chiplet_communication(_in_bits, bonding, 1)
                                dyn += calculate_inter_chiplet_communication(_out_bits, bonding, 1)
                            else:
                                reads = row['i_access']
                                writes = row['o_access']

                                # the same crossings as the main path: per chip, then x tp
                                r_e = calculate_inter_chiplet_communication(_accesses_to_bits(reads), bonding, 1)
                                w_e = calculate_inter_chiplet_communication(_accesses_to_bits(writes), bonding, 1)
                                dyn += r_e + w_e
                                dyn *= tp

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

                        # The die's leakage is read from the row at the batch the base candidate
                        # looks up. An option without that row (only its het row) has no leakage
                        # to charge, so it is not offered: no fall back to another batch's row
                        # or to 0 W.
                        leak_b = 1 if is_attention_layers(op_name) else batch_size
                        r = row_dict.get((op_name, leak_b, seq_len, mapper_idx, tp,
                                          "single", dram_i, dram_o))
                        if r is None:
                            continue
                        sp = r['static_power'] * tp

                        is_pim_chiplet = getattr(chiplet, 'arch_target', '') == 'PIM'
                        if is_pim_chiplet:
                            # The raw DB static_power is the tp computing dies'; add their
                            # PHY/controller leakage, as the critical-path PIM path does. No idle
                            # die: a PIM stage's second die of each pair exists because the pair
                            # is its input boundary's memory, but an off-path op has no boundary
                            # of its own (it reads and writes the chain's, which their stages
                            # charge) and places and pays for tp dies (cost below), all computing.
                            # Each of the tp computing units draws whole-CENT-device standby, not the
                            # DB row's static_power (a NaN row stays NaN); the PHY + controller
                            # leakage is charged per die (global_parameter.py).
                            if not math.isnan(r['static_power']):
                                sp = CENT_DEVICE_STANDBY_W * tp
                            _phy = _cached_area_overheads(0, bonding, 'GDDR7')
                            sp += (_phy["phy_area"] + _phy["ctrl_area"]) * 0.0001 * PIM_DIES_PER_UNIT

                        core_mm2 = _offcp_core_mm2(chiplet)
                        full_area = calculate_actual_area(
                            core_mm2, dram_type=dram_i, bonding=bonding,
                            all_phy=(not is_pim_chiplet))
                        cost = _cached_die_cost(full_area, bonding) * tp
                        if is_pim_chiplet:
                            # A unit is PIM_DIES_PER_UNIT separately priced dies (global_parameter.py)
                            cost *= PIM_DIES_PER_UNIT

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
    # stage -> raw options of a non-parallel stage, weighted by cost (cost_aware) after the MoE
    # expansion; stage -> its input-boundary DRAM
    raw_stages, stage_dram = {}, {}

    # Each stage is priced as a one-group network and charges the boundary in front of it: the
    # one physical boundary between it and the previous stage, sized for the previous stage's
    # output as well as its own input and weights. Only the last stage also charges the network
    # output. Off-CP ops read and write CP boundaries and charge none.
    last_stage = len(cp_groups) - 1
    prev_out_gb = 0.0   # nothing in front of the first stage but the network input

    for grp_idx, fg in enumerate(cp_groups):
        pos_range = group_position_ranges[grp_idx]
        grp_buf_start = pos_range[0]
        grp_buf_end = pos_range[-1] + 1

        group_dram = buffer_config[grp_buf_start]
        n_layers_in_group = len(fg.layers)
        local_buf = [group_dram] * n_layers_in_group + [buffer_config[grp_buf_end]]

        is_parallel = grp_idx in parallel_indices
        charge_out = grp_idx == last_stage

        if is_parallel and len(fg.layers) > 1:
            member_fgs = []
            for layer in fg.layers:
                single_fg = FusionGroup(is_attn=layer.is_attn)
                single_fg.add_layer(layer)
                member_fgs.append(single_fg)
            member_mem = [cal_mem_req_for_fusion_group(net_name, m, batch_size, seq_len)
                          for m in member_fgs]
            # the members read one input boundary: every member is priced with it sized for all
            # of them, and _merge_parallel_configs keeps one member's charge (_shared_dram_source)
            group_in_gb = max([prev_out_gb] + [m[0] for m in member_mem])
            stage_out_gb = max(m[2] for m in member_mem)
            member_configs = []
            for single_fg in member_fgs:
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
                        min_input_boundary_gb=group_in_gb,
                        charge_output_boundary=charge_out,
                    )
                    op_configs.extend(_stage_raw_options(perf, chiplet, ''))
                member_configs.append(op_configs)

            if all(mc for mc in member_configs):
                group_parsed[grp_idx] = _merge_parallel_configs(member_configs, cost_aware)
            else:
                group_parsed[grp_idx] = []
        else:
            stage_out_gb = cal_mem_req_for_fusion_group(net_name, fg, batch_size, seq_len)[2]
            pn = PhysicalNetwork(virtual_network)
            pn.add_fusion_group(fg)

            options = []
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
                    min_input_boundary_gb=prev_out_gb,
                    charge_output_boundary=charge_out,
                )
                options.extend(_stage_raw_options(perf, chiplet, f'@{buffer_config_str}'))
            raw_stages[grp_idx] = options
            stage_dram[grp_idx] = group_dram
        prev_out_gb = stage_out_gb

    # MoE expert parallelism expansion (on the raw options: an EP variant's cost is its ep
    # chiplets', not the one chiplet's)
    moe_config = getattr(virtual_network, 'moe_config', None)
    if moe_config is not None:
        _moe_expand_expert_groups(raw_stages, cp_groups, moe_config,
                                  virtual_network, chiplet_group, stage_dram)
    for grp_idx, options in raw_stages.items():
        group_parsed[grp_idx] = [_finalize_option(o, cost_aware) for o in options]

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
        off_cp_info, chiplet_group, chiplets_data,
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