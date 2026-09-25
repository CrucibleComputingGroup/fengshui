"""Grouped-query attention (GQA): size the KV cache with num_key_value_heads.

THE MODEL ERROR
  The attention workloads src/workloads/<net>/layer0_attn_qk.yaml and layer0_attn_v.yaml give K
  and V the same head dimension H as Q (a Timeloop problem has one H).  In a GQA model the KV cache
  holds only num_key_value_heads heads, each shared by g = num_attention_heads / num_key_value_heads
  query heads, so everything the framework derives from the size of K or V is g times too large:
    - the database's DRAM reads of the KV tensor, the DRAM energy of those reads, and the
      DRAM-bandwidth bound of the row latency (layer0_attn_qk / layer0_attn_v rows);
    - the KV capacity in network_analysis.csv, which sizes every fusion group's DRAM
      (cal_perf_phy_net.cal_mem_req_for_fusion_group) and seeds the GA's buffer config
      (cal_perf_phy_net.cal_buffer_config).
  FLOPs are right: every query head attends over the whole sequence.
  g = 4 llama3.1-8B, 8 llama3.1-70B, 8 qwen3-30B-A3B, 16 qwen3-235B-A22B, 1 everywhere else.

WHERE THE KV TENSOR LIVES
  In both yamls the KV tensor is Inputs2 (attn_qk: Inputs1 = Q, Inputs2 = K; attn_v: Inputs1 = P,
  Inputs2 = V).  parse_stats.py:106-124 maps Inputs2 to the i_* fields, so in the database the KV
  words are exactly column i_access.  layer_size_analysis.calculate_attention_sizes
  (layer_size_analysis.py:263-275) writes K / V to network_analysis.csv as weight_mem.

THE CORRECTION (applied once, where cal_perf_phy_net builds its row dicts)
  Database row of a KV op, non-PIM, i_access > 0:
      i'      = i / g
      E'      = E - (i - i') * word_size * final_e[dram_i] * 1e-12       (J; parse_stats.py:624-629)
      cycles  = max(C, ceil((i  + w) / bw_i), ceil(o / bw_o))            (postprocess_bw.py:93-101)
      cycles' = max(C, ceil((i' + w) / bw_i), ceil(o / bw_o))
      bw_x    = dram_type_bandwidth_width_dict[dram_x]['bandwidth'] * 8 / word_size / tp  (words/cycle)
      C       = Timeloop compute-only cycles of the op on that chiplet and tp: the sweep's
                'Cycles' with DRAM bandwidth x1000 (run_sweep.py:449-453, timeloop_helper.py:1067),
                i.e. the original_cycles postprocess_bw starts from.  Read from
                attn_compute_cycles.csv, which tools/build_attn_compute_cycles.py extracts from
                the Timeloop stats and checks against every attention row of the database.
    `cycles` must reproduce the row's own latency; a mismatch raises (the table does not describe
    this database).  The latency is replaced by cycles' * cycle_time only when cycles' != cycles.
  network_analysis.csv row of a KV op: weight_mem' = weight_mem / g.

ROWS LEFT AS THEY ARE
  - PIM rows: the CENT model supplies lumped latency / energy with no DRAM access counts, so there
    are no KV words to rescale.
  - middle / end rows of a fusion group: the fusion split (parse_stats.py:404-423) zeroes the
    Inputs2 fields, and for these ops Inputs2 is K / V, so the database rows carry no KV read and
    there is nothing to rescale here.  Physically no producer in the fusion group holds the KV
    cache on chip.  cal_perf_phy_net._apply_attention_rows rebuilds attn_v's middle / end rows
    from its corrected 'single' row, so they read V with num_key_value_heads heads; attn_qk's
    middle / end rows (linear path only: on the DAG path attn_qk opens its stage) still read no K.
  - networks with g = 1.

WHERE g COMES FROM (required; no default)
  src/workloads/<net>/NETWORK.yaml architecture.num_attention_heads / num_key_value_heads (each
  cites the model's HF config.json).  A network without a NETWORK.yaml must be listed, with its g,
  in KV_GROUP_WITHOUT_NETWORK_YAML.  Any other network raises when the evaluator first sees it.
"""
import csv
import math
import os

import yaml

from global_parameter import NET_DIR, cycle_time, dram_type_bandwidth_width_dict, word_size
from utility_functions import is_attention_layers

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Attention ops whose Inputs2 is the KV tensor (see module docstring).  Database column holding
# the KV words, and network_analysis.csv column holding the KV capacity:
KV_OPS = ("layer0_attn_qk", "layer0_attn_v")
KV_DB_WORDS = "i_access"
KV_CAPACITY = "weight_mem"

# Compute-only cycles C of the KV ops, per (net, layer_name, arch_target, glb_scale, pe_x_scale,
# pe_y_scale, tp_degree).  Built by tools/build_attn_compute_cycles.py.  Also read by
# cal_perf_phy_net._apply_attention_rows, which rebuilds attn_v's fused rows on it.
COMPUTE_CYCLES_CSV = os.path.join(_THIS_DIR, "attn_compute_cycles.csv")

# Networks with no NETWORK.yaml.  None of them is a GQA model, so g = 1 for each, stated here:
KV_GROUP_WITHOUT_NETWORK_YAML = {
    # OPT-66B: multi-head attention.  HF facebook/opt-66b config.json has num_attention_heads 72 and
    # no num_key_value_heads; src/workloads/gpt_OPT-66B_*/layer0_{1_v,2_q,3_k}.yaml project to the
    # same O = 9216 = 72 heads x 128.
    "gpt_OPT-66B_prefill": 1,
    "gpt_OPT-66B_decode": 1,
    # network_analysis.csv-only entries (no workload directory, not in the database).  Multi-head
    # attention: their q / k / v projections have equal weight_mem in network_analysis.csv.
    "gpt-1.3B_prefill": 1,
    "gpt-1.3B_decode": 1,
    "vit": 1,
    "stable_diffusion": 1,
    # CNNs: no attention operator.
    "efficientnet_b0": 1,
    "mobilenet_v3_small": 1,
    "replknet31b": 1,
    "resnet50": 1,
    "resnet18": 1,          # network_analysis.csv only
    "vgg16": 1,             # network_analysis.csv only
    "comparison": 1,        # network_analysis.csv only: three CNN layers (mobilenet / resnet)
    "comparison_tmp": 1,    # network_analysis.csv only: three CNN layers (resnet / replknet)
}

_KV_GROUP_CACHE = {}
_COMPUTE_CYCLES = None


def kv_group_factor(net_name):
    """g = num_attention_heads / num_key_value_heads of `net_name` (raises if unsourced)."""
    g = _KV_GROUP_CACHE.get(net_name)
    if g is None:
        g = _resolve_kv_group_factor(net_name)
        _KV_GROUP_CACHE[net_name] = g
    return g


def _resolve_kv_group_factor(net_name):
    path = os.path.join(NET_DIR, net_name, "NETWORK.yaml")
    has_yaml = os.path.isfile(path)
    if net_name in KV_GROUP_WITHOUT_NETWORK_YAML:
        if has_yaml:
            raise ValueError(f"{net_name}: has {path}; record its heads there and drop it from "
                             f"gqa_kv.KV_GROUP_WITHOUT_NETWORK_YAML")
        return KV_GROUP_WITHOUT_NETWORK_YAML[net_name]
    if not has_yaml:
        raise KeyError(f"no KV-head count for network {net_name!r}: add num_attention_heads and "
                       f"num_key_value_heads to {path}, or list it in "
                       f"gqa_kv.KV_GROUP_WITHOUT_NETWORK_YAML")
    with open(path) as f:
        arch = yaml.safe_load(f)["architecture"]
    for k in ("num_attention_heads", "num_key_value_heads"):
        if k not in arch:
            raise KeyError(f"{path}: architecture.{k} is required (KV-cache size)")
    h, kvh = int(arch["num_attention_heads"]), int(arch["num_key_value_heads"])
    if kvh <= 0 or h % kvh:
        raise ValueError(f"{path}: num_attention_heads {h} is not a multiple of "
                         f"num_key_value_heads {kvh}")
    return h // kvh


def check_kv_ops(net, layer_names, g):
    """Raise if GQA network `net` (g > 1) has an attention op other than KV_OPS.

    The correction is keyed on the op names in KV_OPS; an attention op with another name
    (utility_functions.is_attention_layers, e.g. the legacy *_qk / *_av) would keep its
    query-head-sized KV cache without notice.
    """
    if g == 1:
        return
    other = sorted({l for l in layer_names if is_attention_layers(l) and l not in KV_OPS})
    if other:
        raise ValueError(f"{net}: g = {g}, but attention ops {other} are not in gqa_kv.KV_OPS "
                         f"{KV_OPS}; their KV tensor cannot be corrected")


def kv_capacity(network_analysis):
    """weight_mem of a network_analysis.csv DataFrame with K / V sized by num_key_value_heads.

    Returns the column with the KV-op rows divided by g; every network in the frame must have a g.
    """
    groups = {net: kv_group_factor(net) for net in network_analysis["net_name"].unique()}
    for net, layers in network_analysis.groupby("net_name")["layer_name"].unique().items():
        check_kv_ops(net, layers, groups[net])
    g = network_analysis["net_name"].map(groups)
    w = network_analysis[KV_CAPACITY]
    return w.where(~network_analysis["layer_name"].isin(KV_OPS), w / g)


def _compute_cycles():
    global _COMPUTE_CYCLES
    if _COMPUTE_CYCLES is None:
        if not os.path.isfile(COMPUTE_CYCLES_CSV):
            raise FileNotFoundError(f"{COMPUTE_CYCLES_CSV} is required to correct the GQA attention "
                                    f"rows; build it with tools/build_attn_compute_cycles.py")
        table = {}
        with open(COMPUTE_CYCLES_CSV, newline="") as f:
            for r in csv.DictReader(f):
                key = (r["net"], r["layer_name"], r["arch_target"], int(r["glb_scale"]),
                       int(r["pe_x_scale"]), int(r["pe_y_scale"]), int(r["tp_degree"]))
                if key in table:
                    raise ValueError(f"{COMPUTE_CYCLES_CSV}: duplicate key {key}")
                table[key] = int(r["compute_cycles"])
        _COMPUTE_CYCLES = table
    return _COMPUTE_CYCLES


def dram_words_per_cycle(dram_type, tp):
    """DRAM bandwidth in words per cycle per chip (postprocess_bw.py:93-94)."""
    return dram_type_bandwidth_width_dict[dram_type]["bandwidth"] * 8 / word_size / tp


def roofline_cycles(compute_cycles, i_words, w_words, o_words, dram_i, dram_o, tp):
    """postprocess_bw.apply_bw_throttling_analytical's latency, in cycles (postprocess_bw.py:93-101)."""
    iw = i_words + w_words
    ci = math.ceil(iw / dram_words_per_cycle(dram_i, tp)) if iw > 0 else 0
    co = math.ceil(o_words / dram_words_per_cycle(dram_o, tp)) if o_words > 0 else 0
    return max(compute_cycles, ci, co)


def compute_cycles(net, layer_name, arch_target, glb_scale, pe_x_scale, pe_y_scale, tp):
    """C of a KV op on a chiplet at tp (attn_compute_cycles.csv); a missing key raises."""
    key = (net, layer_name, arch_target, int(glb_scale), int(pe_x_scale), int(pe_y_scale), int(tp))
    C = _compute_cycles().get(key)
    if C is None:
        raise KeyError(f"{COMPUTE_CYCLES_CSV} has no compute-only cycles for {key}")
    return C


def correct_db_row(row, net, layer_name, arch_target, glb_scale, pe_x_scale, pe_y_scale, tp,
                   dram_i, dram_o, g):
    """Correct one database row dict of a KV op in place (see module docstring).

    `row` holds dynamic_energy, latency, i_access, w_access, o_access; `g` is kv_group_factor(net).
    """
    if g == 1 or arch_target == "PIM":
        return
    i = row[KV_DB_WORDS]
    if not i >= 0:                  # NaN or negative: not a word count
        raise ValueError(f"{net} {layer_name} {arch_target}: {KV_DB_WORDS} = {i!r}")
    if i == 0:                      # middle / end of a fusion group: no KV read (module docstring)
        return
    key = (net, layer_name, arch_target, int(glb_scale), int(pe_x_scale), int(pe_y_scale), int(tp))
    C = compute_cycles(*key)
    w, o = row["w_access"], row["o_access"]
    old = roofline_cycles(C, i, w, o, dram_i, dram_o, tp)
    if not math.isclose(old * cycle_time, row["latency"], rel_tol=1e-9):
        raise ValueError(f"{key} {dram_i}@{dram_o}: latency {row['latency']!r} is not "
                         f"max(C, DRAM bound) = {old} cycles; {COMPUTE_CYCLES_CSV} does not "
                         f"describe this database")
    ni = i / g
    row["dynamic_energy"] = row["dynamic_energy"] - (i - ni) * word_size * \
        dram_type_bandwidth_width_dict[dram_i]["final_e"] * 1e-12
    new = roofline_cycles(C, ni, w, o, dram_i, dram_o, tp)
    if new != old:
        row["latency"] = new * cycle_time
    row[KV_DB_WORDS] = ni
