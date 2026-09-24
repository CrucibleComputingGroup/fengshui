"""Tensor sizes of network_analysis.csv as the models read them: whole tensors, in bf16 GB.

THE MODEL ERROR
  network_analysis.csv gives each op's in_mem / weight_mem / out_mem in GB, one row per
  (net, layer, fused type, batch, seq) and tp degree.  The sizes were read too small twice over:
    - bits per element.  A row's GB are elements x bits / 8e9 at the bits its generator was run
      with, 8 for most rows (layer_size_analysis.py --bits default), while the model's word is
      global_parameter.word_size (bf16).  They were read as bf16 GB: half the bytes.
    - one chip's slice.  A tp = 2 row of a head-split op (attention, softmax) holds half the
      tensor.  cal_perf_phy_net._get_net_mem_dict kept, per key, whichever row came last: the
      tp = 2 row for 8336 of 8736 keys.  A DRAM boundary is charged once whatever the tp (its
      capacity, cost, area and leakage are not multiplied by tp), so it holds the whole tensor:
      the tp = 1 row.
  An attention boundary of an 8-bit network was provisioned for a quarter of its bytes, every
  other tensor for half; of a 16-bit network, an attention boundary for half.

THE CORRECTION
  whole_tensor_sizes() keeps the tp = 1 rows and rescales the three sizes by word_size / bits
  once, after gqa_kv.kv_capacity (K / V sized by num_key_value_heads).  Its readers:
  cal_perf_phy_net (fusion-group DRAM capacity, the linear GA's seed buffer config, MoE expert
  DRAM, the off-path PIM activation crossing), through with_fused_softmax(), and
  cost_model/generate_cost_sweep.py (Fig. 10 memory $), which sums the file's own rows.

THE MISSING SOFTMAX
  The evaluator prices one fused <prefix>softmax op, but the file sizes the llama3.1 / qwen3
  softmax only as its four sub-ops: layer_size_analysis.calculate_transformer_elementwise_sizes
  raises on the fused name.  The capacity code skipped the missing key, so the fused softmax held
  no DRAM.  with_fused_softmax() adds its rows from attn_qk's; a key still missing raises there.

WHERE THE BITS COME FROM (required per network; no default)
  NET_ANALYSIS_BITS, per network family, and NET_ANALYSIS_OP_BITS for single ops written at other
  bits than their network.  Checked on the tp = 1, batch 1 'single' rows: every op of a network
  with a src/workloads directory matches layer_size_analysis's sizing of its yaml at the bits
  listed and not at the other width (the vit_* layer0_softmax, which layer_size_analysis no longer
  sizes, by its heads x seq^2 scores: 12 x 197^2 x 2 B = 9.31416e-4 GB for vit_b16_s197); the
  networks without one match their models' dimensions at 8 bits (e.g. gpt-1.3B q weight
  2048^2 x 1 B = 4.194304e-3 GB, vit q weight 768^2 x 1 B = 5.89824e-4 GB, resnet18 layer2_1_conv2
  output 128 x 28^2 x 1 B = 1.00352e-4 GB).  A network not listed raises.
"""
import re

import pandas as pd

import gqa_kv
from global_parameter import fused_layer_types, word_size

SIZE_FIELDS = ("in_mem", "weight_mem", "out_mem")

# network family (net_name without a _prefill* / _decode* suffix) -> bits per element of its rows
NET_ANALYSIS_BITS = {
    # layer_size_analysis.py (--bits, default 8); the llama3.1-8B / qwen3-30B-A3B batches 16-64 by
    # extend_network_analysis_batches.py (bits_per_word=8)
    "llama3.1_8b": 8,
    "llama3.1_70b": 8,
    "qwen3_30b_a3b": 8,
    "qwen3_235b_a22b": 8,
    "gpt-1.3B": 8,
    "gpt_OPT-66B": 8,
    "vit": 8,
    "stable_diffusion": 8,
    "efficientnet_b0": 8,
    "resnet18": 8,
    "resnet50": 8,
    "vgg16": 8,
    "comparison": 8,
    "comparison_tmp": 8,
    # add_cnn_to_network_analysis.py (BITS_PER_WORD = 16)
    "mobilenet_v3_small": 16,
    "replknet31b": 16,
    # 16: matched at 16 and not at 8; the run that wrote them is not recorded
    "vit_b16_s197": 16,
    "vit_h14_s257": 16,
    "vit_l16_s197": 16,
}

# (family, layer_name) -> bits, for ops written at other bits than their network: the llama3.1 /
# qwen3 softmax sub-ops are at 16 (layer_size_analysis.calculate_sizes_from_shape, which books the
# score matrix as weight_mem)
NET_ANALYSIS_OP_BITS = {
    (fam, op): 16
    for fam in ("llama3.1_8b", "llama3.1_70b", "qwen3_30b_a3b", "qwen3_235b_a22b")
    for op in ("layer0_softmax_max", "layer0_softmax_sub_exp", "layer0_softmax_sum",
               "layer0_softmax_div")
}


def net_family(net_name):
    """net_name without its _prefill* / _decode* suffix (the NET_ANALYSIS_BITS key)."""
    return re.sub(r"_(prefill|decode)(_\w+)?$", "", net_name)


def whole_tensor_sizes(network_analysis):
    """The tp = 1 rows of a network_analysis.csv DataFrame, in_mem / weight_mem / out_mem in GB of
    word_size-bit words, K / V sized by num_key_value_heads.

    Raises if a network has no bits per element or a key has no single tp = 1 row.
    """
    df = network_analysis.copy()
    df[gqa_kv.KV_CAPACITY] = gqa_kv.kv_capacity(network_analysis)
    fam = df["net_name"].map(net_family)
    unknown = sorted(set(fam) - set(NET_ANALYSIS_BITS))
    if unknown:
        raise KeyError(f"network_analysis.csv: no bits per element for {unknown}; add them to "
                       f"network_analysis_sizes.NET_ANALYSIS_BITS")
    bits = pd.Series([NET_ANALYSIS_OP_BITS.get((f, op), NET_ANALYSIS_BITS[f])
                      for f, op in zip(fam, df["layer_name"])], index=df.index)
    for c in SIZE_FIELDS:
        df[c] = df[c] * (word_size / bits)
    key = ["net_name", "layer_name", "fused_layer_type", "batch_size", "sequence_length"]
    whole = df[df["tp"] == 1]
    n_keys, n_whole = len(df[key].drop_duplicates()), len(whole[key].drop_duplicates())
    if n_whole != n_keys or len(whole) != n_whole:
        raise ValueError(f"network_analysis.csv: {n_keys} keys, {n_whole} with a tp = 1 row, "
                         f"{len(whole)} tp = 1 rows; each key needs exactly one")
    return whole


SOFTMAX_SUBOPS = ("softmax_max", "softmax_sub_exp", "softmax_sum", "softmax_div")


def with_fused_softmax(whole):
    """whole (whole_tensor_sizes) plus the fused <prefix>softmax rows the file lacks.

    For every <prefix>attn_qk 'single' row of a network with no <prefix>softmax row, one row per
    fused type at the same (batch, seq): softmax maps the scores S that attn_qk writes (that row's
    out_mem) to probabilities of the same shape, so it reads S where it opens its stage (start,
    single), writes as much where it closes it (end, single) and holds no weights (its row max and
    sum stay in the GLB, softmax_vector).  operations = the four sub-ops' (SOFTMAX_SUBOPS) at the
    same fused type and batch 1, the one batch they are recorded at: like attn_qk's rows and the
    ViT networks' own fused softmax rows, an attention-side row is per item at every batch.

    Raises if a sub-op has no batch-1 row to take the operations from.
    """
    key = ["net_name", "layer_name", "fused_layer_type", "batch_size", "sequence_length"]
    ops = dict(zip(whole[key].itertuples(index=False, name=None), whole["operations"]))
    fused = {(n, l) for n, l in zip(whole["net_name"], whole["layer_name"]) if l.endswith("softmax")}
    qk = whole[(whole["fused_layer_type"] == "single") & whole["layer_name"].str.endswith("attn_qk")]
    rows = []
    for r in qk.itertuples(index=False):
        prefix = r.layer_name[:-len("attn_qk")]
        if (r.net_name, prefix + "softmax") in fused:
            continue
        for ft in fused_layer_types:
            subs = [(r.net_name, prefix + s, ft, 1, r.sequence_length) for s in SOFTMAX_SUBOPS]
            missing = [k for k in subs if k not in ops]
            if missing:
                raise KeyError(f"network_analysis.csv: no rows {missing} to size "
                               f"{prefix}softmax of {r.net_name}")
            rows.append({"net_name": r.net_name, "layer_name": prefix + "softmax",
                         "fused_layer_type": ft,
                         "in_mem": r.out_mem if ft in ("start", "single") else 0.0,
                         "weight_mem": 0.0,
                         "out_mem": r.out_mem if ft in ("end", "single") else 0.0,
                         "operations": sum(ops[k] for k in subs),
                         "tp": 1, "batch_size": r.batch_size, "sequence_length": r.sequence_length})
    return pd.concat([whole, pd.DataFrame(rows, columns=whole.columns)], ignore_index=True)
