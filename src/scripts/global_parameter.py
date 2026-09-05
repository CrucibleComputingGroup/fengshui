import os
import numpy as np
from collections import defaultdict

try:
    import pytimeloop.timeloopfe.v4 as tl
    Specification = tl.Specification
except ImportError:
    tl = None
    Specification = None
THIS_SCRIPT_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
PROJECT_DIR = os.path.normpath(os.path.join(THIS_SCRIPT_DIR, ".."))
# arch/, workloads/, outputs/ are at project root, not inside scripts/
ARCH_DIR = os.path.join(PROJECT_DIR, "arch")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
TOP_JINJA_PATH = os.path.join(ARCH_DIR, "top.yaml.jinja2")
TOP_STATIONARY_JINJA_PATH = os.path.join(ARCH_DIR, "top_stationary.yaml.jinja2")
NET_DIR = os.path.join(PROJECT_DIR, "workloads")

num_mapping_per_arch=1  # energy only

cycle_time = 1e-9
technology_node = 14

mapping_targets = ["energy", "delay"]
tp_degrees = [1,2]
glb_scales = [1,4,9,16]
pe_scales = [1,2,3,4]
bonding_techniques = ["2D", "2.5D"]
on_srams = [True]

arch_targets=['eyeriss_like','simba_like','gemmini_like']
DEFAULT_ARCH_TARGETS = arch_targets

reg_mac_spatial = 4 # for Simba (follow tutorial)

# TODO update this to division
# No longer needed
# tile_sizes = [1, 4, 16, 32, 64]

# batch config:
def batch_configs(transformer: bool=True):
    if transformer:
        return [1,4,8,16]
    return [1]

prefill_seq_lens = [512, 1024, 2048, 4096]
decode_kv_lens = [512, 1024, 2048, 4096]

timeloop_timeout = 300
timeloop_victory_condition = 100
timeloop_num_threads = 4
timeloop_search_strategy = "random"


base_query_points = np.linspace(0.00001, 0.1, 1000).tolist()  # in seconds

word_size = 16 # bf16

glb_base_word = int(1048576*64/word_size)
pe_x_base_size = 64 # scale start from 64


net_topology_dict={
    'efficientnet_b0': [0,9],
    'mobilenet_v3_small': [0, 3, 8],  # block boundaries: idx0=feat.2, idx3=feat.5, idx8=feat.10
    'resnet50': [0,8],
    'replknet31b': [0, 6],  # stage boundary: idx0=stages_0, idx6=stages_1
    'gpt_OPT-66B_decode': [0,1,2,3,4,8,9],
    'gpt_OPT-66B_prefill': [0,1,2,3,4,8,9],
    'gpt-1.3B_decode': [0,1,2,3,4,8,9],
    'gpt-1.3B_prefill': [0,1,2,3,4,8,9],
    'vit': [0,1,2,3,4,8,9],
}

# Auto-generate topology entries for LLM workloads
for _model in ['llama3.1_8b', 'llama3.1_70b']:
    for _phase in ['prefill_s512', 'prefill_s1024', 'prefill_s2048', 'prefill_s4096',
                   'decode_kv512', 'decode_kv1024', 'decode_kv2048', 'decode_kv4096']:
        net_topology_dict[f'{_model}_{_phase}'] = [0]
for _model in ['qwen3_30b_a3b', 'qwen3_235b_a22b']:
    for _phase in ['prefill_s512', 'prefill_s1024', 'prefill_s2048', 'prefill_s4096',
                   'decode_kv512', 'decode_kv1024', 'decode_kv2048', 'decode_kv4096']:
        net_topology_dict[f'{_model}_{_phase}'] = [0]

# ViT workloads: 3 model variants
VIT_MODELS = ['vit_b16_s197', 'vit_l16_s197', 'vit_h14_s257']
for _vit in VIT_MODELS:
    net_topology_dict[_vit] = [0]

# LLaMA BF16 operators
LLAMA_PROJECTION_OPS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LLAMA_ATTENTION_OPS = ["attn_qk", "attn_v"]
LLAMA_SOFTMAX_SUB_OPS = ["softmax_max", "softmax_sub_exp", "softmax_sum", "softmax_div"]  # legacy sub-ops (pre-fused in DB)
LLAMA_SOFTMAX_OPS = ["softmax"]
LLAMA_ALL_OPS = LLAMA_PROJECTION_OPS + LLAMA_ATTENTION_OPS + LLAMA_SOFTMAX_OPS

# LLaMA TP config: which dimension to split for tensor parallelism
# New namespace: Cin/Cout/S/B for projections, B/Nh/Sq/Sk/D for attention
LLAMA_TP_CONFIG = {
    "q_proj": "M",
    "k_proj": "M",
    "v_proj": "M",
    "o_proj": "M",
    "gate_proj": "M",
    "up_proj": "M",
    "down_proj": "M",
    "attn_qk": "H",
    "attn_v": "H",
    "softmax": "H",
    "softmax_max":     "H",
    "softmax_sub_exp": "H",
    "softmax_sum":     "H",
    "softmax_div":     "H",
}

# LLaMA attention ops are batch-agnostic (run once, scale B externally)
LLAMA_BATCH_AGNOSTIC_OPS = ["attn_qk", "attn_v"]

# Qwen3 MOE operators (superset of LLaMA ops)
QWEN_EXTRA_OPS = ["expert_gate_proj", "expert_down_proj", "router", "lm_head"]
QWEN_PROJECTION_OPS = LLAMA_PROJECTION_OPS + ["expert_gate_proj", "expert_down_proj", "router", "lm_head"]
QWEN_ATTENTION_OPS = LLAMA_ATTENTION_OPS

QWEN_TP_CONFIG = {
    "q_proj": "M",
    "k_proj": "M",
    "v_proj": "M",
    "o_proj": "M",
    "expert_gate_proj": "M",
    "expert_down_proj": "M",
    "router": "M",
    "lm_head": "M",
    "attn_qk": "H",
    "attn_v": "H",
    "softmax": "H",
    "softmax_max":     "H",
    "softmax_sub_exp": "H",
    "softmax_sum":     "H",
    "softmax_div":     "H",
}
QWEN_BATCH_AGNOSTIC_OPS = ["attn_qk", "attn_v"]

# ViT operators
# ViT uses standard MHA (no GQA) + GELU MLP (fc1/fc2, no up_proj).
# Operator shapes reuse the same GEMM4D / Attn5D forms as LLaMA:
#   Projection: (B, N, C, M)  — q_proj/k_proj/v_proj/o_proj/gate_proj/down_proj/lm_head
#   Attention:  (B, H, Q, K, D) — attn_qk/attn_v
#   Softmax:    (B, H, Q, K)   — softmax_max/sub_exp/sum/div
# TP does not apply to ViT inference (single-device classification model).
VIT_PROJECTION_OPS = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "down_proj", "lm_head"]
VIT_ATTENTION_OPS  = ["attn_qk", "attn_v"]
VIT_SOFTMAX_OPS    = ["softmax_max", "softmax_sub_exp", "softmax_sum", "softmax_div"]
VIT_ALL_OPS        = VIT_PROJECTION_OPS + VIT_ATTENTION_OPS + VIT_SOFTMAX_OPS

# ViT TP config — operator→dimension mapping (same structure as LLaMA/Qwen).
# TP degree=1 is the default for ViT; these entries exist for consistency but
# are never activated unless tp_degree>1 is explicitly requested.
VIT_TP_CONFIG = {
    "q_proj":    "M",
    "k_proj":    "M",
    "v_proj":    "M",
    "o_proj":    "M",
    "gate_proj": "M",
    "down_proj": "M",
    "lm_head":   "M",
    "attn_qk":   "H",
    "attn_v":    "H",
    "softmax_max":     "H",
    "softmax_sub_exp": "H",
    "softmax_sum":     "H",
    "softmax_div":     "H",
}

# MoE Expert Parallelism
MOE_EP_DEGREES = [1, 4, 8]
MOE_EXPERT_OPS = ["expert_gate_proj", "expert_down_proj"]
SWITCH_ARCH_TARGET = 'switch_8port'

# DRAM bandwidth contention from parallel ops sharing a physical DRAM.
# Maps network_type -> layer_name -> (dram_i_divisor, dram_o_divisor).
# Default (1, 1) for layers not listed (no contention).
# Q/K/V: 3 parallel readers on input DRAM; Q&K share output DRAM;
#         V & attn_qk overlap (off-CP slack) and share output DRAM.
# gate/up: parallel SwiGLU branches share both input and output DRAM.
BW_CONTENTION_MAP = {
    'llama': {
        'q_proj': (3, 2), 'k_proj': (3, 2), 'v_proj': (3, 2),
        'attn_qk': (1, 2),
        'gate_proj': (2, 2), 'up_proj': (2, 2),
    },
    'qwen': {
        'q_proj': (3, 2), 'k_proj': (3, 2), 'v_proj': (3, 2),
        'attn_qk': (1, 2),
        'expert_gate_proj': (2, 2),
    },
    'vit': {
        'q_proj': (3, 2), 'k_proj': (3, 2), 'v_proj': (3, 2),
        'attn_qk': (1, 2),
    },
}

def is_llama_network(net: str) -> bool:
    """Check if a network name is a LLaMA BF16 workload."""
    return "llama" in net.lower()

# Base PE sizes per architecture (for BF16 arch files)
ARCH_BASE_PE = {
    "eyeriss_like": (64, 64),
    "simba_like": (64, 16),
    "gemmini_like": (64, 64),
    "simple_vector": (64, 1),
    "PIM": (1, 1),  # PIM has no traditional PE array
}

dram_options = ['LPDDR5','DDR5', 'GDDR7', 'HBM3']

# ---- PIM die model ----
# A PIM chip is a GDDR module with near-bank processing units (PUs) added.
# Near-bank PUs are fabricated in the DRAM process, reducing memory density.
#
# From CENT [Gu et al., ASPLOS'25, Table 1]:
#   GDDR6-PIM (AiM) retains 75% memory density vs conventional GDDR6.
#   The 25% area loss accommodates near-bank MAC reduction trees.
#
# Die area = conventional_area_per_GB / PIM_memory_density × module_capacity
#   GDDR7: 84 mm²/GB, 2 GB/module → 168 mm² conventional
#   PIM:   168 / 0.75 = 224 mm² (33% larger die for compute logic)
PIM_GDDR7_MODULE_GB = 2                    # one GDDR7 module capacity
PIM_MEMORY_DENSITY = 0.75                  # PIM retains 75% memory density (CENT, ASPLOS'25)
PIM_DIE_AREA_MM2 = (84.0 * PIM_GDDR7_MODULE_GB
                    / PIM_MEMORY_DENSITY)   # = 224.0 mm²

# DRAM config
dram_type_bandwidth_width_dict = {
    #pj/bit
    'LPDDR5': {'bandwidth': 70.4, 'width': 64, 'timeloop': r'"LPDDR5"', 'timeloop_e': 11, 'final_e':11}, #
    'DDR5': {'bandwidth': 70.4, 'width': 64, 'timeloop': r'"DDR5"', 'timeloop_e': 16, 'final_e':16},
    'GDDR7': {'bandwidth': 320, 'width': 64, 'timeloop': r'"DDR5"', 'timeloop_e': 4.5, 'final_e':4.5}, # dual channel
    'HBM3': {'bandwidth': 819, 'width': 1024, 'timeloop': r'"HBM3"', 'timeloop_e': 3.5, 'final_e':3.5}, #
    'HBM3E': {'bandwidth': 1229, 'width': 1024, 'timeloop': r'"HBM3E"', 'timeloop_e': 3.44, 'final_e':3.44}, #
    'Ideal': {'bandwidth': 1229000, 'width': 1024, 'timeloop': r'"HBM3E"', 'timeloop_e': 0, 'final_e':0} #
}

#TODO update parameters
e_inter_per_bit = 1.3e-12

transformer_nets = ['gpt_OPT-66B_prefill', 'gpt_OPT-66B_decode', 'vit', 'gpt-1.3B_prefill', 'gpt-1.3B_decode'] + VIT_MODELS

# Add LLM workloads to transformer_nets
for _model in ['llama3.1_8b', 'llama3.1_70b', 'qwen3_30b_a3b', 'qwen3_235b_a22b']:
    for _phase in ['prefill_s512', 'prefill_s1024', 'prefill_s2048', 'prefill_s4096',
                   'decode_kv512', 'decode_kv1024', 'decode_kv2048', 'decode_kv4096']:
        transformer_nets.append(f'{_model}_{_phase}')

# ============================================================
# Legacy OPT/CNN support — used by database_builder.py,
# parse_stats.py, layer_size_analysis.py, network_dataclass.py,
# cal_perf_phy_net.py, workload_parser.py, etc.
# NOT used by timeloop_helper.py or run_sweep.py.
# ============================================================

arch_vec_targets = ['simple_vector']

fused_layer_types = ["start", "middle", "end", "single"]
layer_forced_fused_dict = {'layer2_max':'start', 'layer3_sn':'middle', 'layer4_sd':'middle', 'layer5_a':'end'}  # legacy OPT only

def seq_configs(transformer: bool=False):
    if transformer:
        return [256,512,1024]
    return [1]

layer_factors = {
    "max": {"compute": 0.001637, "leak": 0.000072, "area": 0.05},
    "exp": {"compute": 5.515, "leak": 5.53, "area": 5.5},
    "div": {"compute": 2.112, "leak": 1.35144, "area": 2.50},
}

xy_dict = {
    "layer0_2_q": {"x": "O", "y": "D"},
    "layer0_3_k": {"x": "O", "y": "D"},
    "layer0_1_v": {"x": "O", "y": "D"},
    "layer1_qk": {"x": "E", "y": "M"},
    "layer6_av": {"x": "M", "y": "F"},
    "layer7_o": {"x": "D", "y": "O"},
    "layer8_ffn1": {"x": "O", "y": "I"},
    "layer9_ffn2": {"x": "I", "y": "O"}
}

TRANSFORMER_CONFIG = {
    "eyeriss_like": {"pe_names": {"x": "PE_column", "y": "PE"}, "factor_keys": xy_dict},
    "simba_like": {"pe_names": {"x": "PE", "y": "distributed_buffers"}, "factor_keys": xy_dict},
    "gemmini_like": {"pe_names": {"x": "PE_column", "y": "PE"}, "factor_keys": xy_dict},
}

TRANSFORMER_TP_CONFIG = {
    "layer0_2_q": "D", "layer0_3_k": "D", "layer0_1_v": "D",
    "layer1_qk": "H", "layer2_max": "H", "layer3_sn": "H",
    "layer4_sd": "H", "layer5_a": "H", "layer6_av": "H",
    "layer7_o": "O", "layer8_ffn1": "I", "layer9_ffn2": "O",
    "layer0_softmax": "H",  # fused softmax (new naming)
}

batch_agnostic_ops = ["layer1_qk", "layer2_max", "layer3_sn", "layer4_sd", "layer5_a", "layer6_av"]

def unifyname(name):
    """Normalize layer names to canonical keys for legacy TP config / xy_dict lookup.
    Handles both legacy naming (layer0_2_q) and new naming (layer0_q_proj, attn_v).
    """
    if not isinstance(name, str):
        return name
    s = name.strip()
    sl = s.lower()
    canonical = {"layer0_2_q","layer0_3_k","layer0_1_v","layer1_qk","layer2_max",
                 "layer3_sn","layer4_sd","layer5_a","layer6_av","layer7_o",
                 "layer8_ffn1","layer9_ffn2"}
    if sl in canonical:
        return s
    two_token_mapping = {
        "q_proj": "layer0_2_q", "k_proj": "layer0_3_k", "v_proj": "layer0_1_v",
        "o_proj": "layer7_o", "gate_proj": "layer8_ffn1", "down_proj": "layer9_ffn2",
        "up_proj": "layer8_ffn1", "attn_qk": "layer1_qk", "attn_v": "layer6_av",
    }
    for pat, canon in two_token_mapping.items():
        if sl.endswith(pat):
            return canon
    suffix = sl.split("_")[-1]
    mapping = {
        "q": "layer0_2_q", "k": "layer0_3_k", "v": "layer0_1_v",
        "qk": "layer1_qk", "max": "layer2_max", "sn": "layer3_sn",
        "sd": "layer4_sd", "a": "layer5_a", "av": "layer6_av",
        "o": "layer7_o", "ffn1": "layer8_ffn1", "ffn2": "layer9_ffn2",
    }
    return mapping.get(suffix, s)
