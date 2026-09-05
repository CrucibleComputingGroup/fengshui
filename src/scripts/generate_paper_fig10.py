"""Generate Figure 10 (fig:big_eval) for the MICRO paper.
2x2 grid: Energy, Energy×Cost, EDP, EDP×Cost across architectural paradigms,
normalized to Homogeneous All Nets.

Paradigms:
  0. GPU                        = gpu-layer-benchmark results
  1. Homogeneous All Nets       = baseline.py --mode=homo_allnet
  2. Homogeneous Per Net        = baseline.py --mode=homo
  3. Chiplet Pool               = SAEO sweep, N=8
  4. Unconstrained              = baseline.py --mode=heter

Reuses the exact broken-Y-axis visualization style from
gpu-layer-benchmark/visualization/plot_norm_two_row.py.
"""
import os
import re
import glob as globmod
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
try:
    from scipy.stats import gmean
except ImportError:
    def gmean(a):
        a = np.asarray(a, dtype=float)
        return float(np.exp(np.mean(np.log(a))))

# ColorBrewer "Set2" (first 5), hardcoded to avoid a seaborn dependency.
palette = [
    (0.400, 0.761, 0.647),
    (0.988, 0.553, 0.384),
    (0.553, 0.627, 0.796),
    (0.906, 0.541, 0.765),
    (0.651, 0.847, 0.329),
]

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Computer Modern Roman']

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = os.environ.get('FENGSHUI_ARCHGYM', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'archgym_results'))
# Chiplet-pool generation to read for the Chiplet-Pool (n=8) and Unconstrained
# (convergence-endpoint) bars: 'ae' = the DETERMINISTIC artifact-evaluation pools
# (per-network GA reseed; supersede v7 for exact reproduction). See fengshui/AE_PLAN.md.
CHAIN_VERSION = 'ae'
BASELINES_DIR = os.path.join(BASE, 'baselines')
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
# Figure output directory. Anonymous, portable default = repo-local figures/;
# override with FENGSHUI_FIG_OUT (our paper build points it at the Overleaf images dir).
OUT_DIR = os.environ.get('FENGSHUI_FIG_OUT',
                         os.path.join(SCRIPTS_DIR, '..', 'figures'))

# ── Network list ──────────────────────────────────────────────────────────
NETWORKS = [
    'llama3.1_8b_prefill_s1024_b1_seq1024',
    'llama3.1_8b_prefill_s1024_b8_seq1024',
    'llama3.1_8b_decode_kv1024_b1_seq1',
    'llama3.1_8b_decode_kv1024_b8_seq1',
    'llama3.1_70b_prefill_s1024_b1_seq1024',
    'llama3.1_70b_prefill_s1024_b8_seq1024',
    'llama3.1_70b_decode_kv1024_b1_seq1',
    'llama3.1_70b_decode_kv1024_b8_seq1',
    'qwen3_30b_a3b_prefill_s1024_b1_seq1024',
    'qwen3_30b_a3b_prefill_s1024_b8_seq1024',
    'qwen3_30b_a3b_decode_kv1024_b1_seq1',
    'qwen3_30b_a3b_decode_kv1024_b8_seq1',
    'qwen3_235b_a22b_prefill_s1024_b1_seq1024',
    'qwen3_235b_a22b_prefill_s1024_b8_seq1024',
    'qwen3_235b_a22b_decode_kv1024_b1_seq1',
    'qwen3_235b_a22b_decode_kv1024_b8_seq1',
    'mobilenet_v3_small_b1_seq1',
    'mobilenet_v3_small_b8_seq1',
    'replknet31b_b1_seq1',
    'replknet31b_b8_seq1',
]

# ── Short display names for x-axis ──────────────────────────────────────
DISPLAY_NAMES = {
    'llama3.1_8b_prefill_s1024_b1_seq1024':    'LLaMA-8B\npf b1',
    'llama3.1_8b_prefill_s1024_b8_seq1024':    'LLaMA-8B\npf b8',
    'llama3.1_8b_decode_kv1024_b1_seq1':       'LLaMA-8B\ndec b1',
    'llama3.1_8b_decode_kv1024_b8_seq1':       'LLaMA-8B\ndec b8',
    'llama3.1_70b_prefill_s1024_b1_seq1024':   'LLaMA-70B\npf b1',
    'llama3.1_70b_prefill_s1024_b8_seq1024':   'LLaMA-70B\npf b8',
    'llama3.1_70b_decode_kv1024_b1_seq1':      'LLaMA-70B\ndec b1',
    'llama3.1_70b_decode_kv1024_b8_seq1':      'LLaMA-70B\ndec b8',
    'qwen3_30b_a3b_prefill_s1024_b1_seq1024':  'Qwen-30B\npf b1',
    'qwen3_30b_a3b_prefill_s1024_b8_seq1024':  'Qwen-30B\npf b8',
    'qwen3_30b_a3b_decode_kv1024_b1_seq1':     'Qwen-30B\ndec b1',
    'qwen3_30b_a3b_decode_kv1024_b8_seq1':     'Qwen-30B\ndec b8',
    'qwen3_235b_a22b_prefill_s1024_b1_seq1024':'Qwen-235B\npf b1',
    'qwen3_235b_a22b_prefill_s1024_b8_seq1024':'Qwen-235B\npf b8',
    'qwen3_235b_a22b_decode_kv1024_b1_seq1':   'Qwen-235B\ndec b1',
    'qwen3_235b_a22b_decode_kv1024_b8_seq1':   'Qwen-235B\ndec b8',
    'mobilenet_v3_small_b1_seq1':              'MobileNet\nb1',
    'mobilenet_v3_small_b8_seq1':              'MobileNet\nb8',
    'replknet31b_b1_seq1':                     'RepLKNet\nb1',
    'replknet31b_b8_seq1':                     'RepLKNet\nb8',
    'geometric_mean':                          'Geo.\nMean',
}

# ── Three-tier x-axis: batch (per column) / phase+len (per pair) / model (per group)
# The same "span to de-duplicate" idea as model names is applied to phase & seqlen,
# so nothing repeats per-column except the batch size that actually varies.

def _batch_of(net):
    """Tier 1 label: batch size, shown under every bar group."""
    if '_b1' in net:
        return 'b1'
    if '_b8' in net:
        return 'b8'
    return ''

# Tier 2: phase, spanning each LLM column pair. Sequence / KV length is fixed
# at 1024 for all LLM points and stated in the caption to save space.
# (start_col, span, label) — column indices into NETWORKS.
PHASE_GROUPS = [
    (0,  2, 'Prefill'), (2,  2, 'Decode'),
    (4,  2, 'Prefill'), (6,  2, 'Decode'),
    (8,  2, 'Prefill'), (10, 2, 'Decode'),
    (12, 2, 'Prefill'), (14, 2, 'Decode'),
]

# Tier 3: model family spanning its group of columns: (label, n_columns).
MODEL_GROUPS = [
    ('LLaMA-8B',  4),
    ('LLaMA-70B', 4),
    ('Qwen-30B',  4),
    ('Qwen-235B', 4),
    ('MobileNet', 2),
    ('RepLKNet',  2),
    ('GM',        1),
]


# ── GPU benchmark paths ───────────────────────────────────────────────────
# bf16 compiled-backend GPU data (matches committed constants: energy 8.3, R 0.174,
# EC 88). Materialized from gpu-layer-benchmark branch `compiler-backend-comparison`
# (*_compiled_bf16 dirs). The old `benchmarks/` (eager) dir gives energy 8.1 / EC 79.
GPU_DIR = os.environ.get('FENGSHUI_GPU_DIR', os.path.join(SCRIPTS_DIR, '..', 'gpu', 'benchmarks_bf16'))
GPU_COST = 10000  # optimistic RTX PRO 6000 acquisition cost ($)

# Mapping: NETWORKS entry → GPU txt file path (relative to GPU_DIR)
GPU_FILE_MAP = {
    'llama3.1_8b_prefill_s1024_b1_seq1024':    'llama3.1_8b_prefill/llama3.1_8b_prefill_b1_s1024.txt',
    'llama3.1_8b_prefill_s1024_b8_seq1024':    'llama3.1_8b_prefill/llama3.1_8b_prefill_b8_s1024.txt',
    'llama3.1_8b_decode_kv1024_b1_seq1':       'llama3.1_8b_decode/llama3.1_8b_decode_b1_kv1024.txt',
    'llama3.1_8b_decode_kv1024_b8_seq1':       'llama3.1_8b_decode/llama3.1_8b_decode_b8_kv1024.txt',
    'llama3.1_70b_prefill_s1024_b1_seq1024':   'llama3.1_70b_prefill/llama3.1_70b_prefill_b1_s1024.txt',
    'llama3.1_70b_prefill_s1024_b8_seq1024':   'llama3.1_70b_prefill/llama3.1_70b_prefill_b8_s1024.txt',
    'llama3.1_70b_decode_kv1024_b1_seq1':      'llama3.1_70b_decode/llama3.1_70b_decode_b1_kv1024.txt',
    'llama3.1_70b_decode_kv1024_b8_seq1':      'llama3.1_70b_decode/llama3.1_70b_decode_b8_kv1024.txt',
    'qwen3_30b_a3b_prefill_s1024_b1_seq1024':  'qwen3_30b_a3b_prefill/qwen3_30b_a3b_prefill_b1_s1024.txt',
    'qwen3_30b_a3b_prefill_s1024_b8_seq1024':  'qwen3_30b_a3b_prefill/qwen3_30b_a3b_prefill_b8_s1024.txt',
    'qwen3_30b_a3b_decode_kv1024_b1_seq1':     'qwen3_30b_a3b_decode/qwen3_30b_a3b_decode_b1_kv1024.txt',
    'qwen3_30b_a3b_decode_kv1024_b8_seq1':     'qwen3_30b_a3b_decode/qwen3_30b_a3b_decode_b8_kv1024.txt',
    'qwen3_235b_a22b_prefill_s1024_b1_seq1024':'qwen3_235b_a22b_prefill/qwen3_235b_a22b_prefill_b1_s1024.txt',
    'qwen3_235b_a22b_prefill_s1024_b8_seq1024':'qwen3_235b_a22b_prefill/qwen3_235b_a22b_prefill_b8_s1024.txt',
    'qwen3_235b_a22b_decode_kv1024_b1_seq1':   'qwen3_235b_a22b_decode/qwen3_235b_a22b_decode_b1_kv1024.txt',
    'qwen3_235b_a22b_decode_kv1024_b8_seq1':   'qwen3_235b_a22b_decode/qwen3_235b_a22b_decode_b8_kv1024.txt',
    'mobilenet_v3_small_b1_seq1':              'mobilenet_v3_small/mobilenet_v3_small_b1.txt',
    'mobilenet_v3_small_b8_seq1':              'mobilenet_v3_small/mobilenet_v3_small_b8.txt',
    'replknet31b_b1_seq1':                     'replknet_31b/replknet31b_b1.txt',
    'replknet31b_b8_seq1':                     'replknet_31b/replknet31b_b8.txt',
}


def _parse_gpu_file(filepath):
    """Parse a GPU benchmark txt file, return (energy_J, latency_ms)."""
    energy_pat = r"Total Pipeline Energy with Idle \+ P2P \(J\):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    lat_pat = r"Bottleneck Latency \(ms\):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    with open(filepath) as f:
        text = f.read()
    m_e = re.search(energy_pat, text)
    m_l = re.search(lat_pat, text)
    if not m_e or not m_l:
        raise ValueError(f'Could not parse energy/latency from {filepath}')
    return float(m_e.group(1)), float(m_l.group(1))


def _homo_t_asic_dict(competing_raw):
    """Per-net homogeneous-baseline latency t_ASIC (SECONDS), used for the
    iso-throughput cost fraction R = t_GPU / t_ASIC. Derived as EDP/energy from the
    Gemini-style (homogeneous) rows: both are DB quantities (energy in J, latency in
    seconds via cycle_time=1e-9), so t_ASIC = EDP[J*s] / energy[J] is in seconds."""
    dfg = pd.read_csv(competing_raw)
    sel = dfg[dfg['framework'] == 'Gemini-style'].copy()
    t = {}
    for net in NETWORKS:
        e = sel[(sel['objective'] == 'energy') & (sel['cost_aware'].astype(str) == 'False')
                & (sel['network'] == net)]
        d = sel[(sel['objective'] == 'edp') & (sel['cost_aware'].astype(str) == 'False')
                & (sel['network'] == net)]
        if len(e) and len(d) and float(e['value'].iloc[0]) > 0:
            t[net] = float(d['value'].iloc[0]) / float(e['value'].iloc[0])
    return t


def load_gpu_dicts(metric_key, competing_raw):
    """Load GPU results as {net: {"min_energy": val}} for a given metric.

    UNITS FIX (was a 1000x bug): the ASIC EDP is energy[J] * latency[SECONDS]
    (utility_functions.py:367, DB latency in s), but this parser reads
    'Bottleneck Latency (ms)'. We convert ms -> s before forming EDP, so the
    GPU/ASIC EDP and EDP*Cost ratios are unit-consistent. Energy and Energy*Cost
    have no latency term and are unchanged.

    COST: the cost-bearing panels charge the GPU only the iso-throughput fraction
    R = t_GPU / t_ASIC of a $GPU_COST card (matches the committed R-geomean 0.174),
    not a flat card."""
    use_cost = 'cost' in metric_key
    use_edp = metric_key in ('edp', 'edp_cost')
    t_asic = _homo_t_asic_dict(competing_raw) if use_cost else {}
    gpu_dict = {}
    for net in NETWORKS:
        rel = GPU_FILE_MAP.get(net)
        if not rel:
            continue
        fpath = os.path.join(GPU_DIR, rel)
        if not os.path.isfile(fpath):
            print(f'  [GPU] WARNING: missing {fpath}')
            continue
        energy_j, latency_ms = _parse_gpu_file(fpath)
        latency_s = latency_ms / 1000.0          # <-- ms -> s (the units fix)
        val = energy_j
        if use_edp:
            val *= latency_s
        if use_cost:
            R = latency_s / t_asic[net]          # iso-throughput device fraction
            val *= GPU_COST * R
        gpu_dict[net] = {"min_energy": val}
    # geometric mean
    vals = [gpu_dict[n]["min_energy"] for n in NETWORKS if n in gpu_dict]
    if vals:
        gpu_dict["geometric_mean"] = {"min_energy": gmean(vals)}
    print(f'  [{metric_key}] GPU: {len(gpu_dict)-1} networks from {GPU_DIR}')
    return gpu_dict


# ═══════════════════════════════════════════════════════════════════════════
# Data loading: build {net_name: {"min_energy": val}} dicts per paradigm
# ═══════════════════════════════════════════════════════════════════════════

def _find_in_dirs(pattern_suffix):
    candidates = []
    for d in [SCRIPTS_DIR, BASELINES_DIR]:
        candidates.extend(globmod.glob(os.path.join(d, pattern_suffix)))
    if not candidates:
        raise FileNotFoundError(f'No files matching {pattern_suffix}')
    return max(candidates, key=os.path.getmtime)


def _find_saeo_sweep_csv(metric_key):
    if metric_key in ('energy', 'energy_cost'):
        obj, suffix = 'energy', '_min_energy'
    else:
        obj, suffix = 'edp', '_min_edp'
    cost = '_cost' if 'cost' in metric_key else ''
    candidates = []
    # Search old-style dirs: {obj}{cost}_saeo*
    dir_prefix = f'{obj}{cost}_saeo'
    for d in globmod.glob(os.path.join(BASE, f'{dir_prefix}*')):
        if not os.path.isdir(d):
            continue
        dirname = os.path.basename(d)
        remainder = dirname[len(dir_prefix):]
        if remainder and not remainder.startswith('_'):
            continue
        for f in globmod.glob(os.path.join(d, 'incremental_chiplet_sweep_*.csv')):
            with open(f) as fh:
                n_lines = sum(1 for _ in fh)
            if n_lines >= 9:  # need at least N=8 row
                candidates.append(f)
    # Search v6 chain dirs (updated chiplet pool)
    chain_prefix = f'{CHAIN_VERSION}_{obj}{cost}_chain' if cost else f'{CHAIN_VERSION}_{obj}_chain'
    for d in globmod.glob(os.path.join(BASE, f'{chain_prefix}*')):
        if not os.path.isdir(d):
            continue
        for f in globmod.glob(os.path.join(d, 'saeo_isaeo_chain_*.csv')):
            with open(f) as fh:
                n_lines = sum(1 for _ in fh)
            if n_lines >= 9:
                candidates.append(f)
    if not candidates:
        raise FileNotFoundError(f'No complete SAEO sweep CSV for {metric_key}')
    return max(candidates, key=os.path.getmtime), suffix


def _obj_and_cost(metric_key):
    obj = 'energy' if metric_key in ('energy', 'energy_cost') else 'edp'
    cost = 'cost' in metric_key
    return obj, cost


def load_paradigm_dicts(metric_key, competing_raw=None):
    """Load data for all paradigms as {net: {"min_energy": val}} dicts."""
    obj, cost = _obj_and_cost(metric_key)
    if competing_raw is None:
        # DRAM-double-count-corrected Gemini baseline (matches the committed Fig 9 +
        # constants.tex: poolEnergyRed 48.7 etc.). The pre-correction competing_full_raw.csv
        # yields stale, over-favorable numbers (71.9) that silently desync from the paper.
        competing_raw = os.path.join(SCRIPTS_DIR, 'arch_impl', 'competing_corrected_raw.csv')

    # P1: Homogeneous all-net with a SINGLE unified memory type (homogenized). Sourced
    # from the competing-frameworks run; replaces the old het-memory `homo_allnet`, which
    # unfairly handed the homogeneous baseline Fengshui's per-layer memory heterogeneity.
    gemini_csv = competing_raw
    dfg = pd.read_csv(gemini_csv)
    sel = dfg[(dfg['framework'] == 'Gemini-style') &
              (dfg['objective'] == obj) &
              (dfg['cost_aware'].astype(str) == str(cost))]
    if sel.empty:
        raise ValueError(f'No Gemini-style rows for obj={obj} cost={cost} in {gemini_csv}')
    homo_allnet = {row['network']: {"min_energy": row['value']} for _, row in sel.iterrows()}
    print(f'  [{metric_key}] Homo/Gemini-style: {gemini_csv} ({len(homo_allnet)} nets)')

    # P2: Homo Per Net
    pernet_csv = _find_in_dirs(f'optimal_single_chiplet_{obj}_{cost}_*.csv')
    df = pd.read_csv(pernet_csv)
    homo_pernet = {}
    for _, row in df.iterrows():
        homo_pernet[row['network']] = {"min_energy": row[obj]}
    print(f'  [{metric_key}] Homo Per Net:  {pernet_csv}')

    # P3: Chiplet Pool (SAEO N=8)
    saeo_csv, suffix = _find_saeo_sweep_csv(metric_key)
    df_saeo = pd.read_csv(saeo_csv)
    chiplet_pool = {}
    row8 = df_saeo[df_saeo['n_chiplets'] == 8].iloc[0]
    for net in NETWORKS:
        col = f'{net}{suffix}'
        chiplet_pool[net] = {"min_energy": row8[col]}
    print(f'  [{metric_key}] Chiplet Pool:  {saeo_csv}')

    # P4: Unconstrained / hetero-ideal = the CONVERGENCE endpoint of the chiplet-pool
    # chain, NOT a hardcoded N=10. The pool is grown until the marginal improvement in
    # the search objective stays <1% for `patience` consecutive steps (see
    # converge_chain.py); we read that converged N. Prefer the `*_converged` chain
    # (which may extend past N=10, e.g. EDP) and fall back to the base chain, then take
    # its largest available n as the endpoint.
    cost_str = '_cost' if cost else ''
    base_chain_dir = f'{CHAIN_VERSION}_{obj}{cost_str}_chain'
    conv_chain_dir = base_chain_dir + '_converged'
    chain_csvs = (sorted(globmod.glob(os.path.join(BASE, conv_chain_dir, 'saeo_isaeo_chain_*.csv')))
                  or sorted(globmod.glob(os.path.join(BASE, base_chain_dir, 'saeo_isaeo_chain_*.csv'))))
    if not chain_csvs:
        raise FileNotFoundError(
            f'No chain CSV in {conv_chain_dir} or {base_chain_dir} under {BASE}')
    chain_csv = chain_csvs[-1]
    df_chain = pd.read_csv(chain_csv)
    conv_n = int(df_chain['n_chiplets'].max())
    row_conv = df_chain[df_chain['n_chiplets'] == conv_n].iloc[0]
    unconstrained = {}
    for net in NETWORKS:
        col = f'{net}_min_{obj}'
        unconstrained[net] = {"min_energy": row_conv[col]}
    print(f'  [{metric_key}] Unconstrained: {chain_csv} (converged n={conv_n})')

    # Add geometric means
    for d in [homo_allnet, homo_pernet, chiplet_pool, unconstrained]:
        vals = [d[n]["min_energy"] for n in NETWORKS if n in d]
        gm = gmean(vals) if vals else 0.0
        d["geometric_mean"] = {"min_energy": gm}

    return homo_allnet, homo_pernet, chiplet_pool, unconstrained


# ═══════════════════════════════════════════════════════════════════════════
# Visualization: reused from plot_norm_two_row.py (broken Y-axis style)
# ═══════════════════════════════════════════════════════════════════════════

def make_broken_y_axes_in(ax_parent, height_ratios=(1, 3), hspace=0.05):
    fig = ax_parent.figure
    spec = ax_parent.get_subplotspec()
    sub = spec.subgridspec(2, 1, height_ratios=height_ratios, hspace=hspace)
    ax_parent.set_visible(False)
    ax_top = fig.add_subplot(sub[0, 0])
    ax_bottom = fig.add_subplot(sub[1, 0], sharex=ax_top)
    return ax_top, ax_bottom


def plot_all_energy_subplot_broken(
    ax_top,
    ax_bottom,
    gpu_energy: dict,
    homo_allnet_energy: dict,
    homo_pernet_energy: dict,
    chip_pool_energy: dict,
    ideal_energy: dict,
    metric: str = "min_energy",
    title: str = "Energy Comparison",
    ylabel: str = "Energy (J)",
    show_legend: bool = False,
    break_point: float = 1.2,
    gap: float = 0.02,
    top_margin: float = 2.6,
    break_mark_size: float = 12.0,
    hide_x_ticks: bool = False
):
    # ---- collect names ----
    net_names = list(NETWORKS) + ["geometric_mean"]

    # "Homogeneous All Nets" uses a single unified memory type (homogenized); per-net
    # homogeneous is shown as a separate paradigm.
    setups = ["GPU", "Homogeneous All Nets", "Homogeneous Per Net",
              "Chiplet Pool", "Unconstrained"]
    setup_dicts = {
        "GPU": gpu_energy,
        "Homogeneous All Nets": homo_allnet_energy,
        "Homogeneous Per Net": homo_pernet_energy,
        "Chiplet Pool": chip_pool_energy,
        "Unconstrained": ideal_energy
    }
    colors = {s: palette[i] for i, s in enumerate(setups)}

    n_nets = len(net_names)
    x = np.arange(n_nets)
    width = 0.16

    # ---- normalization (to the unified-memory homogeneous all-net baseline) ----
    NORM = "Homogeneous All Nets"
    norms = {net: setup_dicts[NORM][net][metric]
             for net in net_names if net in setup_dicts[NORM]}

    # ---- compute values ----
    all_vals, series_vals = [], {}
    for setup in setups:
        vals = []
        for net in net_names:
            if net in setup_dicts[setup]:
                v = setup_dicts[setup][net][metric] / norms[net]
                vals.append(v)
                if np.isfinite(v):
                    all_vals.append(v)
            else:
                vals.append(np.nan)
        series_vals[setup] = vals

    ymax = max(all_vals) if all_vals else break_point
    ymin = min(all_vals) if all_vals else 0
    any_break = ymax > break_point + 1e-12

    top_bar_containers = []
    for i, setup in enumerate(setups):
        vals = series_vals[setup]
        ax_bottom.bar(x + i * width, vals, width, label=setup, color=colors[setup])
        bars_top = ax_top.bar(x + i * width, vals, width, label=setup, color=colors[setup])
        top_bar_containers.append((bars_top, vals))

    # ---- y-limits ----
    low_max = max(break_point - gap, 0)
    high_min = break_point + gap
    high_max = ymax * (1 + top_margin) if any_break else break_point * (1 + top_margin)

    ax_top.set_ylim(high_min, high_max)
    ax_bottom.set_ylim(0, low_max)
    ax_top.set_yscale('log')

    # Always use log scale for consistency across panels
    log_floor = max(ymin * 0.3, 1e-4) if ymin > 0 else 5e-4
    ax_bottom.set_ylim(log_floor, low_max)
    ax_bottom.set_yscale('log')

    # ---- tighten x margins ----
    x_left = -0.3
    x_right = n_nets - 1 + (len(setups) - 1) * width + width + 0.3
    for a in (ax_top, ax_bottom):
        a.set_xlim(x_left, x_right)

    # ---- ticks/labels ----
    group_centers = x + (len(setups) - 1) * width / 2
    offset = (len(setups) - 1) * width / 2
    if hide_x_ticks:
        ax_bottom.set_xticks([])
    else:
        # Tier 1 — batch size under each bar group (the only per-column variation).
        ax_bottom.set_xticks(group_centers)
        ax_bottom.set_xticklabels([_batch_of(n) for n in net_names],
                                  rotation=0, ha="center", fontsize=9)
        ax_bottom.tick_params(axis='x', length=0, pad=2)
        # Network-name block: model family (top, spanning the group) with
        # Prefill/Decode on the same block directly beneath it (per LLM pair).
        start = 0
        for fam, cnt in MODEL_GROUPS:
            cx = start + (cnt - 1) / 2.0 + offset
            fam_fs = 11.5 if cnt >= 4 else 9.0
            ax_bottom.annotate(
                fam,
                xy=(cx, 0), xycoords=('data', 'axes fraction'),
                xytext=(0, -17), textcoords='offset points',
                ha='center', va='top', fontsize=fam_fs, fontweight='bold',
                annotation_clip=False,
            )
            start += cnt
        for start_col, span, label in PHASE_GROUPS:
            cx = start_col + (span - 1) / 2.0 + offset
            ax_bottom.annotate(
                label,
                xy=(cx, 0), xycoords=('data', 'axes fraction'),
                xytext=(0, -31), textcoords='offset points',
                ha='center', va='top', fontsize=8, color='0.30',
                annotation_clip=False,
            )

    if ylabel:
        ax_bottom.set_ylabel(ylabel, fontsize=13, labelpad=8, fontweight='bold')
        ax_bottom.yaxis.set_label_coords(-0.06, 0.8)

    if show_legend:
        ax_top.legend(loc='upper right', fontsize=8)

    for a in (ax_top, ax_bottom):
        a.grid(True, alpha=0.3, axis='y', which='major')
        a.tick_params(axis='y', labelsize=8.5, pad=0, rotation=90)

    if any_break:
        ax_top.spines.bottom.set_visible(False)
        ax_bottom.spines.top.set_visible(False)
        ax_top.xaxis.tick_top()
        ax_top.tick_params(labeltop=False)
        ax_bottom.xaxis.tick_bottom()
    else:
        ax_top.set_visible(False)

    if any_break:
        for bars_top, vals in top_bar_containers:
            for rect, v in zip(bars_top, vals):
                if not (np.isfinite(v) and v > break_point):
                    continue
                cx = rect.get_x() + rect.get_width() / 2.0
                ax_top.annotate(
                    f"{v:.1f}X",
                    xy=(cx, v),
                    xytext=(7, 5),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=7.5, clip_on=False, rotation=90,
                )

    if any_break:
        d = .5
        kwargs = dict(marker=[(-1, -d), (1, d)],
                      markersize=break_mark_size, linestyle="none",
                      color='k', mec='k', mew=1, clip_on=False)
        ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
        ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kwargs)

    # Add model-group separators (vertical lines between model families)
    # Groups: LLaMA-8B(4), LLaMA-70B(4), Qwen-30B(4), Qwen-235B(4), MobileNet(2), RepLKNet(2), GM(1)
    group_boundaries = [4, 8, 12, 16, 18, 20]
    for gb in group_boundaries:
        xpos = gb - 0.5
        for a in (ax_top, ax_bottom):
            if a.get_visible():
                a.axvline(x=xpos, color='grey', linewidth=0.5,
                          linestyle='--', alpha=0.4)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main(competing_raw=None, chain_version=None):
    """Build Fig 11 (fig:big_eval) with the A2 iso-throughput cost model and print
    the recomputed constants.tex macros. A2 = every paradigm's cost is charged its
    OWN latency (device count is proportional to latency at iso-throughput), then
    normalized to the homogeneous (Gemini-style) baseline. Energy and EDP carry no
    cost term. Validated to reproduce the committed constants on the v7 pool; run on
    the deterministic `ae` pool for the artifact. Recovered from build_a2_figure.py.
    """
    import json as _json
    if competing_raw is None:
        competing_raw = os.path.join(SCRIPTS_DIR, 'arch_impl', 'competing_ae_raw.csv')
    chain = chain_version or CHAIN_VERSION
    ASSETS = os.path.join(SCRIPTS_DIR, 'arch_impl', 'fig11_a2_assets')
    C10 = GPU_COST
    print(f'=== Fig 11 (A2 iso-throughput cost) | chain={chain} | '
          f'competing={os.path.basename(competing_raw)} ===')

    # ── data sources ────────────────────────────────────────────────────────
    dfg = pd.read_csv(competing_raw)
    def gem(net, obj, cost):
        r = dfg[(dfg.framework == 'Gemini-style') & (dfg.objective == obj)
                & (dfg.cost_aware.astype(str) == str(cost)) & (dfg.network == net)]
        return float(r['value'].iloc[0])

    def _chain_csv(k):
        for d in (f'{chain}_{k}_chain_converged', f'{chain}_{k}_chain'):
            cands = sorted(globmod.glob(os.path.join(BASE, d, 'saeo_isaeo_chain_*.csv')))
            if cands:
                return cands[-1]
        raise FileNotFoundError(f'No chain CSV for {k} under {chain}_{k}_chain[_converged]')
    CH = {k: pd.read_csv(_chain_csv(k)) for k in ['energy', 'energy_cost', 'edp', 'edp_cost']}
    def mx(k): return int(CH[k].n_chiplets.max())
    def cv(k, n, net, suf): return float(CH[k][CH[k].n_chiplets == n].iloc[0][f'{net}_{suf}'])

    homo_lat = _json.load(open(os.path.join(ASSETS, 'homo_latency.json')))
    pern_lat = _json.load(open(os.path.join(ASSETS, 'pernet_latency.json')))

    def _pncsv(obj, cost):
        f = sorted(globmod.glob(os.path.join(SCRIPTS_DIR,
                   f'optimal_single_chiplet_{obj}_{cost}_*.csv')))[-1]
        d = pd.read_csv(f)
        return {r['network']: float(r[obj]) for _, r in d.iterrows()}
    PN = {'energy': _pncsv('energy', 'False'), 'edp': _pncsv('edp', 'False'),
          'ec': _pncsv('energy', 'True'), 'edpc': _pncsv('edp', 'True')}

    def _gpu(net):
        e, lat_ms = _parse_gpu_file(os.path.join(GPU_DIR, GPU_FILE_MAP[net]))
        return e, lat_ms / 1000.0

    # ── per-metric per-paradigm dicts (A2 iso: cost × own-latency) ───────────
    P = ["GPU", "Homogeneous All Nets", "Homogeneous Per Net", "Chiplet Pool", "Unconstrained"]
    def D(): return {s: {} for s in P}
    EN, ECd, EDP, EDPC = D(), D(), D(), D()
    for net in NETWORKS:
        e_g, t_g = _gpu(net)
        EN["GPU"][net] = {"min_energy": e_g}
        EN["Homogeneous All Nets"][net] = {"min_energy": gem(net, 'energy', False)}
        EN["Homogeneous Per Net"][net] = {"min_energy": PN['energy'][net]}
        EN["Chiplet Pool"][net] = {"min_energy": cv('energy', 8, net, 'min_energy')}
        EN["Unconstrained"][net] = {"min_energy": cv('energy', mx('energy'), net, 'min_energy')}
        EDP["GPU"][net] = {"min_energy": e_g * t_g}
        EDP["Homogeneous All Nets"][net] = {"min_energy": gem(net, 'edp', False)}
        EDP["Homogeneous Per Net"][net] = {"min_energy": PN['edp'][net]}
        EDP["Chiplet Pool"][net] = {"min_energy": cv('edp', 8, net, 'min_edp')}
        EDP["Unconstrained"][net] = {"min_energy": cv('edp', mx('edp'), net, 'min_edp')}
        ECd["GPU"][net] = {"min_energy": e_g * C10 * t_g}
        ECd["Homogeneous All Nets"][net] = {"min_energy": gem(net, 'energy', True) * homo_lat['energy_cost']['latency'][net]}
        ECd["Homogeneous Per Net"][net] = {"min_energy": PN['ec'][net] * pern_lat['energy_cost']['latency'][net]}
        ECd["Chiplet Pool"][net] = {"min_energy": cv('energy_cost', 8, net, 'min_energy') * cv('energy_cost', 8, net, 'latency')}
        ECd["Unconstrained"][net] = {"min_energy": cv('energy_cost', mx('energy_cost'), net, 'min_energy') * cv('energy_cost', mx('energy_cost'), net, 'latency')}
        EDPC["GPU"][net] = {"min_energy": e_g * t_g * t_g * C10}
        EDPC["Homogeneous All Nets"][net] = {"min_energy": gem(net, 'edp', True) * homo_lat['edp_cost']['latency'][net]}
        EDPC["Homogeneous Per Net"][net] = {"min_energy": PN['edpc'][net] * pern_lat['edp_cost']['latency'][net]}
        EDPC["Chiplet Pool"][net] = {"min_energy": cv('edp_cost', 8, net, 'min_edp') * cv('edp_cost', 8, net, 'latency')}
        EDPC["Unconstrained"][net] = {"min_energy": cv('edp_cost', mx('edp_cost'), net, 'min_edp') * cv('edp_cost', mx('edp_cost'), net, 'latency')}
    for dd in (EN, ECd, EDP, EDPC):
        for s in P:
            vals = [dd[s][n]["min_energy"] for n in NETWORKS]
            dd[s]["geometric_mean"] = {"min_energy": gmean(vals)}

    # ── figure (2×2 broken-y, reuse committed plot fns) ─────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 7.9))
    def _panel(ax, dd, ylabel, bp, hidex, tm=2.6):
        at, ab = make_broken_y_axes_in(ax, height_ratios=(1, 2.5), hspace=0.05)
        plot_all_energy_subplot_broken(
            at, ab, gpu_energy=dd["GPU"], homo_allnet_energy=dd["Homogeneous All Nets"],
            homo_pernet_energy=dd["Homogeneous Per Net"], chip_pool_energy=dd["Chiplet Pool"],
            ideal_energy=dd["Unconstrained"], ylabel=ylabel, break_point=bp,
            hide_x_ticks=hidex, top_margin=tm)
    _panel(axes[0, 0], EN, "Normalized Energy", 1.1, True)
    _panel(axes[0, 1], ECd, "Normalized Energy × Cost", 1.1, True)
    _panel(axes[1, 0], EDP, "Normalized EDP", 1.1, False)
    _panel(axes[1, 1], EDPC, "Normalized EDP × Cost", 1.5, False)
    handles, labels = None, None
    for a in fig.axes:
        hh, ll = a.get_legend_handles_labels()
        if ll:
            handles, labels = hh, ll
            break
    plt.tight_layout(rect=[0, 0.05, 1, 0.965], w_pad=1.5, h_pad=1.0)
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.975),
               ncol=5, frameon=False, fontsize=14)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_pdf = os.path.join(OUT_DIR, 'combined_comparison_norm_two_row_new.pdf')
    out_png = os.path.join(OUT_DIR, 'combined_comparison_norm_two_row_new.png')
    fig.savefig(out_pdf, dpi=200, bbox_inches='tight')
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_pdf}')

    # ── recomputed constants/constants.tex macros ───────────────────────────
    DECODE  = [n for n in NETWORKS if 'decode' in n]
    PREFILL = [n for n in NETWORKS if 'prefill' in n]
    CNN     = [n for n in NETWORKS if 'mobilenet' in n or 'replknet' in n]
    NOREP   = [n for n in NETWORKS if 'replknet' not in n]
    def RR(dd, p, nets):
        return gmean([dd[p][n]["min_energy"] / dd["Homogeneous All Nets"][n]["min_energy"]
                      for n in nets])
    def UNC(dd, nets):
        return (gmean([dd["Chiplet Pool"][n]["min_energy"] / dd["Unconstrained"][n]["min_energy"]
                       for n in nets]) - 1) * 100
    gaps = [UNC(dd, NETWORKS) for dd in (EN, ECd, EDP, EDPC)]
    print('\n=== RECOMPUTED constants/constants.tex (A2 iso; homo baseline = Gemini-style) ===')
    print(f'  \\poolEnergyRed        = {100*(1-RR(EN,"Chiplet Pool",NETWORKS)):.1f}')
    print(f'  \\poolECRed            = {100*(1-RR(ECd,"Chiplet Pool",NETWORKS)):.1f}')
    print(f'  \\poolEDPRed           = {100*(1-RR(EDP,"Chiplet Pool",NETWORKS)):.1f}')
    print(f'  \\poolEDPCRed          = {100*(1-RR(EDPC,"Chiplet Pool",NETWORKS)):.1f}')
    print(f'  \\gpuVsHomoEnergy      = {RR(EN,"GPU",NETWORKS):.1f}')
    print(f'  \\gpuVsHomoEC          = {RR(ECd,"GPU",NETWORKS):.1f}')
    print(f'  \\gpuVsHomoEDP         = {RR(EDP,"GPU",NETWORKS):.2f}')
    print(f'  \\gpuVsHomoEDPC        = {RR(EDPC,"GPU",NETWORKS):.1f}')
    print(f'  \\gpuVsHomoEnergyNoRep = {RR(EN,"GPU",NOREP):.1f}')
    print(f'  \\gpuVsHomoEDPNoRep    = {RR(EDP,"GPU",NOREP):.2f}')
    print(f'  \\decodeEnergyRed      = {100*(1-RR(EN,"Chiplet Pool",DECODE)):.0f}')
    print(f'  \\decodeEDPRed         = {100*(1-RR(EDP,"Chiplet Pool",DECODE)):.0f}')
    print(f'  \\prefillEnergyRed     = {100*(1-RR(EN,"Chiplet Pool",PREFILL)):.0f}')
    print(f'  \\cnnEnergyRed         = {100*(1-RR(EN,"Chiplet Pool",CNN)):.0f}')
    # Peak GPU-vs-homogeneous ratio on the batch-1 CNNs, per cost-weighted panel.
    # These are the two extremes the \gls{ec}/\gls{edpc} prose cites; they differ by
    # ~70x, so a single order-of-magnitude figure describes neither panel.
    CNN_B1 = [n for n in CNN if '_b1_' in n]
    def MAXR(dd, p_, nets):
        return max(dd[p_][n]['min_energy'] / dd['Homogeneous All Nets'][n]['min_energy']
                   for n in nets)
    print(f'  \\gpuVsHomoECMaxCnn    = {MAXR(ECd,"GPU",CNN_B1):.0f}')
    print(f'  \\gpuVsHomoEDPCMaxCnn  = {MAXR(EDPC,"GPU",CNN_B1):.0f}')
    print(f'  \\poolVsUnconGap       = {max(gaps):.1f}   (per-metric E/EC/EDP/EDPC: '
          + ', '.join(f'{g:.1f}' for g in gaps) + ')')


if __name__ == '__main__':
    import argparse
    _ap = argparse.ArgumentParser(
        description='Generate Fig 11 (fig:big_eval, A2 iso cost) + print constants.tex macros')
    _ap.add_argument('--competing', default=os.path.join(SCRIPTS_DIR, 'arch_impl', 'competing_ae_raw.csv'),
                     help="competing raw CSV; its Gemini-style rows are the homogeneous baseline")
    _ap.add_argument('--chain-version', default=CHAIN_VERSION,
                     help="chiplet-pool chain generation for Pool/Unconstrained (default 'ae')")
    _args = _ap.parse_args()
    main(competing_raw=_args.competing, chain_version=_args.chain_version)
