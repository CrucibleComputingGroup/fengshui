"""Figure 9 (fig:big_eval): the chiplet pool against GPU, homogeneous and unconstrained designs.

Four panels (energy, energy x cost, EDP, EDP x cost) over the 20 networks of Figure 8 and
their geometric mean, each bar normalized to the Homogeneous All Nets design of its network.
A cell is one (metric, network, paradigm):

  GPU                   measured bf16 runs, src/gpu/benchmarks_bf16/, at GPU_COST per GPU
  Homogeneous All Nets  one chiplet design for all 20 networks: of the 192 Timeloop designs
                        (eyeriss_like, simba_like, gemmini_like at every buffer and PE scale),
                        the one with the lowest geomean objective. GDDR7 at every stage
                        boundary, no larger batch for off-critical-path operators, inner GA
                        30 x 30
  Homogeneous Per Net   for each network, the best of the same 192 designs. Memory chosen per
                        boundary, off-critical-path operators batched, inner GA 10 x 10
  Chiplet Pool          the N = 8 pool of the chain search, mapped per network with the inner
                        GA 30 x 30; the pool comes from the chain CSV in
                        src/archgym_results/ae_<metric>_chain_converged/ (the directory must
                        hold exactly one saeo_isaeo_chain_*.csv)
  Unconstrained         the largest pool in that file, i.e. the chain's convergence endpoint,
                        mapped the same way

The inner GA is seeded per network (FENGSHUI_DETERMINISTIC=1, FENGSHUI_GA_SEED=0). The two
x cost panels multiply each design's cost-weighted objective by its own latency
(iso-throughput: the number of devices grows with latency).

Run from src/scripts/:
  python3 generate_paper_fig10.py
      Replot from the shipped cells, src/archgym_results/figure9/figure9_cells.csv (no
      database, seconds), and print the abstract's numbers as LaTeX macros.
  python3 generate_paper_fig10.py --recompute --database ../unified_database.csv --workers 16
      Run the 30,880 design-network evaluations behind the 400 cells from the database, the
      pools above and the GPU measurements (measured with the full database on 16 workers:
      22 minutes, about 5 CPU-hours, 48 GB of memory in all, about 3 GB per worker), then
      write figure9_cells.csv and selections.json in the shipped format to --out (default
      src/archgym_results/repro_figure9/, ignored by git), plot, print the macros and count the
      cells that equal the shipped ones. Exits with status 1 unless all 400 do. An interrupted
      run resumes from --out/records.jsonl.
"""
import os

# One BLAS/OpenMP thread per process: --recompute forks one worker per CPU. Set before numpy loads.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import contextlib
import csv
import glob
import hashlib
import json
import math
import multiprocessing as mp
import re
import sys
import time
import traceback

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.normpath(os.path.join(SCRIPTS_DIR, '..'))
ARCHGYM = os.path.join(SRC_DIR, 'archgym_results')
SHIPPED_CELLS = os.path.join(ARCHGYM, 'figure9')            # figure9_cells.csv, selections.json
RECOMPUTE_OUT = os.path.join(ARCHGYM, 'repro_figure9')      # --recompute default; git-ignored
# Figure output directory for the replot: src/figures/ unless FENGSHUI_FIG_OUT is set.
OUT_DIR = os.environ.get('FENGSHUI_FIG_OUT', os.path.join(SRC_DIR, 'figures'))
FIGURE_NAME = 'combined_comparison_norm_two_row_new'

METRICS = {"energy": ("energy", False), "energy_cost": ("energy", True),
           "edp": ("edp", False), "edp_cost": ("edp", True)}
PARADIGMS = ["GPU", "Homogeneous All Nets", "Homogeneous Per Net", "Chiplet Pool", "Unconstrained"]
HOMOGENEOUS_ARCHS = ["eyeriss_like", "simba_like", "gemmini_like"]
CELL_COLUMNS = ["metric", "network", "paradigm", "design",
                "figure_value", "normalized", "value", "latency_s"]

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

# ── GPU measurements ──────────────────────────────────────────────────────
# Per-layer bf16 measurements, one GPU per pipeline stage. A GPU cell uses the file's
# "Total Pipeline Energy with Idle + P2P (J)" and "Bottleneck Latency (ms)".
GPU_DIR = os.path.join(SCRIPTS_DIR, '..', 'gpu', 'benchmarks_bf16')
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


# ═══════════════════════════════════════════════════════════════════════════
# Plotting (broken y axis)
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
# Cells: one row per (metric, network, paradigm)
# ═══════════════════════════════════════════════════════════════════════════

def gm(values):
    """Geometric mean of positive, finite values."""
    assert values and all(v is not None and math.isfinite(v) and v > 0 for v in values)
    return math.exp(sum(math.log(v) for v in values) / len(values))


def cell_keys():
    return [(m, n, p) for m in METRICS for n in NETWORKS for p in PARADIGMS]


def write_cells(directory, cells, selections):
    """Write figure9_cells.csv (CELL_COLUMNS; numbers in Python's exact repr) and selections.json."""
    order = {key: i for i, key in enumerate(cell_keys())}
    keys = [(c["metric"], c["network"], c["paradigm"]) for c in cells]
    assert len(keys) == len(set(keys)) == len(order) and set(keys) == set(order), \
        "need exactly one cell per metric, network and paradigm"
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "figure9_cells.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CELL_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for c in sorted(cells, key=lambda c: order[c["metric"], c["network"], c["paradigm"]]):
            writer.writerow({k: "" if c[k] is None else c[k] for k in CELL_COLUMNS})
    with open(os.path.join(directory, "selections.json"), "w") as f:
        f.write(json.dumps(selections, indent=2, sort_keys=True) + "\n")


def read_cells(directory):
    """The cells of directory/figure9_cells.csv, numbers parsed back exactly."""
    path = os.path.join(directory, "figure9_cells.csv")
    with open(path, newline="") as f:
        cells = [{k: (float(v) if v != "" else None) if k in CELL_COLUMNS[4:] else v
                  for k, v in row.items()} for row in csv.DictReader(f)]
    keys = [(c["metric"], c["network"], c["paradigm"]) for c in cells]
    if len(keys) != len(set(keys)) or set(keys) != set(cell_keys()):
        raise ValueError(f"{path}: expected one row per metric, network and paradigm "
                         f"({len(cell_keys())} rows), got {len(keys)}")
    return cells


def claims(cells):
    """Per metric, the Chiplet Pool's reduction vs Homogeneous All Nets and its gap to
    Unconstrained, both in % of the geomean over the 20 networks."""
    cell = {(c["metric"], c["network"], c["paradigm"]): c for c in cells}
    reduction, gap = {}, {}
    for m in METRICS:
        reduction[m] = 100 * (1 - gm([cell[m, n, "Chiplet Pool"]["normalized"] for n in NETWORKS]))
        gap[m] = 100 * (gm([cell[m, n, "Chiplet Pool"]["figure_value"] / cell[m, n, "Unconstrained"]["figure_value"]
                            for n in NETWORKS]) - 1)
    return reduction, gap


def print_macros(cells):
    """The abstract's numbers, as the paper's LaTeX macros."""
    reduction, gap = claims(cells)
    gaps = [gap[m] for m in METRICS]
    print('\n=== Abstract numbers: Chiplet Pool vs Homogeneous All Nets, geomean over the 20 networks ===')
    print(f'  \\poolEnergyRed        = {reduction["energy"]:.1f}')
    print(f'  \\poolECRed            = {reduction["energy_cost"]:.1f}')
    print(f'  \\poolEDPRed           = {reduction["edp"]:.1f}')
    print(f'  \\poolEDPCRed          = {reduction["edp_cost"]:.1f}')
    print(f'  \\poolVsUnconGap       = {max(gaps):.1f}   (per-metric E/EC/EDP/EDPC: '
          + ', '.join(f'{g:.1f}' for g in gaps) + ')')


def plot(cells, out_dir):
    """The 2 x 2 figure (broken y axes), saved as PDF and PNG in out_dir."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 7.9))
    for ax, (metric, ylabel, bp, hidex) in zip(axes.flat, (
            ("energy", "Normalized Energy", 1.1, True),
            ("energy_cost", "Normalized Energy × Cost", 1.1, True),
            ("edp", "Normalized EDP", 1.1, False),
            ("edp_cost", "Normalized EDP × Cost", 1.5, False))):
        dd = {}
        for p in PARADIGMS:
            dd[p] = {c["network"]: {"min_energy": c["figure_value"]} for c in cells
                     if c["metric"] == metric and c["paradigm"] == p}
            dd[p]["geometric_mean"] = {"min_energy": gm([v["min_energy"] for v in dd[p].values()])}
        at, ab = make_broken_y_axes_in(ax, height_ratios=(1, 2.5), hspace=0.05)
        plot_all_energy_subplot_broken(
            at, ab, gpu_energy=dd[PARADIGMS[0]], homo_allnet_energy=dd[PARADIGMS[1]],
            homo_pernet_energy=dd[PARADIGMS[2]], chip_pool_energy=dd[PARADIGMS[3]],
            ideal_energy=dd[PARADIGMS[4]], ylabel=ylabel, break_point=bp,
            hide_x_ticks=hidex, top_margin=2.6)
    handles, labels = next(a.get_legend_handles_labels() for a in fig.axes
                           if a.get_legend_handles_labels()[1])
    plt.tight_layout(rect=[0, 0.05, 1, 0.965], w_pad=1.5, h_pad=1.0)
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.975),
               ncol=5, frameon=False, fontsize=14)
    os.makedirs(out_dir, exist_ok=True)
    paths = [os.path.join(out_dir, f'{FIGURE_NAME}.{ext}') for ext in ('pdf', 'png')]
    for path in paths:
        fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return paths


# ═══════════════════════════════════════════════════════════════════════════
# --recompute: evaluate every cell from the database
# ═══════════════════════════════════════════════════════════════════════════

G = {}  # evaluator modules, networks and candidate designs; set by boot(), inherited by the workers


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def pool_idents(row):
    """Chiplet identifiers of one row of a chain CSV."""
    return [f"{row[f'chiplet_{i}_arch']}@glb{int(float(row[f'chiplet_{i}_glb_scale']))}"
            f"@pe_x_scale{int(float(row[f'chiplet_{i}_pe_x_scale']))}"
            f"@pe_y_scale{int(float(row[f'chiplet_{i}_pe_y_scale']))}"
            for i in range(1, int(row["n_chiplets"]) + 1)]


def frozen_pools():
    """Per metric, the Chiplet Pool (N = 8) and Unconstrained (largest N) designs of the chain
    CSV in ae_<metric>_chain_converged/, which must hold exactly one saeo_isaeo_chain_*.csv."""
    pools = {}
    for metric in METRICS:
        d = os.path.join(ARCHGYM, f"ae_{metric}_chain_converged")
        choices = sorted(glob.glob(os.path.join(d, "saeo_isaeo_chain_*.csv")))
        if len(choices) != 1:
            raise RuntimeError(f"{os.path.relpath(d, SRC_DIR)} must hold exactly one saeo_isaeo_chain_*.csv; "
                               f"found {len(choices)}: {[os.path.basename(c) for c in choices]}")
        path = choices[0]
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        nmax = max(int(r["n_chiplets"]) for r in rows)
        pools[metric] = {}
        for kind, n in (("pool", 8), ("unconstrained", nmax)):
            row = next(r for r in rows if int(r["n_chiplets"]) == n)
            pools[metric][kind] = {"n": n, "chiplets": pool_idents(row),
                                   "source": os.path.relpath(path, SRC_DIR)}
    return pools


def boot(database):
    """Import the evaluator, load the 20 networks and preload the database (shared by fork)."""
    import cal_perf_phy_net as cp
    import chiplet_sel as cs
    import genetic_algo_opt_phy_net as ga
    import global_parameter as gp
    from baseline import _build_virtual_nets
    from chiplet_dataclass import ChipletConfig, generate_all_chiplet_configs
    print(f"Loading the 20 networks and the database {database} ...", flush=True)
    t0 = time.time()
    vnets = {v.get_unique_name(): v for v in _build_virtual_nets(database_file=database)}
    assert set(vnets) == set(NETWORKS), sorted(set(vnets) ^ set(NETWORKS))
    candidates = [c.get_identifier() for c in generate_all_chiplet_configs(arch_targets=HOMOGENEOUS_ARCHS)]
    assert len(candidates) == 192, len(candidates)
    print(f"Ready in {time.time() - t0:.0f}s: {len(vnets)} networks, {len(candidates)} homogeneous designs",
          flush=True)
    G.update(cp=cp, cs=cs, ga=ga, gp=gp, ChipletConfig=ChipletConfig, vnets=vnets,
             candidates=candidates, db=database)


def make_tasks(pools):
    """Per metric: each of the 192 homogeneous designs on each network twice ('allnet': GDDR7
    memory, no off-path batching, the candidates for Homogeneous All Nets; 'pernet': free
    memory and batching, the candidates for Homogeneous Per Net), and the two pools."""
    tasks = []
    for metric in METRICS:
        for kind in ("allnet", "pernet"):
            for chip in G["candidates"]:
                for net in NETWORKS:
                    tasks.append({"kind": kind, "metric": metric, "network": net, "chiplets": [chip]})
        for kind in ("pool", "unconstrained"):
            for net in NETWORKS:
                tasks.append({"kind": kind, "metric": metric, "network": net,
                              "chiplets": pools[metric][kind]["chiplets"]})
    return tasks


def task_key(t):
    return t["kind"], t["metric"], t["network"], tuple(t["chiplets"])


@contextlib.contextmanager
def homogenize_memory(dram_type):
    """Offer only dram_type at every stage boundary while the inner GA runs."""
    ga, cp, gp = G["ga"], G["cp"], G["gp"]
    saved = (ga.dram_options, cp.dram_options, gp.dram_options)
    ga.dram_options = [dram_type]
    cp.dram_options = [dram_type]
    gp.dram_options = [dram_type]
    try:
        yield
    finally:
        ga.dram_options, cp.dram_options, gp.dram_options = saved


def design(task):
    chiplets = [G["ChipletConfig"].from_identifier(c) for c in task["chiplets"]]
    if task["kind"] == "allnet":
        for c in chiplets:
            c.dram_type = "GDDR7"
    return chiplets


def evaluate(task):
    """Map one design onto one network with the inner GA; return the task's record."""
    start = time.time()
    try:
        obj, cost = METRICS[task["metric"]]
        vn = G["vnets"][task["network"]]
        ctx = homogenize_memory("GDDR7") if task["kind"] == "allnet" else contextlib.nullcontext()
        budget = "10" if task["kind"] == "pernet" else "30"
        os.environ.update(INNER_GA_POP=budget, INNER_GA_GEN=budget)
        try:
            with ctx:
                name, gene, value, config = G["cs"]._eval_single_net_dag(
                    (vn, design(task), obj, G["db"], cost, None, task["kind"] != "allnet"))
        finally:
            os.environ.update(INNER_GA_POP="30", INNER_GA_GEN="30")
        if not config or not math.isfinite(value):
            return {**task, "status": "infeasible", "seconds": time.time() - start}
        latency = float(config["latency"])
        funcs = [{k: getattr(f, k) for k in ("x1", "a", "b", "id")} for f in config["functions"]]
        # same design: the objective is the selected options' energy (times latency for EDP)
        expected = sum(f["a"] * latency + f["b"] for f in funcs)
        if obj == "edp":
            expected *= latency
        assert latency > 0 and value > 0 and math.isclose(value, expected, rel_tol=1e-10), (value, expected)
        return {**task, "status": "ok", "value": float(value), "latency_s": latency,
                "gene": gene, "functions": funcs, "seconds": time.time() - start}
    except Exception:
        return {**task, "status": "error", "traceback": traceback.format_exc(), "seconds": time.time() - start}


def clean(value):
    """JSON-safe copy: numpy scalars to Python, tuples to lists, non-finite floats to None."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return clean(value.item())
    return value


def compute(tasks, out, workers):
    """Evaluate the tasks on a fork pool, appending each record to out/records.jsonl as it
    finishes; records already in that file are reused."""
    path = os.path.join(out, "records.jsonl")
    done = {}
    if os.path.exists(path):
        with open(path, "rb+") as f:   # drop a last line cut short by an interrupted run
            data = f.read()
            f.truncate(data.rfind(b"\n") + 1)
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                done[task_key(r)] = r
    pending = [t for t in tasks if task_key(t) not in done]
    print(f"Tasks: {len(tasks)}; {len(tasks) - len(pending)} already in {path}; "
          f"{len(pending)} to evaluate on {workers} workers", flush=True)
    start = time.time()
    with mp.get_context("fork").Pool(workers) as pool, open(path, "a") as f:
        for i, result in enumerate(pool.imap_unordered(evaluate, pending, chunksize=1), 1):
            if result["status"] == "error":
                raise RuntimeError(f"{task_key(result)[:3]} failed:\n{result['traceback']}")
            line = json.dumps(clean(result), sort_keys=True, allow_nan=False)
            f.write(line + "\n")
            f.flush()
            done[task_key(result)] = json.loads(line)
            if i % 500 == 0 or i == len(pending):
                print(f"Evaluated {i}/{len(pending)} in {time.time() - start:.0f}s", flush=True)
    return [done[task_key(t)] for t in tasks]


def assemble(records):
    """The 400 cells and the homogeneous selections from the evaluated records."""
    rec = {task_key(r): r for r in records}
    pools = {(r["kind"], r["metric"], r["network"]): r for r in records
             if r["kind"] in ("pool", "unconstrained")}
    cells, selections = [], {}
    for metric, (obj, cost) in METRICS.items():
        allnet = []
        for chip in G["candidates"]:
            rs = [rec["allnet", metric, net, (chip,)] for net in NETWORKS]
            if all(r["status"] == "ok" for r in rs):
                allnet.append((gm([r["value"] for r in rs]), chip, rs))
        _, winner, wins = min(allnet, key=lambda x: (x[0], G["candidates"].index(x[1])))
        selections[metric] = {"allnet_design": winner, "pernet_designs": {}}
        for net in NETWORKS:
            pernet = min((rec["pernet", metric, net, (chip,)] for chip in G["candidates"]
                          if rec["pernet", metric, net, (chip,)]["status"] == "ok"),
                         key=lambda r: (r["value"], G["candidates"].index(r["chiplets"][0])))
            selections[metric]["pernet_designs"][net] = pernet["chiplets"][0]
            for paradigm, r in ((PARADIGMS[1], next(r for r in wins if r["network"] == net)),
                                (PARADIGMS[2], pernet),
                                (PARADIGMS[3], pools["pool", metric, net]),
                                (PARADIGMS[4], pools["unconstrained", metric, net])):
                assert r["status"] == "ok", (paradigm, metric, net, r["status"])
                cells.append({"metric": metric, "network": net, "paradigm": paradigm,
                              "design": "+".join(r["chiplets"]),
                              "figure_value": r["value"] * (r["latency_s"] if cost else 1.0),
                              "value": r["value"], "latency_s": r["latency_s"]})
            e, t_ms = _parse_gpu_file(os.path.join(GPU_DIR, GPU_FILE_MAP[net]))
            t = t_ms / 1000
            cells.append({"metric": metric, "network": net, "paradigm": PARADIGMS[0],
                          "design": os.path.join("gpu", "benchmarks_bf16", GPU_FILE_MAP[net]),
                          "figure_value": e * (t if obj == "edp" else 1) * (GPU_COST * t if cost else 1),
                          "value": None, "latency_s": t})
    base = {(c["metric"], c["network"]): c["figure_value"] for c in cells if c["paradigm"] == PARADIGMS[1]}
    for c in cells:
        c["normalized"] = c["figure_value"] / base[c["metric"], c["network"]]
        assert math.isfinite(c["normalized"]) and c["normalized"] > 0, c
    return cells, selections


def compare_to_shipped(cells):
    """Count the cells equal to the shipped ones (every field exactly); True if all are."""
    path = os.path.join(SHIPPED_CELLS, "figure9_cells.csv")
    if not os.path.exists(path):
        print(f"\nMISMATCH: no shipped cells to compare with ({os.path.relpath(path, SRC_DIR)} is missing)")
        return False
    shipped = {(c["metric"], c["network"], c["paradigm"]): c for c in read_cells(SHIPPED_CELLS)}
    differ = [c for c in cells if any(c[k] != shipped[c["metric"], c["network"], c["paradigm"]][k]
                                      for k in CELL_COLUMNS)]
    print(f"\n{len(cells) - len(differ)} of {len(cells)} cells equal the shipped cells "
          f"({os.path.relpath(path, SRC_DIR)})")
    if not differ:
        return True
    relative = [abs(c[k] - s[k]) / abs(s[k]) for c in differ
                for s in [shipped[c["metric"], c["network"], c["paradigm"]]]
                for k in CELL_COLUMNS[4:] if c[k] is not None and s[k] is not None and s[k] != 0]
    print(f"MISMATCH: {len(differ)} cells differ from the shipped cells; largest relative difference "
          f"of a numeric field: {max(relative, default=float('nan')):.3g}")
    for c in differ[:10]:
        s = shipped[c["metric"], c["network"], c["paradigm"]]
        fields = [k for k in CELL_COLUMNS if c[k] != s[k]]
        print(f"  differs: {c['metric']} {c['network']} {c['paradigm']} ({', '.join(fields)}): "
              f"figure_value {c['figure_value']!r} vs shipped {s['figure_value']!r}")
    return False


def recompute(database, workers, out):
    os.environ.update(FENGSHUI_DETERMINISTIC="1", FENGSHUI_GA_SEED="0",
                      INNER_GA_POP="30", INNER_GA_GEN="30")
    pools = frozen_pools()
    st = os.stat(database)
    manifest = {"database": {"path": database, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns},
                "pools": pools,
                "scripts_sha256": {os.path.basename(f): digest(f)
                                   for f in sorted(glob.glob(os.path.join(SCRIPTS_DIR, "*.py")))}}
    os.makedirs(out, exist_ok=True)
    mpath = os.path.join(out, "manifest.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            if json.load(f) != manifest:
                sys.exit(f"{out} holds records of other inputs (its manifest.json differs); "
                         f"pass another --out or delete it")
    else:
        with open(mpath, "w") as f:
            f.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for metric, p in pools.items():
        print(f"{metric}: Chiplet Pool N = {p['pool']['n']}, Unconstrained N = {p['unconstrained']['n']} "
              f"({p['pool']['source']})")
    os.chdir(SCRIPTS_DIR)  # the performance model reads network_analysis.csv by relative path
    boot(database)
    records = compute(make_tasks(pools), out, workers)
    cells, selections = assemble(records)
    write_cells(out, cells, selections)
    print(f"Wrote {os.path.join(out, 'figure9_cells.csv')} and selections.json")
    for path in plot(cells, out):
        print(f"Saved {path}")
    print_macros(cells)
    if not compare_to_shipped(cells):
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(
        description="Figure 9: replot from the shipped cells (default) or recompute every cell "
                    "from the database (--recompute). See the module docstring.")
    p.add_argument("--recompute", action="store_true",
                   help="evaluate every cell from the database instead of replotting the shipped cells")
    p.add_argument("--database", help="with --recompute (required): the performance database, "
                                      "e.g. ../unified_database.csv after tools/download_data.sh")
    p.add_argument("--workers", type=int, help="with --recompute (required): worker processes, one CPU each")
    p.add_argument("--out", help="with --recompute: output directory "
                                 "(default src/archgym_results/repro_figure9/, ignored by git; "
                                 "not the shipped src/archgym_results/figure9/)")
    a = p.parse_args()
    if not a.recompute:
        if a.database or a.workers or a.out:
            p.error("--database, --workers and --out go with --recompute")
        cells = read_cells(SHIPPED_CELLS)
        for path in plot(cells, OUT_DIR):
            print(f"Saved {path}")
        print_macros(cells)
        return
    if not a.database or not a.workers:
        p.error("--recompute needs --database and --workers")
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    if not 1 <= a.workers <= cpus:
        p.error(f"--workers must be between 1 and {cpus}")
    database = os.path.abspath(a.database)
    if not os.path.isfile(database):
        p.error(f"no database at {database} (bash tools/download_data.sh)")
    out = os.path.abspath(a.out or RECOMPUTE_OUT)
    if os.path.realpath(out) == os.path.realpath(SHIPPED_CELLS):
        p.error("--out must not be the shipped cells' directory "
                f"({os.path.relpath(SHIPPED_CELLS, SRC_DIR)}): the recomputed cells are compared with them")
    recompute(database, a.workers, out)


if __name__ == '__main__':
    main()
