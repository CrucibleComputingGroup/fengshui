"""Generate Figure 2 (fig:dp1_memory) for the MICRO paper — heterogeneous memory motivation.

Canonical, reproducible generator extracted from architecture_insight.ipynb (cells 2-3).
Per-operator roofline analysis on a fixed 64x64 PE array @ 1 GHz (8-bit words, batch 1):
each operator is assigned the cheapest DRAM under which it remains compute-bound;
operators memory-bound even under HBM3E keep HBM3E, so latency is unchanged by
construction (asserted below).

The notebook's cost dict held placeholder *relative* factors (DDR5 1 / GDDR7 1.5 /
HBM3E 5.5), which do NOT reproduce the published caption range. The published
25.4--96.7% comes from market $/GB prices (sources cited in the caption:
wikipedia_hbm, wikipedia_lpddr, samsung_k4z80325bc_datasheet, jedec_hbm3_2022).
This script uses those prices and asserts the caption endpoints, then emits
  - the figure PDF/PNG (scripts/images/ + Overleaf src/motivation/images/)
  - src/constants/fig2_constants.tex (single source of truth for the paper numbers)

Run from chiplet_timeloop/scripts/ (reads ./workloads/<net>/):
    python generate_paper_fig2.py [--no-overleaf]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.legend_handler import HandlerBase
from matplotlib import rcParams

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)

from workload_parser import WorkloadParser, AcceleratorConfig, RooflineAnalysis  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
# Portable, repo-local defaults; override with FENGSHUI_FIG_OUT / FENGSHUI_CONST_OUT
# (our paper build points these at the Overleaf tree).
OVERLEAF_IMG_DIR = os.environ.get('FENGSHUI_FIG_OUT',
                                  os.path.join(_HERE, '..', 'figures'))
OVERLEAF_CONST = os.path.join(
    os.environ.get('FENGSHUI_CONST_OUT', os.path.join(_HERE, '..', 'constants')),
    'fig2_constants.tex')
FIG_NAME = 'insight1_final_visualization_correct_lines'

# Bandwidth (GB/s) and market price ($/GB); price sources are cited in the caption.
MEMORY_SPECS = {
    'DDR5':  {'bandwidth': 70.4,  'price': 3.3},
    'GDDR7': {'bandwidth': 320.0, 'price': 7.5},
    'HBM3E': {'bandwidth': 1229.0, 'price': 110.0},
}
# Cheapest-first downgrade order for compute-bound operators.
DOWNGRADE_ORDER = ['DDR5', 'GDDR7']

GPU_CONFIG = {'pe_array_size': 64, 'vector_array_size': 64, 'frequency_ghz': 1.0}
WORD_SIZE_BITS = 8
BATCH_SIZE = 1

NETWORKS = ['resnet50', 'efficientnet_b0', 'mobilenet_v3_small', 'replknet31b',
            'gpt_OPT-66B_prefill', 'gpt_OPT-66B_decode']
LABEL_MAP = {
    'resnet50': 'resnet',
    'efficientnet_b0': 'efficientnet',
    'mobilenet_v3_small': 'mobilenet',
    'replknet31b': 'replknet',
    'gpt_OPT-66B_prefill': 'gpt-66B\nprefill',
    'gpt_OPT-66B_decode': 'gpt-66B\ndecode',
}

# Published caption endpoints (sanity gate against silent drift).
EXPECTED_MIN_SAVINGS = 25.4
EXPECTED_MAX_SAVINGS = 96.7


def make_config(memory_name):
    spec = MEMORY_SPECS[memory_name]
    return AcceleratorConfig(
        pe_array_size=GPU_CONFIG['pe_array_size'],
        vector_array_size=GPU_CONFIG['vector_array_size'],
        pe_frequency_ghz=GPU_CONFIG['frequency_ghz'],
        dram_bandwidth_gbps=spec['bandwidth'],
        word_size=WORD_SIZE_BITS,
    )


def run_analysis():
    """Roofline every operator under each memory; assign cheapest compute-bound option."""
    parser = WorkloadParser(word_size=WORD_SIZE_BITS, batch_size=BATCH_SIZE)
    rooflines = {m: RooflineAnalysis(make_config(m)) for m in MEMORY_SPECS}

    per_layer = []
    for net in NETWORKS:
        parsed = parser.analyze_network(f'workloads/{net}')
        for layer_name, layer in parsed['layers'].items():
            analyses = {
                m: rooflines[m].analyze_layer(
                    macs=layer['macs'],
                    memory_bits=layer['memory_access_bits'],
                    layer_name=layer_name,
                    layer_type=layer['layer_type'],
                ) for m in MEMORY_SPECS
            }
            if analyses['HBM3E']['bottleneck'] == 'Compute Bound':
                chosen = 'HBM3E'
                for m in DOWNGRADE_ORDER:
                    if analyses[m]['bottleneck'] == 'Compute Bound':
                        chosen = m
                        break
            else:
                chosen = 'HBM3E'  # memory-bound: needs the bandwidth
            per_layer.append({
                'network': net,
                'layer': layer_name,
                'memory': chosen,
                'time_ms': analyses[chosen]['execution_time_ms'],
                'time_hbm_ms': analyses['HBM3E']['execution_time_ms'],
            })

    # Latency must be bit-identical to all-HBM3E (the figure's central claim).
    t_het = sum(l['time_ms'] for l in per_layer)
    t_hbm = sum(l['time_hbm_ms'] for l in per_layer)
    assert abs(t_het / t_hbm - 1.0) < 1e-9, f'latency changed: {t_het/t_hbm}'
    return per_layer


def summarize(per_layer):
    hbm_price = MEMORY_SPECS['HBM3E']['price']
    stats = {}
    for net in NETWORKS:
        layers = [l for l in per_layer if l['network'] == net]
        counts = {m: sum(1 for l in layers if l['memory'] == m) for m in MEMORY_SPECS}
        n = len(layers)
        cost = sum(MEMORY_SPECS[m]['price'] * c for m, c in counts.items())
        stats[net] = {
            'counts': counts,
            'n_layers': n,
            'cost_share': cost / (n * hbm_price),
            'savings_pct': 100.0 * (1.0 - cost / (n * hbm_price)),
        }
    return stats


class HomoGradient:
    pass


def plot(stats, out_paths):
    rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams.update({'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16,
                         'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14})

    nets = sorted(NETWORKS)  # match the published x-order (alphabetical)
    x = np.arange(len(nets)) / 4
    width = 0.1
    tab20 = plt.get_cmap('tab20').colors
    color_het = {'HBM3E': tab20[2], 'GDDR7': tab20[4], 'DDR5': tab20[6]}
    blues = ['#045a8d', '#0570b0', '#3690c0']
    hatch = {'HBM3E': '\\', 'GDDR7': 'x', 'DDR5': '/'}
    hbm_price = MEMORY_SPECS['HBM3E']['price']

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for i, net in enumerate(nets):
        s = stats[net]
        n = s['n_layers']
        frac = {m: s['counts'][m] / n for m in MEMORY_SPECS}
        cx = x[i] - width / 2
        xmin, xmax = cx - width / 2, cx + width / 2

        # Homogeneous bar (height 1.0), shaded by where the operators migrate.
        y1 = frac['HBM3E']
        y2 = y1 + frac['GDDR7']
        segs = [(0.0, y1, blues[0], 'HBM3E'), (y1, y2, blues[1], 'GDDR7'), (y2, 1.0, blues[2], 'DDR5')]
        for j, (lo, hi, color, mem) in enumerate(segs):
            if hi - lo <= 1e-12:
                continue
            ax.bar(cx, hi - lo, width, bottom=lo, color=color, align='center', zorder=1,
                   label=('Homogeneous (All HBM3E)' if (i == len(nets) - 1 and j == 0) else None))
            ax.add_patch(patches.Rectangle((xmin, lo), width, hi - lo, facecolor='none',
                                           hatch=hatch[mem], edgecolor='black', lw=0, zorder=3))
        for boundary in (y1, y2):
            if 1e-12 < boundary < 1.0 - 1e-12:
                ax.hlines(y=boundary, xmin=xmin, xmax=xmax, color='black',
                          linestyle='--', linewidth=1.2, zorder=2)

        # Heterogeneous bar: actual cost share by memory type.
        bottom = 0.0
        for mem in ['HBM3E', 'GDDR7', 'DDR5']:
            share = MEMORY_SPECS[mem]['price'] * s['counts'][mem] / (n * hbm_price)
            if share <= 1e-12:
                bottom += share
                continue
            ax.bar(x[i] + width / 2, share, width, bottom=bottom, color=color_het[mem],
                   hatch=hatch[mem], zorder=1,
                   label=(f'Hetero: {mem} Cost' if i == len(nets) - 1 else None))
            bottom += share
        ax.annotate(f'−{s["savings_pct"]:.1f}%', xy=(x[i] + width / 2, bottom),
                    xytext=(0, 4), textcoords='offset points', ha='center', fontsize=13)

    ax.set_ylabel('Average Memory Cost Factor')
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_MAP[n] for n in nets], rotation=0, ha='center')
    ax.set_ylim(0, 1.12)
    ax.grid(True, alpha=0.3)

    class HandlerHomoGradient(HandlerBase):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, w, h, fontsize, trans):
            seg = w / 3.0
            rects = [patches.Rectangle((xdescent + k * seg, ydescent), seg, h, transform=trans,
                                       facecolor=blues[k], edgecolor='none') for k in range(3)]
            rects.append(patches.Rectangle((xdescent, ydescent), w, h, transform=trans,
                                           facecolor='none', edgecolor='black', linewidth=1.0))
            return rects

    proxy = {
        'Homogeneous: GDDR7 (proxy)': patches.Patch(facecolor=blues[1], hatch='xx', edgecolor='black', linewidth=1.0),
        'Homogeneous: DDR5 (proxy)': patches.Patch(facecolor=blues[2], hatch='//', edgecolor='black', linewidth=1.0),
        'Hetero: HBM3E Cost': patches.Patch(facecolor=color_het['HBM3E'], hatch='\\\\', edgecolor='black', linewidth=1.0),
        'Hetero: GDDR7 Cost': patches.Patch(facecolor=color_het['GDDR7'], hatch='xx', edgecolor='black', linewidth=1.0),
        'Hetero: DDR5 Cost': patches.Patch(facecolor=color_het['DDR5'], hatch='//', edgecolor='black', linewidth=1.0),
    }
    labels = ['Homogeneous (All HBM3E)'] + list(proxy.keys())
    handles = [HomoGradient()] + list(proxy.values())
    leg = ax.legend(handles, labels, handler_map={HomoGradient: HandlerHomoGradient()},
                    loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False,
                    handlelength=2.2, columnspacing=1.2)
    plt.setp(leg.get_texts(), fontsize=14)

    for p in out_paths:
        fig.savefig(p, dpi=200, bbox_inches='tight')
        print(f'wrote {p}')
    plt.close(fig)


def write_constants(stats, path):
    sav = {net: stats[net]['savings_pct'] for net in NETWORKS}
    lines = [
        '% AUTO-GENERATED by chiplet_timeloop/scripts/generate_paper_fig2.py — do not edit by hand.',
        '% Figure 2 (fig:dp1_memory): per-operator roofline, cheapest compute-bound DRAM per operator.',
        f'\\newcommand{{\\FigTwoMemCostMin}}{{{min(sav.values()):.1f}}}',
        f'\\newcommand{{\\FigTwoMemCostMax}}{{{max(sav.values()):.1f}}}',
        f'\\newcommand{{\\FigTwoPriceDDR}}{{{MEMORY_SPECS["DDR5"]["price"]:.1f}}}',
        f'\\newcommand{{\\FigTwoPriceGDDR}}{{{MEMORY_SPECS["GDDR7"]["price"]:.1f}}}',
        f'\\newcommand{{\\FigTwoPriceHBM}}{{{MEMORY_SPECS["HBM3E"]["price"]:.0f}}}',
        f'\\newcommand{{\\FigTwoBWDDR}}{{{MEMORY_SPECS["DDR5"]["bandwidth"]:.1f}}}',
        f'\\newcommand{{\\FigTwoBWGDDR}}{{{MEMORY_SPECS["GDDR7"]["bandwidth"]:.0f}}}',
        f'\\newcommand{{\\FigTwoBWHBM}}{{{MEMORY_SPECS["HBM3E"]["bandwidth"]:.0f}}}',
        f'\\newcommand{{\\FigTwoOpCount}}{{{sum(stats[n]["n_layers"] for n in NETWORKS)}}}',
    ]
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'wrote {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-overleaf', action='store_true',
                    help='only write scripts/images/, skip the Overleaf copies')
    args = ap.parse_args()

    per_layer = run_analysis()
    stats = summarize(per_layer)

    print(f'{"network":25s} {"DDR5":>5s} {"GDDR7":>6s} {"HBM3E":>6s} {"savings":>9s}')
    for net in NETWORKS:
        s = stats[net]
        c = s['counts']
        print(f'{net:25s} {c["DDR5"]:5d} {c["GDDR7"]:6d} {c["HBM3E"]:6d} {s["savings_pct"]:8.1f}%')

    sav = [stats[n]['savings_pct'] for n in NETWORKS]
    lo, hi = min(sav), max(sav)
    assert abs(lo - EXPECTED_MIN_SAVINGS) < 0.05, f'min savings drifted: {lo:.2f} vs {EXPECTED_MIN_SAVINGS}'
    assert abs(hi - EXPECTED_MAX_SAVINGS) < 0.05, f'max savings drifted: {hi:.2f} vs {EXPECTED_MAX_SAVINGS}'
    print(f'range {lo:.1f}--{hi:.1f}% (matches published caption); latency ratio 1.000')

    os.makedirs('images', exist_ok=True)
    os.makedirs(OVERLEAF_IMG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OVERLEAF_CONST), exist_ok=True)
    outs = [f'images/{FIG_NAME}.pdf', f'images/{FIG_NAME}.png']
    if not args.no_overleaf:
        outs.append(os.path.join(OVERLEAF_IMG_DIR, f'{FIG_NAME}.pdf'))
    plot(stats, outs)
    if not args.no_overleaf:
        write_constants(stats, OVERLEAF_CONST)


if __name__ == '__main__':
    main()
