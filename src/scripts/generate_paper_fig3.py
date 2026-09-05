"""Generate Figure 3 (fig:dp2_batching) for the MICRO paper — per-operator batch sensitivity.

Canonical, reproducible generator (reviewer D-Q4 / W7). v2 layout: left panel
latency-vs-batch, right panel throughput-vs-batch; four series each =
{prefill, decode} x {uniform batching, per-operator batching}. "Latency" is the
slowest pipeline stage of the transformer block (the pipeline interval), so the
operator story is visible as turning points: decode-uniform latency leaves its
flat region when the linearly-scaling attention operators overtake the
weight-amortizing projection/FFN stage; per-operator batching (attention
replicated at batch-1 latency) defers that turn until FFN itself turns
compute-bound.

Analysis: OPT-66B prefill/decode block operators, grouped into "FFN & projection"
(weight-sharing) and "attention calc" (batch-agnostic), roofline latency on a
fixed 128x128 PE array @ 1 GHz with GDDR7 (320 GB/s, 8-bit), batch 1-128.

Run from chiplet_timeloop/scripts/ (reads ./workloads/gpt_OPT-66B_*):
    python generate_paper_fig3.py [--no-overleaf]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    'fig3_constants.tex')
FIG_NAME = 'insight2_batch_sensitivity'

PE_ARRAY = 128            # 128x128 PEs
FREQ_GHZ = 1.0
GDDR7_BW_GBPS = 320.0
WORD_SIZE_BITS = 8
N_BLOCKS = 64             # OPT-66B decoder blocks; workload models one block
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
NETWORKS = {'gpt_OPT-66B_prefill': 'Prefill', 'gpt_OPT-66B_decode': 'Decode'}

# Operator groups (module -> workload layer names). "attention calc" has no weight
# reuse (batch-agnostic); "FFN & projection" amortizes weight streaming with batch.
MODULE_MAPPING = {
    'FFN & projection': ['layer0_1_v', 'layer0_2_q', 'layer0_3_k', 'layer7_o', 'layer8_ffn1', 'layer9_ffn2'],
    'attention calc': ['layer1_qk', 'layer6_av', 'layer2_max', 'layer3_sn', 'layer4_sd', 'layer5_a'],
}
BATCH_AGNOSTIC = {'attention calc'}


def run_analysis():
    config = AcceleratorConfig(
        pe_array_size=PE_ARRAY,
        vector_array_size=PE_ARRAY,
        pe_frequency_ghz=FREQ_GHZ,
        dram_bandwidth_gbps=GDDR7_BW_GBPS,
        word_size=WORD_SIZE_BITS,
    )
    roofline = RooflineAnalysis(config)

    results = {}  # net -> bs -> module -> summed latency (ms)
    for net in NETWORKS:
        results[net] = {}
        for bs in BATCH_SIZES:
            parser = WorkloadParser(word_size=WORD_SIZE_BITS, batch_size=bs)
            data = parser.analyze_network(f'workloads/{net}')
            per_layer = {
                name: roofline.analyze_layer(
                    macs=l['macs'], memory_bits=l['memory_access_bits'],
                    layer_name=name, layer_type=l['layer_type'],
                )['execution_time_ms']
                for name, l in data['layers'].items()
            }
            results[net][bs] = {
                module: sum(per_layer[l] for l in layers if l in per_layer)
                for module, layers in MODULE_MAPPING.items()
            }
    return results


def latencies(results, net):
    """Slowest-pipeline-stage latency (ms) under both batching schemes."""
    uniform, per_op = [], []
    for bs in BATCH_SIZES:
        stage = results[net][bs]
        uniform.append(max(stage.values()))
        adjusted = {m: (results[net][1][m] if m in BATCH_AGNOSTIC else t) for m, t in stage.items()}
        per_op.append(max(adjusted.values()))
    return uniform, per_op


def throughputs(results, net):
    """Pipeline tokens/sec (prefill: prompts/sec) under both batching schemes."""
    lat_u, lat_p = latencies(results, net)
    uniform = [bs * 1000.0 / N_BLOCKS / l for bs, l in zip(BATCH_SIZES, lat_u)]
    per_op = [bs * 1000.0 / N_BLOCKS / l for bs, l in zip(BATCH_SIZES, lat_p)]
    return uniform, per_op


def plot(results, out_paths):
    plt.rcParams.update({
        'font.family': 'DejaVu Serif',
        'font.size': 15, 'axes.titlesize': 17, 'axes.labelsize': 16,
        'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 13.5,
    })
    fig, (ax_lat, ax_thr) = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)

    phase_color = {'gpt_OPT-66B_prefill': '#ff7f0e', 'gpt_OPT-66B_decode': '#1f77b4'}
    styles = {'uniform': dict(linestyle='-', marker='o'),
              'per_op': dict(linestyle='--', marker='v')}

    handles, labels = [], []
    for net, phase in NETWORKS.items():
        c = phase_color[net]
        lat_u, lat_p = latencies(results, net)
        thr_u, thr_p = throughputs(results, net)
        for sched, lat, thr in [('uniform', lat_u, thr_u), ('per_op', lat_p, thr_p)]:
            h, = ax_lat.plot(BATCH_SIZES, lat, color=c, linewidth=2, markersize=5.5, **styles[sched])
            ax_thr.plot(BATCH_SIZES, thr, color=c, linewidth=2, markersize=5.5, **styles[sched])
            handles.append(h)
            labels.append(f'{phase}, {"uniform" if sched == "uniform" else "per-operator"} batching')

    for ax, ylab in [(ax_lat, 'Pipeline stage latency (ms)'), (ax_thr, 'Throughput (Tokens/sec)')]:
        ax.set_xlabel('Batch Size')
        ax.set_ylabel(ylab)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks(BATCH_SIZES)
        ax.set_xticklabels([str(b) for b in BATCH_SIZES])
        ax.grid(True, alpha=0.3)

    # Annotations: the decode-uniform kink (attention overtakes FFN) and the
    # resulting throughput gap at batch 128.
    dec_lat_u, _ = latencies(results, 'gpt_OPT-66B_decode')
    dec_thr_u, dec_thr_p = throughputs(results, 'gpt_OPT-66B_decode')
    kink = BATCH_SIZES.index(32)
    ax_lat.annotate('attention becomes\nthe slowest stage',
                    xy=(BATCH_SIZES[kink], dec_lat_u[kink]), xytext=(-8, 52),
                    textcoords='offset points', ha='center', va='bottom', fontsize=12.5,
                    arrowprops=dict(arrowstyle='->', lw=1.2, shrinkA=0, shrinkB=2))
    gain = dec_thr_p[-1] / dec_thr_u[-1]
    ax_thr.annotate('', xy=(BATCH_SIZES[-1], dec_thr_p[-1]), xytext=(BATCH_SIZES[-1], dec_thr_u[-1]),
                    arrowprops=dict(arrowstyle='<->', lw=1.3, shrinkA=2, shrinkB=2))
    ax_thr.annotate(f'{gain:.1f}×', xy=(BATCH_SIZES[-1], np.sqrt(dec_thr_p[-1] * dec_thr_u[-1])),
                    xytext=(-6, 0), textcoords='offset points', ha='right', va='center', fontsize=13)
    ax_thr.annotate('uniform saturates',
                    xy=(BATCH_SIZES[5], dec_thr_u[5]), xytext=(10, -38),
                    textcoords='offset points', ha='left', va='top', fontsize=12.5,
                    arrowprops=dict(arrowstyle='->', lw=1.2, shrinkA=0, shrinkB=2))

    fig.legend(handles, labels, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.14), frameon=False, columnspacing=1.8)

    for p in out_paths:
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f'wrote {p}')
    plt.close(fig)


def write_constants(results, path):
    pre_u, pre_p = throughputs(results, 'gpt_OPT-66B_prefill')
    dec_u, dec_p = throughputs(results, 'gpt_OPT-66B_decode')
    div = next(j for j in range(len(BATCH_SIZES)) if dec_p[j] / dec_u[j] > 1.05)
    lines = [
        '% AUTO-GENERATED by chiplet_timeloop/scripts/generate_paper_fig3.py — do not edit by hand.',
        '% Figure 3 (fig:dp2_batching): per-operator batch sensitivity, OPT-66B on 128x128 PE + GDDR7.',
        f'\\newcommand{{\\FigThreeDivergeBatch}}{{{BATCH_SIZES[div - 1]}}}',
        f'\\newcommand{{\\FigThreeDecodeGain}}{{{dec_p[-1] / dec_u[-1]:.1f}}}',
        f'\\newcommand{{\\FigThreePrefillGain}}{{{pre_p[-1] / pre_u[-1]:.2f}}}',
    ]
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'wrote {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-overleaf', action='store_true')
    args = ap.parse_args()

    results = run_analysis()

    pre_u, pre_p = throughputs(results, 'gpt_OPT-66B_prefill')
    dec_u, dec_p = throughputs(results, 'gpt_OPT-66B_decode')
    lat_du, lat_dp = latencies(results, 'gpt_OPT-66B_decode')
    print(f'{"bs":>4s} {"dec_lat_u":>10s} {"dec_lat_p":>10s} {"dec_thr_u":>10s} {"dec_thr_p":>10s} {"ratio":>6s}')
    for j, bs in enumerate(BATCH_SIZES):
        print(f'{bs:4d} {lat_du[j]:10.2f} {lat_dp[j]:10.2f} {dec_u[j]:10.2f} {dec_p[j]:10.2f} {dec_p[j]/dec_u[j]:6.2f}')

    # Prefill: per-operator batching must coincide with uniform batching.
    assert all(abs(p / u - 1.0) < 0.01 for p, u in zip(pre_p, pre_u)), 'prefill curves diverged'
    # Decode: per-operator batching must eventually beat uniform batching.
    assert dec_p[-1] / dec_u[-1] > 1.5, f'decode gain too small: {dec_p[-1]/dec_u[-1]:.2f}'

    os.makedirs('images', exist_ok=True)
    os.makedirs(OVERLEAF_IMG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OVERLEAF_CONST), exist_ok=True)
    outs = [f'images/{FIG_NAME}.pdf', f'images/{FIG_NAME}.png']
    if not args.no_overleaf:
        outs.append(os.path.join(OVERLEAF_IMG_DIR, f'{FIG_NAME}.pdf'))
    plot(results, outs)
    if not args.no_overleaf:
        write_constants(results, OVERLEAF_CONST)


if __name__ == '__main__':
    main()
