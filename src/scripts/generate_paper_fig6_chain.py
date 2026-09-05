"""Generate Figure 8 (fig:pool_size) — chiplet pool optimization: SA vs SAEO+I-SAEO.

Reads the per-N incremental sweep CSVs and plots a 2x2 grid (four metrics) of the geometric
mean across all networks, normalized to N=1. The SA baseline uses the shipped SA runs
(`<SA_VER>_<metric>_sa/`) and the SAEO+I-SAEO curve uses the deterministic pool
(`<POOL_VER>_<metric>_chain/`). Paths and versions are env-overridable (see below).
"""
import os
import glob as globmod
import pandas as pd
import numpy as np
from scipy.stats import gmean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
})

SA_COLOR = '#2171b5'
CHAIN_COLOR = '#cb181d'

_HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get('FENGSHUI_ARCHGYM', os.path.join(_HERE, '..', 'archgym_results'))
OUT_DIR = os.environ.get('FENGSHUI_FIG_OUT', os.path.join(_HERE, '..', 'figures'))

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


def find_latest_csv(directory):
    """Find the most recent incremental_chiplet_sweep CSV or chain CSV in directory."""
    patterns = [
        os.path.join(directory, 'incremental_chiplet_sweep_*.csv'),
        os.path.join(directory, 'saeo_isaeo_chain_*.csv'),
    ]
    candidates = []
    for p in patterns:
        candidates.extend(globmod.glob(p))
    if not candidates:
        raise FileNotFoundError(f'No sweep CSV found in {directory}')
    return max(candidates, key=os.path.getmtime)


def compute_gmean_series(csv_path, objective):
    """Read a sweep CSV, compute geometric mean of per-network metric across N values."""
    df = pd.read_csv(csv_path)
    suffix = f'_min_{objective}'
    ns = df['n_chiplets'].values
    gmeans = []
    for _, row in df.iterrows():
        vals = []
        for net in NETWORKS:
            col = f'{net}{suffix}'
            if col in row and pd.notna(row[col]) and row[col] > 0:
                vals.append(row[col])
        gmeans.append(gmean(vals) if vals else np.nan)
    return ns, np.array(gmeans)


# Metric configs: (title, objective, cost_aware, sa_dir, chain_dir).
# SA baseline = shipped SA runs (default v7); SAEO+I-SAEO = the deterministic AE pool.
SA_VER   = os.environ.get('FENGSHUI_SA_VER', 'v7')
POOL_VER = os.environ.get('FENGSHUI_POOL_VER', 'ae')
METRICS = [
    ('Energy',                'energy', False, f'{SA_VER}_energy_sa',      f'{POOL_VER}_energy_chain'),
    ('Energy $\\times$ Cost', 'energy', True,  f'{SA_VER}_energy_cost_sa', f'{POOL_VER}_energy_cost_chain'),
    ('EDP',                   'edp',    False, f'{SA_VER}_edp_sa',         f'{POOL_VER}_edp_chain'),
    ('EDP $\\times$ Cost',    'edp',    True,  f'{SA_VER}_edp_cost_sa',    f'{POOL_VER}_edp_cost_chain'),
]

fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
axes = axes.flatten()

for idx, (title, objective, cost_aware, sa_dir, chain_dir) in enumerate(METRICS):
    ax = axes[idx]

    sa_csv = find_latest_csv(os.path.join(BASE, sa_dir))
    chain_csv = find_latest_csv(os.path.join(BASE, chain_dir))
    print(f'[{title}] SA:    {sa_csv}')
    print(f'[{title}] Chain: {chain_csv}')

    ns_sa, sa_vals = compute_gmean_series(sa_csv, objective)
    ns_chain, chain_vals = compute_gmean_series(chain_csv, objective)

    # Normalize to N=1
    sa_norm = sa_vals / sa_vals[0]
    chain_norm = chain_vals / chain_vals[0]

    ax.plot(ns_sa, sa_norm, '-o', color=SA_COLOR, label='SA',
            markersize=4, linewidth=1.5, markeredgewidth=0.5, markeredgecolor='white')
    ax.plot(ns_chain, chain_norm, '-s', color=CHAIN_COLOR, label='SAEO+I-SAEO',
            markersize=4, linewidth=1.5, markeredgewidth=0.5, markeredgecolor='white')

    # Dashed line at N=8
    ax.axvline(x=8, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.annotate('Our Chiplet\nPool', xy=(8, 0.5), xycoords=('data', 'axes fraction'),
                xytext=(-6, 0), textcoords='offset points',
                fontsize=7, fontstyle='italic', fontweight='bold',
                ha='right', va='center', color='#333333')

    ax.set_xlabel('Number of Chiplets')
    ax.set_ylabel('Normalized Value')
    ax.set_title(title)
    n_max = max(ns_sa.max(), ns_chain.max())
    ax.set_xticks(range(1, int(n_max) + 1))
    ax.set_xlim(0.5, n_max + 0.5)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', framealpha=0.9)

plt.tight_layout(h_pad=1.0, w_pad=0.8)
os.makedirs(OUT_DIR, exist_ok=True)

out_pdf = os.path.join(OUT_DIR, 'num_chiplet_sweep.pdf')
out_png = os.path.join(OUT_DIR, 'num_chiplet_sweep.png')
plt.savefig(out_pdf)
plt.savefig(out_png)
plt.close()
print(f'Saved {out_pdf}')
print(f'Saved {out_png}')
