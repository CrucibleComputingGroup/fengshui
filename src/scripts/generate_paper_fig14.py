#!/usr/bin/env python3
"""Generate Figure 14: Vision DET normalized energy×cost and energy.

Homo baseline: pick the best single NON-PIM chiplet from the v7 hetero pool
(same pool used by hetero). Evaluate each candidate across ALL vision workloads
with GA(30,30), take the chiplet with lowest aggregate. This ensures homo uses
the same design space as hetero but restricted to one chiplet type.

Hetero (Mozart): extract n=8 chiplet pool (including PIM) from v7_*_chain CSVs,
run GA(30,30) with 5+1 trials per workload (1 seeded with homo solution +
5 random-init), take best.

All vision workloads are evaluated through the GA directly.
Random seed is fixed (SEED=42) for reproducibility.

Usage:
    cd scripts/
    python generate_paper_fig14.py
"""
import argparse
import os
import sys
import glob as globmod
import time
import random
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Fix random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib import rcParams

# ---- Style matching original anon.py ----
rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams.update({
    'font.size': 24, 'axes.titlesize': 24, 'axes.labelsize': 24,
    'xtick.labelsize': 24, 'ytick.labelsize': 24, 'legend.fontsize': 24,
    'font.weight': 500, 'axes.titleweight': 500, 'axes.labelweight': 500,
})
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = [
    'Noto Serif CJK SC', 'STIXGeneral', 'Times New Roman', 'DejaVu Serif']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['axes.formatter.use_mathtext'] = True
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'STIXGeneral'
mpl.rcParams['mathtext.it'] = 'STIXGeneral:italic'
mpl.rcParams['mathtext.bf'] = 'STIXGeneral:bold'

# ── Paths ──────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.normpath(os.path.join(THIS_DIR, '..'))
BASE = os.path.join(PROJECT_DIR, 'archgym_results')
# CORRECTED canonical root DB (DRAM double-count + ViT-softmax fixed, 2026-06-15).
# NOT timeloop_experiments/unified_database.csv (the pre-fix buggy snapshot).
DATABASE = os.path.join(PROJECT_DIR, 'unified_database.csv')
VIT_PIM_DATABASE = os.path.join(PROJECT_DIR, 'timeloop_experiments', 'vit_pim_database.csv')
NET_DIR = os.path.join(PROJECT_DIR, 'workloads')
# Repo-relative by default; the paper's Overleaf tree is not part of this repo.
# (The edge-AV figure itself was dropped from the camera-ready -- the case study is
# prose-only -- but this script still emits RESULTS_CSV, which backs the macros.)
DEFAULT_OUT = os.environ.get(
    'FENGSHUI_FIG_OUT',
    os.path.join(PROJECT_DIR, 'figures', 'comparison_plot_automibile.pdf'))
# Per-network results CSV: single source of truth for the AV case-study macros
# (\AVEnergyRed / \AVECRed). build_case_study_constants.py reads this so the
# paper numbers can never silently drift from the compute again.
RESULTS_CSV = os.path.join(PROJECT_DIR, 'case_study', 'av_vision_results.csv')

# v7 = corrected-DB pool (energy double-count + ViT-softmax fixed, 2026-06-15),
# NOTE: v7 is intentional here. The shipped av_vision_results.csv -- the single
# source of truth for the paper's \AVEnergyRed / \AVECRed macros -- was produced
# on the v7 pool, so this script must read v7 to reproduce those numbers.
# generate_paper_fig10.py moved to CHAIN_VERSION='ae' for the main results;
# this case study was not regenerated. Override with FENGSHUI_CHAIN_VERSION.
_CHAIN_VER = os.environ.get('FENGSHUI_CHAIN_VERSION', 'v7')
V7_ENERGY_DIR = os.path.join(BASE, f'{_CHAIN_VER}_energy_chain')
V7_ENERGY_COST_DIR = os.path.join(BASE, f'{_CHAIN_VER}_energy_cost_chain')

# All vision workloads: (net_name, display_name, seq_len, is_vit)
ALL_WORKLOADS = [
    ('mobilenet_v3_small', 'MobileNet', 1, False),
    ('replknet31b',        'RepLKNet',  1, False),
    ('vit_l16_s197',       'ViT-L/16',  1, True),
    ('vit_h14_s257',       'ViT-H/14',  1, True),
]

DET_DEADLINES_MS = [10, 33]
NETWORK_ORDER = ['MobileNet', 'RepLKNet', 'ViT-L/16', 'ViT-H/14']

# GA parameters
GA_POP = 30
GA_GEN = 30
N_TRIALS_HETERO = 5

# ── Brute-force homo search space (non-PIM only) ──
HOMO_ARCHS = ['eyeriss_like', 'simba_like', 'gemmini_like']
HOMO_GLB_SCALES = [1, 4, 9, 16]
HOMO_PE_SCALES = [1, 2, 3, 4]


def _find_latest_csv(directory, pattern_prefix='saeo_isaeo_chain_'):
    for prefix in [pattern_prefix, 'incremental_chiplet_sweep_']:
        pattern = os.path.join(directory, f'{prefix}*.csv')
        candidates = globmod.glob(pattern)
        if candidates:
            return max(candidates, key=os.path.getmtime)
    raise FileNotFoundError(f'No sweep/chain CSV found in {directory}')


def _build_database_for_workload(net_name, is_vit):
    """Return the correct database path for a workload.
    ViT needs merged unified + PIM database; CNN uses unified directly.
    """
    if not is_vit:
        return DATABASE
    # Build merged ViT database (cached). Invalidate the cache when either source
    # DB is newer than it — otherwise a stale merge silently masks DB fixes (e.g.
    # the 2026-06-14 ViT fused-softmax rows would be dropped by a pre-fix cache).
    import tempfile
    # Encode the source DB's mtime in the cache name so an updated/corrected DB can
    # never be silently masked by a stale merge (the 2026-06-15 DRAM-fix landmine:
    # the old cache was newer than the buggy source, so invalidation never fired).
    _dbsig = int(os.path.getmtime(DATABASE))
    merged_path = os.path.join(tempfile.gettempdir(), f'vit_merged_database_{_dbsig}.csv')
    _srcs = [DATABASE] + ([VIT_PIM_DATABASE] if os.path.exists(VIT_PIM_DATABASE) else [])
    if (os.path.exists(merged_path) and
            os.path.getmtime(merged_path) >= max(os.path.getmtime(s) for s in _srcs)):
        return merged_path

    import csv as csv_mod
    vit_nets = {wl[0] for wl in ALL_WORKLOADS if wl[3]}
    with open(merged_path, 'w', newline='') as out_f:
        with open(DATABASE) as in_f:
            reader = csv_mod.DictReader(in_f)
            writer = csv_mod.DictWriter(out_f, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if row['net'] in vit_nets:
                    writer.writerow(row)
        if os.path.exists(VIT_PIM_DATABASE):
            with open(VIT_PIM_DATABASE) as pim_f:
                for row in csv_mod.DictReader(pim_f):
                    writer.writerow(row)
    n_lines = sum(1 for _ in open(merged_path)) - 1
    print(f"  Merged ViT database: {n_lines} rows")
    return merged_path


def _run_ga_single(vn, chiplet_list, results_file, cost_aware,
                   pop=GA_POP, gen=GA_GEN, prev_best_gene=None):
    """Run GA once, return (best_value, best_gene). Returns (inf, None) on failure."""
    from genetic_algo_opt_phy_net import genetic_algo_opt_phy_net
    import global_parameter
    try:
        gene, val, _ = genetic_algo_opt_phy_net(
            vn, chiplet_list, objective="energy",
            population_size=pop, generations=gen,
            query_points=global_parameter.base_query_points,
            results_file=results_file, cost_aware=cost_aware,
            use_sequential=True, prev_best_gene=prev_best_gene)
        return val, gene
    except Exception:
        return float('inf'), None


def bruteforce_homo(vn, results_file, cost_aware):
    """Brute-force all 192 non-PIM single-chiplet configs.
    Returns (best_energy, best_chip_label, best_chiplet_config, best_gene)."""
    from chiplet_dataclass import ChipletConfig
    best_val = float('inf')
    best_chip = None
    best_chiplet = None
    best_gene = None

    for arch in HOMO_ARCHS:
        for glb in HOMO_GLB_SCALES:
            for px in HOMO_PE_SCALES:
                for py in HOMO_PE_SCALES:
                    chiplet = ChipletConfig(
                        arch_target=arch,
                        global_buffer_size_scale=glb,
                        pe_x_scale=px, pe_y_scale=py)
                    val, gene = _run_ga_single(vn, [chiplet], results_file, cost_aware)
                    if val < best_val:
                        best_val = val
                        best_chip = f"{arch} glb={glb} px={px} py={py}"
                        best_chiplet = chiplet
                        best_gene = gene

    return best_val, best_chip, best_chiplet, best_gene


def _make_hetero_seed_gene(homo_chiplet, homo_gene, hetero_pool, num_layers):
    """Create a seed gene for the hetero GA from the homo solution.
    Maps the homo chiplet to its index in the hetero pool,
    sets all layers to that chiplet, and copies the DRAM config.
    """
    # Find the homo chiplet's index in the hetero pool
    target_idx = None
    for i, c in enumerate(hetero_pool):
        if (c.arch_target == homo_chiplet.arch_target and
            c.global_buffer_size_scale == homo_chiplet.global_buffer_size_scale and
            c.pe_x_scale == homo_chiplet.pe_x_scale and
            c.pe_y_scale == homo_chiplet.pe_y_scale):
            target_idx = i
            break

    if target_idx is None:
        # Homo chiplet not in hetero pool; can't seed directly
        return None

    seed = {
        'binary_string': str(target_idx) * num_layers,
        'buffer_config': homo_gene.get('buffer_config', ['HBM3'] * (num_layers + 1)),
    }
    return seed


def evaluate_hetero(vn, pool, results_file, cost_aware,
                    homo_val, homo_chiplet, homo_gene):
    """Run GA with full hetero pool. Seeds with homo solution to guarantee
    hetero >= homo, plus N_TRIALS_HETERO additional random-init trials."""
    # Start with homo as the floor
    best_val = homo_val

    # Trial 0: seeded with homo solution
    seed_gene = _make_hetero_seed_gene(
        homo_chiplet, homo_gene, pool, len(vn.layers))
    val, _ = _run_ga_single(vn, pool, results_file, cost_aware,
                            prev_best_gene=seed_gene)
    if val < best_val:
        best_val = val

    # Additional random-init trials
    for trial in range(N_TRIALS_HETERO):
        val, _ = _run_ga_single(vn, pool, results_file, cost_aware)
        if val < best_val:
            best_val = val

    return best_val


def evaluate_all_workloads(pool, cost_aware, label):
    """Evaluate all vision workloads with correct homo baseline.

    Homo: brute-force all 192 non-PIM chiplets. For EACH chiplet, evaluate
    ALL workloads and sum energy. Pick the ONE chiplet with lowest total.
    Then report that chiplet's per-workload energy as the homo baseline.

    Hetero: for each workload, run GA with the full pool (seeded with homo).
    """
    from network_dataclass import VirtualNetwork
    from chiplet_dataclass import ChipletConfig

    # ── Step 1: Load all workloads ──
    workload_info = []  # (vn, db_file, display_name)
    for net_name, display_name, seq_len, is_vit in ALL_WORKLOADS:
        db_file = _build_database_for_workload(net_name, is_vit)
        _db = pd.read_csv(db_file)
        db_layers = set(_db[_db['net'] == net_name]['layer_name'].unique())
        vn = VirtualNetwork(net_name, batch_size=1, sequence_length=seq_len)
        vn.load_from_dir(os.path.join(NET_DIR, net_name), db_layers=db_layers)
        if len(vn.layers) == 0:
            print(f"    WARNING: No layers for {net_name}, skipping")
            continue
        workload_info.append((vn, db_file, display_name))
        print(f"  Loaded {display_name} ({net_name}, {len(vn.layers)} layers)")

    # ── Step 2: Homo — best single non-PIM chiplet from the SAME pool ──
    # Try each non-PIM chiplet in the pool across ALL workloads, pick best total
    homo_candidates = [c for c in pool
                       if c.arch_target in ('eyeriss_like', 'gemmini_like',
                                            'simba_like')]
    print(f"\n  [{label}] Homo: testing {len(homo_candidates)} non-PIM chiplets "
          f"from pool across {len(workload_info)} workloads...")
    t0 = time.time()

    best_total = float('inf')
    best_homo_chiplet = None
    best_homo_label = None
    best_homo_per_wl = {}

    for chiplet in homo_candidates:
        total = 0.0
        per_wl = {}
        feasible = True
        for vn, db_file, display_name in workload_info:
            val, gene = _run_ga_single(
                vn, [chiplet], db_file, cost_aware)
            if val == float('inf'):
                feasible = False
                break
            total += val
            per_wl[display_name] = (val, gene)

        if feasible and total < best_total:
            best_total = total
            best_homo_chiplet = chiplet
            best_homo_label = (f"{chiplet.arch_target} glb="
                               f"{chiplet.global_buffer_size_scale} "
                               f"px={chiplet.pe_x_scale} "
                               f"py={chiplet.pe_y_scale}")
            best_homo_per_wl = per_wl

    t_homo = time.time() - t0
    print(f"  Best homo chiplet: {best_homo_label} "
          f"(total={best_total:.6e}) [{t_homo:.1f}s]")
    for dn, (val, _) in best_homo_per_wl.items():
        print(f"    {dn}: {val:.6e}")

    # ── Step 3: Hetero — per-workload GA, seeded with homo ──
    pool_filtered = [c for c in pool
                     if c.arch_target in ('eyeriss_like', 'gemmini_like',
                                          'simba_like', 'PIM')]

    results = []
    for vn, db_file, display_name in workload_info:
        homo_val, homo_gene = best_homo_per_wl.get(
            display_name, (float('inf'), None))

        t1 = time.time()
        hetero_val = evaluate_hetero(
            vn, pool_filtered, db_file, cost_aware,
            homo_val, best_homo_chiplet, homo_gene)
        t_hetero = time.time() - t1

        ratio = hetero_val / homo_val if homo_val > 0 else float('inf')
        print(f"  [{label}] {display_name}: "
              f"Homo={homo_val:.6e}, Hetero={hetero_val:.6e} "
              f"(ratio={ratio:.4f}) [{t_hetero:.1f}s]")

        results.append({
            'network': display_name,
            'homo_energy': homo_val,
            'hetero_energy': hetero_val,
        })

    return results


def plot_figure14(out_path):
    energy_csv = _find_latest_csv(V7_ENERGY_DIR)
    energy_cost_csv = _find_latest_csv(V7_ENERGY_COST_DIR)
    print(f'Energy chain:      {energy_csv}')
    print(f'Energy×Cost chain: {energy_cost_csv}')

    # Extract hetero pools
    from chiplet_dataclass import ChipletConfig
    pool_energy = ChipletConfig.from_csv_for_n_chiplets(8, energy_csv)
    pool_cost = ChipletConfig.from_csv_for_n_chiplets(8, energy_cost_csv)

    print(f"\nEnergy pool ({len(pool_energy)} chiplets):")
    for c in pool_energy:
        print(f"  {c.arch_target} glb={c.global_buffer_size_scale} "
              f"pe_x={c.pe_x_scale} pe_y={c.pe_y_scale}")
    print(f"\nEnergy×Cost pool ({len(pool_cost)} chiplets):")
    for c in pool_cost:
        print(f"  {c.arch_target} glb={c.global_buffer_size_scale} "
              f"pe_x={c.pe_x_scale} pe_y={c.pe_y_scale}")

    # Build merged ViT database once
    print("\nBuilding merged ViT database...")
    _build_database_for_workload('vit_b16_s197', True)

    # Evaluate: Energy (cost_aware=False)
    print("\n" + "=" * 60)
    print("ENERGY (cost_aware=False)")
    print("=" * 60)
    energy_data = evaluate_all_workloads(pool_energy, False, "Energy")

    # Evaluate: Energy×Cost (cost_aware=True)
    print("\n" + "=" * 60)
    print("ENERGY × COST (cost_aware=True)")
    print("=" * 60)
    energy_cost_data = evaluate_all_workloads(pool_cost, True, "Energy×Cost")

    # ── Write per-network results CSV (single source of truth for paper macros) ──
    import csv as _csv
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, 'w', newline='') as _fh:
        _w = _csv.writer(_fh)
        _w.writerow(['net', 'cost_aware', 'homo', 'hetero', 'ratio', 'reduction_pct'])
        for _data, _aware in [(energy_data, False), (energy_cost_data, True)]:
            for _d in _data:
                _homo, _het = _d['homo_energy'], _d['hetero_energy']
                if _homo in (0, float('inf')) or _het == float('inf'):
                    continue
                _ratio = _het / _homo
                _w.writerow([_d['network'], _aware, f'{_homo:.6e}', f'{_het:.6e}',
                             f'{_ratio:.6f}', f'{(1.0 - _ratio) * 100:.4f}'])
    print(f"\nResults CSV saved to {RESULTS_CSV}")

    # ── Build flat DataFrame ──
    rows = []
    for data, cost_aware in [(energy_cost_data, True), (energy_data, False)]:
        for d in data:
            if d['homo_energy'] == float('inf') or d['hetero_energy'] == float('inf'):
                print(f"WARNING: Skipping {d['network']} (inf energy)")
                continue
            for deadline_ms in DET_DEADLINES_MS:
                rows.append({
                    'network_type': d['network'], 'cost_aware': cost_aware,
                    'latency_requirment': deadline_ms / 1000.0,
                    'homogenous': True, 'best_value': d['homo_energy'],
                })
                rows.append({
                    'network_type': d['network'], 'cost_aware': cost_aware,
                    'latency_requirment': deadline_ms / 1000.0,
                    'homogenous': False, 'best_value': d['hetero_energy'],
                })

    df = pd.DataFrame(rows)

    # Normalize to Homo baseline
    keys = ['network_type', 'cost_aware', 'latency_requirment']
    homo_ref = (
        df[df['homogenous'] == True][keys + ['best_value']]
        .rename(columns={'best_value': '_homo_ref'})
    )
    df = df.merge(homo_ref, on=keys, how='left')
    df['_homo_ref'] = df['_homo_ref'].replace(0, np.nan)
    df['plot_value'] = np.where(df['homogenous'] == True, 1.0,
                                df['best_value'] / df['_homo_ref'])
    df = df.dropna(subset=['plot_value'])

    # Group labels
    df['latency_ms'] = (df['latency_requirment'] * 1000).round().astype('Int64')
    df['homogeneity'] = df['homogenous'].map({True: 'Homo', False: 'Mozart'})
    df['group'] = (
        df['homogeneity'] + ' (e2e latency $\\leq$ ' +
        df['latency_ms'].astype(str) + ' ms)')

    wanted_latencies = sorted(df['latency_requirment'].unique())
    type_order = [n for n in NETWORK_ORDER if n in df['network_type'].values]
    group_order = []
    for L in wanted_latencies:
        ms = int(round(L * 1000))
        group_order += [f'Homo (e2e latency $\\leq$ {ms} ms)',
                        f'Mozart (e2e latency $\\leq$ {ms} ms)']

    sns.set_palette("Set2")
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    plot_cfg = [
        (axes[0], True, 'Normalized Energy × Cost'),
        (axes[1], False, 'Normalized Energy'),
    ]
    handles_all, labels_all = None, None

    for ax, aware, ylabel in plot_cfg:
        data = df[df['cost_aware'] == aware].copy()
        if data.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=24)
            ax.axis('off')
            continue

        sns.barplot(data=data, x='network_type', y='plot_value', hue='group',
                    order=type_order, hue_order=group_order,
                    ax=ax, dodge=True, edgecolor='none', linewidth=0)
        ax.axhline(1.0, linewidth=1, linestyle=':', alpha=0.6)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        ax.yaxis.get_offset_text().set_visible(False)
        ax.set_xlabel(None)
        ax.set_ylabel(ylabel, fontsize=24)
        ax.tick_params(axis='x', rotation=0)
        ax.tick_params(axis='y', labelrotation=45, labelsize=18)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        mozart = data[data['homogenous'] == False]
        drops = []
        for (net, lat), grp in mozart.groupby(['network_type', 'latency_requirment']):
            ref = grp['_homo_ref'].iloc[0]
            val = grp['best_value'].iloc[0]
            if ref > 0 and not np.isnan(ref):
                drops.append((ref - val) / ref)
        metric = 'ENERGY × COST' if aware else 'ENERGY'
        if drops:
            print(f"[{metric}] Mozart AVG DECREASE: {np.mean(drops)*100:.2f}%")

        if handles_all is None:
            handles_all, labels_all = ax.get_legend_handles_labels()
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    if handles_all and labels_all:
        fig.legend(handles_all, labels_all, loc='lower center', ncol=4,
                   bbox_to_anchor=(0.55, 0.1), frameon=False, fontsize=20)

    plt.tight_layout(rect=[0, 0.15, 1, 1])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to {out_path}")
    png_path = out_path.rsplit('.', 1)[0] + '.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"PNG saved to {png_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 14: Vision DET energy plots")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output figure path")
    args = parser.parse_args()
    plot_figure14(args.out)


if __name__ == '__main__':
    main()
