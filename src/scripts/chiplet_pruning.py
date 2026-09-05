"""
chiplet_pruning.py — Roofline-inspired search space pruning for chiplet pool DSE.

Pre-filters the 768 (arch x glb x pe_x x pe_y x dram) chiplet configurations
down to a much smaller set of "promising" configs by analyzing the performance
database. Configs that are never competitive for ANY network are pruned.

This reduces the outer search space from 768^N to K^N (K << 768), dramatically
improving convergence of all search algorithms (SA, GA, BO, Random).

Usage:
    from chiplet_pruning import get_pruned_configs
    pruned = get_pruned_configs(database_file, top_pct=30, needed_nets=needed_nets)
    # pruned is a list of (arch, glb, pe_x, pe_y, dram) tuples
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Set
from collections import Counter

# Add paths
_CHIPLET_TL_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _CHIPLET_TL_DIR not in sys.path:
    sys.path.insert(0, _CHIPLET_TL_DIR)
_EXP_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'timeloop_experiments'))
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

from chiplet_dataclass import ChipletConfig
from global_parameter import dram_options, arch_targets as DEFAULT_ARCH_TARGETS, glb_scales, pe_scales


def get_pruned_configs(
    database_file: str,
    top_pct: int = 30,
    needed_nets: Optional[Set[str]] = None,
    cost_aware: bool = False,
) -> List[Tuple]:
    """Compute the pruned set of chiplet configurations.

    For each network in the database, ranks all configs by energy (or energy*cost
    proxy) and keeps the top top_pct%. Returns the union across all networks —
    a config is kept if it's competitive for ANY network.

    Args:
        database_file: Path to the performance CSV
        top_pct: Percentage of configs to keep per network (default 30)
        needed_nets: Optional set of network names to consider
        cost_aware: If True, use energy * area as proxy for energy * cost

    Returns:
        List of (arch_target, glb_scale, pe_x_scale, pe_y_scale, dram_type) tuples
    """
    df = pd.read_csv(database_file)

    if needed_nets:
        df = df[df['net'].isin(needed_nets)]

    # Compute per-config, per-network mean energy (and area for cost-aware)
    agg_cols = {'dynamic_energy': 'mean'}
    if cost_aware:
        agg_cols['area'] = 'mean'

    config_net = df.groupby(
        ['arch_target', 'glb_scale', 'pe_x_scale', 'pe_y_scale', 'dram_i', 'net']
    ).agg(agg_cols).reset_index()

    if cost_aware:
        config_net['score'] = config_net['dynamic_energy'] * config_net['area']
    else:
        config_net['score'] = config_net['dynamic_energy']

    nets = config_net['net'].unique()
    useful_configs = set()

    for net in nets:
        net_df = config_net[config_net['net'] == net].sort_values('score')
        top_k = max(1, int(len(net_df) * top_pct / 100))
        for _, row in net_df.head(top_k).iterrows():
            useful_configs.add((
                row['arch_target'],
                int(row['glb_scale']),
                int(row['pe_x_scale']),
                int(row['pe_y_scale']),
                row['dram_i'],
            ))

    return sorted(useful_configs)


def pruned_configs_to_options(pruned_configs: List[Tuple]):
    """Extract the unique values per dimension from pruned configs.

    Returns:
        dict with keys 'arch_targets', 'glb_scales', 'pe_x_scales',
        'pe_y_scales', 'dram_types' — each a sorted list of valid values.
    """
    archs = sorted(set(c[0] for c in pruned_configs))
    glbs = sorted(set(c[1] for c in pruned_configs))
    pe_xs = sorted(set(c[2] for c in pruned_configs))
    pe_ys = sorted(set(c[3] for c in pruned_configs))
    drams = sorted(set(c[4] for c in pruned_configs))
    return {
        'arch_targets': archs,
        'glb_scales': glbs,
        'pe_x_scales': pe_xs,
        'pe_y_scales': pe_ys,
        'dram_types': drams,
    }


def generate_pruned_chiplet_group(
    n_chiplets: int,
    pruned_configs: List[Tuple],
    seed: Optional[int] = None,
) -> List[ChipletConfig]:
    """Generate a random chiplet group from the pruned config set."""
    import random
    if seed is not None:
        random.seed(seed)

    chiplets = []
    for _ in range(n_chiplets):
        cfg = random.choice(pruned_configs)
        chiplets.append(ChipletConfig(
            arch_target=cfg[0],
            global_buffer_size_scale=cfg[1],
            pe_x_scale=cfg[2],
            pe_y_scale=cfg[3],
            dram_type=cfg[4],
        ))
    return chiplets


def print_pruning_summary(pruned_configs: List[Tuple], total: int = 768):
    """Print a summary of the pruning results."""
    n = len(pruned_configs)
    archs = Counter(c[0] for c in pruned_configs)
    drams = Counter(c[4] for c in pruned_configs)
    print(f"Pruned configs: {n}/{total} ({n/total*100:.0f}%)")
    print(f"  Architectures: {dict(archs)}")
    print(f"  DRAM types:    {dict(drams)}")
    print(f"  Search space per chiplet: {n} (was {total})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="../timeloop_experiments/csv/llama_qwen_all_dram.csv")
    parser.add_argument("--top-pct", type=int, default=30)
    parser.add_argument("--cost-aware", action="store_true")
    args = parser.parse_args()

    pruned = get_pruned_configs(args.database, args.top_pct, cost_aware=args.cost_aware)
    print_pruning_summary(pruned)
    opts = pruned_configs_to_options(pruned)
    print(f"\nPer-dimension options:")
    for k, v in opts.items():
        print(f"  {k}: {v}")
