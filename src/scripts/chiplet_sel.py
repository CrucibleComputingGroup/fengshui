import os
import time
import math
import random
import logging
import numpy as np
import csv
import threading
try:
    import pytimeloop.timeloopfe.v4 as tl
except ImportError:
    tl = None
import joblib
from typing import List, Tuple, Optional, Set, Dict
import sys
import logging
from parse_stats import *
from convex_hull import convex_hull_min_e, process_layer_convex_hull
from chiplet_dataclass import *
from utility_functions import *
from global_parameter import *
import global_parameter
from network_dataclass import *
from genetic_algo_opt_phy_net import genetic_algo_opt_phy_net, genetic_algo_opt_phy_net_dag
import copy
import hashlib
from contextlib import contextmanager

# ── Determinism ──────────────────────────────────────────────────────────────
# The inner GA (genetic_algo_opt_phy_net{,_dag}) draws from Python's *global*
# `random` stream, so unseeded its results depend on call order and worker count
# (values move by 1-2%). Each per-network GA therefore runs under a global-RNG
# state that is
#   (1) SAVED before the GA and RESTORED after — so an enclosing SA pool-search
#       keeps its own RNG stream intact (the GA and SA share `random`), and
#   (2) SEEDED from a STABLE hash of the network's unique name — hashlib, NOT
#       builtin hash(), which is PYTHONHASHSEED-salted and differs across runs.
# The GA touches only `random` (never np.random), so np.random is left alone.
# The shipped results use the defaults below. FENGSHUI_DETERMINISTIC=0 leaves the
# GA unseeded; FENGSHUI_GA_SEED shifts the whole seed family.
_GA_SEED_BASE = int(os.environ.get('FENGSHUI_GA_SEED', '0'))

def _stable_seed(name: str) -> int:
    h = hashlib.sha256(f'{_GA_SEED_BASE}:{name}'.encode()).hexdigest()
    return int(h[:8], 16)

@contextmanager
def deterministic_ga_rng(name: str):
    """Seed the global `random` stream from a stable per-network key, then
    restore the prior state so an enclosing search's RNG is untouched."""
    if os.environ.get('FENGSHUI_DETERMINISTIC', '1') == '0':
        yield
        return
    _state = random.getstate()
    random.seed(_stable_seed(name))
    try:
        yield
    finally:
        random.setstate(_state)

def generate_neighbor_chiplet_optimized(
    chiplet,
    arch_targets: List[str],
    glb_scale_options: np.ndarray,
    pe_scale_options: np.ndarray,
    temperature: float = 1.0,
    dram_type_options: Optional[List[str]] = None
):
    """
    Generate a neighbor chiplet with possible architecture type change.
    PE_x and PE_y can have different scales.
    DRAM type is NOT mutated — every compute chiplet supports all DRAM types;
    the inner GA decides which DRAM to use per layer.
    """
    # Decide whether to change architecture type based on temperature
    if random.random() < temperature:
        current_arch_idx = arch_targets.index(chiplet.arch_target)
        available_indices = list(range(len(arch_targets)))
        available_indices.remove(current_arch_idx)
        new_arch_target = arch_targets[random.choice(available_indices)]
    else:
        new_arch_target = chiplet.arch_target

    # Get current indices for scaling factors
    glb_scale_options = np.array(glb_scale_options)
    pe_scale_options = np.array(pe_scale_options)

    current_glb_idx = np.abs(glb_scale_options - chiplet.global_buffer_size_scale).argmin()
    current_pe_x_idx = np.abs(pe_scale_options - chiplet.pe_x_scale).argmin()
    current_pe_y_idx = np.abs(pe_scale_options - chiplet.pe_y_scale).argmin()

    # Calculate max index changes based on temperature
    max_idx_change = max(1, int(len(glb_scale_options) * 0.5 * temperature))

    # Generate new indices with numpy operations
    new_glb_idx = np.clip(
        current_glb_idx + random.randint(-max_idx_change, max_idx_change),
        0, len(glb_scale_options) - 1
    )
    new_pe_x_idx = np.clip(
        current_pe_x_idx + random.randint(-max_idx_change, max_idx_change),
        0, len(pe_scale_options) - 1
    )
    new_pe_y_idx = np.clip(
        current_pe_y_idx + random.randint(-max_idx_change, max_idx_change),
        0, len(pe_scale_options) - 1
    )

    return ChipletConfig(
        arch_target=new_arch_target,
        global_buffer_size_scale=glb_scale_options[new_glb_idx],
        pe_x_scale=pe_scale_options[new_pe_x_idx],
        pe_y_scale=pe_scale_options[new_pe_y_idx],
    )

_FIXED_ARCH_TARGETS = frozenset({SWITCH_ARCH_TARGET, 'PIM'})

def generate_neighbor_group_optimized(
    current_group,
    arch_targets: List[str],
    glb_scale_options: np.ndarray,
    pe_scale_options: np.ndarray,
    temperature: float = 1.0
):
    """Optimized neighbor generation with architecture type changes.
    Fixed chiplets (switch, PIM) are preserved as-is — only compute chiplets are mutated.
    """
    new_group = []
    # Identify compute chiplets (exclude switch and PIM) for modification
    compute_indices = [i for i, c in enumerate(current_group)
                       if getattr(c, 'arch_target', '') not in _FIXED_ARCH_TARGETS]
    if not compute_indices:
        return list(current_group)
    n_modify = max(1, int(len(compute_indices) * temperature))
    modify_set = set(random.sample(compute_indices, min(n_modify, len(compute_indices))))

    for i, chiplet in enumerate(current_group):
        if i in modify_set:
            new_group.append(generate_neighbor_chiplet_optimized(
                chiplet,
                arch_targets,
                glb_scale_options,
                pe_scale_options,
                temperature
            ))
        else:
            new_group.append(chiplet)

    return new_group

def _eval_single_net(args):
    """Worker function for parallel network evaluation via mp.Pool.
    Accepts a tuple to be compatible with pool.map().
    """
    virtual_net, chiplet_group, objective, results_file, cost_aware, prev_gene, dag = args
    with deterministic_ga_rng(virtual_net.get_unique_name()):
        best_gene, best_value, best_config = genetic_algo_opt_phy_net(
            virtual_net,
            chiplet_group,
            objective=objective,
            population_size=10,
            generations=10,
            query_points=base_query_points,
            results_file=results_file,
            cost_aware=cost_aware,
            prev_best_gene=prev_gene,
            use_sequential=True,  # always sequential inside worker (reuse forked cache)
            dag=dag,
        )
    return (virtual_net.get_unique_name(), best_gene, best_value, best_config)


def _eval_single_net_dag(args):
    """Worker function for DAG-CP network evaluation via mp.Pool."""
    virtual_net, chiplet_group, objective, results_file, cost_aware, prev_gene, v_het_batch = args
    from cal_perf_phy_net import create_cp_spec
    cp_spec = create_cp_spec(virtual_net)
    _ga_pop = int(os.environ.get('INNER_GA_POP', '30'))
    _ga_gen = int(os.environ.get('INNER_GA_GEN', '30'))
    with deterministic_ga_rng(virtual_net.get_unique_name()):
        best_gene, best_value, best_config = genetic_algo_opt_phy_net_dag(
            virtual_net,
            chiplet_group,
            cp_spec=cp_spec,
            objective=objective,
            population_size=_ga_pop,
            generations=_ga_gen,
            results_file=results_file,
            cost_aware=cost_aware,
            prev_best_gene=prev_gene,
            use_sequential=True,
            v_het_batch=v_het_batch,
        )
    return (virtual_net.get_unique_name(), best_gene, best_value, best_config)


def run_single_optimization(
    virtual_nets: List[VirtualNetwork],
    chiplet_group: List,
    objective: str = "energy",
    logger: Optional[logging.Logger] = None,
    results_file: Optional[str] = None,
    cost_aware: bool=False,
    prev_best_genes: Optional[Dict[str, Dict]] = None,
    use_sequential: bool=False,
    n_workers: Optional[int] = None,
    dag=None,
    use_dag_cp: bool = False,
    v_het_batch: bool = True,
) -> Tuple[List, Dict]:
    """
    Run optimization for a single chiplet configuration across all networks.

    Args:
        virtual_nets: List of VirtualNetwork objects
        chiplet_group: List of chiplet configurations
        objective: Optimization objective, either "energy" or "edp"
        logger: Logger instance
        results_file: CSV file to store results
        cost_aware: Whether to consider cost
        prev_best_genes: Previous best genes for warm-starting
        use_sequential: If True, evaluate GA genes sequentially (cache-friendly)
        n_workers: Number of parallel workers for network-level parallelism.
        dag: OperatorDAG for legacy DAG mode (ignored when use_dag_cp=True)
        use_dag_cp: If True, use the critical-path DAG GA instead of linear GA.
        v_het_batch: If True and use_dag_cp, off-CP ops use larger batch.

    Returns:
        Tuple containing chiplet group and results dictionary
    """
    results = {}

    # Build task args
    task_args = []
    for virtual_net in virtual_nets:
        prev_gene = None
        if prev_best_genes and virtual_net.get_unique_name() in prev_best_genes:
            prev_gene = prev_best_genes[virtual_net.get_unique_name()]
        if use_dag_cp:
            task_args.append((virtual_net, chiplet_group, objective, results_file,
                              cost_aware, prev_gene, v_het_batch))
        else:
            task_args.append((virtual_net, chiplet_group, objective, results_file,
                              cost_aware, prev_gene, dag))

    worker_fn = _eval_single_net_dag if use_dag_cp else _eval_single_net

    # Decide parallelism: use mp.Pool(fork) when we have workers and cache is preloaded
    effective_workers = n_workers or 1
    if effective_workers > 1 and use_sequential and len(virtual_nets) > 1:
        import multiprocessing as mp
        try:
            mp.set_start_method('fork', force=True)
        except RuntimeError:
            pass  # already set
        with mp.Pool(min(effective_workers, len(virtual_nets))) as pool:
            net_results = pool.map(worker_fn, task_args)
    else:
        # Sequential fallback
        net_results = [worker_fn(args) for args in task_args]

    # Collect results
    for unique_name, best_gene, best_value, best_config in net_results:
        results[unique_name] = {
            'best_gene': best_gene,
            'min_value': best_value,
            'best_config': best_config,
            'best_latency': float(best_config.get('latency', global_parameter.base_query_points[best_config['idx']])) if best_config else float("inf")
        }
        if logger:
            metric_name = "EDP" if objective == "edp" else "energy"
            mode = "DAG-CP" if use_dag_cp else "linear"
            logger.info(f"[{mode}] Network {unique_name} - Best {metric_name} (cost aware: {cost_aware}): {best_value:.2e}")

    return chiplet_group, results
