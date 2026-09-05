"""
run_archgym_chiplet.py — Compare optimization algorithms for chiplet pool selection.

Runs SA (baseline from chiplet_sel.py), Bayesian Optimization, and Genetic Algorithm
from arch-gym on the same chiplet pool optimization problem. Outputs convergence curves
and final results to CSV for analysis.

Usage:
    python3 run_archgym_chiplet.py \
        --objective energy \
        --n-chiplets 8 \
        --database final_database.csv \
        --n-evals 200 \
        --algorithms sa bo ga random \
        --seed 42
"""

import os
import sys
import time
import copy
import random
import logging
import argparse
import csv
import json
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

# Ensure local imports work
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
# Add chiplet_timeloop parent dir so local packages (timeloop/, etc.) are importable
_CHIPLET_TL_DIR = os.path.normpath(os.path.join(THIS_DIR, '..'))
if _CHIPLET_TL_DIR not in sys.path:
    sys.path.insert(0, _CHIPLET_TL_DIR)
_EXP_DIR = os.path.normpath(os.path.join(THIS_DIR, '..', 'timeloop_experiments'))
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

from chiplet_dataclass import ChipletConfig, generate_chiplet_group, create_switch_chiplet
from global_parameter import (
    arch_targets as DEFAULT_ARCH_TARGETS,
    glb_scales, pe_scales, NET_DIR, SWITCH_ARCH_TARGET,
)
import global_parameter
from network_dataclass import VirtualNetwork
from utility_functions import calculate_average_opt_value
from chiplet_sel import run_single_optimization, simulated_annealing_optimization, generate_neighbor_group_optimized
from chiplet_pool_env import ChipletPoolEnv, ARCH_INDEX, GLB_INDEX, PE_INDEX
from chiplet_pool_estimator import (
    ChipletPoolEstimator,
    build_skopt_optimizer,
    decode_skopt_point,
)
from cal_perf_phy_net import preload_database
from chiplet_pruning import get_pruned_configs, generate_pruned_chiplet_group, print_pruning_summary

_SINGLETON_ARCH_TARGETS = frozenset({'PIM', SWITCH_ARCH_TARGET})


def _build_arch_targets(include_pim=False, include_switch=False):
    """Build the architecture target list, optionally including PIM and switch.
    When included, algorithms can naturally discover these special chiplets
    without forced injection."""
    targets = list(DEFAULT_ARCH_TARGETS)
    if include_pim and 'PIM' not in targets:
        targets.append('PIM')
    if include_switch and SWITCH_ARCH_TARGET not in targets:
        targets.append(SWITCH_ARCH_TARGET)
    return targets


def _enforce_singleton_specials(chiplet_group, arch_targets=None):
    """Ensure at most 1 PIM and 1 switch in the pool.
    Duplicate specials are replaced with a random compute chiplet.
    Also normalizes PIM to its canonical config (glb=1, pe=1x1, GDDR7)."""
    if arch_targets is None:
        arch_targets = DEFAULT_ARCH_TARGETS
    compute_targets = [a for a in arch_targets if a not in _SINGLETON_ARCH_TARGETS]
    if not compute_targets:
        compute_targets = list(DEFAULT_ARCH_TARGETS)
    seen_specials = set()
    for i, c in enumerate(chiplet_group):
        if c.arch_target in _SINGLETON_ARCH_TARGETS:
            if c.arch_target in seen_specials:
                # Replace duplicate with random compute chiplet
                chiplet_group[i] = ChipletConfig(
                    arch_target=random.choice(compute_targets),
                    global_buffer_size_scale=random.choice(glb_scales),
                    pe_x_scale=random.choice(pe_scales),
                    pe_y_scale=random.choice(pe_scales))
            else:
                seen_specials.add(c.arch_target)
                # Normalize PIM to canonical config
                if c.arch_target == 'PIM':
                    c.global_buffer_size_scale = 1
                    c.pe_x_scale = 1
                    c.pe_y_scale = 1
                    c.dram_type = 'GDDR7'
    return chiplet_group


def _encode_initial_group(initial_group, n_chiplets, arch_targets=None):
    """Encode an initial_group (List[ChipletConfig]) into the float array
    used by DE/PSO/memetic/EDA.  4 dims per chiplet: arch, glb, pe_x, pe_y."""
    if arch_targets is None:
        arch_targets = DEFAULT_ARCH_TARGETS
    arch_map = {a: i for i, a in enumerate(arch_targets)}
    glb_map = {g: i for i, g in enumerate(glb_scales)}
    pe_map = {p: i for i, p in enumerate(pe_scales)}
    x = []
    for c in initial_group:
        x.extend([arch_map.get(c.arch_target, 0),
                   glb_map.get(c.global_buffer_size_scale, 0),
                   pe_map.get(c.pe_x_scale, 0),
                   pe_map.get(c.pe_y_scale, 0)])
    while len(x) < n_chiplets * 4:
        x.append(0)
    return np.array(x, dtype=float)


# ---------------------------------------------------------------------------
# Evaluation helper (shared across all algorithms)
# ---------------------------------------------------------------------------

OUTER_WORKERS = 64  # parallel outer evals (chiplet groups evaluated in parallel)


# Module-level flags set from main() — read by worker functions via fork
_USE_DAG_CP = False
_V_HET_BATCH = True


def _eval_group_worker(args):
    """Worker for outer-level parallel evaluation. Each worker evaluates one
    chiplet group using sequential inner path (reuses forked cache)."""
    chiplet_group, virtual_nets, objective, database_file, cost_aware = args
    _, results = run_single_optimization(
        virtual_nets=virtual_nets,
        chiplet_group=chiplet_group,
        objective=objective,
        results_file=database_file,
        cost_aware=cost_aware,
        use_sequential=True,
        n_workers=1,  # no inner parallelism; outer pool handles it
        use_dag_cp=_USE_DAG_CP,
        v_het_batch=_V_HET_BATCH,
    )
    avg_value, network_results = calculate_average_opt_value(results, objective)
    return chiplet_group, avg_value, network_results


def evaluate_chiplet_group(chiplet_group, virtual_nets, objective, database_file,
                           cost_aware=False, prev_best_genes=None):
    """Evaluate a single chiplet group (sequential, for BO ask/tell)."""
    _, results = run_single_optimization(
        virtual_nets=virtual_nets,
        chiplet_group=chiplet_group,
        objective=objective,
        results_file=database_file,
        cost_aware=cost_aware,
        prev_best_genes=prev_best_genes,
        use_sequential=True,
        n_workers=8,  # inner parallelism for single-eval path (BO)
        use_dag_cp=_USE_DAG_CP,
        v_het_batch=_V_HET_BATCH,
    )
    avg_value, network_results = calculate_average_opt_value(results, objective)
    return avg_value, network_results


def evaluate_batch_parallel(groups, virtual_nets, objective, database_file,
                            cost_aware=False, n_workers=None):
    """Evaluate multiple chiplet groups in parallel using outer-level mp.Pool(fork).
    Returns list of (chiplet_group, value, network_results) tuples."""
    import multiprocessing as mp
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass

    workers = n_workers or OUTER_WORKERS
    args = [(g, virtual_nets, objective, database_file, cost_aware) for g in groups]
    with mp.Pool(min(workers, len(groups))) as pool:
        results = pool.map(_eval_group_worker, args)
    return results


# ---------------------------------------------------------------------------
# Algorithm: Random Search (baseline)
# ---------------------------------------------------------------------------

def run_random_search(virtual_nets, n_chiplets, objective, database_file,
                      n_evals, cost_aware=False, seed=42,
                      include_pim=False, include_switch=False):
    """Random search baseline — evaluates all groups in parallel batches."""
    random.seed(seed)
    np.random.seed(seed)

    arch_targets = _build_arch_targets(include_pim, include_switch)

    # Generate all random groups upfront
    groups = []
    for _ in range(n_evals):
        g = generate_chiplet_group(
            n_chiplets=n_chiplets,
            arch_targets=arch_targets,
            glb_scale_options=glb_scales,
            pe_scale_options=pe_scales,
        )
        groups.append(_enforce_singleton_specials(g, arch_targets))

    # Evaluate all in parallel
    t0 = time.perf_counter()
    batch_results = evaluate_batch_parallel(
        groups, virtual_nets, objective, database_file, cost_aware)
    total_dt = time.perf_counter() - t0

    # Build history
    best_value = float('inf')
    best_group = None
    history = []
    for i, (group, value, _) in enumerate(batch_results):
        if value < best_value:
            best_value = value
            best_group = copy.deepcopy(group)
        history.append({
            'eval': i + 1,
            'value': value,
            'best_value': best_value,
            'eval_time': total_dt / n_evals,
        })

    print(f"[Random] {n_evals} evals in {total_dt:.1f}s ({total_dt/n_evals:.1f}s/eval) | Best: {best_value:.4e}")
    return best_group, best_value, history


# ---------------------------------------------------------------------------
# Algorithm: Simulated Annealing (existing baseline)
# ---------------------------------------------------------------------------

def run_sa(virtual_nets, n_chiplets, objective, database_file,
           n_evals, cost_aware=False, seed=42,
           initial_temp=1.0, cooling_rate=0.95, min_temp=0.01,
           include_pim=False, include_switch=False,
           initial_group=None):
    """Domain-specific Simulated Annealing with per-eval history tracking.

    Uses the same temperature-aware neighbor generation from chiplet_sel.py
    (architecture-aware mutations, temperature-scaled perturbations) but with
    batch-parallel evaluation and per-eval convergence tracking.
    PIM and switch chiplets are preserved by neighbor generation (not mutated).
    """
    import global_parameter

    random.seed(seed)
    np.random.seed(seed)

    # Build temperature schedule
    temp_schedule = []
    temp = initial_temp
    while temp > min_temp:
        temp_schedule.append(temp)
        temp *= cooling_rate

    # Calculate iterations per temperature to match total budget
    iters_per_temp = max(1, n_evals // len(temp_schedule))
    total_planned = len(temp_schedule) * iters_per_temp
    print(f"[SA] temp_steps={len(temp_schedule)}, iter_per_temp={iters_per_temp}, total≈{total_planned}")

    arch_targets = _build_arch_targets(include_pim, include_switch)

    # Generate initial solution (warm-start if provided)
    if initial_group is not None:
        current_group = copy.deepcopy(initial_group)
        print(f"  [SA] Warm-started from previous best")
    else:
        current_group = generate_chiplet_group(
            n_chiplets=n_chiplets,
            arch_targets=arch_targets,
            glb_scale_options=glb_scales,
            pe_scale_options=pe_scales,
        )

    # Evaluate initial solution
    t0_total = time.perf_counter()
    current_value, _ = evaluate_chiplet_group(
        current_group, virtual_nets, objective, database_file, cost_aware)

    best_group = copy.deepcopy(current_group)
    best_value = current_value
    history = []
    eval_count = 1
    history.append({'eval': 1, 'value': current_value, 'best_value': best_value, 'eval_time': 0})

    for temp_idx, temperature in enumerate(temp_schedule):
        if eval_count >= n_evals:
            break

        # Generate batch of neighbors for this temperature
        batch_size = min(iters_per_temp, n_evals - eval_count)
        neighbors = []
        for _ in range(batch_size):
            neighbors.append(generate_neighbor_group_optimized(
                current_group,
                arch_targets,
                glb_scale_options=global_parameter.glb_scales,
                pe_scale_options=global_parameter.pe_scales,
                temperature=temperature,
            ))

        # Evaluate all neighbors in parallel
        batch_results = evaluate_batch_parallel(
            neighbors, virtual_nets, objective, database_file, cost_aware)

        # Process results with SA acceptance criterion
        for neighbor_group, neighbor_value, _ in batch_results:
            eval_count += 1
            delta_e = neighbor_value - current_value

            # SA acceptance: always accept better, probabilistically accept worse
            if delta_e < 0 or (current_value > 0 and
                               random.random() < math.exp(-delta_e / (temperature * current_value))):
                current_group = copy.deepcopy(neighbor_group)
                current_value = neighbor_value

                if current_value < best_value:
                    best_value = current_value
                    best_group = copy.deepcopy(current_group)

            history.append({
                'eval': eval_count,
                'value': neighbor_value,
                'best_value': best_value,
                'eval_time': 0,
            })

        print(f"\r[SA] T={temperature:.4f} | Evals: {eval_count}/{n_evals} | Best: {best_value:.4e}", end="")

    total_dt = time.perf_counter() - t0_total
    for h in history:
        h['eval_time'] = total_dt / len(history)
    print(f"\n[SA] {eval_count} evals in {total_dt:.1f}s ({total_dt/eval_count:.1f}s/eval)")
    return best_group, best_value, history


# ---------------------------------------------------------------------------
# Algorithm: Bayesian Optimization (via skopt)
# ---------------------------------------------------------------------------

def run_bo(virtual_nets, n_chiplets, objective, database_file,
           n_evals, cost_aware=False, seed=42,
           n_initial_points=10, acq_func="EI"):
    """Bayesian Optimization using scikit-optimize with batch-parallel evaluation.

    Initial random points are evaluated in one parallel batch.
    Subsequent BO iterations use batches of suggestions evaluated in parallel.
    """
    random.seed(seed)
    np.random.seed(seed)

    bo_batch_size = min(OUTER_WORKERS, max(1, n_evals // 10))  # batch size for BO

    optimizer, dim_names = build_skopt_optimizer(
        n_chiplets=n_chiplets,
        n_initial_points=min(n_initial_points, n_evals // 2),
        acq_func=acq_func,
    )

    best_value = float('inf')
    best_group = None
    history = []
    eval_count = 0
    t0_total = time.perf_counter()

    while eval_count < n_evals:
        # Ask for a batch of points
        batch_size = min(bo_batch_size, n_evals - eval_count)
        suggestions = [optimizer.ask() for _ in range(batch_size)]
        groups = [decode_skopt_point(s, n_chiplets) for s in suggestions]

        # Evaluate batch in parallel
        batch_results = evaluate_batch_parallel(
            groups, virtual_nets, objective, database_file, cost_aware)

        # Tell optimizer all results
        for suggestion, (group, value, _) in zip(suggestions, batch_results):
            optimizer.tell(suggestion, value)
            eval_count += 1

            if value < best_value:
                best_value = value
                best_group = copy.deepcopy(group)

            history.append({
                'eval': eval_count,
                'value': value,
                'best_value': best_value,
                'eval_time': 0,  # batch timing
            })

        print(f"\r[BO] {eval_count}/{n_evals} | Best: {best_value:.4e}", end="")

    total_dt = time.perf_counter() - t0_total
    # Backfill eval_time
    for h in history:
        h['eval_time'] = total_dt / len(history)

    print(f"\n[BO] {n_evals} evals in {total_dt:.1f}s ({total_dt/n_evals:.1f}s/eval)")
    return best_group, best_value, history


# ---------------------------------------------------------------------------
# Algorithm: Genetic Algorithm (from arch-gym sko)
# ---------------------------------------------------------------------------

def run_ga(virtual_nets, n_chiplets, objective, database_file,
           n_evals, cost_aware=False, seed=42,
           pop_size=20, mutation_rate=0.1,
           initial_group=None, pruned_configs=None,
           include_pim=False, include_switch=False):
    """Genetic Algorithm for chiplet pool optimization.

    Args:
        initial_group: Optional List[ChipletConfig] to warm-start the population.
        pruned_configs: Optional list of (arch, glb, pe_x, pe_y, dram) tuples.
                        If provided, samples/mutates only within this set.
        include_pim: If True, inject PIM into generated groups at n>=2.
        include_switch: If True, inject switch into generated groups at n>=2.
    """
    random.seed(seed)
    np.random.seed(seed)

    arch_targets = _build_arch_targets(include_pim, include_switch)

    # If pruned configs provided, use index-based encoding over the pruned set
    if pruned_configs is not None:
        n_configs = len(pruned_configs)
        n_dim = n_chiplets  # one index per chiplet
        lb = np.zeros(n_dim)
        ub = np.array([n_configs - 1] * n_chiplets, dtype=float)
    else:
        n_configs = None
        # 4 dims per chiplet: arch, glb, pe_x, pe_y (DRAM decided by inner GA)
        n_dim = n_chiplets * 4
        lb = np.zeros(n_dim)
        ub = np.array([len(arch_targets) - 1, len(glb_scales) - 1,
                        len(pe_scales) - 1, len(pe_scales) - 1] * n_chiplets, dtype=float)

    best_value = float('inf')
    best_group = None
    history = []
    eval_count = 0
    t0_total = time.perf_counter()
    n_generations = max(1, n_evals // pop_size)

    def encode_group(chiplet_group):
        """Convert ChipletConfig list to float array."""
        if pruned_configs is not None:
            cfg_to_idx = {cfg: i for i, cfg in enumerate(pruned_configs)}
            x = []
            for c in chiplet_group:
                key = (c.arch_target, c.global_buffer_size_scale, c.pe_x_scale, c.pe_y_scale, c.dram_type)
                idx = cfg_to_idx.get(key, 0)
                x.append(idx)
            while len(x) < n_chiplets:
                x.append(0)
            return np.array(x, dtype=float)
        else:
            arch_map = {a: i for i, a in enumerate(arch_targets)}
            glb_map = {g: i for i, g in enumerate(glb_scales)}
            pe_map = {p: i for i, p in enumerate(pe_scales)}
            x = []
            for c in chiplet_group:
                x.extend([arch_map.get(c.arch_target, 0), glb_map.get(c.global_buffer_size_scale, 0),
                           pe_map.get(c.pe_x_scale, 0), pe_map.get(c.pe_y_scale, 0)])
            while len(x) < n_chiplets * 4:
                x.append(0)
            return np.array(x, dtype=float)

    def decode_individual(x):
        """Convert float array to chiplet group."""
        chiplets = []
        if pruned_configs is not None:
            for i in range(n_chiplets):
                idx = int(np.clip(round(x[i]), 0, n_configs - 1))
                cfg = pruned_configs[idx]
                chiplets.append(ChipletConfig(
                    arch_target=cfg[0], global_buffer_size_scale=cfg[1],
                    pe_x_scale=cfg[2], pe_y_scale=cfg[3], dram_type=cfg[4]))
        else:
            for i in range(n_chiplets):
                base = i * 4
                arch_idx = int(np.clip(round(x[base]), 0, len(arch_targets) - 1))
                glb_idx = int(np.clip(round(x[base + 1]), 0, len(glb_scales) - 1))
                pe_x_idx = int(np.clip(round(x[base + 2]), 0, len(pe_scales) - 1))
                pe_y_idx = int(np.clip(round(x[base + 3]), 0, len(pe_scales) - 1))
                chiplets.append(ChipletConfig(
                    arch_target=arch_targets[arch_idx],
                    global_buffer_size_scale=glb_scales[glb_idx],
                    pe_x_scale=pe_scales[pe_x_idx],
                    pe_y_scale=pe_scales[pe_y_idx]))
        return _enforce_singleton_specials(chiplets, arch_targets)

    # Initialize population — warm-start if initial_group provided
    population = []
    if initial_group is not None:
        seed_individual = encode_group(initial_group)
        population.append(seed_individual)
        # Create mutations: only mutate the LAST chiplet (the new one)
        n_seed_mutants = max(1, pop_size // 4)  # 25% of pop from seed
        if pruned_configs is not None:
            # In pruned mode, last chiplet is last index
            for _ in range(n_seed_mutants):
                mutant = seed_individual.copy()
                mutant[-1] = random.randint(0, int(ub[-1]))
                population.append(mutant)
        else:
            prev_n = n_chiplets - 1
            for _ in range(n_seed_mutants):
                mutant = seed_individual.copy()
                for d in range(prev_n * 5, n_dim):
                    mutant[d] = random.randint(0, int(ub[d]))
                population.append(mutant)

    # Fill remaining with random individuals
    while len(population) < pop_size:
        individual = np.array([
            random.randint(0, int(u)) for u in ub
        ], dtype=float)
        population.append(individual)
    population = np.array(population[:pop_size])

    for gen in range(n_generations):
        # Evaluate entire population in parallel
        remaining = n_evals - eval_count
        batch_size = min(pop_size, remaining)
        if batch_size <= 0:
            break

        groups = [decode_individual(population[j]) for j in range(batch_size)]
        batch_results = evaluate_batch_parallel(
            groups, virtual_nets, objective, database_file, cost_aware)

        fitness = np.full(pop_size, float('inf'))
        for j, (group, value, _) in enumerate(batch_results):
            fitness[j] = value
            eval_count += 1

            if value < best_value:
                best_value = value
                best_group = copy.deepcopy(group)

            history.append({
                'eval': eval_count,
                'value': value,
                'best_value': best_value,
                'eval_time': 0,
            })

        print(f"\r[GA] Gen {gen+1}/{n_generations} | Evals: {eval_count}/{n_evals} | Best: {best_value:.4e}", end="")

        if eval_count >= n_evals:
            break

        # Selection (tournament)
        next_pop = []
        # Elitism: keep top 2
        elite_idx = np.argsort(fitness)[:2]
        for idx in elite_idx:
            next_pop.append(population[idx].copy())

        while len(next_pop) < pop_size:
            # Tournament selection
            t_idx = np.random.choice(pop_size, size=3, replace=False)
            t_fitness = fitness[t_idx]
            winner = t_idx[np.argmin(t_fitness)]
            parent1 = population[winner].copy()

            t_idx = np.random.choice(pop_size, size=3, replace=False)
            t_fitness = fitness[t_idx]
            winner = t_idx[np.argmin(t_fitness)]
            parent2 = population[winner].copy()

            # Crossover (uniform)
            mask = np.random.random(n_dim) < 0.5
            child = np.where(mask, parent1, parent2)

            # Mutation
            for d in range(n_dim):
                if random.random() < mutation_rate:
                    child[d] = random.randint(0, int(ub[d]))

            next_pop.append(child)

        population = np.array(next_pop[:pop_size])

    total_dt = time.perf_counter() - t0_total
    for h in history:
        h['eval_time'] = total_dt / len(history) if history else 0
    print(f"[GA] {eval_count} evals in {total_dt:.1f}s ({total_dt/max(1,eval_count):.1f}s/eval)")
    return best_group, best_value, history


# ---------------------------------------------------------------------------
# Algorithm: Differential Evolution (DE)
# ---------------------------------------------------------------------------

def run_de(virtual_nets, n_chiplets, objective, database_file,
           n_evals, cost_aware=False, seed=42,
           pop_size=20, F=0.8, CR=0.7,
           include_pim=False, include_switch=False,
           initial_group=None):
    """Differential Evolution — uses difference vectors for mutation.
    Better at escaping local optima than GA's random mutation."""
    random.seed(seed)
    np.random.seed(seed)
    arch_targets = _build_arch_targets(include_pim, include_switch)

    n_dim = n_chiplets * 4
    ub = np.array([len(arch_targets) - 1, len(glb_scales) - 1,
                    len(pe_scales) - 1, len(pe_scales) - 1] * n_chiplets, dtype=float)

    def decode(x):
        chiplets = []
        for i in range(n_chiplets):
            b = i * 4
            chiplets.append(ChipletConfig(
                arch_target=arch_targets[int(np.clip(round(x[b]), 0, ub[b]))],
                global_buffer_size_scale=glb_scales[int(np.clip(round(x[b+1]), 0, ub[b+1]))],
                pe_x_scale=pe_scales[int(np.clip(round(x[b+2]), 0, ub[b+2]))],
                pe_y_scale=pe_scales[int(np.clip(round(x[b+3]), 0, ub[b+3]))]))
        return _enforce_singleton_specials(chiplets, arch_targets)

    best_value, best_group = float('inf'), None
    history = []
    eval_count = 0
    t0 = time.perf_counter()

    # Init population (warm-start first individual if provided)
    pop = np.array([[random.randint(0, int(u)) for u in ub] for _ in range(pop_size)], dtype=float)
    if initial_group is not None:
        pop[0] = _encode_initial_group(initial_group, n_chiplets, arch_targets)
    groups = [decode(pop[j]) for j in range(pop_size)]
    batch = evaluate_batch_parallel(groups, virtual_nets, objective, database_file, cost_aware)
    fitness = np.full(pop_size, float('inf'))
    for j, (g, v, _) in enumerate(batch):
        fitness[j] = v
        eval_count += 1
        if v < best_value:
            best_value, best_group = v, copy.deepcopy(g)
        history.append({'eval': eval_count, 'value': v, 'best_value': best_value, 'eval_time': 0})

    while eval_count < n_evals:
        trials = []
        for i in range(pop_size):
            idxs = [x for x in range(pop_size) if x != i]
            a, b, c = [pop[j] for j in random.sample(idxs, 3)]
            mutant = np.clip(np.round(a + F * (b - c)), 0, ub)
            # Binomial crossover
            trial = pop[i].copy()
            j_rand = random.randint(0, n_dim - 1)
            for d in range(n_dim):
                if random.random() < CR or d == j_rand:
                    trial[d] = mutant[d]
            trials.append(trial)

        remaining = n_evals - eval_count
        batch_size = min(pop_size, remaining)
        trial_groups = [decode(trials[j]) for j in range(batch_size)]
        batch = evaluate_batch_parallel(trial_groups, virtual_nets, objective, database_file, cost_aware)

        for j, (g, v, _) in enumerate(batch):
            eval_count += 1
            if v <= fitness[j]:  # DE selection: replace if better or equal
                pop[j] = trials[j]
                fitness[j] = v
            if v < best_value:
                best_value, best_group = v, copy.deepcopy(g)
            history.append({'eval': eval_count, 'value': v, 'best_value': best_value, 'eval_time': 0})

        print(f"\r[DE] Evals: {eval_count}/{n_evals} | Best: {best_value:.4e}", end="")

    dt = time.perf_counter() - t0
    for h in history: h['eval_time'] = dt / len(history) if history else 0
    print(f"\n[DE] {eval_count} evals in {dt:.1f}s | Best: {best_value:.4e}")
    return best_group, best_value, history


# ---------------------------------------------------------------------------
# Algorithm: Particle Swarm Optimization (PSO)
# ---------------------------------------------------------------------------

def run_pso(virtual_nets, n_chiplets, objective, database_file,
            n_evals, cost_aware=False, seed=42,
            pop_size=20, w=0.7, c1=1.5, c2=1.5,
            include_pim=False, include_switch=False,
            initial_group=None):
    """Discrete PSO — particles move through config space guided by personal
    and global best positions."""
    random.seed(seed)
    np.random.seed(seed)
    arch_targets = _build_arch_targets(include_pim, include_switch)

    n_dim = n_chiplets * 4
    ub = np.array([len(arch_targets) - 1, len(glb_scales) - 1,
                    len(pe_scales) - 1, len(pe_scales) - 1] * n_chiplets, dtype=float)

    def decode(x):
        chiplets = []
        for i in range(n_chiplets):
            b = i * 4
            chiplets.append(ChipletConfig(
                arch_target=arch_targets[int(np.clip(round(x[b]), 0, ub[b]))],
                global_buffer_size_scale=glb_scales[int(np.clip(round(x[b+1]), 0, ub[b+1]))],
                pe_x_scale=pe_scales[int(np.clip(round(x[b+2]), 0, ub[b+2]))],
                pe_y_scale=pe_scales[int(np.clip(round(x[b+3]), 0, ub[b+3]))]))
        return _enforce_singleton_specials(chiplets, arch_targets)

    best_value, best_group = float('inf'), None
    history = []
    eval_count = 0
    t0 = time.perf_counter()

    # Init (warm-start first particle if provided)
    pos = np.array([[random.randint(0, int(u)) for u in ub] for _ in range(pop_size)], dtype=float)
    if initial_group is not None:
        pos[0] = _encode_initial_group(initial_group, n_chiplets, arch_targets)
    vel = np.zeros_like(pos)
    pbest_pos = pos.copy()
    pbest_val = np.full(pop_size, float('inf'))
    gbest_pos = pos[0].copy()
    gbest_val = float('inf')

    while eval_count < n_evals:
        remaining = n_evals - eval_count
        batch_size = min(pop_size, remaining)
        groups = [decode(pos[j]) for j in range(batch_size)]
        batch = evaluate_batch_parallel(groups, virtual_nets, objective, database_file, cost_aware)

        for j, (g, v, _) in enumerate(batch):
            eval_count += 1
            if v < pbest_val[j]:
                pbest_val[j] = v
                pbest_pos[j] = pos[j].copy()
            if v < gbest_val:
                gbest_val = v
                gbest_pos = pos[j].copy()
            if v < best_value:
                best_value, best_group = v, copy.deepcopy(g)
            history.append({'eval': eval_count, 'value': v, 'best_value': best_value, 'eval_time': 0})

        # Update velocities and positions
        r1 = np.random.random((pop_size, n_dim))
        r2 = np.random.random((pop_size, n_dim))
        vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos)
        pos = np.clip(np.round(pos + vel), 0, ub)

        print(f"\r[PSO] Evals: {eval_count}/{n_evals} | Best: {best_value:.4e}", end="")

    dt = time.perf_counter() - t0
    for h in history: h['eval_time'] = dt / len(history) if history else 0
    print(f"\n[PSO] {eval_count} evals in {dt:.1f}s | Best: {best_value:.4e}")
    return best_group, best_value, history


# ---------------------------------------------------------------------------
# Algorithm: Memetic Algorithm (GA + Local Search)
# ---------------------------------------------------------------------------

def run_memetic(virtual_nets, n_chiplets, objective, database_file,
                n_evals, cost_aware=False, seed=42,
                pop_size=20, mutation_rate=0.1, ls_budget_pct=0.3,
                include_pim=False, include_switch=False,
                initial_group=None):
    """GA + greedy local search on the best solution each generation."""
    random.seed(seed)
    np.random.seed(seed)
    arch_targets = _build_arch_targets(include_pim, include_switch)

    n_dim = n_chiplets * 4
    ub = np.array([len(arch_targets) - 1, len(glb_scales) - 1,
                    len(pe_scales) - 1, len(pe_scales) - 1] * n_chiplets, dtype=float)

    def decode(x):
        chiplets = []
        for i in range(n_chiplets):
            b = i * 4
            chiplets.append(ChipletConfig(
                arch_target=arch_targets[int(np.clip(round(x[b]), 0, ub[b]))],
                global_buffer_size_scale=glb_scales[int(np.clip(round(x[b+1]), 0, ub[b+1]))],
                pe_x_scale=pe_scales[int(np.clip(round(x[b+2]), 0, ub[b+2]))],
                pe_y_scale=pe_scales[int(np.clip(round(x[b+3]), 0, ub[b+3]))]))
        return _enforce_singleton_specials(chiplets, arch_targets)

    def encode(group):
        arch_map = {a: i for i, a in enumerate(arch_targets)}
        glb_map = {g: i for i, g in enumerate(glb_scales)}
        pe_map = {p: i for i, p in enumerate(pe_scales)}
        x = []
        for c in group:
            x.extend([arch_map.get(c.arch_target, 0), glb_map.get(c.global_buffer_size_scale, 0),
                       pe_map.get(c.pe_x_scale, 0), pe_map.get(c.pe_y_scale, 0)])
        return np.array(x, dtype=float)

    best_value, best_group, best_enc = float('inf'), None, None
    history = []
    eval_count = 0
    t0 = time.perf_counter()
    ga_budget = int(n_evals * (1 - ls_budget_pct))

    # --- GA phase --- (warm-start first individual if provided)
    pop = np.array([[random.randint(0, int(u)) for u in ub] for _ in range(pop_size)], dtype=float)
    if initial_group is not None:
        pop[0] = _encode_initial_group(initial_group, n_chiplets, arch_targets)
    fitness = np.full(pop_size, float('inf'))
    n_gen = max(1, ga_budget // pop_size)

    for gen in range(n_gen):
        remaining = ga_budget - eval_count
        bs = min(pop_size, remaining)
        if bs <= 0: break

        groups = [decode(pop[j]) for j in range(bs)]
        batch = evaluate_batch_parallel(groups, virtual_nets, objective, database_file, cost_aware)
        for j, (g, v, _) in enumerate(batch):
            fitness[j] = v
            eval_count += 1
            if v < best_value:
                best_value, best_group, best_enc = v, copy.deepcopy(g), pop[j].copy()
            history.append({'eval': eval_count, 'value': v, 'best_value': best_value, 'eval_time': 0})

        if eval_count >= ga_budget: break

        # GA operators
        next_pop = []
        elite = np.argsort(fitness)[:2]
        for idx in elite: next_pop.append(pop[idx].copy())
        while len(next_pop) < pop_size:
            t_idx = np.random.choice(pop_size, size=min(3, pop_size), replace=False)
            p1 = pop[t_idx[np.argmin(fitness[t_idx])]].copy()
            t_idx = np.random.choice(pop_size, size=min(3, pop_size), replace=False)
            p2 = pop[t_idx[np.argmin(fitness[t_idx])]].copy()
            mask = np.random.random(n_dim) < 0.5
            child = np.where(mask, p1, p2)
            for d in range(n_dim):
                if random.random() < mutation_rate:
                    child[d] = random.randint(0, int(ub[d]))
            next_pop.append(child)
        pop = np.array(next_pop[:pop_size])

        print(f"\r[MEM] GA Gen {gen+1}/{n_gen} | Evals: {eval_count}/{n_evals} | Best: {best_value:.4e}", end="")

    print(f"\n[MEM] GA done: {eval_count} evals, best={best_value:.4e}. Starting local search...")

    # --- Local search: try swapping each chiplet param one at a time ---
    if best_enc is None:
        dt = time.perf_counter() - t0
        for h in history: h['eval_time'] = dt / len(history) if history else 0
        return best_group, best_value, history

    current = best_enc.copy()
    current_val = best_value
    improved = True
    ls_round = 0

    while improved and eval_count < n_evals:
        improved = False
        ls_round += 1
        positions = list(range(n_dim))
        random.shuffle(positions)

        for d in positions:
            if eval_count >= n_evals: break
            # Try all values for this dimension
            candidates = []
            candidate_encs = []
            for val in range(int(ub[d]) + 1):
                if val == int(current[d]): continue
                trial = current.copy()
                trial[d] = val
                candidates.append(decode(trial))
                candidate_encs.append(trial)

            if not candidates: continue
            remaining = n_evals - eval_count
            candidates = candidates[:remaining]
            candidate_encs = candidate_encs[:remaining]

            batch = evaluate_batch_parallel(candidates, virtual_nets, objective, database_file, cost_aware)
            for k, (g, v, _) in enumerate(batch):
                eval_count += 1
                if v < best_value:
                    best_value, best_group = v, copy.deepcopy(g)
                if v < current_val:
                    current_val = v
                    current = candidate_encs[k].copy()
                    improved = True
                history.append({'eval': eval_count, 'value': v, 'best_value': best_value, 'eval_time': 0})

        print(f"\r[MEM] LS round {ls_round} | Evals: {eval_count}/{n_evals} | Best: {best_value:.4e}", end="")

    dt = time.perf_counter() - t0
    for h in history: h['eval_time'] = dt / len(history) if history else 0
    print(f"\n[MEM] {eval_count} evals in {dt:.1f}s | Best: {best_value:.4e}")
    return best_group, best_value, history


# ---------------------------------------------------------------------------
# Algorithm: EDA (Estimation of Distribution Algorithm)
# ---------------------------------------------------------------------------

def run_eda(virtual_nets, n_chiplets, objective, database_file,
            n_evals, cost_aware=False, seed=42,
            pop_size=30, elite_ratio=0.3, lr=0.3,
            include_pim=False, include_switch=False,
            initial_group=None):
    """EDA: learn per-position marginal distributions from elites, sample new pop."""
    random.seed(seed)
    np.random.seed(seed)
    arch_targets = _build_arch_targets(include_pim, include_switch)

    n_dim = n_chiplets * 4
    ub = np.array([len(arch_targets) - 1, len(glb_scales) - 1,
                    len(pe_scales) - 1, len(pe_scales) - 1] * n_chiplets, dtype=float)
    # Per-dimension probability tables
    dim_sizes = [int(u) + 1 for u in ub]

    def decode(x):
        chiplets = []
        for i in range(n_chiplets):
            b = i * 4
            chiplets.append(ChipletConfig(
                arch_target=arch_targets[int(np.clip(round(x[b]), 0, ub[b]))],
                global_buffer_size_scale=glb_scales[int(np.clip(round(x[b+1]), 0, ub[b+1]))],
                pe_x_scale=pe_scales[int(np.clip(round(x[b+2]), 0, ub[b+2]))],
                pe_y_scale=pe_scales[int(np.clip(round(x[b+3]), 0, ub[b+3]))]))
        return _enforce_singleton_specials(chiplets, arch_targets)

    # Init probability tables (bias toward initial_group if provided)
    probs = [np.ones(s) / s for s in dim_sizes]
    if initial_group is not None:
        seed_enc = _encode_initial_group(initial_group, n_chiplets, arch_targets)
        for d in range(n_dim):
            idx = int(np.clip(seed_enc[d], 0, dim_sizes[d] - 1))
            probs[d][idx] += 0.5  # bias toward warm-start values
            probs[d] /= probs[d].sum()

    def sample_individual():
        return np.array([np.random.choice(dim_sizes[d], p=probs[d]) for d in range(n_dim)], dtype=float)

    best_value, best_group = float('inf'), None
    history = []
    eval_count = 0
    t0 = time.perf_counter()
    n_elite = max(2, int(pop_size * elite_ratio))
    stagnant = 0
    prev_best = float('inf')

    while eval_count < n_evals:
        remaining = n_evals - eval_count
        bs = min(pop_size, remaining)
        if bs <= 0: break

        pop_arr = np.array([sample_individual() for _ in range(bs)])
        groups = [decode(pop_arr[j]) for j in range(bs)]
        batch = evaluate_batch_parallel(groups, virtual_nets, objective, database_file, cost_aware)

        values = []
        for j, (g, v, _) in enumerate(batch):
            values.append(v)
            eval_count += 1
            if v < best_value:
                best_value, best_group = v, copy.deepcopy(g)
            history.append({'eval': eval_count, 'value': v, 'best_value': best_value, 'eval_time': 0})

        # Select elite (handle inf)
        finite_vals = [(v if np.isfinite(v) else 1e18, j) for j, v in enumerate(values)]
        sorted_idx = [j for _, j in sorted(finite_vals)]
        elite = pop_arr[sorted_idx[:n_elite]]

        # Update distributions
        for d in range(n_dim):
            counts = np.zeros(dim_sizes[d])
            for ind in elite:
                counts[int(ind[d])] += 1
            empirical = counts / len(elite)
            probs[d] = (1 - lr) * probs[d] + lr * empirical
            probs[d] = probs[d] * 0.95 + 0.05 / dim_sizes[d]  # smoothing
            probs[d] /= probs[d].sum()

        # Adaptive restart
        if best_value < prev_best - 1e-6:
            stagnant = 0
            prev_best = best_value
        else:
            stagnant += 1
        if stagnant >= 5:
            for d in random.sample(range(n_dim), n_dim // 2):
                probs[d] = np.ones(dim_sizes[d]) / dim_sizes[d]
            stagnant = 0

        print(f"\r[EDA] Evals: {eval_count}/{n_evals} | Best: {best_value:.4e}", end="")

    dt = time.perf_counter() - t0
    for h in history: h['eval_time'] = dt / len(history) if history else 0
    print(f"\n[EDA] {eval_count} evals in {dt:.1f}s | Best: {best_value:.4e}")
    return best_group, best_value, history


# ---------------------------------------------------------------------------
# Algorithm: Surrogate-Assisted Evolutionary Optimization (SAEO)
# ---------------------------------------------------------------------------

def _encode_group_for_surrogate(chiplet_group, n_chiplets, arch_list=None):
    """Encode chiplet group for RF surrogate. Per-chiplet sorted + aggregate features."""
    if arch_list is None:
        arch_list = DEFAULT_ARCH_TARGETS

    # Per-chiplet features: arch one-hot + glb/pe_x/pe_y (no DRAM — decided by inner GA)
    feat_per_chiplet = len(arch_list) + 3

    def encode_one(c):
        arch_oh = [1.0 if c.arch_target == a else 0.0 for a in arch_list]
        return arch_oh + [c.global_buffer_size_scale / 16.0,
                          c.pe_x_scale / 4.0,
                          c.pe_y_scale / 4.0]

    sorted_chiplets = sorted(chiplet_group, key=lambda c: c.get_identifier())
    features = []
    for c in sorted_chiplets:
        features.extend(encode_one(c))
    while len(features) < n_chiplets * feat_per_chiplet:
        features.extend([0.0] * feat_per_chiplet)

    n = len(chiplet_group)
    for a in arch_list:
        features.append(sum(1 for c in chiplet_group if c.arch_target == a) / max(n, 1))
    pe_areas = [c.pe_x_scale * c.pe_y_scale for c in chiplet_group]
    features.append(np.mean(pe_areas) / 16.0 if pe_areas else 0)
    features.append(np.max(pe_areas) / 16.0 if pe_areas else 0)
    features.append(np.min(pe_areas) / 16.0 if pe_areas else 0)
    glb_vals = [c.global_buffer_size_scale for c in chiplet_group]
    features.append(np.sum(glb_vals) / (16.0 * max(n, 1)))
    features.append(np.max(glb_vals) / 16.0 if glb_vals else 0)
    features.append(len(set(c.arch_target for c in chiplet_group)) / len(arch_list))
    return features


def run_saeo(virtual_nets, n_chiplets, objective, database_file,
             n_evals, cost_aware=False, seed=42,
             n_bootstrap=None, candidates_per_round=2000, top_k=None,
             mutation_rate=0.3, include_pim=False, include_switch=False,
             initial_group=None, pruned_configs=None):
    """Surrogate-Assisted Evolutionary Optimization.
    RF surrogate pre-screens large candidate pools generated via mutation/crossover.
    When pruned_configs is provided, constrains the search to roofline-pruned configs."""
    from sklearn.ensemble import RandomForestRegressor

    random.seed(seed)
    np.random.seed(seed)

    arch_targets = _build_arch_targets(include_pim, include_switch)

    # When pruned_configs is set, constrain all generation to the pruned space
    if pruned_configs is not None:
        from chiplet_pruning import pruned_configs_to_options
        _pruned_opts = pruned_configs_to_options(pruned_configs)
        _p_archs = _pruned_opts['arch_targets']
        _p_glbs = _pruned_opts['glb_scales']
        _p_pe_xs = _pruned_opts['pe_x_scales']
        _p_pe_ys = _pruned_opts['pe_y_scales']
        _p_drams = _pruned_opts['dram_types']
        print(f"[SAEO] Pruning active: {len(pruned_configs)} configs "
              f"({len(_p_archs)} archs, {len(_p_glbs)} glbs, "
              f"{len(_p_pe_xs)}x{len(_p_pe_ys)} PEs, {len(_p_drams)} DRAMs)")
    else:
        _p_archs = arch_targets
        _p_glbs = glb_scales
        _p_pe_xs = pe_scales
        _p_pe_ys = pe_scales
        _p_drams = None  # no DRAM mutation when not pruning

    if n_bootstrap is None:
        n_bootstrap = max(20, n_evals // 5)
    if top_k is None:
        top_k = min(OUTER_WORKERS, max(4, n_evals // 10))

    X_train, y_train = [], []
    archive_groups, archive_values = [], []
    best_value, best_group = float('inf'), None
    history = []
    eval_count = 0
    t0_total = time.perf_counter()

    # Bootstrap (include warm-start initial_group if provided)
    print(f"[SAEO] Phase 1: Bootstrap ({n_bootstrap} random evals)...")
    bootstrap_groups = []
    if initial_group is not None:
        warm = copy.deepcopy(initial_group)
        bootstrap_groups.append(warm)
        # Also add mutations of the warm-start to seed the surrogate
        for _ in range(min(9, n_bootstrap - 1)):
            mutant = copy.deepcopy(warm)
            for c in mutant:
                if random.random() < 0.3:
                    c.arch_target = random.choice(_p_archs)
                if random.random() < 0.3:
                    c.global_buffer_size_scale = random.choice(_p_glbs)
                if random.random() < 0.3:
                    c.pe_x_scale = random.choice(_p_pe_xs)
                if random.random() < 0.3:
                    c.pe_y_scale = random.choice(_p_pe_ys)
                if _p_drams and random.random() < 0.3:
                    c.dram_type = random.choice(_p_drams)
            bootstrap_groups.append(_enforce_singleton_specials(mutant, arch_targets))
    for _ in range(n_bootstrap - len(bootstrap_groups)):
        if pruned_configs is not None:
            g = generate_pruned_chiplet_group(n_chiplets, pruned_configs)
        else:
            g = generate_chiplet_group(
                n_chiplets=n_chiplets, arch_targets=arch_targets,
                glb_scale_options=glb_scales, pe_scale_options=pe_scales)
        bootstrap_groups.append(_enforce_singleton_specials(g, arch_targets))

    batch_results = evaluate_batch_parallel(
        bootstrap_groups, virtual_nets, objective, database_file, cost_aware)

    for group, value, _ in batch_results:
        eval_count += 1
        X_train.append(_encode_group_for_surrogate(group, n_chiplets, arch_targets))
        y_train.append(value)
        archive_groups.append(group)
        archive_values.append(value)
        if value < best_value:
            best_value = value
            best_group = copy.deepcopy(group)
        history.append({'eval': eval_count, 'value': value,
                        'best_value': best_value, 'eval_time': 0})

    print(f"[SAEO] Bootstrap done: {eval_count} evals, best={best_value:.4e}")

    # Surrogate-guided rounds
    round_num = 0
    while eval_count < n_evals:
        round_num += 1
        remaining = n_evals - eval_count
        batch_this_round = min(top_k, remaining)
        if batch_this_round <= 0:
            break

        # Filter inf for surrogate training
        X_arr = np.array(X_train)
        y_arr = np.array(y_train)
        finite_mask = np.isfinite(y_arr)
        if finite_mask.sum() < 5:
            # Not enough finite data — do random exploration
            rand_groups = []
            for _ in range(batch_this_round):
                if pruned_configs is not None:
                    g = generate_pruned_chiplet_group(n_chiplets, pruned_configs)
                else:
                    g = generate_chiplet_group(
                        n_chiplets=n_chiplets, arch_targets=arch_targets,
                        glb_scale_options=glb_scales, pe_scale_options=pe_scales)
                rand_groups.append(_enforce_singleton_specials(g, arch_targets))
            batch_results = evaluate_batch_parallel(
                rand_groups, virtual_nets, objective, database_file, cost_aware)
            for group, value, _ in batch_results:
                eval_count += 1
                X_train.append(_encode_group_for_surrogate(group, n_chiplets, arch_targets))
                y_train.append(value)
                archive_groups.append(group)
                archive_values.append(value)
                if value < best_value:
                    best_value = value
                    best_group = copy.deepcopy(group)
                history.append({'eval': eval_count, 'value': value,
                                'best_value': best_value, 'eval_time': 0})
            print(f"\r[SAEO] Round {round_num} | Evals: {eval_count}/{n_evals} | "
                  f"Best: {best_value:.4e} | (random, {finite_mask.sum()} finite)", end="")
            continue

        surrogate = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=2, n_jobs=-1,
            random_state=seed + round_num)
        surrogate.fit(X_arr[finite_mask], y_arr[finite_mask])

        top_indices = np.argsort(archive_values)[:max(10, len(archive_values) // 4)]
        candidates = []
        candidate_features = []

        for _ in range(candidates_per_round):
            parent = archive_groups[random.choice(top_indices)]
            if random.random() < 0.5 and len(top_indices) > 1:
                other = archive_groups[random.choice(top_indices)]
                child = [copy.deepcopy(parent[i] if random.random() < 0.5 else other[i])
                         for i in range(min(len(parent), len(other)))]
            else:
                child = [copy.deepcopy(c) for c in parent]

            for c in child:
                if random.random() < mutation_rate:
                    c.arch_target = random.choice(_p_archs)
                if random.random() < mutation_rate:
                    c.global_buffer_size_scale = random.choice(_p_glbs)
                if random.random() < mutation_rate:
                    c.pe_x_scale = random.choice(_p_pe_xs)
                if random.random() < mutation_rate:
                    c.pe_y_scale = random.choice(_p_pe_ys)
                if _p_drams and random.random() < mutation_rate:
                    c.dram_type = random.choice(_p_drams)

            if random.random() < 0.1:
                if pruned_configs is not None:
                    child = generate_pruned_chiplet_group(n_chiplets, pruned_configs)
                else:
                    child = generate_chiplet_group(
                        n_chiplets=n_chiplets, arch_targets=arch_targets,
                        glb_scale_options=glb_scales, pe_scale_options=pe_scales)

            child = _enforce_singleton_specials(child, arch_targets)
            candidates.append(child)
            candidate_features.append(_encode_group_for_surrogate(child, n_chiplets, arch_targets))

        predicted = surrogate.predict(np.array(candidate_features))
        top_k_idx = np.argsort(predicted)[:batch_this_round]
        selected = [candidates[i] for i in top_k_idx]

        batch_results = evaluate_batch_parallel(
            selected, virtual_nets, objective, database_file, cost_aware)

        for group, value, _ in batch_results:
            eval_count += 1
            X_train.append(_encode_group_for_surrogate(group, n_chiplets, arch_targets))
            y_train.append(value)
            archive_groups.append(group)
            archive_values.append(value)
            if value < best_value:
                best_value = value
                best_group = copy.deepcopy(group)
            history.append({'eval': eval_count, 'value': value,
                            'best_value': best_value, 'eval_time': 0})

        sel_pred = predicted[top_k_idx]
        actual = [r[1] for r in batch_results]
        finite_actual = [(p, a) for p, a in zip(sel_pred, actual) if np.isfinite(a)]
        rank_corr = 0.0
        if len(finite_actual) > 2:
            pa, aa = zip(*finite_actual)
            rank_corr = np.corrcoef(
                np.argsort(np.argsort(pa)),
                np.argsort(np.argsort(aa))
            )[0, 1]

        print(f"\r[SAEO] Round {round_num} | Evals: {eval_count}/{n_evals} | "
              f"Best: {best_value:.4e} | rank-corr: {rank_corr:.2f}", end="")

    total_dt = time.perf_counter() - t0_total
    for h in history:
        h['eval_time'] = total_dt / len(history)

    print(f"\n[SAEO] {eval_count} evals in {total_dt:.1f}s ({total_dt/eval_count:.1f}s/eval) | "
          f"Best: {best_value:.4e}")
    return best_group, best_value, history


# ---------------------------------------------------------------------------
# Algorithm: Incremental-SAEO (I-SAEO)
# ---------------------------------------------------------------------------

def run_isaeo(virtual_nets, n_chiplets, objective, database_file,
              n_evals, cost_aware=False, seed=42,
              n_bootstrap=None, candidates_per_round=2000, top_k=None,
              mutation_rate=0.3, include_pim=False, include_switch=False,
              initial_group=None, pruned_configs=None, phase2_frac=0.2):
    """Incremental-aware Surrogate-Assisted Evolutionary Optimization.

    Designed for incremental chiplet pool expansion. When initial_group is
    provided (from the previous n-1 round), the search is structured in two
    phases:
      Phase 1 (80% budget): Fix the first n-1 chiplets ("prefix"), only
          mutate/search the last chiplet. Guarantees monotonic improvement.
      Phase 2 (20% budget): Fine-tune all chiplets with low mutation rate,
          allowing small global adjustments.

    Without initial_group, falls back to standard SAEO behavior.
    """
    from sklearn.ensemble import RandomForestRegressor

    random.seed(seed)
    np.random.seed(seed)

    arch_targets = _build_arch_targets(include_pim, include_switch)

    if pruned_configs is not None:
        _p_archs = sorted(set(c[0] for c in pruned_configs))
        _p_glbs = sorted(set(c[1] for c in pruned_configs))
        _p_pe_xs = sorted(set(c[2] for c in pruned_configs))
        _p_pe_ys = sorted(set(c[3] for c in pruned_configs))
        _p_drams = sorted(set(c[4] for c in pruned_configs))
    else:
        from global_parameter import dram_options as DRAM_OPTIONS
        _p_archs = arch_targets
        _p_glbs = list(glb_scales)
        _p_pe_xs = list(pe_scales)
        _p_pe_ys = list(pe_scales)
        _p_drams = list(DRAM_OPTIONS)

    if n_bootstrap is None:
        n_bootstrap = max(20, n_evals // 5)
    if top_k is None:
        top_k = min(OUTER_WORKERS, max(4, n_evals // 10))

    # Determine prefix length (fixed chiplets from previous round)
    has_prefix = initial_group is not None and len(initial_group) >= n_chiplets
    prefix_len = len(initial_group) - 1 if has_prefix else 0
    # Budget split: (1-phase2_frac) prefix-fixed, phase2_frac global fine-tune.
    # phase2_frac=0 => pure prefix-fixed: the fixed prefix is NEVER altered
    # (phase 2 is gated on phase2_budget > 0 below, so it never activates).
    _p2 = phase2_frac if has_prefix else 0.0
    phase1_budget = int(n_evals * (1.0 - _p2)) if has_prefix else n_evals
    phase2_budget = n_evals - phase1_budget

    def _mutate_chiplet(c):
        """Mutate a single chiplet."""
        new_c = copy.deepcopy(c)
        if random.random() < mutation_rate:
            new_c.arch_target = random.choice(_p_archs)
        if random.random() < mutation_rate:
            new_c.global_buffer_size_scale = random.choice(_p_glbs)
        if random.random() < mutation_rate:
            new_c.pe_x_scale = random.choice(_p_pe_xs)
        if random.random() < mutation_rate:
            new_c.pe_y_scale = random.choice(_p_pe_ys)
        if _p_drams and random.random() < mutation_rate:
            new_c.dram_type = random.choice(_p_drams)
        return new_c

    def _random_chiplet():
        """Generate a random chiplet."""
        return ChipletConfig(
            arch_target=random.choice(_p_archs),
            global_buffer_size_scale=random.choice(_p_glbs),
            pe_x_scale=random.choice(_p_pe_xs),
            pe_y_scale=random.choice(_p_pe_ys),
            dram_type=random.choice(_p_drams) if _p_drams else 'LPDDR5')

    def _make_candidate_phase1(parent_group):
        """Phase 1: only mutate the last chiplet, keep prefix fixed."""
        child = [copy.deepcopy(c) for c in parent_group]
        if random.random() < 0.3:
            # Fully randomize last chiplet
            child[-1] = _random_chiplet()
        else:
            child[-1] = _mutate_chiplet(child[-1])
        return _enforce_singleton_specials(child, arch_targets)

    def _make_candidate_phase2(parent_group):
        """Phase 2: fine-tune all chiplets with low mutation rate."""
        child = [copy.deepcopy(c) for c in parent_group]
        low_rate = 0.1  # lower than normal 0.3
        for c in child:
            if random.random() < low_rate:
                c.arch_target = random.choice(_p_archs)
            if random.random() < low_rate:
                c.global_buffer_size_scale = random.choice(_p_glbs)
            if random.random() < low_rate:
                c.pe_x_scale = random.choice(_p_pe_xs)
            if random.random() < low_rate:
                c.pe_y_scale = random.choice(_p_pe_ys)
            if _p_drams and random.random() < low_rate:
                c.dram_type = random.choice(_p_drams)
        return _enforce_singleton_specials(child, arch_targets)

    X_train, y_train = [], []
    archive_groups, archive_values = [], []
    best_value, best_group = float('inf'), None
    history = []
    eval_count = 0
    t0_total = time.perf_counter()

    # --- Bootstrap ---
    bootstrap_n = min(n_bootstrap, phase1_budget)
    print(f"[I-SAEO] Phase 1: Bootstrap ({bootstrap_n} evals, prefix_len={prefix_len})...")
    bootstrap_groups = []

    if has_prefix:
        # Include a constraint-normalized copy of the initial group. This also
        # prevents the newly appended chiplet from duplicating PIM or switch.
        bootstrap_groups.append(
            _enforce_singleton_specials(copy.deepcopy(initial_group), arch_targets)
        )
        # Generate variants: only vary the last chiplet
        for _ in range(min(bootstrap_n - 1, 19)):
            g = copy.deepcopy(initial_group)
            g[-1] = _random_chiplet()
            bootstrap_groups.append(_enforce_singleton_specials(g, arch_targets))

    # Fill remaining with random groups
    for _ in range(bootstrap_n - len(bootstrap_groups)):
        if has_prefix:
            # Random last chiplet, keep prefix
            g = copy.deepcopy(initial_group)
            g[-1] = _random_chiplet()
            bootstrap_groups.append(_enforce_singleton_specials(g, arch_targets))
        else:
            g = generate_chiplet_group(
                n_chiplets=n_chiplets, arch_targets=arch_targets,
                glb_scale_options=glb_scales, pe_scale_options=pe_scales)
            bootstrap_groups.append(_enforce_singleton_specials(g, arch_targets))

    batch_results = evaluate_batch_parallel(
        bootstrap_groups, virtual_nets, objective, database_file, cost_aware)

    for group, value, _ in batch_results:
        eval_count += 1
        X_train.append(_encode_group_for_surrogate(group, n_chiplets, arch_targets))
        y_train.append(value)
        archive_groups.append(group)
        archive_values.append(value)
        if value < best_value:
            best_value = value
            best_group = copy.deepcopy(group)
        history.append({'eval': eval_count, 'value': value,
                        'best_value': best_value, 'eval_time': 0})

    print(f"[I-SAEO] Bootstrap done: {eval_count} evals, best={best_value:.4e}")

    # --- Surrogate-guided Phase 1: search only new chiplet ---
    round_num = 0
    current_phase = 1
    phase1_limit = phase1_budget

    while eval_count < n_evals:
        # Switch to phase 2 when budget reached
        if current_phase == 1 and eval_count >= phase1_limit and phase2_budget > 0:
            current_phase = 2
            print(f"\n[I-SAEO] Switching to Phase 2: global fine-tune ({phase2_budget} evals remaining)")

        round_num += 1
        remaining = n_evals - eval_count
        batch_this_round = min(top_k, remaining)
        if batch_this_round <= 0:
            break

        X_arr = np.array(X_train)
        y_arr = np.array(y_train)
        finite_mask = np.isfinite(y_arr)

        if finite_mask.sum() < 5:
            # Not enough data — random exploration
            rand_groups = []
            for _ in range(batch_this_round):
                if has_prefix and current_phase == 1:
                    g = copy.deepcopy(initial_group)
                    g[-1] = _random_chiplet()
                    rand_groups.append(_enforce_singleton_specials(g, arch_targets))
                else:
                    g = generate_chiplet_group(
                        n_chiplets=n_chiplets, arch_targets=arch_targets,
                        glb_scale_options=glb_scales, pe_scale_options=pe_scales)
                    rand_groups.append(_enforce_singleton_specials(g, arch_targets))
            batch_results = evaluate_batch_parallel(
                rand_groups, virtual_nets, objective, database_file, cost_aware)
            for group, value, _ in batch_results:
                eval_count += 1
                X_train.append(_encode_group_for_surrogate(group, n_chiplets, arch_targets))
                y_train.append(value)
                archive_groups.append(group)
                archive_values.append(value)
                if value < best_value:
                    best_value = value
                    best_group = copy.deepcopy(group)
                history.append({'eval': eval_count, 'value': value,
                                'best_value': best_value, 'eval_time': 0})
            continue

        surrogate = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=2, n_jobs=-1,
            random_state=seed + round_num)
        surrogate.fit(X_arr[finite_mask], y_arr[finite_mask])

        top_indices = np.argsort(archive_values)[:max(10, len(archive_values) // 4)]
        candidates = []
        candidate_features = []

        make_candidate = _make_candidate_phase1 if current_phase == 1 else _make_candidate_phase2

        for _ in range(candidates_per_round):
            parent = archive_groups[random.choice(top_indices)]

            if current_phase == 1:
                # Phase 1: crossover only on last chiplet
                if random.random() < 0.5 and len(top_indices) > 1:
                    other = archive_groups[random.choice(top_indices)]
                    child = [copy.deepcopy(c) for c in parent]
                    child[-1] = copy.deepcopy(random.choice([parent[-1], other[-1]]))
                    child[-1] = _mutate_chiplet(child[-1])
                else:
                    child = make_candidate(parent)
                # 10% chance: fully random last chiplet
                if random.random() < 0.1:
                    child = copy.deepcopy(parent)
                    child[-1] = _random_chiplet()
            else:
                # Phase 2: crossover on all positions, low mutation
                if random.random() < 0.5 and len(top_indices) > 1:
                    other = archive_groups[random.choice(top_indices)]
                    child = [copy.deepcopy(parent[i] if random.random() < 0.5 else other[i])
                             for i in range(min(len(parent), len(other)))]
                else:
                    child = [copy.deepcopy(c) for c in parent]
                child = _make_candidate_phase2(child)

            child = _enforce_singleton_specials(child, arch_targets)
            candidates.append(child)
            candidate_features.append(_encode_group_for_surrogate(child, n_chiplets, arch_targets))

        predicted = surrogate.predict(np.array(candidate_features))
        top_k_idx = np.argsort(predicted)[:batch_this_round]
        selected = [candidates[i] for i in top_k_idx]

        batch_results = evaluate_batch_parallel(
            selected, virtual_nets, objective, database_file, cost_aware)

        for group, value, _ in batch_results:
            eval_count += 1
            X_train.append(_encode_group_for_surrogate(group, n_chiplets, arch_targets))
            y_train.append(value)
            archive_groups.append(group)
            archive_values.append(value)
            if value < best_value:
                best_value = value
                best_group = copy.deepcopy(group)
            history.append({'eval': eval_count, 'value': value,
                            'best_value': best_value, 'eval_time': 0})

        sel_pred = predicted[top_k_idx]
        actual = [r[1] for r in batch_results]
        finite_actual = [(p, a) for p, a in zip(sel_pred, actual) if np.isfinite(a)]
        rank_corr = 0.0
        if len(finite_actual) > 2:
            pa, aa = zip(*finite_actual)
            rank_corr = np.corrcoef(
                np.argsort(np.argsort(pa)),
                np.argsort(np.argsort(aa))
            )[0, 1]

        phase_tag = f"P{current_phase}"
        print(f"\r[I-SAEO] {phase_tag} Round {round_num} | Evals: {eval_count}/{n_evals} | "
              f"Best: {best_value:.4e} | rank-corr: {rank_corr:.2f}", end="")

    total_dt = time.perf_counter() - t0_total
    for h in history:
        h['eval_time'] = total_dt / len(history)

    print(f"\n[I-SAEO] {eval_count} evals in {total_dt:.1f}s ({total_dt/eval_count:.1f}s/eval) | "
          f"Best: {best_value:.4e}")
    return best_group, best_value, history


# ---------------------------------------------------------------------------
# Main: load workloads, run algorithms, save results
# ---------------------------------------------------------------------------

def setup_virtual_nets(database_file, include_cnn=False):
    """Load virtual networks (same as chiplet_sel.py main)."""
    virtual_nets = []
    for model in ['llama3.1_8b', 'llama3.1_70b', 'qwen3_30b_a3b', 'qwen3_235b_a22b']:
        virtual_nets.append(VirtualNetwork(f"{model}_prefill_s1024", batch_size=1, sequence_length=1024))
        virtual_nets.append(VirtualNetwork(f"{model}_prefill_s1024", batch_size=8, sequence_length=1024))
        virtual_nets.append(VirtualNetwork(f"{model}_decode_kv1024", batch_size=1, sequence_length=1))
        virtual_nets.append(VirtualNetwork(f"{model}_decode_kv1024", batch_size=8, sequence_length=1))

    if include_cnn:
        for cnn in ['mobilenet_v3_small', 'replknet31b']:
            virtual_nets.append(VirtualNetwork(cnn, batch_size=1, sequence_length=1))
            virtual_nets.append(VirtualNetwork(cnn, batch_size=8, sequence_length=1))

    # Load database to get available layers per network
    _db = pd.read_csv(database_file)
    _db_layers_per_net = {
        net: set(_db[_db['net'] == net]['layer_name'].unique())
        for net in _db['net'].unique()
    }

    for virtual_net in virtual_nets:
        db_layers = _db_layers_per_net.get(virtual_net.network_name, None)
        try:
            virtual_net.load_from_dir(os.path.join(NET_DIR, virtual_net.network_name), db_layers=db_layers)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load {virtual_net.network_name}: {e}")
    virtual_nets = [vn for vn in virtual_nets if len(vn.layers) > 0]

    # Pre-load and filter database for fast evaluation
    needed_nets = set(vn.network_name for vn in virtual_nets)
    preload_database(database_file, needed_nets=needed_nets)

    return virtual_nets


def save_results(all_results, output_dir, objective, n_chiplets):
    """Save comparison results to CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Save convergence history for each algorithm
    for algo_name, result in all_results.items():
        csv_file = os.path.join(output_dir, f"convergence_{algo_name}_{timestamp}.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['eval', 'value', 'best_value', 'eval_time'])
            writer.writeheader()
            for row in result['history']:
                writer.writerow(row)

    # Save summary comparison
    summary = {}
    for algo_name, result in all_results.items():
        summary[algo_name] = {
            'best_value': result['best_value'],
            'total_evals': len(result['history']),
            'total_time': sum(h['eval_time'] for h in result['history']),
            'best_chiplets': [c.get_identifier() for c in result['best_group']] if result['best_group'] else [],
        }

    summary_file = os.path.join(output_dir, f"summary_{objective}_n{n_chiplets}_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"Results Summary: {objective}, n_chiplets={n_chiplets}")
    print("=" * 70)
    metric = "EDP" if objective == "edp" else "Energy"
    print(f"{'Algorithm':<12} {'Best ' + metric:<18} {'Evals':<8} {'Time (s)':<12}")
    print("-" * 50)
    for algo_name, s in summary.items():
        print(f"{algo_name:<12} {s['best_value']:<18.4e} {s['total_evals']:<8} {s['total_time']:<12.1f}")

    print(f"\nResults saved to {output_dir}/")
    return summary_file


def run_incremental_sweep(virtual_nets, objective, database_file,
                          n_start, n_end, evals_per_n, cost_aware=False,
                          seed=42, pop_size=20, mutation_rate=0.1,
                          pruned_configs=None,
                          include_pim=False, include_switch=False,
                          algorithm='ga', algo_kwargs=None,
                          results_file=None,
                          initial_best_group=None):
    """Incremental chiplet pool sweep: n=K+1 warm-starts from n=K's best + 1 random chiplet.

    This mirrors chiplet_sel.py's run_incremental_n_chiplet_sweep but using the
    selected algorithm. Results should be monotonically improving (larger pool = superset).

    Args:
        initial_best_group: Optional chiplet group to seed the first n_start round.
            Used for chaining (e.g., SAEO n=1-6 → I-SAEO n=7-10).

    Writes a chiplet_sel-style CSV with per-network metrics and per-chiplet composition.

    Returns:
        Dict mapping n_chiplets -> {best_group, best_value, history, network_results}
    """
    from utility_functions import config_to_ids

    if algo_kwargs is None:
        algo_kwargs = {}

    # --- Setup CSV with chiplet_sel-style columns ---
    if results_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f'incremental_chiplet_sweep_{objective}_{timestamp}.csv'

    fieldnames = ['n_chiplets', f'best_{objective}']
    for vn in virtual_nets:
        uname = vn.get_unique_name()
        fieldnames.append(f'{uname}_min_{objective}')
        fieldnames.append(f'{uname}_latency')
        fieldnames.append(f'{uname}_config')
        fieldnames.append(f'{uname}_gene')
    for i in range(1, n_end + 1):
        fieldnames.extend([
            f'chiplet_{i}_arch',
            f'chiplet_{i}_glb_scale',
            f'chiplet_{i}_pe_x_scale',
            f'chiplet_{i}_pe_y_scale',
        ])

    with open(results_file, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    all_results = {}
    prev_best_group = copy.deepcopy(initial_best_group) if initial_best_group is not None else None

    for n_chiplets in range(n_start, n_end + 1):
        print(f"\n{'='*60}")
        print(f"Incremental {algorithm.upper()}: n_chiplets={n_chiplets}")
        print(f"{'='*60}")

        # Build initial group: previous best + 1 random chiplet
        initial_group = None
        if prev_best_group is not None and n_chiplets > len(prev_best_group):
            initial_group = copy.deepcopy(prev_best_group)
            while len(initial_group) < n_chiplets:
                if pruned_configs is not None:
                    cfg = random.choice(pruned_configs)
                    initial_group.append(ChipletConfig(
                        arch_target=cfg[0], global_buffer_size_scale=cfg[1],
                        pe_x_scale=cfg[2], pe_y_scale=cfg[3], dram_type=cfg[4]))
                else:
                    sweep_arch_targets = _build_arch_targets(include_pim, include_switch)
                    initial_group.append(ChipletConfig(
                        arch_target=random.choice(sweep_arch_targets),
                        global_buffer_size_scale=random.choice(glb_scales),
                        pe_x_scale=random.choice(pe_scales),
                        pe_y_scale=random.choice(pe_scales)))
            print(f"  Warm-started from n={n_chiplets-1} best + 1 random chiplet")
        else:
            print(f"  Cold start (no previous solution)")

        t0 = time.perf_counter()
        common = dict(virtual_nets=virtual_nets, n_chiplets=n_chiplets,
                      objective=objective, database_file=database_file,
                      n_evals=evals_per_n, cost_aware=cost_aware, seed=seed,
                      include_pim=include_pim, include_switch=include_switch)
        if algorithm == 'ga':
            best_group, best_value, history = run_ga(
                **common, pop_size=pop_size, mutation_rate=mutation_rate,
                initial_group=initial_group, pruned_configs=pruned_configs)
        elif algorithm == 'sa':
            best_group, best_value, history = run_sa(
                **common, initial_group=initial_group, **algo_kwargs)
        elif algorithm == 'random':
            best_group, best_value, history = run_random_search(**common)
        elif algorithm == 'bo':
            best_group, best_value, history = run_bo(
                virtual_nets, n_chiplets, objective, database_file,
                evals_per_n, cost_aware, seed, **algo_kwargs)
        elif algorithm == 'de':
            best_group, best_value, history = run_de(
                **common, initial_group=initial_group)
        elif algorithm == 'pso':
            best_group, best_value, history = run_pso(
                **common, initial_group=initial_group)
        elif algorithm == 'memetic':
            best_group, best_value, history = run_memetic(
                **common, initial_group=initial_group)
        elif algorithm == 'eda':
            best_group, best_value, history = run_eda(
                **common, initial_group=initial_group)
        elif algorithm == 'saeo':
            best_group, best_value, history = run_saeo(
                **common, initial_group=initial_group,
                pruned_configs=pruned_configs, **algo_kwargs)
        elif algorithm == 'isaeo':
            best_group, best_value, history = run_isaeo(
                **common, initial_group=initial_group,
                pruned_configs=pruned_configs, **algo_kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        dt = time.perf_counter() - t0

        # Monotonic clamp: if new result is worse than previous, clamp the value
        # but keep the current n-sized group (don't copy the smaller group)
        prev_best_value = all_results[n_chiplets - 1]['best_value'] if (n_chiplets - 1) in all_results else float('inf')
        if best_value > prev_best_value and prev_best_group is not None:
            print(f"\n  [CLAMP] n={n_chiplets} result ({best_value:.4f}) worse than "
                  f"n={n_chiplets-1} ({prev_best_value:.4f}), clamping value")
            best_value = prev_best_value

        prev_best_group = copy.deepcopy(best_group)

        # Re-evaluate best group to get per-network results (cached, fast)
        net_results = {}
        if best_group is not None:
            _, raw_results = run_single_optimization(
                virtual_nets=virtual_nets,
                chiplet_group=best_group,
                objective=objective,
                results_file=database_file,
                cost_aware=cost_aware,
                use_sequential=True,
                n_workers=8,
                use_dag_cp=_USE_DAG_CP,
                v_het_batch=_V_HET_BATCH,
            )
            net_results = raw_results

        all_results[n_chiplets] = {
            'best_group': best_group,
            'best_value': best_value,
            'history': history,
            'time': dt,
            'network_results': net_results,
        }

        # --- Write CSV row (chiplet_sel style) ---
        row = {'n_chiplets': n_chiplets, f'best_{objective}': best_value}
        for vn in virtual_nets:
            uname = vn.get_unique_name()
            nr = net_results.get(uname, {})
            row[f'{uname}_min_{objective}'] = nr.get('min_value', float('inf'))
            row[f'{uname}_latency'] = nr.get('best_latency', float('inf'))
            try:
                row[f'{uname}_config'] = config_to_ids(nr['best_config']) if nr.get('best_config') else ''
            except Exception:
                row[f'{uname}_config'] = ''
            row[f'{uname}_gene'] = nr.get('best_gene', '')
        if best_group is not None:
            for i, chiplet in enumerate(best_group):
                row[f'chiplet_{i+1}_arch'] = chiplet.arch_target
                row[f'chiplet_{i+1}_glb_scale'] = chiplet.global_buffer_size_scale
                row[f'chiplet_{i+1}_pe_x_scale'] = chiplet.pe_x_scale
                row[f'chiplet_{i+1}_pe_y_scale'] = chiplet.pe_y_scale

        try:
            with open(results_file, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
            print(f"  Results written to {results_file}")
        except Exception as e:
            print(f"  WARNING: CSV write failed: {e}")

        # --- Console summary ---
        metric_name = "EDP" if objective == "edp" else "Energy"
        print(f"\n--- Summary for n_chiplets = {n_chiplets} ---")
        print(f"Best {metric_name}: {best_value:.2e}")
        if best_group is not None:
            print("Chiplet configurations:")
            for i, chiplet in enumerate(best_group):
                print(f"  Chiplet {i+1}:")
                print(f"    Architecture: {chiplet.arch_target}")
                print(f"    GLB Scale: {chiplet.global_buffer_size_scale}")
                print(f"    PE X Scale: {chiplet.pe_x_scale}")
                print(f"    PE Y Scale: {chiplet.pe_y_scale}")
            print(f"Network-specific {metric_name.lower()} results:")
            for uname, nr in net_results.items():
                print(f"  {uname}: {nr['min_value']:.2e}")
        else:
            print("  WARNING: No valid solution found for this n_chiplets")

    # Print final summary
    print(f"\n{'='*60}")
    print("Incremental Sweep Summary")
    print(f"{'='*60}")
    metric_name = "EDP" if objective == "edp" else "Energy"
    print(f"{'n':>3} {'best_value':>12} {'time(s)':>8}")
    for n in sorted(all_results.keys()):
        r = all_results[n]
        print(f"{n:3d} {r['best_value']:12.4e} {r['time']:8.1f}")

    best_n = min(all_results.items(), key=lambda x: x[1]['best_value'])
    print(f"\nOverall best: n_chiplets = {best_n[0]}, {metric_name.lower()} = {best_n[1]['best_value']:.5e}")
    print(f"\nResults saved to {results_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Compare optimization algorithms for chiplet pool selection")
    parser.add_argument("--objective", type=str, choices=["energy", "edp"], default="energy")
    parser.add_argument("--n-chiplets", type=int, default=8)
    parser.add_argument("--database", type=str, default="final_database.csv")
    parser.add_argument("--n-evals", type=int, default=200,
                        help="Total evaluation budget per algorithm")
    parser.add_argument("--algorithms", nargs='+', default=["random", "sa", "bo", "ga"],
                        choices=["random", "sa", "bo", "ga", "de", "pso", "memetic", "eda", "saeo", "isaeo"],
                        help="Algorithms to run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cost-aware", action="store_true", default=False)
    parser.add_argument("--output-dir", type=str, default="archgym_results",
                        help="Directory for output files")
    # BO-specific
    parser.add_argument("--bo-acq-func", type=str, default="EI",
                        choices=["EI", "LCB", "PI", "gp_hedge"])
    parser.add_argument("--bo-initial-points", type=int, default=10)
    # GA-specific
    parser.add_argument("--ga-pop-size", type=int, default=20)
    parser.add_argument("--ga-mutation-rate", type=float, default=0.1)
    # SA-specific
    parser.add_argument("--sa-initial-temp", type=float, default=1.0)
    parser.add_argument("--sa-cooling-rate", type=float, default=0.95)
    parser.add_argument("--sa-min-temp", type=float, default=0.01)
    # Incremental sweep
    parser.add_argument("--method", type=str, choices=["compare", "incremental"], default="compare",
                        help="'compare': compare algorithms at fixed n_chiplets. "
                             "'incremental': sweep n_chiplets with warm-start (n=K+1 from n=K best + 1 random)")
    parser.add_argument("--n-start", type=int, default=1, help="Start n_chiplets for incremental sweep")
    parser.add_argument("--n-end", type=int, default=16, help="End n_chiplets for incremental sweep")
    # Pruning
    parser.add_argument("--prune-pct", type=int, default=0,
                        help="Roofline pruning: keep top X%% configs per network (0=disabled, 30 recommended)")
    # Inner GA tuning
    parser.add_argument("--inner-ga-pop", type=int, default=10,
                        help="Inner GA population size (layer fusion optimization)")
    parser.add_argument("--inner-ga-gen", type=int, default=10,
                        help="Inner GA generations (layer fusion optimization)")
    # PIM / switch / CNN / DAG
    parser.add_argument("--pim", action="store_true", default=True,
                        help="Include PIM chiplet at n>=2 (default: True)")
    parser.add_argument("--no-pim", dest="pim", action="store_false")
    parser.add_argument("--switch", action="store_true", default=True,
                        help="Include switch chiplet for MoE EP at n>=2 (default: True)")
    parser.add_argument("--no-switch", dest="switch", action="store_false")
    parser.add_argument("--cnn", action="store_true", default=True,
                        help="Include CNN workloads (MobileNetV3, RepLKNet)")
    parser.add_argument("--dag", action="store_true", default=True,
                        help="Use DAG critical-path inner GA (default: True)")
    parser.add_argument("--no-dag", dest="dag", action="store_false")
    parser.add_argument("--v-het-batch", type=str, default="True",
                        help="Off-CP ops use larger batch for energy amortization")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Set module-level flags for worker processes (read via fork)
    global _USE_DAG_CP, _V_HET_BATCH
    _USE_DAG_CP = args.dag
    _V_HET_BATCH = args.v_het_batch != "False"

    logger.info(f"Objective: {args.objective}, n_chiplets: {args.n_chiplets}")
    logger.info(f"Algorithms: {args.algorithms}, n_evals: {args.n_evals}")
    logger.info(f"PIM: {args.pim}, Switch: {args.switch}, CNN: {args.cnn}, DAG: {args.dag}")

    # Set fork start method for mp.Pool (must be before any Pool creation)
    import multiprocessing as mp
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass

    # Load virtual networks
    logger.info("Loading virtual networks...")
    virtual_nets = setup_virtual_nets(args.database, include_cnn=args.cnn)
    logger.info(f"Loaded {len(virtual_nets)} virtual networks")

    # Roofline pruning
    pruned_configs = None
    if args.prune_pct > 0:
        needed_nets = set(vn.network_name for vn in virtual_nets)
        pruned_configs = get_pruned_configs(
            args.database, top_pct=args.prune_pct,
            needed_nets=needed_nets, cost_aware=args.cost_aware)
        print_pruning_summary(pruned_configs)

    if args.method == "incremental":
        # Incremental sweep: n=n_start to n_end, warm-starting each from previous
        algo = args.algorithms[0]  # use first algorithm for incremental sweep
        logger.info(f"Incremental sweep ({algo.upper()}): n={args.n_start} to {args.n_end}, {args.n_evals} evals per n")
        # Build algorithm-specific kwargs
        algo_kwargs = {}
        if algo == 'sa':
            algo_kwargs = dict(initial_temp=args.sa_initial_temp,
                               cooling_rate=args.sa_cooling_rate,
                               min_temp=args.sa_min_temp)
        elif algo == 'bo':
            algo_kwargs = dict(n_initial_points=args.bo_initial_points,
                               acq_func=args.bo_acq_func)

        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_csv = os.path.join(
            args.output_dir,
            f'incremental_chiplet_sweep_{args.objective}_{timestamp}.csv')

        sweep_results = run_incremental_sweep(
            virtual_nets, args.objective, args.database,
            n_start=args.n_start, n_end=args.n_end,
            evals_per_n=args.n_evals, cost_aware=args.cost_aware,
            seed=args.seed, pop_size=args.ga_pop_size,
            mutation_rate=args.ga_mutation_rate,
            pruned_configs=pruned_configs,
            include_pim=args.pim, include_switch=args.switch,
            algorithm=algo, algo_kwargs=algo_kwargs,
            results_file=results_csv,
        )

        # Also save per-n convergence history
        for n, result in sweep_results.items():
            csv_file = os.path.join(args.output_dir, f"convergence_{algo}_n{n}_{timestamp}.csv")
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['eval', 'value', 'best_value', 'eval_time'])
                writer.writeheader()
                for row in result['history']:
                    writer.writerow(row)

    else:
        # Compare algorithms at fixed n_chiplets
        all_results = {}

        algorithm_runners = {
            'random': lambda: run_random_search(
                virtual_nets, args.n_chiplets, args.objective, args.database,
                args.n_evals, args.cost_aware, args.seed,
                include_pim=args.pim, include_switch=args.switch),
            'sa': lambda: run_sa(
                virtual_nets, args.n_chiplets, args.objective, args.database,
                args.n_evals, args.cost_aware, args.seed,
                args.sa_initial_temp, args.sa_cooling_rate, args.sa_min_temp,
                include_pim=args.pim, include_switch=args.switch),
            'bo': lambda: run_bo(
                virtual_nets, args.n_chiplets, args.objective, args.database,
                args.n_evals, args.cost_aware, args.seed,
                args.bo_initial_points, args.bo_acq_func),
            'ga': lambda: run_ga(
                virtual_nets, args.n_chiplets, args.objective, args.database,
                args.n_evals, args.cost_aware, args.seed,
                args.ga_pop_size, args.ga_mutation_rate,
                include_pim=args.pim, include_switch=args.switch),
            'de': lambda: run_de(
                virtual_nets, args.n_chiplets, args.objective, args.database,
                args.n_evals, args.cost_aware, args.seed,
                include_pim=args.pim, include_switch=args.switch),
            'pso': lambda: run_pso(
                virtual_nets, args.n_chiplets, args.objective, args.database,
                args.n_evals, args.cost_aware, args.seed,
                include_pim=args.pim, include_switch=args.switch),
            'memetic': lambda: run_memetic(
                virtual_nets, args.n_chiplets, args.objective, args.database,
                args.n_evals, args.cost_aware, args.seed,
                include_pim=args.pim, include_switch=args.switch),
            'eda': lambda: run_eda(
                virtual_nets, args.n_chiplets, args.objective, args.database,
                args.n_evals, args.cost_aware, args.seed,
                include_pim=args.pim, include_switch=args.switch),
            'saeo': lambda: run_saeo(
                virtual_nets, args.n_chiplets, args.objective, args.database,
                args.n_evals, args.cost_aware, args.seed,
                include_pim=args.pim, include_switch=args.switch,
                pruned_configs=pruned_configs),
            'isaeo': lambda: run_isaeo(
                virtual_nets, args.n_chiplets, args.objective, args.database,
                args.n_evals, args.cost_aware, args.seed,
                include_pim=args.pim, include_switch=args.switch,
                pruned_configs=pruned_configs),
        }

        for algo in args.algorithms:
            print(f"\n{'='*60}")
            print(f"Running {algo.upper()}...")
            print(f"{'='*60}")

            t0 = time.perf_counter()
            best_group, best_value, history = algorithm_runners[algo]()
            total_time = time.perf_counter() - t0

            all_results[algo] = {
                'best_group': best_group,
                'best_value': best_value,
                'history': history,
                'total_time': total_time,
            }

            metric = "EDP" if args.objective == "edp" else "Energy"
            print(f"[{algo.upper()}] Best {metric}: {best_value:.4e} in {total_time:.1f}s")
            if best_group:
                for i, c in enumerate(best_group):
                    print(f"  Chiplet {i}: {c.get_identifier()}")

        # Save all results
        save_results(all_results, args.output_dir, args.objective, args.n_chiplets)


if __name__ == "__main__":
    main()
