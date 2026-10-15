"""
run_archgym_chiplet.py — incremental chiplet-pool search (Figure 8).

run_incremental_sweep grows a pool one chiplet at a time: the search at N = K + 1 starts from
the best N = K pool plus one random chiplet. Each N is searched with simulated annealing
(run_sa, Figure 8's SA baseline, run from this script's command line), SAEO (run_saeo) or
I-SAEO (run_isaeo); run_saeo_isaeo_chain.py and converge_chain.py drive the last two. A
pool's score is the geomean over the 20 networks of each network's inner-GA mapping
(chiplet_sel.run_single_optimization).

Usage (Figure 8's SA baseline; add --cost-aware for the x cost metrics):
    python3 run_archgym_chiplet.py --method incremental --algorithms sa \
        --objective energy --database ../unified_database.csv \
        --n-start 1 --n-end 10 --n-evals 500 --seed 42 --output-dir DIR
"""

import os
import sys
import time
import copy
import random
import logging
import argparse
import csv
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
from chiplet_sel import run_single_optimization, generate_neighbor_group_optimized
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
    """Evaluate one chiplet group (the first evaluation of run_sa)."""
    _, results = run_single_optimization(
        virtual_nets=virtual_nets,
        chiplet_group=chiplet_group,
        objective=objective,
        results_file=database_file,
        cost_aware=cost_aware,
        prev_best_genes=prev_best_genes,
        use_sequential=True,
        n_workers=8,  # inner parallelism: the networks of this one group
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
# Main: load workloads, run the incremental sweep, save results
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


def run_incremental_sweep(virtual_nets, objective, database_file,
                          n_start, n_end, evals_per_n, *, algorithm,
                          cost_aware=False, seed=42,
                          pruned_configs=None,
                          include_pim=False, include_switch=False,
                          algo_kwargs=None,
                          results_file=None,
                          initial_best_group=None):
    """Incremental chiplet pool sweep: n=K+1 warm-starts from n=K's best + 1 random chiplet.

    algorithm is 'sa', 'saeo' or 'isaeo'. Results should be monotonically improving
    (larger pool = superset).

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
        if algorithm == 'sa':
            best_group, best_value, history = run_sa(
                **common, initial_group=initial_group, **algo_kwargs)
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
    parser = argparse.ArgumentParser(
        description="Incremental chiplet-pool search with simulated annealing (Figure 8's SA baseline)")
    parser.add_argument("--method", type=str, choices=["incremental"], required=True,
                        help="'incremental': sweep n_chiplets with warm-start (n=K+1 from n=K best + 1 random)")
    parser.add_argument("--algorithms", nargs='+', choices=["sa"], required=True,
                        help="search algorithm: 'sa' (simulated annealing)")
    parser.add_argument("--objective", type=str, choices=["energy", "edp"], default="energy")
    parser.add_argument("--database", type=str, default="../unified_database.csv")
    parser.add_argument("--n-evals", type=int, default=500,
                        help="Evaluation budget per n_chiplets (Figure 8: 500)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cost-aware", action="store_true", default=False)
    parser.add_argument("--output-dir", type=str, default="archgym_results",
                        help="Directory for output files")
    # SA-specific
    parser.add_argument("--sa-initial-temp", type=float, default=1.0)
    parser.add_argument("--sa-cooling-rate", type=float, default=0.95)
    parser.add_argument("--sa-min-temp", type=float, default=0.01)
    # Incremental sweep
    parser.add_argument("--n-start", type=int, default=1, help="Start n_chiplets for incremental sweep")
    parser.add_argument("--n-end", type=int, default=10, help="End n_chiplets for incremental sweep (Figure 8: 10)")
    # Pruning
    parser.add_argument("--prune-pct", type=int, default=0,
                        help="Roofline pruning: keep top X%% configs per network (0 = no pruning, as in Figure 8)")
    # PIM / switch / CNN / DAG
    parser.add_argument("--pim", action="store_true", default=True,
                        help="Include PIM chiplet at n>=2 (default: True)")
    parser.add_argument("--no-pim", dest="pim", action="store_false")
    parser.add_argument("--switch", action="store_true", default=True,
                        help="Include switch chiplet for MoE EP at n>=2 (default: True)")
    parser.add_argument("--no-switch", dest="switch", action="store_false")
    parser.add_argument("--cnn", action="store_true", default=True,
                        help="Include CNN workloads (MobileNetV3, RepLKNet); always on")
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

    logger.info(f"Objective: {args.objective}, cost-aware: {args.cost_aware}")
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

    # Incremental sweep: n=n_start to n_end, warm-starting each from previous
    algo = args.algorithms[0]
    logger.info(f"Incremental sweep ({algo.upper()}): n={args.n_start} to {args.n_end}, {args.n_evals} evals per n")
    algo_kwargs = dict(initial_temp=args.sa_initial_temp,
                       cooling_rate=args.sa_cooling_rate,
                       min_temp=args.sa_min_temp)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_csv = os.path.join(
        args.output_dir,
        f'incremental_chiplet_sweep_{args.objective}_{timestamp}.csv')

    sweep_results = run_incremental_sweep(
        virtual_nets, args.objective, args.database,
        n_start=args.n_start, n_end=args.n_end,
        evals_per_n=args.n_evals, cost_aware=args.cost_aware,
        seed=args.seed,
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


if __name__ == "__main__":
    main()
