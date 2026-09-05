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

# ── Determinism for artifact evaluation ──────────────────────────────────────
# The inner GA (genetic_algo_opt_phy_net{,_dag}) draws from Python's *global*
# `random` stream, so results depend on call order and worker count (the
# "run-order shifts values 1–2%" landmine). For reproducible artifact
# evaluation we run each per-network GA under a global-RNG state that is
#   (1) SAVED before the GA and RESTORED after — so an enclosing SA pool-search
#       keeps its own RNG stream intact (the GA and SA share `random`), and
#   (2) SEEDED from a STABLE hash of the network's unique name — hashlib, NOT
#       builtin hash(), which is PYTHONHASHSEED-salted and differs across runs.
# The GA touches only `random` (never np.random), so np.random is left alone.
# Set FENGSHUI_DETERMINISTIC=0 to restore the legacy stochastic behavior;
# FENGSHUI_GA_SEED shifts the whole seed family (stability-across-seeds checks).
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


def evaluate_neighbor(current_group, temp, thread_id, virtual_nets, arch_targets, objective,
                      logger, results_file, cost_aware, prev_best_genes=None,
                      use_dag_cp=False, v_het_batch=True):
    """Function to evaluate a neighbor solution"""
    # Generate neighbor

    neighbor_group = generate_neighbor_group_optimized(
        current_group,
        arch_targets,
        glb_scale_options=global_parameter.glb_scales,
        pe_scale_options=global_parameter.pe_scales,
        temperature=temp
    )

    try:
        start_time = time.perf_counter()
        neighbor_results = run_single_optimization(
            virtual_nets=virtual_nets,
            chiplet_group=neighbor_group,
            objective=objective,
            logger=logger,
            results_file=results_file,
            cost_aware=cost_aware,
            prev_best_genes=prev_best_genes,
            use_dag_cp=use_dag_cp,
            v_het_batch=v_het_batch,
        )[1]
        
        neighbor_value, neighbor_network_results = calculate_average_opt_value(neighbor_results, objective)
        eval_time = time.perf_counter() - start_time
        
        if logger:
            logger.debug(f"Thread {thread_id}: Evaluated solution in {eval_time:.2f}s")
        
        return neighbor_group, neighbor_results, neighbor_value, neighbor_network_results
    except Exception as e:
        if logger:
            logger.error(f"Thread {thread_id}: Error evaluating solution: {e}")
        else:
            print(f"Thread {thread_id}: Error evaluating solution: {e}")
        return None, None, None, None

def simulated_annealing_optimization(
    arch_targets: List[str],
    virtual_nets: List[VirtualNetwork],
    n_chiplets: int,
    objective: str = "energy",
    initial_temp: float = 1.0,
    cooling_rate: float = 0.95,
    min_temp: float = 0.01,
    iterations_per_temp: int = 5,
    results_file: str = "test_results.csv",
    logger: Optional[logging.Logger] = None,
    n_workers: Optional[int] = None,  # Number of parallel workers
    early_termination_threshold: int = 10,  # Number of consecutive non-improving temperature levels
    early_termination_tolerance: float = 1e-5,  # Relative improvement threshold
    cost_aware: bool=False,
    prev_best_genes: Optional[Dict[str, Dict]] = None,
    initial_group: Optional[List] = None,
    use_dag_cp: bool = False,
    v_het_batch: bool = True,
) -> Tuple[List, Dict, float, Dict]:
    """
    Optimized simulated annealing implementation with architecture type changes.
    Uses multi-threading to parallelize iterations at each temperature level.
    Includes early termination if no significant improvement is found.

    Parameters:
    ----------
    (original parameters)

    early_termination_threshold: int
        Stop optimization if no significant improvement occurs for this many consecutive temperature levels
    early_termination_tolerance: float
        Minimum relative improvement to consider significant (improvement/best_value)
    use_dag_cp: bool
        If True, use DAG critical-path GA instead of linear GA.
    v_het_batch: bool
        If True and use_dag_cp, off-CP ops use larger batch for energy amortization.
    """
    import concurrent.futures
    import multiprocessing
    import time
    
    # Initialize current solution
    if initial_group is not None:
        # Use the provided initial group
        current_group = copy.deepcopy(initial_group)
        if logger:
            logger.info(f"Starting with provided initial group of {len(current_group)} chiplets")
    else:
        # Generate a random initial group
        current_group = generate_chiplet_group(
            n_chiplets=n_chiplets,
            arch_targets=arch_targets,
            glb_scale_options=global_parameter.glb_scales,
            pe_scale_options=global_parameter.pe_scales
        )
        if logger:
            logger.info("Starting with randomly generated chiplet group")
        
    # Evaluate initial solution
    t0 = time.perf_counter()
    current_results = run_single_optimization(
        virtual_nets=virtual_nets,
        objective=objective,
        logger=logger if logger else logging.getLogger(),
        chiplet_group=current_group,
        results_file=results_file,
        cost_aware=cost_aware,
        prev_best_genes=prev_best_genes,
        use_dag_cp=use_dag_cp,
        v_het_batch=v_het_batch,
    )[1]
    current_value, current_network_results = calculate_average_opt_value(current_results, objective)
    
    # Track best solution
    best_group = copy.deepcopy(current_group)
    best_results = current_results
    best_value = current_value
    best_network_results = current_network_results
    
    # Early termination tracking
    non_improving_count = 0
    previous_best_value = best_value
    
    # Pre-calculate temperature schedule
    temperature = initial_temp
    temp_schedule = []
    while temperature > min_temp:
        temp_schedule.append(temperature)
        temperature *= cooling_rate
    
    total_iterations = 0
    total_start_time = time.perf_counter()
    
    # Main optimization loop
    for temp_idx, temperature in enumerate(temp_schedule):
        temp_start_time = time.perf_counter()
        # For each temperature level, collect neighbor tasks
        tasks = []
        for i in range(iterations_per_temp):
            tasks.append((copy.deepcopy(current_group), temperature, i))
        
        # Collection to store results from all threads
        thread_results = []

        # In simulated_annealing_optimization:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers or multiprocessing.cpu_count()) as executor:
            # Submit all tasks to the process pool with all required parameters
            future_to_task = {
                executor.submit(
                    evaluate_neighbor,
                    task[0],                  # current_group_copy
                    task[1],                  # temperature
                    task[2],                  # thread_id
                    virtual_nets,
                    arch_targets,
                    objective,
                    logger,
                    results_file,
                    cost_aware,
                    prev_best_genes,
                    use_dag_cp,
                    v_het_batch,
                ): task for task in tasks
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    neighbor_group, neighbor_results, neighbor_value, neighbor_network_results = future.result()
                    
                    if neighbor_group is not None:
                        # Add to thread_results for evaluation
                        thread_results.append((
                            neighbor_group, 
                            neighbor_results, 
                            neighbor_value, 
                            neighbor_network_results
                        ))
                except Exception as e:
                    if logger:
                        logger.error(f"Error getting result from thread: {e}")
                    else:
                        print(f"Error getting result from thread: {e}")
        
        # Process all neighbors sequentially now that we have all results
        best_improved_this_temp = False
        for neighbor_group, neighbor_results, neighbor_value, neighbor_network_results in thread_results:
            # Acceptance criteria
            delta_e = neighbor_value - current_value
            # For energy minimization, accept if new value is lower or based on probability
            if delta_e < 0 or random.random() < math.exp(-delta_e / (temperature * current_value)):
                current_group = copy.deepcopy(neighbor_group)
                current_results = neighbor_results
                current_value = neighbor_value
                current_network_results = neighbor_network_results
                
                # Update best solution if current is better
                if current_value < best_value:
                    relative_improvement = (best_value - current_value) / best_value
                    best_group = copy.deepcopy(current_group)
                    best_results = current_results
                    best_value = current_value
                    best_network_results = current_network_results
                    
                    # Check if improvement is significant
                    if relative_improvement > early_termination_tolerance:
                        best_improved_this_temp = True
        
        # Check early termination condition
        relative_improvement = (previous_best_value - best_value) / previous_best_value
        if relative_improvement > early_termination_tolerance:
            non_improving_count = 0
        else:
            non_improving_count += 1
        
        previous_best_value = best_value
        
        total_iterations += len(thread_results)
        temp_time = time.perf_counter() - temp_start_time
        
        # Print progress
        metric_name = "EDP" if objective == "edp" else "Energy"
        progress = (temp_idx + 1) / len(temp_schedule) * 100
        print(f"\r[{progress:.1f}%] T: {temperature:.4f}, Current {metric_name}: {current_value:.2e}, Best {metric_name}: {best_value:.2e}, Time: {temp_time:.2f}s, Non-improving: {non_improving_count}/{early_termination_threshold}", end="")
        
        # Early termination check
        if non_improving_count >= early_termination_threshold:
            print(f"\nEarly termination after {temp_idx + 1}/{len(temp_schedule)} temperature levels ({(temp_idx + 1) / len(temp_schedule) * 100:.1f}%) due to {non_improving_count} consecutive non-improving iterations")
            break
    
    total_time = time.perf_counter() - total_start_time
    print(f"\nCompleted {total_iterations} iterations in {total_time:.2f}s ({total_iterations/total_time:.2f} iter/s)")
    
    return best_group, best_results, best_value, best_network_results
    

def run_incremental_n_chiplet_sweep(
    n_start: int,
    n_end: int,
    virtual_nets: List[VirtualNetwork],
    objective: str = "energy",
    initial_temp: float = 1.0,
    cooling_rate: float = 0.95,
    min_temp: float = 0.01,
    iterations_per_temp: int = 5,
    results_file: Optional[str] = None,
    database_file: str = "chiplet_optimization_database.csv",
    cost_aware: bool= False,
    use_dag_cp: bool = False,
    v_het_batch: bool = True,
    include_pim: bool = False,
    include_switch: bool = False,
):
    """
    Run simulated annealing optimization with incrementally increasing number of chiplets.
    Each new run starts with the best configuration from the previous run plus one new chiplet.
    
    Args:
        n_start: Starting number of chiplets
        n_end: Final number of chiplets
        arch_targets: List of architecture targets
        nets: List of network names
        net_layers_dict: Dictionary of network layers
        cycle_time: Cycle time parameter
        glb_scales: List of global buffer scaling options
        pe_scales: List of PE scaling options
        objective: Optimization objective ("energy" or "edp")
        initial_temp: Initial temperature for SA
        cooling_rate: Cooling rate for SA
        min_temp: Minimum temperature for SA
        iterations_per_temp: Number of iterations per temperature
        results_file: Optional file name for results
    
    Returns:
        Dictionary of results for each n_chiplets value
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Generate timestamp for results file if not provided
    if results_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f'incremental_chiplet_sweep_{objective}_{timestamp}.csv'
    
    # Setup CSV file with header
    # Define fieldnames for CSV header
    fieldnames = ['n_chiplets', f'best_{objective}']
    # Add network-specific fields
    for virtual_net in virtual_nets:
        fieldnames.append(f'{virtual_net.get_unique_name()}_min_{objective}')
        fieldnames.append(f'{virtual_net.get_unique_name()}_latency')
        fieldnames.append(f'{virtual_net.get_unique_name()}_config')
        fieldnames.append(f'{virtual_net.get_unique_name()}_gene')
    
    # Add fields for chiplet configurations (maximum possible number in range)
    for i in range(1, n_end + 1):
        fieldnames.extend([
            f'chiplet_{i}_arch',
            f'chiplet_{i}_glb_scale',
            f'chiplet_{i}_pe_x_scale',
            f'chiplet_{i}_pe_y_scale'
        ])
    
    # Create CSV file with header
    with open(results_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    # Dictionary to store results for each n_chiplets
    all_results = {}
    
    # Start with no best group
    best_group = None

    # Store best genes for each network from previous iterations
    prev_best_genes = {virtual_net.get_unique_name(): None for virtual_net in virtual_nets}
    
    # Iterate through chiplet numbers
    for n_chiplets in range(n_start, n_end + 1):
        print(f"\n{'='*60}")
        print(f"Starting optimization for n_chiplets = {n_chiplets}")
        print(f"{'='*60}")

        # If there's a best group from the previous iteration, use it as a starting point
        # by adding one random chiplet
        if best_group is not None and n_chiplets > len(best_group):
            # Create a copy of the best group from the previous iteration
            current_group = copy.deepcopy(best_group)

            # Add one random chiplet to reach the current n_chiplets
            # PIM/switch can be discovered naturally via arch_targets
            while len(current_group) < n_chiplets:
                new_chiplet = ChipletConfig(
                    arch_target=random.choice(arch_targets),
                    global_buffer_size_scale=random.choice(global_parameter.glb_scales),
                    pe_x_scale=random.choice(global_parameter.pe_scales),
                    pe_y_scale=random.choice(global_parameter.pe_scales)
                )
                current_group.append(new_chiplet)

            print(f"Starting with previous best configuration plus {n_chiplets - len(best_group)} new chiplet(s)")
        else:
            # No previous best group, generate a completely new random group
            current_group = generate_chiplet_group(
                n_chiplets=n_chiplets,
                arch_targets=arch_targets,
                glb_scale_options=global_parameter.glb_scales,
                pe_scale_options=global_parameter.pe_scales
            )
            print("Starting with completely new random configuration")
        
        # Run SA optimization for the current n_chiplets
        best_group, best_results, best_value, best_network_results = simulated_annealing_optimization(
            logger=logger,
            arch_targets=arch_targets,
            virtual_nets=virtual_nets,
            n_chiplets=n_chiplets,
            objective=objective,
            initial_temp=initial_temp,
            cooling_rate=cooling_rate,
            min_temp=min_temp,
            results_file=database_file,  # Use the database file for SA
            iterations_per_temp=iterations_per_temp,
            n_workers=None,
            cost_aware=cost_aware,
            prev_best_genes=prev_best_genes,
            initial_group=current_group,
            use_dag_cp=use_dag_cp,
            v_het_batch=v_het_batch,
        )

        # Update previous best genes for next iteration
        for virtual_net in virtual_nets:
            prev_best_genes[virtual_net.get_unique_name()] = best_results[virtual_net.get_unique_name()]['best_gene']
        
        # Prepare results dictionary
        result = {
            'n_chiplets': n_chiplets,
            f'best_{objective}': best_value
        }
        
        # Add network-specific results
        for virtual_net in virtual_nets:
            result[f'{virtual_net.get_unique_name()}_min_{objective}'] = best_network_results[virtual_net.get_unique_name()]['min_value']
            result[f'{virtual_net.get_unique_name()}_config'] = config_to_ids(best_results[virtual_net.get_unique_name()]['best_config'])
            result[f'{virtual_net.get_unique_name()}_gene'] = best_results[virtual_net.get_unique_name()]['best_gene']
            result[f'{virtual_net.get_unique_name()}_latency'] = best_results[virtual_net.get_unique_name()]['best_latency']
        #    print(result[f'{virtual_net.get_unique_name()}_latency'])

        # Add chiplet configurations
        for i, chiplet in enumerate(best_group):
            result[f'chiplet_{i+1}_arch'] = chiplet.arch_target
            result[f'chiplet_{i+1}_glb_scale'] = chiplet.global_buffer_size_scale
            result[f'chiplet_{i+1}_pe_x_scale'] = chiplet.pe_x_scale
            result[f'chiplet_{i+1}_pe_y_scale'] = chiplet.pe_y_scale
        
        # Store in all_results
        all_results[n_chiplets] = result
        
        # Write result to CSV immediately after each round
        try:
            with open(results_file, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(result)
            print(f"Results for n_chiplets = {n_chiplets} written to {results_file}")
        except Exception as e:
            logger.error(f"Error writing to CSV file: {e}")
        
        # Print summary for this n_chiplets
        metric_name = "EDP" if objective == "edp" else "Energy"
        print(f"\n--- Summary for n_chiplets = {n_chiplets} ---")
        print(f"Best {metric_name}: {best_value:.2e}")
        print("Chiplet configurations:")
        for i, chiplet in enumerate(best_group):
            print(f"  Chiplet {i+1}:")
            print(f"    Architecture: {chiplet.arch_target}")
            print(f"    GLB Scale: {chiplet.global_buffer_size_scale}")
            print(f"    PE X Scale: {chiplet.pe_x_scale}")
            print(f"    PE Y Scale: {chiplet.pe_y_scale}")
        
        print(f"Network-specific {metric_name.lower()} results:")
        for net_name, net_result in best_network_results.items():
            print(f"  {net_name}: {net_result['min_value']:.2e}")
    
    # Print final summary across all n_chiplets
    print("\n" + "="*60)
    print("Final results summary across all n_chiplets")
    print("="*60)
    
    metric_name = "EDP" if objective == "edp" else "Energy"
    print(f"| n_chiplets | {metric_name} {'':15} |")
    print("|" + "-"*11 + "|" + "-"*23 + "|")
    
    for n in range(n_start, n_end + 1):
        result = all_results[n]
        print(f"| {n:9d} | {result[f'best_{objective}']:.5e} |")
    
    # Find overall best configuration
    best_n = min(all_results.items(), key=lambda x: x[1][f'best_{objective}'])
    print(f"\nOverall best configuration: n_chiplets = {best_n[0]}")
    print(f"Overall best {metric_name.lower()}: {best_n[1][f'best_{objective}']:.5e}")
    
    print(f"\nResults saved to {results_file}")
    
    return all_results

if __name__ == "__main__":
    import argparse
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Chiplet optimization with simulated annealing")
    parser.add_argument("--objective", type=str, choices=["energy", "edp"], default="energy",
                        help="Optimization objective: 'energy' or 'edp'")
    parser.add_argument("--method", type=str, choices=["incremental", "sweep", "single"], default="incremental",
                        help="Optimization method: 'incremental', 'sweep', or 'single'")
    parser.add_argument("--n-start", type=int, default=1, help="Starting number of chiplets for incremental method")
    parser.add_argument("--n-end", type=int, default=8, help="Ending number of chiplets for incremental or sweep method")
    parser.add_argument("--n-chiplets", type=int, default=4, help="Fixed number of chiplets for single method")
    parser.add_argument("--initial-temp", type=float, default=1.0, help="Initial temperature for simulated annealing")
    parser.add_argument("--cooling-rate", type=float, default=0.95, help="Cooling rate for simulated annealing")
    parser.add_argument("--min-temp", type=float, default=0.01, help="Minimum temperature for simulated annealing")
    parser.add_argument("--iterations", type=int, default=5, help="Iterations per temperature level")
    # Add database file as command-line argument
    parser.add_argument("--database", type=str, default="chiplet_optimization_database.csv",
                        help="Database file for storing optimization results")
    parser.add_argument("--cost-aware", type=str, default="False",
                        help="Whether or not to take cost into consideration")
    parser.add_argument("--dag", action="store_true", default=True,
                        help="Use DAG critical-path GA (default: True)")
    parser.add_argument("--no-dag", dest="dag", action="store_false",
                        help="Use linear GA instead of DAG")
    parser.add_argument("--v-het-batch", type=str, default="True",
                        help="If True and --dag, off-CP ops (V proj) use larger batch")
    parser.add_argument("--pim", action="store_true", default=True,
                        help="Include PIM chiplet in pool at n>=2 (default: True)")
    parser.add_argument("--no-pim", dest="pim", action="store_false",
                        help="Disable PIM chiplet injection")
    parser.add_argument("--switch", action="store_true", default=True,
                        help="Include switch chiplet for MoE EP at n>=2 (default: True)")
    parser.add_argument("--no-switch", dest="switch", action="store_false",
                        help="Disable switch chiplet injection")
    parser.add_argument("--cnn", action="store_true", default=True,
                        help="Include CNN workloads (MobileNetV3, RepLKNet) (default: True)")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Base configuration
    arch_targets = DEFAULT_ARCH_TARGETS
    virtual_nets = []
    # LLaMA 3.1 and Qwen3 models
    for model in ['llama3.1_8b', 'llama3.1_70b', 'qwen3_30b_a3b', 'qwen3_235b_a22b']:
        virtual_nets.append(VirtualNetwork(f"{model}_prefill_s1024", batch_size=1, sequence_length=1024))
        virtual_nets.append(VirtualNetwork(f"{model}_prefill_s1024", batch_size=8, sequence_length=1024))
        virtual_nets.append(VirtualNetwork(f"{model}_decode_kv1024", batch_size=1, sequence_length=1))
        virtual_nets.append(VirtualNetwork(f"{model}_decode_kv1024", batch_size=8, sequence_length=1))

    # CNN workloads (optional)
    if args.cnn:
        for cnn in ['mobilenet_v3_small', 'replknet31b']:
            virtual_nets.append(VirtualNetwork(cnn, batch_size=1, sequence_length=1))
            virtual_nets.append(VirtualNetwork(cnn, batch_size=8, sequence_length=1))
    # Load database to get available layers per network
    import pandas as pd
    _db = pd.read_csv(args.database)
    _db_layers_per_net = {net: set(_db[_db['net']==net]['layer_name'].unique()) for net in _db['net'].unique()}

    for virtual_net in virtual_nets:
        db_layers = _db_layers_per_net.get(virtual_net.network_name, None)
        try:
            virtual_net.load_from_dir(os.path.join(NET_DIR, virtual_net.network_name), db_layers=db_layers)
        except Exception as e:
            logger.warning(f"Failed to load {virtual_net.network_name}: {e}")
    # Filter out networks that failed to load
    virtual_nets = [vn for vn in virtual_nets if len(vn.layers) > 0]
    
    # nets = list(set(nets))

    # Get layers for each network
    # net_layers_dict = {}
    # net_layers_dict = gen_net_layers_dict(nets, NET_DIR)
    
    # SA parameters from command-line arguments
    initial_temp = args.initial_temp
    cooling_rate = args.cooling_rate
    min_temp = args.min_temp
    iterations_per_temp = args.iterations
    
    # Objective from command-line arguments
    objective = args.objective  # "energy" or "edp"
    cost_aware = True
    if args.cost_aware == "False":
        cost_aware = False
    use_dag_cp = args.dag
    v_het_batch = args.v_het_batch != "False"
    # Log the configuration
    logger.info(f"Starting optimization with objective: {objective} (cost aware: {cost_aware})")
    if use_dag_cp:
        logger.info(f"Using DAG critical-path GA (v_het_batch={v_het_batch})")
    logger.info(f"PIM: {args.pim}, Switch: {args.switch}, CNN: {args.cnn}")
    logger.info(f"Networks: {len(virtual_nets)} workloads loaded")
    logger.info(f"Method: {args.method}")
    if args.method == "incremental" or args.method == "sweep":
        logger.info(f"Chiplet range: {args.n_start} to {args.n_end}")
    else:
        logger.info(f"Number of chiplets: {args.n_chiplets}")
    
    # Method from command-line arguments
    run_method = args.method
    
    # Generate timestamp for output results file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Get database file from command-line arguments
    database_file = args.database
    
    if run_method == "incremental":
        # Run incremental optimization (start with fewer chiplets, gradually increase)
        n_start = args.n_start
        n_end = args.n_end
        
        # Create output results file name (different from database file)
        output_file = f'incremental_chiplet_sweep_{timestamp}_{objective}_{cost_aware}.csv'
        
        # Run the incremental optimization
        results = run_incremental_n_chiplet_sweep(
            n_start=n_start,
            n_end=n_end,
            virtual_nets = virtual_nets,
            objective=objective,
            initial_temp=initial_temp,
            cooling_rate=cooling_rate,
            min_temp=min_temp,
            iterations_per_temp=iterations_per_temp,
            results_file=output_file,
            database_file=database_file,
            cost_aware = cost_aware,
            use_dag_cp=use_dag_cp,
            v_het_batch=v_het_batch,
            include_pim=args.pim,
            include_switch=args.switch,
        )
        
    elif run_method == "sweep":
        print("not supported")
    else:  # "single" method
        print("not supported")

# nohup python3 chiplet_sel.py --objective=energy --method=incremental --n-start=1 --n-end=16 --database=final_database.csv --cost-aware=False > e_f.out &
# nohup python3 chiplet_sel.py --objective=energy --method=incremental --n-start=1 --n-end=16 --database=final_database.csv --cost-aware=True > e_t.out &
# nohup python3 chiplet_sel.py --objective=edp --method=incremental --n-start=1 --n-end=16 --database=final_database.csv --cost-aware=False > edp_f.out &
# nohup python3 chiplet_sel.py --objective=edp --method=incremental --n-start=1 --n-end=16 --database=final_database.csv --cost-aware=True > edp_t.out &