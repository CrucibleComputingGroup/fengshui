import random
import os
import time
from copy import deepcopy
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set, Union
import functools
from global_parameter import *
from chiplet_dataclass import *
from network_dataclass import *
from convex_hull import *
from utility_functions import cal_opt_val_fused
from cal_perf_phy_net import cal_perf_phy_net, get_chiplet_data, cal_buffer_config

# Move evaluate_gene function outside the main function
def evaluate_gene(gene, virtual_network, chiplet_group, chiplets_data, results_file, objective, verbose, cost_aware, dag=None):
    try:
        # For MoE workloads: force expert layers into own fusion groups
        moe_config = getattr(virtual_network, 'moe_config', None)
        if moe_config is not None:
            gene = _force_expert_fusion_gene(gene, virtual_network)

        # Generate physical network from gene
        physical_network = create_physical_network_from_gene(virtual_network, gene)

        (min_e, min_e_config), (min_edp, min_edp_config) = \
            cal_perf_phy_net(chiplet_group, chiplets_data, physical_network,
                            res_csv_file=results_file,
                            buffer_config=gene["buffer_config"],
                            dag=dag, cost_aware=cost_aware,
                            verbose=verbose,
                            moe_config=moe_config)

        if objective == "energy":
            fitness = min_e
            config = min_e_config
        else:  # edp
            fitness = min_edp
            config = min_edp_config

        return gene, fitness, config, None

    except Exception as e:
        # If gene produces invalid network, return error
        return gene, float('inf'), None, str(e)

def genetic_algo_opt_phy_net(virtual_network, chiplet_group,objective="energy",
                             population_size=10, generations=20, mutation_rate=0.2,
                             crossover_rate=0.8, tournament_size=3, elite_size=2,
                             query_points=base_query_points, results_file=None, verbose=False,
                             max_workers=None, cost_aware=False,prev_best_gene=None,
                             use_sequential=False, dag=None):
    """
    Genetic algorithm with new gene encoding to find optimal physical network configuration.
    Each function call creates its own independent thread pool for internal parallelization.
    
    Args:
        virtual_network: The VirtualNetwork object to optimize
        chiplet_group: chiplet group available to use
        objective: "energy" or "edp" to specify optimization target
        population_size: Size of the population in each generation
        generations: Number of generations to evolve
        mutation_rate: Probability of mutation for each gene
        crossover_rate: Probability of crossover for each pair
        tournament_size: Number of individuals in each tournament selection
        elite_size: Number of top individuals to preserve in each generation
        arch_targets: List of architecture targets for chiplet generation
        glb_scales: GLB scale options for chiplet configuration
        pe_scales: PE scale options for chiplet configuration
        query_points: List of time points for performance evaluation
        results_file: Timeloop results file path
        max_workers: Maximum number of worker threads to use (None = auto)
        
    Returns:
        best_gene: The optimal gene configuration
        best_value: The optimal objective value (energy or EDP)
        best_config: The optimal chiplet configuration
    """
    import concurrent.futures
    import time
    from copy import deepcopy
    import random
    
    if results_file is None:
        results_file = "timeloop_fitted_results_20250304_191750.csv"
    
    # Initialize population with random genes
    population = []
    num_layers = len(virtual_network.layers)
    chiplets_data = []
    for chiplet_idx, chiplet_config in enumerate(chiplet_group):
        # Use original_name if available, otherwise fall back to network_name

        chiplets_data.append(get_chiplet_data(
            results_file,
            chiplet_config.arch_target,
            chiplet_config.global_buffer_size_scale,
            chiplet_config.pe_x_scale,
            chiplet_config.pe_y_scale,
            virtual_network.network_name
        ))

    remaining_empty_population_size = population_size

    # Every compute chiplet supports all DRAM types — use full set
    pool_dram_types = dram_options

    # Add the previous best gene if available
    if prev_best_gene is not None and isinstance(prev_best_gene, dict):
        # Make sure the gene is compatible with the current number of layers
        if len(prev_best_gene.get('binary_string', '')) == num_layers:
            population.append(prev_best_gene)
            remaining_empty_population_size -= 1
            if verbose:
                print(f"Including previous best gene in initial population: {prev_best_gene['binary_string']}")

    # always have non-fused version in the initial population. A non-PIM stage of the fused
    # layer<N>_softmax alone is rejected (calculate_network_performance_with_memory_check), so
    # softmax is fused onto a linear neighbour (its predecessor, or its successor when first).
    separate = ['1'] * num_layers
    for i, layer in enumerate(virtual_network.layers):
        if layer.name.endswith('_softmax') and num_layers > 1:
            separate[i if i > 0 else 1] = '0'
    separate = ''.join(separate)
    physical_network = create_physical_network_from_gene_binary_string(virtual_network=virtual_network, binary_string=separate)
    buffer_config = cal_buffer_config(get_max_pes(chiplet_group), physical_network, virtual_network.batch_size, virtual_network.sequence_length, max(tp_degrees))

    if remaining_empty_population_size > 0:
        population.append({
                'binary_string': separate,
                'buffer_config': buffer_config
            })
    remaining_empty_population_size -= 1

    for _ in range(remaining_empty_population_size):
        # Generate random binary string (with first bit always 1)
        binary = ['1']
        for i in range(1, num_layers):
            # 50% chance to start a new fusion group
            binary.append('1' if random.random() < 0.5 else '0')

        gene = {
            'binary_string': ''.join(binary),
            'buffer_config': [random.choice(pool_dram_types) for _ in range(num_layers + 1)]
        }

        population.append(gene)
        
    
    best_gene = None
    best_value = float('inf')  # We're minimizing
    best_config = None
    if verbose:
        print(f"Starting genetic algorithm optimization for {objective}...")
    start_time = time.time()
    
    # Set up thread pool outside the generation loop
    max_workers = population_size if max_workers is None else min(max_workers,population_size)

    # Main GA loop
    from functools import partial
    eval_func = partial(
        evaluate_gene,
        virtual_network=virtual_network,
        chiplet_group=chiplet_group,
        chiplets_data=chiplets_data,
        results_file=results_file,
        objective=objective,
        verbose=verbose,
        cost_aware=cost_aware,
        dag=dag
    )

    def _run_generation_sequential(population, fitness_scores):
        """Evaluate population sequentially (reuses in-process caches)."""
        for i, gene in enumerate(population):
            gene, fitness, config, error = eval_func(gene)
            fitness_scores[i] = fitness
            if fitness < best_value_holder[0]:
                best_value_holder[0] = fitness
                best_holder[0] = deepcopy(gene)
                best_config_holder[0] = deepcopy(config)
                if verbose:
                    print(f"New best {objective}: {best_value_holder[0]}")
                    print(f"Gene: {gene['binary_string']}")
                    if config:
                        print(f"Query point: {best_config_holder[0].get('latency', 'N/A')}")
                        for group_config in best_config_holder[0]['functions']:
                            print(f"Group config ID: {group_config.id}")
            if error and verbose:
                print(f"Error evaluating gene: {error}, {gene}")

    def _run_generation_parallel(population, fitness_scores, executor):
        """Evaluate population in parallel via ProcessPoolExecutor."""
        futures = {}
        for i, gene in enumerate(population):
            future = executor.submit(eval_func, gene)
            futures[future] = i
        for future in concurrent.futures.as_completed(futures):
            gene, fitness, config, error = future.result()
            idx = futures[future]
            fitness_scores[idx] = fitness
            if fitness < best_value_holder[0]:
                best_value_holder[0] = fitness
                best_holder[0] = deepcopy(gene)
                best_config_holder[0] = deepcopy(config)
                if verbose:
                    print(f"New best {objective}: {best_value_holder[0]}")
                    print(f"Gene: {gene['binary_string']}")
                    if config:
                        print(f"Query point: {best_config_holder[0].get('latency', 'N/A')}")
                        for group_config in best_config_holder[0]['functions']:
                            print(f"Group config ID: {group_config.id}")
            if error and verbose:
                print(f"Error evaluating gene: {error}, {gene}")

    # Mutable holders so inner functions can update best
    best_value_holder = [best_value]
    best_holder = [best_gene]
    best_config_holder = [best_config]

    def _ga_loop(run_gen_fn, executor=None):
        nonlocal population
        for generation in range(generations):
            gen_start_time = time.time()
            fitness_scores = [None] * len(population)

            if executor is not None:
                run_gen_fn(population, fitness_scores, executor)
            else:
                run_gen_fn(population, fitness_scores)

            valid_scores = [s for s in fitness_scores if s is not None and s != float('inf')]
            if valid_scores:
                gen_best = min(valid_scores)
                if verbose:
                    print(f"Generation {generation+1}/{generations}:")
                    print(f"  Best: {gen_best:.6f}")
                    print(f"  Valid solutions: {len(valid_scores)}/{population_size}")
            else:
                if verbose:
                    print(f"Generation {generation+1}/{generations}: No valid solutions found")
            
            # Create next generation
            next_population = []

            # Elitism: keep the best individuals
            combined = list(zip(population, fitness_scores))
            sorted_population = [x[0] for x in sorted(combined, key=lambda x: x[1])]
            next_population.extend(deepcopy(sorted_population[:elite_size]))

            # Fill the rest of the population with crossover and mutation
            while len(next_population) < population_size:
                # Tournament selection
                parent1 = tournament_selection(population, fitness_scores, tournament_size)
                parent2 = tournament_selection(population, fitness_scores, tournament_size)

                # Crossover
                if random.random() < crossover_rate:
                    child1, child2 = crossover(parent1, parent2, virtual_network.network_name)
                else:
                    child1, child2 = deepcopy(parent1), deepcopy(parent2)

                # Mutation
                child1 = mutate(child1, mutation_rate, virtual_network.network_name, pool_dram_types)
                child2 = mutate(child2, mutation_rate, virtual_network.network_name, pool_dram_types)

                # Add to next generation
                next_population.append(child1)
                if len(next_population) < population_size:
                    next_population.append(child2)

            # Update population
            population = next_population

            gen_end_time = time.time()
            if verbose:
                print(f"  Generation time: {gen_end_time - gen_start_time:.2f} seconds")
                print("--------------------------------------------------")

    # Dispatch: sequential (cache-friendly) or parallel (ProcessPoolExecutor)
    if use_sequential:
        _ga_loop(_run_generation_sequential)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            _ga_loop(_run_generation_parallel, executor)

    # Extract results from holders
    best_value = best_value_holder[0]
    best_gene = best_holder[0]
    best_config = best_config_holder[0]

    end_time = time.time()
    if verbose:
        print(f"Genetic algorithm completed in {end_time - start_time:.2f} seconds")
        print(f"Best {objective}: {best_value}")

    # Return the best gene, value, configuration
    return best_gene, best_value, best_config


def tournament_selection(population, fitness_scores, tournament_size):
    """Tournament selection with new gene format."""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    
    # Find the best individual in the tournament (lowest score since minimizing)
    best_idx = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
    return deepcopy(population[best_idx])


def crossover(parent1, parent2, net_name):
    """Crossover operation for new gene format."""
    # Ensure parents have the same length
    binary_len = len(parent1['binary_string'])
    
    # Choose random crossover point
    crossover_point = random.randint(1, binary_len - 1)
    
    # Create children
    child1 = {
        'binary_string': parent1['binary_string'][:crossover_point] + parent2['binary_string'][crossover_point:],
        'buffer_config': parent1['buffer_config'][:crossover_point] + parent2['buffer_config'][crossover_point:]
    }
    
    child2 = {
        'binary_string': parent2['binary_string'][:crossover_point] + parent1['binary_string'][crossover_point:],
        'buffer_config': parent2['buffer_config'][:crossover_point] + parent1['buffer_config'][crossover_point:]
    }
    
    # Ensure first bit is 1 for both children
    if child1['binary_string'][0] != '1':
        child1['binary_string'] = '1' + child1['binary_string'][1:]
    
    if child2['binary_string'][0] != '1':
        child2['binary_string'] = '1' + child2['binary_string'][1:]
    
    # Harmonize parameters within fusion groups
    child1 = harmonize_gene(child1,net_name)
    child2 = harmonize_gene(child2,net_name)
    
    return child1, child2

def harmonize_gene(gene, net_name):
    """
    Ensure parameters within fusion groups are consistent.
    This function now harmonizes both the binary_string and buffer_config.
    Only unify the first num_layers items, the last item is a placeholder/endpoint, keep free.
    """
    binary_list = list(gene['binary_string'])
    buffer_conf = gene['buffer_config']
    num_layers = len(binary_list)

    # 1. Harmonize net topology first (as in the original code)
    # str is immutable
    if net_name in net_topology_dict: # Check if key exists
        for idx in net_topology_dict[net_name]:
            if idx < len(binary_list):
                 binary_list[idx] = '1'
    
    # 2. Harmonize buffer_config within each fusion group
    current_group_start_idx = 0
    for i in range(1, num_layers):
        if binary_list[i] == '1':
            # This is the start of a new group.
            # Harmonize the previous group (from current_group_start_idx to i-1).
            group_buffer_type = buffer_conf[current_group_start_idx]
            for j in range(current_group_start_idx, i):
                buffer_conf[j] = group_buffer_type
            
            # Update the start index for the new group
            current_group_start_idx = i

    # Harmonize the last group (from the last '1' to the end)
    last_group_buffer_type = buffer_conf[current_group_start_idx]
    for j in range(current_group_start_idx, num_layers):
        buffer_conf[j] = last_group_buffer_type

    # Update the gene with harmonized values
    gene['binary_string'] = ''.join(binary_list)
    gene['buffer_config'] = buffer_conf # buffer_conf was modified in-place

    return gene

def _force_expert_fusion_gene(gene, virtual_network):
    """Force each expert layer into its own fusion group for MoE workloads."""
    layers = virtual_network.layers
    binary = list(gene['binary_string'])
    for i, layer in enumerate(layers):
        if layer.name in MOE_EXPERT_OPS:
            if i < len(binary):
                binary[i] = '1'
            if i + 1 < len(binary):
                binary[i + 1] = '1'
    gene = dict(gene)
    gene['binary_string'] = ''.join(binary)
    return gene


def mutate(gene, mutation_rate, net_name, allowed_dram_types=None):
    """Mutation operation for new gene format."""
    if allowed_dram_types is None:
        allowed_dram_types = dram_options
    binary = list(gene['binary_string'])

    # Mutate binary string (fusion boundaries)
    for i in range(1, len(binary)):  # Skip first bit to ensure it stays 1
        if random.random() < mutation_rate:
            binary[i] = '1' if binary[i] == '0' else '0'
    
    new_buffer_config = []
    
    for current_buffer in gene['buffer_config']:
        if random.random() < mutation_rate:
            # Choose a different buffer configuration — restricted to allowed DRAM types
            available_options = [opt for opt in allowed_dram_types if opt != current_buffer]
            if available_options:
                new_buffer = random.choice(available_options)
            else:
                new_buffer = current_buffer
        else:
            new_buffer = current_buffer
        new_buffer_config.append(new_buffer)
    
    # Create mutated gene
    mutated_gene = {
        'binary_string': ''.join(binary),
        'buffer_config': new_buffer_config
    }
    
    # Harmonize parameters within fusion groups
    mutated_gene = harmonize_gene(mutated_gene, net_name)
    
    return mutated_gene


# ============================================================
# DAG GA: Critical-Path-Based Gene Encoding
# ============================================================

from cal_perf_phy_net import CriticalPathSpec


def _random_dag_gene(cp_spec: CriticalPathSpec, pool_dram_types: List[str]) -> dict:
    """Generate a random gene for the DAG GA."""
    n = cp_spec.gene_length
    binary = ['1' if i in cp_spec.forced_boundaries else
              ('1' if random.random() < 0.5 else '0')
              for i in range(n)]
    binary[0] = '1'
    gene = {
        'binary_string': ''.join(binary),
        'buffer_config': [random.choice(pool_dram_types)
                          for _ in range(cp_spec.buffer_config_length)],
    }
    return harmonize_dag_gene(gene, cp_spec)


def harmonize_dag_gene(gene: dict, cp_spec: CriticalPathSpec) -> dict:
    """Harmonize a DAG gene: enforce forced boundaries, unify DRAM within groups."""
    binary_list = list(gene['binary_string'])
    buffer_conf = gene['buffer_config']
    n = len(binary_list)

    # 1. Force boundaries
    for idx in cp_spec.forced_boundaries:
        if idx < n:
            binary_list[idx] = '1'

    # 2. Unify buffer_config within each fusion group
    current_group_start = 0
    for i in range(1, n):
        if binary_list[i] == '1':
            group_type = buffer_conf[current_group_start]
            for j in range(current_group_start, i):
                buffer_conf[j] = group_type
            current_group_start = i
    # Last group
    group_type = buffer_conf[current_group_start]
    for j in range(current_group_start, n):
        buffer_conf[j] = group_type

    gene['binary_string'] = ''.join(binary_list)
    gene['buffer_config'] = buffer_conf
    return gene


def crossover_dag(parent1: dict, parent2: dict, cp_spec: CriticalPathSpec) -> Tuple[dict, dict]:
    """Single-point crossover for DAG genes."""
    n = len(parent1['binary_string'])
    pt = random.randint(1, n - 1)

    child1 = {
        'binary_string': parent1['binary_string'][:pt] + parent2['binary_string'][pt:],
        'buffer_config': parent1['buffer_config'][:pt] + parent2['buffer_config'][pt:],
    }
    child2 = {
        'binary_string': parent2['binary_string'][:pt] + parent1['binary_string'][pt:],
        'buffer_config': parent2['buffer_config'][:pt] + parent1['buffer_config'][pt:],
    }
    return harmonize_dag_gene(child1, cp_spec), harmonize_dag_gene(child2, cp_spec)


def mutate_dag(gene: dict, mutation_rate: float, cp_spec: CriticalPathSpec,
               allowed_dram_types: List[str] = None) -> dict:
    """Mutation for DAG genes. Skips forced-boundary positions."""
    if allowed_dram_types is None:
        allowed_dram_types = dram_options
    forced = set(cp_spec.forced_boundaries)
    binary = list(gene['binary_string'])

    for i in range(len(binary)):
        if i in forced:
            continue  # never mutate forced boundaries
        if random.random() < mutation_rate:
            binary[i] = '1' if binary[i] == '0' else '0'

    new_buf = []
    for cur in gene['buffer_config']:
        if random.random() < mutation_rate:
            opts = [o for o in allowed_dram_types if o != cur]
            new_buf.append(random.choice(opts) if opts else cur)
        else:
            new_buf.append(cur)

    mutated = {'binary_string': ''.join(binary), 'buffer_config': new_buf}
    return harmonize_dag_gene(mutated, cp_spec)


def evaluate_dag_gene(gene, cp_spec, virtual_network, chiplet_group,
                      chiplets_data,
                      results_file, objective, verbose, cost_aware,
                      v_het_batch=True):
    """Evaluate a DAG gene. Returns (gene, fitness, config, error)."""
    try:
        from cal_perf_phy_net import cal_perf_phy_net_dag_cp
        (min_e, min_e_config), (min_edp, min_edp_config) = cal_perf_phy_net_dag_cp(
            cp_spec=cp_spec,
            virtual_network=virtual_network,
            chiplet_group=chiplet_group,
            chiplets_data=chiplets_data,
            gene=gene,
            res_csv_file=results_file,
            cost_aware=cost_aware,
            verbose=verbose,
            v_het_batch=v_het_batch,
        )
        if objective == "energy":
            return gene, min_e, min_e_config, None
        else:
            return gene, min_edp, min_edp_config, None
    except Exception as e:
        return gene, float('inf'), None, str(e)


def genetic_algo_opt_phy_net_dag(
    virtual_network,
    chiplet_group,
    cp_spec: CriticalPathSpec,
    objective="energy",
    population_size=10,
    generations=20,
    mutation_rate=0.2,
    crossover_rate=0.8,
    tournament_size=3,
    elite_size=2,
    results_file=None,
    verbose=False,
    cost_aware=False,
    prev_best_gene=None,
    use_sequential=True,
    v_het_batch=True,
):
    """Genetic algorithm using critical-path-based DAG gene encoding.

    Gene length = len(cp_spec.positions), NOT num_layers.
    Parallel ops at a CP position form virtual groups (cross-product in convex hull).
    Off-CP ops (V proj) evaluated separately with het-batch + relaxed latency.

    Args:
        v_het_batch: If True, off-CP ops use larger batch for energy amortization.
            If False, off-CP ops use base batch (still get relaxed latency from slack).
    """
    import concurrent.futures
    from functools import partial

    if results_file is None:
        results_file = "timeloop_fitted_results_20250304_191750.csv"

    # Every compute chiplet supports all DRAM types
    pool_dram_types = dram_options

    # Pre-load chiplet data
    chiplets_data = []
    for chiplet in chiplet_group:
        chiplets_data.append(get_chiplet_data(
            results_file, chiplet.arch_target,
            chiplet.global_buffer_size_scale,
            chiplet.pe_x_scale, chiplet.pe_y_scale,
            virtual_network.network_name))

    # --- Initialize population ---
    population = []
    remaining = population_size

    if prev_best_gene is not None and isinstance(prev_best_gene, dict):
        if len(prev_best_gene.get('binary_string', '')) == cp_spec.gene_length:
            population.append(prev_best_gene)
            remaining -= 1

    # Seed one all-separate gene per DRAM type so PIM (needs GDDR7),
    # HBM3, etc. all get a fair initial evaluation.
    for dram in pool_dram_types:
        if remaining <= 0:
            break
        seed = {
            'binary_string': '1' * cp_spec.gene_length,
            'buffer_config': [dram] * cp_spec.buffer_config_length,
        }
        population.append(harmonize_dag_gene(seed, cp_spec))
        remaining -= 1

    for _ in range(remaining):
        population.append(_random_dag_gene(cp_spec, pool_dram_types))

    # --- GA loop ---
    best_value = float('inf')
    best_gene = None
    best_config = None

    eval_func = partial(
        evaluate_dag_gene,
        cp_spec=cp_spec,
        virtual_network=virtual_network,
        chiplet_group=chiplet_group,
        chiplets_data=chiplets_data,
        results_file=results_file,
        objective=objective,
        verbose=verbose,
        cost_aware=cost_aware,
        v_het_batch=v_het_batch,
    )

    start_time = time.time()
    for generation in range(generations):
        gen_start = time.time()
        fitness_scores = [None] * len(population)

        # Evaluate
        for i, gene in enumerate(population):
            _, fitness, config, error = eval_func(gene)
            fitness_scores[i] = fitness
            if fitness < best_value:
                best_value = fitness
                best_gene = deepcopy(gene)
                best_config = deepcopy(config)
                if verbose:
                    print(f"  New best {objective}: {best_value}")
                    print(f"  Gene: {gene['binary_string']}")
            if error and verbose:
                print(f"  Error: {error}")

        valid = [s for s in fitness_scores if s is not None and s != float('inf')]
        if verbose:
            print(f"Gen {generation+1}/{generations}: best={min(valid):.6f}, "
                  f"valid={len(valid)}/{population_size}, "
                  f"time={time.time()-gen_start:.1f}s")

        # Next generation
        combined = list(zip(population, fitness_scores))
        sorted_pop = [x[0] for x in sorted(combined, key=lambda x: x[1])]
        next_pop = deepcopy(sorted_pop[:elite_size])

        while len(next_pop) < population_size:
            p1 = tournament_selection(population, fitness_scores, tournament_size)
            p2 = tournament_selection(population, fitness_scores, tournament_size)
            if random.random() < crossover_rate:
                c1, c2 = crossover_dag(p1, p2, cp_spec)
            else:
                c1, c2 = deepcopy(p1), deepcopy(p2)
            c1 = mutate_dag(c1, mutation_rate, cp_spec, pool_dram_types)
            c2 = mutate_dag(c2, mutation_rate, cp_spec, pool_dram_types)
            next_pop.append(c1)
            if len(next_pop) < population_size:
                next_pop.append(c2)

        population = next_pop

    if verbose:
        print(f"DAG GA completed in {time.time()-start_time:.1f}s, best {objective}: {best_value}")

    return best_gene, best_value, best_config


if __name__ == "__main__":
    # Create a virtual network
    net_name="gpt-1.3B_prefill"
    network = VirtualNetwork(f"{net_name}", batch_size=1, sequence_length=256)
    network.load_from_dir(os.path.join(NET_DIR, f"{net_name}"))
    
    # Query points
    #query_points = np.linspace(0.001, 1.0, 1000).tolist()
    
    # Architecture targets
    arch_targets = DEFAULT_ARCH_TARGETS
    
    # Create chiplet group
    # chiplet_group = generate_all_chiplet_configs(
    #         arch_targets=arch_targets,
    #         glb_scale_options=glb_scales,
    #         pe_scale_options=pe_scales
    #     )
    chiplet_group = generate_chiplet_group(
        n_chiplets = 4,
        arch_targets=arch_targets,
        glb_scale_options=glb_scales,
        pe_scale_options=pe_scales
    )
    # Run genetic algorithm for energy optimization with new encoding
    best_gene_energy, best_energy, best_config_energy = genetic_algo_opt_phy_net(
        network,
        chiplet_group,
        objective="energy",
        population_size=10,
        generations=10,
        query_points=base_query_points,
        results_file="llm_13.csv",
        max_workers=10,
        verbose = True,
        cost_aware= True
    )
    
    print("\n=== Best Energy Configuration ===")
    print(f"Energy: {best_energy}")
    print(f"Gene: {best_gene_energy['binary_string']}")
    #print(f"Query point: {query_points[best_config_energy['idx']]}")
    print("Group configurations:")
    for group_config in best_config_energy['functions']:
        print(f"  {group_config.id}")
    