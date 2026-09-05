import os
import time
import random
import math
import logging
import numpy as np
import csv
from typing import List, Tuple, Optional, Dict
import copy
import multiprocessing
import concurrent.futures

from parse_stats import *
from convex_hull import convex_hull_min_e, process_layer_convex_hull
from chiplet_dataclass import *
from utility_functions import *
from global_parameter import *
from network_dataclass import *
from genetic_algo_opt_phy_net import genetic_algo_opt_phy_net, genetic_algo_opt_phy_net_dag
from chiplet_sel import run_single_optimization as _chiplet_sel_run_single_optimization
from chiplet_sel import simulated_annealing_optimization, generate_neighbor_group_optimized
from cal_perf_phy_net import preload_database, create_cp_spec


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_virtual_nets(database_file="unified_database.csv"):
    """Build virtual networks for baseline evaluation.
    Matches setup_virtual_nets in run_archgym_chiplet.py (b1/b8, include CNN)."""
    virtual_nets = []
    for model in ['llama3.1_8b', 'llama3.1_70b', 'qwen3_30b_a3b', 'qwen3_235b_a22b']:
        virtual_nets.append(VirtualNetwork(f"{model}_prefill_s1024", batch_size=1, sequence_length=1024))
        virtual_nets.append(VirtualNetwork(f"{model}_prefill_s1024", batch_size=8, sequence_length=1024))
        virtual_nets.append(VirtualNetwork(f"{model}_decode_kv1024", batch_size=1, sequence_length=1))
        virtual_nets.append(VirtualNetwork(f"{model}_decode_kv1024", batch_size=8, sequence_length=1))
    for cnn in ['mobilenet_v3_small', 'replknet31b']:
        virtual_nets.append(VirtualNetwork(cnn, batch_size=1, sequence_length=1))
        virtual_nets.append(VirtualNetwork(cnn, batch_size=8, sequence_length=1))

    import pandas as pd
    _db = pd.read_csv(database_file)
    _db_layers_per_net = {
        net: set(_db[_db['net'] == net]['layer_name'].unique())
        for net in _db['net'].unique()
    }
    for vn in virtual_nets:
        db_layers = _db_layers_per_net.get(vn.network_name, None)
        try:
            vn.load_from_dir(os.path.join(NET_DIR, vn.network_name), db_layers=db_layers)
        except Exception as e:
            print(f"Warning: failed to load {vn.network_name}: {e}")
    virtual_nets = [vn for vn in virtual_nets if len(vn.layers) > 0]
    needed_nets = set(vn.network_name for vn in virtual_nets)
    preload_database(database_file, needed_nets=needed_nets)
    return virtual_nets


import global_parameter as _gp
DEFAULT_ARCH_TARGETS = _gp.DEFAULT_ARCH_TARGETS


def _inject_special_chiplets(chiplet_group):
    """Inject PIM and switch chiplets into a chiplet group if not already present."""
    existing_archs = {c.arch_target for c in chiplet_group}
    if 'PIM' not in existing_archs:
        chiplet_group.append(ChipletConfig('PIM', 1, 1, 1, dram_type='GDDR7'))
    if _gp.SWITCH_ARCH_TARGET not in existing_archs:
        chiplet_group.append(create_switch_chiplet())
    return chiplet_group


# ---------------------------------------------------------------------------
# Mode 1: heterogeneous global baseline (formerly baseline_heter_glb.py)
# Uses chiplet_sel.run_single_optimization which supports DAG-CP and het-batch.
# ---------------------------------------------------------------------------

def run_single_optimization(
    virtual_nets: List[VirtualNetwork],
    chiplet_group: List[ChipletConfig],
    objective: str = "energy",
    logger: Optional[logging.Logger] = None,
    results_file: Optional[str] = None,
    cost_aware: bool = False,
    use_dag_cp: bool = False,
    v_het_batch: bool = True,
) -> Tuple[List, Dict]:
    """
    Run optimization for a single chiplet configuration across all networks.
    Delegates to chiplet_sel.run_single_optimization for DAG-CP and het-batch support.
    """
    return _chiplet_sel_run_single_optimization(
        virtual_nets=virtual_nets,
        chiplet_group=chiplet_group,
        objective=objective,
        logger=logger,
        results_file=results_file,
        cost_aware=cost_aware,
        use_dag_cp=use_dag_cp,
        v_het_batch=v_het_batch,
    )


def all_chiplets_optimization(
    arch_targets: List[str],
    virtual_nets: List[VirtualNetwork],
    objective: str = "energy",
    results_file: str = "all_chiplets_results.csv",
    database_file: str = "all_chiplets_database.csv",
    logger: Optional[logging.Logger] = None,
    cost_aware: bool = False,
    use_dag_cp: bool = False,
    v_het_batch: bool = True,
) -> Tuple[List, Dict, float, Dict]:
    """
    Run optimization using all possible chiplet configurations as the chiplet group.
    (Heterogeneous global baseline)
    Includes PIM and switch chiplets for MoE expert parallelism.
    """
    all_chiplets = generate_all_chiplet_configs(
        arch_targets=arch_targets,
        glb_scale_options=glb_scales,
        pe_scale_options=pe_scales
    )
    # Inject PIM and switch chiplets
    all_chiplets = _inject_special_chiplets(all_chiplets)

    print(f"Generated {len(all_chiplets)} unique chiplet configurations (incl. PIM + switch)")
    chiplet_group = all_chiplets

    print(f"Starting optimization using all {len(chiplet_group)} chiplets...")
    start_time = time.perf_counter()

    _, results = run_single_optimization(
        virtual_nets=virtual_nets,
        chiplet_group=chiplet_group,
        objective=objective,
        logger=logger,
        results_file=database_file,
        cost_aware=cost_aware,
        use_dag_cp=use_dag_cp,
        v_het_batch=v_het_batch,
    )

    best_value, best_network_results = calculate_average_opt_value(results, objective)

    total_time = time.perf_counter() - start_time
    print(f"Optimization completed in {total_time:.2f}s")

    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = [f'min_{objective}']
        for virtual_net in virtual_nets:
            fieldnames.append(f'{virtual_net.get_unique_name()}_min_{objective}')
            fieldnames.append(f'{virtual_net.get_unique_name()}_config')
            fieldnames.append(f'{virtual_net.get_unique_name()}_gene')
            fieldnames.append(f'{virtual_net.get_unique_name()}_latency')
        for i, chiplet in enumerate(chiplet_group):
            fieldnames.extend([
                f'chiplet_{i+1}_arch',
                f'chiplet_{i+1}_glb_scale',
                f'chiplet_{i+1}_pe_x_scale',
                f'chiplet_{i+1}_pe_y_scale'
            ])

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        row = {f'min_{objective}': best_value}
        for virtual_net in virtual_nets:
            row[f'{virtual_net.get_unique_name()}_min_{objective}'] = best_network_results[virtual_net.get_unique_name()]['min_value']
            row[f'{virtual_net.get_unique_name()}_config'] = config_to_ids(results[virtual_net.get_unique_name()]['best_config'])
            row[f'{virtual_net.get_unique_name()}_gene'] = results[virtual_net.get_unique_name()]['best_gene']
            row[f'{virtual_net.get_unique_name()}_latency'] = results[virtual_net.get_unique_name()]['best_latency']
        for i, chiplet in enumerate(chiplet_group):
            row[f'chiplet_{i+1}_arch'] = chiplet.arch_target
            row[f'chiplet_{i+1}_glb_scale'] = chiplet.global_buffer_size_scale
            row[f'chiplet_{i+1}_pe_x_scale'] = chiplet.pe_x_scale
            row[f'chiplet_{i+1}_pe_y_scale'] = chiplet.pe_y_scale
        writer.writerow(row)

    metric_name = "EDP" if objective == "edp" else "Energy"
    print(f"\n=== Best {metric_name} Result ===")
    print(f"Overall {metric_name}: {best_value:.4e}")
    print(f"\nNetwork-specific {metric_name.lower()} results:")
    for net_name, net_result in best_network_results.items():
        print(f"  {net_name}: {net_result['min_value']:.4e}")
    print(f"\nResults saved to {results_file}")

    return chiplet_group, results, best_value, best_network_results


# ---------------------------------------------------------------------------
# Mode 2: homogeneous separate baseline (formerly baseline_homo_sep.py)
# ---------------------------------------------------------------------------

def find_optimal_single_chiplet_per_network(
    arch_targets: List[str],
    virtual_networks: List[VirtualNetwork],
    objective: str = "energy",
    results_file: Optional[str] = None,
    database_file: str = "final_database.csv",
    logger: Optional[logging.Logger] = None,
    cost_aware: bool = False
) -> Dict[str, Dict]:
    """
    Find the optimal single chiplet configuration for each network individually.
    (Homogeneous separate baseline)
    Uses DAG-CP evaluation to properly handle all layers (avoids silent 0-energy bug).
    """
    if results_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f'optimal_single_chiplet_{objective}_cost_{cost_aware}_{timestamp}.csv'

    fieldnames = ['network', 'arch_target', 'glb_scale', 'pe_x_scale', 'pe_y_scale', objective, 'config', 'gene']
    with open(results_file, 'w', newline='') as csvfile:
        csv.DictWriter(csvfile, fieldnames=fieldnames).writeheader()

    best_results = {}

    for virtual_net in virtual_networks:
        network_name = virtual_net.get_unique_name()
        if logger:
            logger.info(f"Finding optimal chiplet for network: {network_name}")

        # Pre-compute DAG critical-path spec for this network
        cp_spec = create_cp_spec(virtual_net)

        best_value = float('inf')
        best_chiplet = None
        best_result = None

        total_configs = len(arch_targets) * len(_gp.pe_scales) * len(_gp.pe_scales) * len(_gp.glb_scales)
        processed_configs = 0

        for arch_target in arch_targets:
            for glb_scale in _gp.glb_scales:
                for pe_x_scale in _gp.pe_scales:
                    for pe_y_scale in _gp.pe_scales:
                        chiplet = ChipletConfig(
                            arch_target=arch_target,
                            global_buffer_size_scale=glb_scale,
                            pe_x_scale=pe_x_scale,
                            pe_y_scale=pe_y_scale
                        )
                        chiplet_group = [chiplet]

                        try:
                            best_gene, value, best_config = genetic_algo_opt_phy_net_dag(
                                virtual_network=virtual_net,
                                chiplet_group=chiplet_group,
                                cp_spec=cp_spec,
                                objective=objective,
                                population_size=10,
                                generations=10,
                                results_file=database_file,
                                cost_aware=cost_aware,
                                use_sequential=True,
                                v_het_batch=True,
                            )

                            if value < best_value:
                                best_value = value
                                best_chiplet = chiplet
                                best_result = {
                                    'best_gene': best_gene,
                                    'min_value': value,
                                    'best_config': best_config
                                }
                        except Exception as e:
                            msg = f"Error evaluating chiplet {chiplet}: {e}"
                            logger.error(msg) if logger else print(msg)

                        processed_configs += 1
                        if processed_configs % 10 == 0 or processed_configs == total_configs:
                            progress = (processed_configs / total_configs) * 100
                            print(f"\rProgress for {network_name}: {progress:.1f}% ({processed_configs}/{total_configs})", end="")

        print()

        best_results[network_name] = {'chiplet': best_chiplet, 'results': best_result}

        if best_chiplet:
            with open(results_file, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({
                    'network': network_name,
                    'arch_target': best_chiplet.arch_target,
                    'glb_scale': best_chiplet.global_buffer_size_scale,
                    'pe_x_scale': best_chiplet.pe_x_scale,
                    'pe_y_scale': best_chiplet.pe_y_scale,
                    objective: best_result['min_value'],
                    'config': str(config_to_ids(best_result['best_config'])),
                    'gene': str(best_result['best_gene'])
                })

        metric_name = "EDP" if objective == "edp" else "Energy"
        if logger:
            logger.info(f"Best chiplet for {network_name}:")
            logger.info(f"  Architecture: {best_chiplet.arch_target}")
            logger.info(f"  GLB Scale: {best_chiplet.global_buffer_size_scale}")
            logger.info(f"  PE X Scale: {best_chiplet.pe_x_scale}")
            logger.info(f"  PE Y Scale: {best_chiplet.pe_y_scale}")
            logger.info(f"  {metric_name}: {best_result['min_value']:.2e}")
        else:
            print(f"Best chiplet for {network_name}:")
            print(f"  Architecture: {best_chiplet.arch_target}")
            print(f"  GLB Scale: {best_chiplet.global_buffer_size_scale}")
            print(f"  PE X Scale: {best_chiplet.pe_x_scale}")
            print(f"  PE Y Scale: {best_chiplet.pe_y_scale}")
            print(f"  {metric_name}: {best_result['min_value']:.2e}")

    print(f"\nResults saved to {results_file}")
    return best_results


def find_optimal_single_chiplet_all_networks(
    arch_targets: List[str],
    virtual_networks: List[VirtualNetwork],
    objective: str = "energy",
    results_file: Optional[str] = None,
    database_file: str = "unified_database.csv",
    logger: Optional[logging.Logger] = None,
    cost_aware: bool = False
) -> Dict:
    """
    Find the single best chiplet configuration shared across ALL networks.
    (Homogeneous all-net baseline: one chiplet design for all workloads.)
    Uses DAG-CP evaluation for consistency with per-net baseline.
    """
    if results_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f'homo_allnet_{objective}_cost_{cost_aware}_{timestamp}.csv'

    best_geomean = float('inf')
    best_chiplet = None
    best_per_net = None

    total_configs = len(arch_targets) * len(_gp.pe_scales) * len(_gp.pe_scales) * len(_gp.glb_scales)
    processed = 0

    # Pre-compute CP specs for all networks
    cp_specs = {vn.get_unique_name(): create_cp_spec(vn) for vn in virtual_networks}

    for arch_target in arch_targets:
        for glb_scale in _gp.glb_scales:
            for pe_x_scale in _gp.pe_scales:
                for pe_y_scale in _gp.pe_scales:
                    chiplet = ChipletConfig(
                        arch_target=arch_target,
                        global_buffer_size_scale=glb_scale,
                        pe_x_scale=pe_x_scale,
                        pe_y_scale=pe_y_scale
                    )
                    chiplet_group = [chiplet]

                    # Evaluate across all networks
                    net_values = {}
                    valid = True
                    for vn in virtual_networks:
                        try:
                            _, value, config = genetic_algo_opt_phy_net_dag(
                                virtual_network=vn,
                                chiplet_group=chiplet_group,
                                cp_spec=cp_specs[vn.get_unique_name()],
                                objective=objective,
                                population_size=10,
                                generations=10,
                                results_file=database_file,
                                cost_aware=cost_aware,
                                use_sequential=True,
                                v_het_batch=True,
                            )
                            net_values[vn.get_unique_name()] = value
                        except Exception:
                            valid = False
                            break

                    if valid and net_values:
                        vals = list(net_values.values())
                        geomean = np.exp(np.mean(np.log(np.array(vals) + 1e-30)))
                        if geomean < best_geomean:
                            best_geomean = geomean
                            best_chiplet = chiplet
                            best_per_net = dict(net_values)

                    processed += 1
                    if processed % 10 == 0 or processed == total_configs:
                        print(f"\rHomo All-Net: {processed}/{total_configs} configs, best geomean={best_geomean:.4e}", end="")

    print()

    # Write results: one row per network (same format as per-net for easy comparison)
    fieldnames = ['network', 'arch_target', 'glb_scale', 'pe_x_scale', 'pe_y_scale', objective]
    with open(results_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        if best_chiplet and best_per_net:
            for net_name, value in best_per_net.items():
                writer.writerow({
                    'network': net_name,
                    'arch_target': best_chiplet.arch_target,
                    'glb_scale': best_chiplet.global_buffer_size_scale,
                    'pe_x_scale': best_chiplet.pe_x_scale,
                    'pe_y_scale': best_chiplet.pe_y_scale,
                    objective: value,
                })

    if best_chiplet:
        msg = (f"Best all-net chiplet: {best_chiplet.arch_target} "
               f"glb={best_chiplet.global_buffer_size_scale} "
               f"pe=({best_chiplet.pe_x_scale},{best_chiplet.pe_y_scale}) "
               f"geomean={best_geomean:.4e}")
        logger.info(msg) if logger else print(msg)

    print(f"Results saved to {results_file}")
    return {'chiplet': best_chiplet, 'geomean': best_geomean, 'per_net': best_per_net}


def run_all_homo_configurations(database_file="unified_database.csv"):
    """Run homo-sep baseline for all objective/cost_aware combinations."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    arch_targets = DEFAULT_ARCH_TARGETS
    virtual_nets = _build_virtual_nets(database_file=database_file)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    configs = [
        ("energy", False), ("energy", True),
        ("edp",    False), ("edp",    True),
    ]
    result_files = {}
    for objective, cost_aware in configs:
        label = f"{objective}_{'True' if cost_aware else 'False'}"
        logger.info(f"Starting optimization for {label}")
        fname = f'optimal_single_chiplet_{label}_{timestamp}.csv'
        find_optimal_single_chiplet_per_network(
            arch_targets=arch_targets,
            virtual_networks=virtual_nets,
            objective=objective,
            results_file=fname,
            database_file=database_file,
            logger=logger,
            cost_aware=cost_aware
        )
        result_files[label] = fname

    print("\nResults saved to:")
    for label, fname in result_files.items():
        print(f"  {label}: {fname}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chiplet baseline optimization")
    parser.add_argument("--mode", type=str, choices=["heter", "homo", "homo_allnet"], required=True,
                        help="'heter': heterogeneous global baseline (all chiplets); "
                             "'homo': homogeneous per-net baseline (best single chiplet per network); "
                             "'homo_allnet': homogeneous all-net baseline (one chiplet for all networks)")
    parser.add_argument("--objective", type=str, choices=["energy", "edp"], default="energy")
    parser.add_argument("--cost-aware", type=str, default="True")
    parser.add_argument("--database", type=str, default="unified_database.csv")
    parser.add_argument("--use-dag-cp", action="store_true", default=False,
                        help="Use DAG critical-path GA for transformer workloads")
    parser.add_argument("--no-het-batch", action="store_true", default=False,
                        help="Disable heterogeneous batching for off-CP ops")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    objective = args.objective.strip().lower()
    cost_aware = args.cost_aware.strip() != "False"
    v_het_batch = not args.no_het_batch

    if args.mode == "heter":
        virtual_nets = _build_virtual_nets(database_file=args.database)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f'all_chiplets_{timestamp}_{objective}_{cost_aware}.csv'
        all_chiplets_optimization(
            arch_targets=DEFAULT_ARCH_TARGETS,
            virtual_nets=virtual_nets,
            objective=objective,
            results_file=results_file,
            database_file=args.database,
            logger=logger,
            cost_aware=cost_aware,
            use_dag_cp=args.use_dag_cp,
            v_het_batch=v_het_batch,
        )
    elif args.mode == "homo_allnet":
        virtual_nets = _build_virtual_nets(database_file=args.database)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        for obj, ca in [("energy", False), ("energy", True), ("edp", False), ("edp", True)]:
            fname = f'homo_allnet_{obj}_cost_{ca}_{timestamp}.csv'
            logger.info(f"Homo All-Net: {obj}, cost_aware={ca}")
            find_optimal_single_chiplet_all_networks(
                arch_targets=DEFAULT_ARCH_TARGETS,
                virtual_networks=virtual_nets,
                objective=obj,
                results_file=fname,
                database_file=args.database,
                logger=logger,
                cost_aware=ca,
            )
    else:  # homo
        run_all_homo_configurations(database_file=args.database)

# nohup python3 baseline.py --mode=heter --objective=energy --cost-aware=True --use-dag-cp > best_e_t.out &
# nohup python3 baseline.py --mode=heter --objective=energy --cost-aware=False --use-dag-cp > best_e_f.out &
# nohup python3 baseline.py --mode=heter --objective=edp    --cost-aware=True --use-dag-cp > best_edp_t.out &
# nohup python3 baseline.py --mode=heter --objective=edp    --cost-aware=False --use-dag-cp > best_edp_f.out &
# nohup python3 baseline.py --mode=homo  --database=unified_database.csv > homo_baseline.out &
