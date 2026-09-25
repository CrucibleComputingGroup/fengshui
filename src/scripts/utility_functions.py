import os
import logging
from typing import List, Tuple, Optional, Set, Dict
from datetime import datetime
import sys
import argparse
from global_parameter import *
import glob
import numpy as np
import math
from itertools import product


# Legacy softmax detection — used by database_builder.py, parse_stats.py,
# cal_perf_phy_net.py, network_dataclass.py, workload_parser.py, layer_size_analysis.py
def is_softmax_layers(problem_name):
    # Legacy naming: layer2_max, layer3_sn, layer4_sd, layer5_a
    # New naming: layer0_softmax_max, layer0_softmax_sub_exp, layer0_softmax_sum, layer0_softmax_div
    if "softmax" in problem_name:
        return True
    return problem_name.endswith("_max") or problem_name.endswith("_sn") or problem_name.endswith("_sd") or problem_name.endswith("_a")

def is_projection_layers(problem_name):
    if "attn_" in problem_name:
        return False
    if problem_name.endswith("_proj"):
        return True
    # Legacy naming (layer0_2_q, layer0_3_k, etc.)
    return problem_name.endswith("_q") or problem_name.endswith("_k") or problem_name.endswith("_v") or problem_name.endswith("_o") or problem_name.endswith("_ffn1") or problem_name.endswith("_ffn2")
def is_attention_layers(problem_name):
    if "attn_qk" in problem_name or "attn_v" in problem_name:
        return True
    # Legacy naming
    return problem_name.endswith("_qk") or problem_name.endswith("_av")
def is_new_style_projection(problem_name):
    """Check if this is a new-style projection layer (e.g., q_proj, gate_proj)."""
    return problem_name.endswith("_proj")
def is_gemm_layer(problem_name):
    """Check if this is any GEMM-based layer (projection, expert, lm_head, router)."""
    return is_projection_layers(problem_name) or problem_name.endswith("lm_head") or problem_name.endswith("router") or "expert_" in problem_name

def config_to_ids(min_e_config):
    res_ids = []
    for group_config in min_e_config['functions']:
        res_ids.append(group_config.id)
    return res_ids

def dram_config_to_id(dram_config):
    return f"{dram_config['I']}@{dram_config['O']}"


def all_fillings(choices_per_pos):
    """
    choices_per_pos: list of iterables; e.g. [[1,2], ['a','b','c'], [True, False]]
    returns a list of lists, each a valid filling
    """
    return [list(p) for p in product(*choices_per_pos)]

def get_piecewise_linear_value(x, x_points, y_points):
    """
    Get value using piecewise linear interpolation between filtered points.
    Automatically filters out pairs where either x or y is infinity.
    
    Args:
        x: The x value to evaluate
        x_points: Array of x values
        y_points: Array of y values
        
    Returns:
        Interpolated y value
    """
    # Filter out infinity and NaN values
    valid_indices = []
    for i in range(len(x_points)):
        if (i < len(y_points) and 
            np.isfinite(x_points[i]) and 
            np.isfinite(y_points[i])):
            valid_indices.append(i)
    
    # Extract only the valid points
    filtered_x = np.array([x_points[i] for i in valid_indices])
    filtered_y = np.array([y_points[i] for i in valid_indices])
    
    # Check if we have enough points after filtering
    if len(filtered_x) == 0:
        return float('inf')  # Return infinity when no valid points exist
    
    if len(filtered_x) == 1:
        # Only one valid point, return its y value
        return filtered_y[0]
    
    # Find the two closest points for interpolation
    if x < min(filtered_x):
        # Extrapolate below smallest x using the first two points
        i0, i1 = 0, 1
    elif x > max(filtered_x):
        # Extrapolate above largest x using the last two points
        i0, i1 = len(filtered_x) - 2, len(filtered_x) - 1
    else:
        # Interpolate between two points
        i1 = np.searchsorted(filtered_x, x)
        i0 = i1 - 1
    
    # Linear interpolation formula: y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    x0, x1 = filtered_x[i0], filtered_x[i1]
    y0, y1 = filtered_y[i0], filtered_y[i1]
    
    # Avoid division by zero
    if x0 == x1:
        return y0
    
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    
    
def get_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--clear-outputs",
        default=False,
        action="store_true",
        help="Clear all generated outputs",
    )
    argparser.add_argument(
        "--architecture",
        type=str,
        default="eyeriss_like",
        help="Architecture to run in the example_designs directory. "
        "If 'all' is given, all architectures will be run.",
    )
    argparser.add_argument(
        "--generate-ref-outputs",
        default=False,
        action="store_true",
        help="Generate reference outputs instead of outputs",
    )
    argparser.add_argument(
        "--problem",
        type=str,
        default=None,
        help="Problem to run in the layer_shapes directory. If a directory is "
        "specified, all problems in the directory will be run. If not specified, "
        "the default problem will be run.",
    )
    argparser.add_argument(
        "--n_jobs", type=int, default=None, help="Number of jobs to run in parallel"
    )
    argparser.add_argument(
        "--remove-sparse-opts",
        default=False,
        action="store_true",
        help="Remove sparse optimizations",
    )
    return argparser.parse_args()

def clean_directory(dir_path, keep_file ="timeloop-mapper.stats.txt" ):
    # Ensure directory exists
    if not os.path.exists(dir_path):
        return
    
    stats_file = os.path.join(dir_path, keep_file)
    mapper_stats_file = os.path.join(dir_path, "timeloop-mapper.map.txt")
    input_file = os.path.join(dir_path, "parsed-processed-input.yaml")
    # too large and can be easily recreated 
    # input_file = os.path.join(dir_path, "parsed-processed-input.yaml")
    # Check if the specific file exists
    if os.path.isfile(stats_file):
        # Get all files in the directory
        all_files = glob.glob(os.path.join(dir_path, "*"))
        
        # Remove all files except the stats file
        for file_path in all_files:
            if file_path != stats_file and file_path != mapper_stats_file and file_path != input_file and os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
        

def get_layer_name_from_yaml(path: str) -> str:
    """Extract base layer name without prefix/extension from path."""
    return os.path.splitext(os.path.basename(path))[0]

def gen_net_layers_dict(nets,NET_DIR=None):
    # generate net_layers_dict from nets
    # net 1 -> {layer1, layer2}
    # net 2 -> {layer1, layer2}
    if NET_DIR == None:
        THIS_SCRIPT_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
        NET_DIR = os.path.join(THIS_SCRIPT_DIR, "workloads")
    # Get layers for each network
    net_layers_dict = {}
    for net in nets:
        net_layers_dir = os.path.join(NET_DIR, net)
        net_layers_dict[net] = [                
            os.path.join(net_layers_dir, f)
            for f in os.listdir(net_layers_dir)
            if not f.endswith(".ipynb_checkpoints")
        ]
    return net_layers_dict

def setup_logging(base_dir: str) -> logging.Logger:
    """Setup logging to both file and console."""
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(base_dir, f"optimization_run_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger('optimization')
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def calculate_average_opt_value(results: Dict, objective: str = "energy", geomean=True) -> Tuple[float, Dict]:
    """
    Calculate average optimization value across all networks.
    
    Args:
        results: Dictionary of optimization results
        objective: Optimization objective, either "energy" or "edp"
        geomean: If True (default), calculate geometric mean. If False, calculate arithmetic mean.
        
    Returns:
        Tuple containing average value and network-specific results
    """
    network_results = {}
    
    if geomean:
        # Geometric mean over ALL networks. If any network returns inf,
        # the pool cannot handle all workloads so the result is inf.
        product_value = 1
        n_total = len(results)

        for net_name, net_result in results.items():
            min_value = net_result['min_value']
            network_results[net_name] = {
                'min_value': min_value
            }
            if not math.isfinite(min_value) or min_value <= 0:
                # Pool can't handle this network — overall result is inf
                return float('inf'), network_results
            product_value *= min_value

        avg_value = product_value ** (1/n_total) if n_total > 0 else float('inf')
    else:
        # Original arithmetic mean calculation
        total_value = 0
        
        for net_name, net_result in results.items():
            min_value = net_result['min_value']
            total_value += min_value
            
            network_results[net_name] = {
                'min_value': min_value
            }
        
        # Arithmetic mean
        avg_value = total_value / len(results) if len(results) > 0 else 0
    
    return avg_value, network_results
    
def cal_opt_val_fused(buffer_config_optimized_results: Dict,
                      query_points=None,
                      query_points_dict=None):
    """Calculate minimum energy for one physical network.
    optimized_results: buffer_config_str -> fusion_group_idx -> res List(min_e,id)

    query_points_dict: buffer_config_str -> list of query points (adaptive mode)
    query_points: single list of query points for all configs (legacy mode)

    Returns infinity if any network has no valid results."""
    # Backwards compatibility: if query_points given but not dict, broadcast
    if query_points is None and query_points_dict is None:
        query_points = base_query_points
    if query_points_dict is None:
        query_points_dict = {k: query_points for k in buffer_config_optimized_results}

    try:
        e_results = {}
        edp_results = {}
        for buffer_config_str,optimized_results in buffer_config_optimized_results.items():
            e_results[buffer_config_str] = {}
            edp_results[buffer_config_str] = {}

            qp = query_points_dict[buffer_config_str]

            groups = list(optimized_results.keys())
            if not groups:
                raise ValueError("No fusion groups with valid results")

            # Get the first layer's results
            first_layer_group = optimized_results[groups[0]]

            if not first_layer_group:
                raise ValueError(f"Fusion group {groups[0]} has no results")

            # Initialize summed results with first layer
            summed_results = [(e, [f]) for e, f in first_layer_group]

            # Sum with subsequent layers elementwise
            for group in groups[1:]:
                group_results = optimized_results[group]

                if not group_results:
                    raise ValueError(f"Group {group} has no results")

                # Ensure same number of elements in each layer
                if len(group_results) != len(summed_results):
                    raise ValueError(f"Layer length mismatch in {group} has {len(group_results)} results, expected {len(summed_results)}")

                new_summed = []
                # Sum elementwise
                for i in range(len(summed_results)):
                    sum_e, sum_funcs = summed_results[i]
                    group_e, group_func = group_results[i]
                    new_summed.append((
                        sum_e + group_e,
                        sum_funcs + [group_func]
                    ))
                summed_results = new_summed

            # Find minimum energy sum and its functions
            min_idx = min(range(len(summed_results)), key=lambda i: summed_results[i][0])
            min_sum_e, min_funcs = summed_results[min_idx]

            if min_sum_e <= 0:
                raise ValueError(f"Network has non-positive minimum energy: {min_sum_e} \n{min_funcs}")

            e_temp_result = {
                'min_val': min_sum_e,
                'functions': min_funcs,
                'idx': min_idx,
                'latency': qp[min_idx] if min_idx < len(qp) else None
            }

            e_results[buffer_config_str] = e_temp_result

            # Calculate EDP = e * query[index]
            edp_values = [(e * qp[i], i) for i, (e, _) in enumerate(summed_results)]
            # Find the index with minimum EDP
            min_edp_idx = min(range(len(edp_values)), key=lambda i: edp_values[i][0])
            # Get the original tuple corresponding to minimum EDP
            _, min_edp_funcs = summed_results[min_edp_idx]

            # If you want the actual EDP value as well
            min_edp_value = edp_values[min_edp_idx][0]

            edp_temp_results = {
                'min_val': min_edp_value,
                'functions': min_edp_funcs,
                'idx': min_edp_idx,
                'latency': qp[min_edp_idx] if min_edp_idx < len(qp) else None
            }

            edp_results[buffer_config_str] = edp_temp_results
            if min_edp_value <= 0:
                raise ValueError(f"Network has non-positive minimum edp: {min_edp_value}")


        # New shape: buffer_cfg_idx -> fusion_group_idx -> [(min_e, id)]
        best_energy = (float('inf'), {'min_val': float('inf'), 'functions': [], 'idx': -1, 'buffer_idx': None})
        best_edp    = (float('inf'), {'min_val': float('inf'), 'functions': [], 'idx': -1, 'buffer_idx': None})

        for buf_cfg, _ in buffer_config_optimized_results.items():

            # Track global min ENERGY
            if e_results[buf_cfg]['min_val'] < best_energy[0]:
                best_energy = (
                    e_results[buf_cfg]['min_val'],
                    dict(e_results[buf_cfg], buffer_idx=[buf_cfg])
                )

            # Track global min EDP
            if edp_results[buf_cfg]['min_val'] < best_edp[0]:
                best_edp = (
                    edp_results[buf_cfg]['min_val'],
                    dict(edp_results[buf_cfg], buffer_idx=[buf_cfg])
                )

        return best_energy, best_edp

    except Exception:
        return (float('inf'), None), (float('inf'), None)



def print_accelerator_mappings(logger: logging.Logger, 
                             optimized_results: Dict,
                             query_points: List[float]):
    """Print accelerator mappings for each network and layer with change detection."""
    logger.info("\n=== Accelerator Mappings ===")
    
    for net in optimized_results:
        logger.info(f"\nNetwork: {net}")
        
        # Get sorted layer names
        layers = sorted(optimized_results[net].keys())
        
        # Print layer chain
        layer_chain = " -> ".join(layers)
        logger.info(f"Layer chain: {layer_chain}")
        
        # Keep track of previous mappings to detect changes
        prev_mappings = None
        prev_latency = None
        
        # Print mappings for each latency point
        for i, latency in enumerate(query_points):
            current_mappings = []
            
            for layer in layers:
                opt_val, opt_func = optimized_results[net][layer][i]
                if opt_func is None:
                    current_mappings.append("NO_VALID_MAPPING")
                else:
                    current_mappings.append(opt_func.id)
            
            # Check if mappings changed from previous latency point
            if current_mappings != prev_mappings:
                # If this isn't the first change, print a summary of the previous range
                if prev_mappings is not None and prev_latency is not None:
                    logger.info(f"  (Above mapping maintained from {prev_latency:.5f}s to {query_points[i-1]:.5f}s)")
                
                logger.info(f"\nLatency = {latency:.5f}s:")
                mapping_chain = " -> ".join(current_mappings)
                logger.info(f"Mappings: {mapping_chain}")
                
                # Update previous values
                prev_mappings = current_mappings
                prev_latency = latency
        
        # Print final range if we had valid mappings
        if prev_mappings is not None and prev_latency is not None:
            logger.info(f"  (Above mapping maintained from {prev_latency:.5f}s to {query_points[-1]:.5f}s)")

