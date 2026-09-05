from typing import Dict, List, Optional, Tuple, Set
import os
import sys
from unittest import skip
import joblib
import yaml
import csv
import json
from datetime import datetime
import logging
import numpy as np
from tqdm import tqdm

# Add scripts/ to path for shared modules (works on host and in Docker container)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in [os.path.join(_THIS_DIR, "..", "scripts"), os.path.join(_THIS_DIR, "..")]:
    if os.path.isfile(os.path.join(_candidate, "utility_functions.py")) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import pytimeloop.timeloopfe.v4 as tl
import timeloop_helper
from utility_functions import *
from global_parameter import *
from network_dataclass import *
from parse_stats import *
from fit_tile_size import fit_power_function
import fcntl

class TimeloopDBBuilder:
    def __init__(self, 
                arch_targets: List[str],
                nets: List[str],
                net_layers_dict: Dict[str, List[str]],
                base_dir: str, # log dir; workload dir
                batch_sizes_to_run: List[int] = [1,32],
                sequence_lengths_to_run: List[int] = [256],
                dram_configs: str = '[{"I":"LPDDR5","O":"LPDDR5"},{"I":"GDDR7","O":"GDDR7"}]',
                fused_layer_types_to_run: List[str] = ['single'],
                chunk_size: int = 50,
                cycle_time: float = 1e-9,
                is_transformer: bool = False,
                skip_softmax = False,
                softmax_only = False,
                output_base_dir = None # output dir 
                ):  # Default cycle time: 1 ns
        
        self.nets = nets
        self.net_layers_dict = net_layers_dict
        self.batch_sizes = batch_sizes_to_run
        self.sequence_lengths_to_run = sequence_lengths_to_run
        self.fused_layer_types = fused_layer_types_to_run
        self.is_transformer = is_transformer
        self.skip_softmax = skip_softmax
        self.softmax_only = softmax_only
        self.output_base_dir = output_base_dir
        # use TP degree from global parameter
        self.tp_degrees=tp_degrees
        
        self.base_dir = base_dir
        self.chunk_size = chunk_size
        self.cycle_time = cycle_time
        self.logger = self._setup_logging()
        self.processed_masters = set()
        # Define parameter ranges for grid search

        self.arch_targets = arch_targets
        self.glb_scales = glb_scales
        self.pe_scales = pe_scales
        self.dram_configs = json.loads(dram_configs)

        # Parse all layers and identify identical ones
        self.layer_parser = LayerParser(nets, net_layers_dict, base_dir, transformer=is_transformer)
        self.all_layers = self.layer_parser.parse_all_layers()
        
        self.unique_layers = self.layer_parser.get_unique_layers()
        
        # Track parsed results for CSV export
        self.parsed_results = []
        
        # Clear previous log files
        for file in ['timeloop_output.log', 'timeloop_errors.log']:
            log_path = os.path.join(self.base_dir, file)
            if os.path.exists(log_path):
                os.remove(log_path)
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging to both file and console."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.base_dir, f"timeloop_db_{timestamp}.log")
        self.progress_file = os.path.join(self.base_dir, f"progress_{timestamp}.log")
        
        logger = logging.getLogger('timeloop_db')
        logger.setLevel(logging.INFO)
        logger.handlers = []
        
        file_handler = logging.FileHandler(log_file)
        progress_handler = logging.FileHandler(self.progress_file)
        console_handler = logging.StreamHandler()
        
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        progress_formatter = logging.Formatter('%(message)s')
        
        file_handler.setFormatter(file_formatter)
        progress_handler.setFormatter(progress_formatter)
        console_handler.setFormatter(progress_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(progress_handler)
        logger.addHandler(console_handler)
        
        return logger
        
    def build_database(self, run_id=None, total_runs=None, shared_csv_path=None, networks_to_run=None):
        """Build database of timeloop results with layer deduplication.
        
        Args:
            run_id (int, optional): ID of this run for parallel processing.
            total_runs (int, optional): Total number of parallel runs.
            shared_csv_path (str, optional): Path to shared CSV file for results.
            networks_to_run (List[str], optional): List of specific networks to process.
        """
        self.processed_masters = set()
        # Get unique layers to run (only masters, not duplicates)
        unique_layers = self.layer_parser.get_unique_layers()
        
        # Initialize storage for parsed results
        self.parsed_results = []
    
        # Store shared CSV path
        self.shared_csv_path = shared_csv_path
    
        layer_keys_to_process = []
        # Get list of all layer_keys
        all_layer_keys = list(unique_layers.keys())
        
        # Filter layer keys by network if specified
        if networks_to_run:
            self.logger.info(f"Filtering for networks: {networks_to_run}")
            filtered_layer_keys = [
                layer_key for layer_key in all_layer_keys 
                if layer_key.split("@", 1)[0] in networks_to_run
            ]
            all_layer_keys = filtered_layer_keys
            self.logger.info(f"Found {len(all_layer_keys)} layers for the specified networks")
    
        # If run_id is specified, only process a subset of layers
        if run_id is not None and total_runs is not None:
            layers_per_run = max(1, len(all_layer_keys) // total_runs)
            start_idx = run_id * layers_per_run
            end_idx = min(start_idx + layers_per_run, len(all_layer_keys))
            
            # Select subset of layers for this run
            layer_keys_to_process = all_layer_keys[start_idx:end_idx]
            self.logger.info(f"Node {run_id} processing {len(layer_keys_to_process)} layers")
            
        else:
            # Process all layers
            layer_keys_to_process = all_layer_keys
            
        # Process only the assigned layers
        for layer_key in layer_keys_to_process:
            layer_data = unique_layers[layer_key]
            net, layer_name = layer_key.split("@", 1)

            layer_path = layer_data["yaml_path"]
            layer_config = layer_data["layer_config"]

            self.logger.info(f"Processing unique layer: {layer_key}")

            # re parse to get problem dimension
            # not optimal but acceptable
            with open(layer_path, "r") as f:
                yaml_data = yaml.safe_load(f)
            layer_config = yaml_data.get("problem", {}).get("instance", {})

            # Detect LLaMA BF16 workload
            is_llama = is_llama_network(net)

            if is_llama:
                # LLaMA BF16 code path — uses DSE-proven constraints
                op_suffix = layer_name.split("_", 1)[-1] if "_" in layer_name else layer_name
                for prefix in ["layer0_", "layer1_", "layer2_"]:
                    if layer_name.startswith(prefix):
                        op_suffix = layer_name[len(prefix):]
                        break

                tp_dim = LLAMA_TP_CONFIG.get(op_suffix, "M")
                output_feature_size = layer_config.get(tp_dim, 1)

                output_tile_channels = []
                for tp_degree in self.tp_degrees:
                    output_tile_channels.append(int(math.floor(output_feature_size / tp_degree)))
                layer_tp_degrees = self.tp_degrees

                if output_feature_size == 1:
                    output_tile_channels = [1]
                    layer_tp_degrees = [1]

                temp_arch_targets = self.arch_targets

                # Sequence length: prefill uses configured lengths, decode uses 1
                if "prefill" in net:
                    temp_sequence_lengths = self.sequence_lengths_to_run
                else:
                    temp_sequence_lengths = [1]

                temp_batch_sizes = self.batch_sizes
                is_attention_op = op_suffix in LLAMA_BATCH_AGNOSTIC_OPS
                if is_attention_op:
                    temp_batch_sizes = [1]

                # Deduplicate (batch, seq) combos that produce the same effective N.
                # For projections: N_eff = batch * seq (from workload file, seq already baked into N).
                # Since run_mapper_llama multiplies N by batch_size, combos with the
                # same batch*1 (decode) or batch*seq (prefill) product are redundant.
                # We keep one representative per unique N_eff and record the mapping
                # so results can be duplicated later.
                base_N = layer_config.get("N", 1)
                seen_n_eff = {}  # n_eff -> (batch_size, sequence_length)
                dedup_batch_seq = []
                dedup_map = {}  # (batch, seq) -> (master_batch, master_seq)
                for batch_size in temp_batch_sizes:
                    for seq in temp_sequence_lengths:
                        n_eff = base_N * batch_size  # seq is already in base_N for prefill workloads
                        if n_eff not in seen_n_eff:
                            seen_n_eff[n_eff] = (batch_size, seq)
                            dedup_batch_seq.append((batch_size, seq))
                        else:
                            dedup_map[(batch_size, seq)] = seen_n_eff[n_eff]

                if dedup_map:
                    orig_count = len(temp_batch_sizes) * len(temp_sequence_lengths)
                    self.logger.info(
                        f"  Dedup (batch,seq) for {op_suffix}: {orig_count} -> {len(dedup_batch_seq)} "
                        f"(skipped {list(dedup_map.keys())})")

                configs = [
                    (net, layer_path, batch_size, sequence_length, mapper_idx, 'single',
                     output_tile_channel, layer_tp_degrees[otc_idx],
                     arch, glb_scale, pe_x_scale, pe_y_scale, dram_config, self.output_base_dir)
                    for batch_size, sequence_length in dedup_batch_seq
                    for mapper_idx in range(num_mapping_per_arch)
                    for otc_idx, output_tile_channel in enumerate(output_tile_channels)
                    for arch in temp_arch_targets
                    for glb_scale in self.glb_scales
                    for pe_x_scale in self.pe_scales
                    for pe_y_scale in self.pe_scales
                    for dram_config in self.dram_configs
                ]
                mapper_fn = timeloop_helper.run_mapper_llama
            else:
                # Legacy GPT/CNN code path
                if self.is_transformer:
                    output_feature_size = layer_config[TRANSFORMER_TP_CONFIG[unifyname(layer_name)]]
                else:
                    output_feature_size = layer_config["M"]

                output_tile_channels = []
                for tp_degree in self.tp_degrees:
                    output_tile_channels += [int(math.floor(output_feature_size/tp_degree))]

                layer_tp_degrees = self.tp_degrees

                if output_feature_size == 1:
                    output_tile_channels = [1]
                    layer_tp_degrees = [1]

                temp_arch_targets = self.arch_targets
                if is_softmax_layers(layer_name):
                    temp_arch_targets = arch_vec_targets

                if "prefill" in net:
                    temp_sequence_lengths = self.sequence_lengths_to_run
                else:
                    temp_sequence_lengths = [1]

                temp_batch_sizes = self.batch_sizes
                if unifyname(layer_name) in batch_agnostic_ops:
                    temp_batch_sizes = [1]

                configs = [
                    (net, layer_path, batch_size, sequence_length, mapper_idx, fused_layer_type if not is_softmax_layers(layer_name) else layer_forced_fused_dict[unifyname(layer_name)], output_tile_channel, layer_tp_degrees[output_tile_channel_index],
                     arch, glb_scale, pe_x_scale, pe_y_scale, dram_config, self.is_transformer, True, 'eq', {}, self.output_base_dir)
                    for batch_size in temp_batch_sizes
                    for sequence_length in temp_sequence_lengths
                    for mapper_idx in range(num_mapping_per_arch)
                    for fused_layer_type in self.fused_layer_types
                    for output_tile_channel_index, output_tile_channel in enumerate(output_tile_channels)
                    for arch in temp_arch_targets
                    for glb_scale in self.glb_scales
                    for pe_x_scale in self.pe_scales
                    for pe_y_scale in self.pe_scales
                    for dram_config in self.dram_configs
                ]
                mapper_fn = timeloop_helper.run_mapper

            if not is_llama:
                if utility_functions.is_softmax_layers(layer_name):
                    if self.skip_softmax:
                        configs = []
                if self.softmax_only:
                    if not utility_functions.is_softmax_layers(layer_name):
                        configs = []

            total_configs = len(configs)
            self.logger.info(f"Total configurations for {layer_key}: {total_configs}")

            # Process configurations in chunks
            layer_results = []
            with tqdm(total=total_configs, desc=f"{layer_key}") as pbar:
                for i in range(0, len(configs), self.chunk_size):
                    chunk = configs[i:i + self.chunk_size]
                    chunk_results = joblib.Parallel(n_jobs=60)(
                        joblib.delayed(mapper_fn)(*config)
                        for config in chunk
                    )
                    layer_results.extend([r for r in chunk_results if r[0] is not None])
                    pbar.update(len(chunk))
                    
                    # Log progress
                    progress = (i + len(chunk)) / total_configs * 100
                    with open(self.progress_file, 'a') as f:
                        f.write(f"{layer_key}: {progress:.2f}% complete\n")
            
            # Process successful results
            output_dir = OUTPUT_DIR
            if self.output_base_dir:
                output_dir = os.path.join(self.output_base_dir, "outputs")

            for config_id, problem_id, _ in layer_results:
                parsed_result = process_mapping_results(config_id, problem_id, output_dir)
                if parsed_result:
                    parsed_result["is_duplicate"] = False
                    parsed_result["master_layer"] = layer_key
                    self.parsed_results.append(parsed_result)
            
            # Mark this master layer as processed
            self.processed_masters.add(layer_key)
            
            # Log completion and success rate
            successful = len(layer_results)
            if total_configs:
                self.logger.info(
                    f"Processed {layer_key}: "
                    f"Found {successful}/{total_configs} existing results "
                    f"({successful/total_configs*100:.2f}%)"
                )
            else:
                self.logger.info(
                    f"Skipped {layer_key}: "
                )

    def export_only(self, shared_csv_path=None, networks_to_run=None):
        """
        Process existing results and export to CSV without re-running timeloop simulations.
        This function reads existing mapping results from the output directory.
        
        Args:
            shared_csv_path (str, optional): Path to shared CSV file for results.
            networks_to_run (List[str], optional): List of specific networks to process.
        """
        # Initialize empty list for parsed results
        self.processed_masters = set()
    
        self.parsed_results = []
        
        # Get unique layers to process
        unique_layers = self.layer_parser.get_unique_layers()
        
        # Filter layer keys by network if specified
        if networks_to_run:
            self.logger.info(f"Filtering for networks: {networks_to_run}")
            filtered_layers = {
                layer_key: layer_data for layer_key, layer_data in unique_layers.items()
                if layer_key.split("@", 1)[0] in networks_to_run
            }
            unique_layers = filtered_layers
            self.logger.info(f"Found {len(unique_layers)} layers for the specified networks")
        
        # 预处理：根据 outputs 目录情况统一主从关系（实现见下方）
        # 简化：封装为一个小函数，按 outputs 目录统一主从关系
        base_output_dir = os.path.join(self.output_base_dir, "outputs") if self.output_base_dir else OUTPUT_DIR

        def _has_data(net: str, name: str) -> bool:
            p = os.path.join(base_output_dir, net, os.path.splitext(name)[0])
            try:
                return os.path.isdir(p) and any(os.scandir(p))
            except Exception:
                return False

        def _normalize_masters_by_outputs(unique_layers_dict: Dict[str, Dict]):
            for mk in list(unique_layers_dict.keys()):
                net_m, name_m = mk.split("@", 1)
                if _has_data(net_m, name_m):
                    continue
                alias = [k for k, v in self.all_layers.items() if v.get("master_layer_key") == mk and k.split("@", 1)[0] == net_m]
                repl = next((ak for ak in alias if _has_data(net_m, ak.split("@", 1)[1])), None)
                if not repl:
                    continue
                for kk in [mk] + alias:
                    info = self.all_layers.get(kk)
                    if not info:
                        continue
                    info["flag"] = kk != repl
                    info["master_layer_key"] = None if kk == repl else repl
                unique_layers_dict.pop(mk, None)
                rep = self.all_layers.get(repl)
                if rep:
                    unique_layers_dict[repl] = {
                        "layer_config": rep["layer_config"],
                        "yaml_path": rep["yaml_path"],
                        "flag": False,
                        "master_layer_key": rep["master_layer_key"],
                    }

        _normalize_masters_by_outputs(unique_layers)

        # Process each unique layer
        for layer_key, layer_data in unique_layers.items():
            net, layer_name = layer_key.split("@", 1)
            layer_path = layer_data["yaml_path"]
            layer_config = layer_data["layer_config"]
            
            self.logger.info(f"Processing existing results for layer: {layer_key}")
            
            # re parse to get problem dimension
            # not optimal but acceptable
            with open(layer_path, "r") as f:
                yaml_data = yaml.safe_load(f)
            layer_config = yaml_data.get("problem", {}).get("instance", {})
            
            # TP parallelism
            if self.is_transformer:
                tp_dim = TRANSFORMER_TP_CONFIG.get(unifyname(layer_name))
                if tp_dim and tp_dim in layer_config:
                    output_feature_size = layer_config[tp_dim]
                elif "M" in layer_config:
                    # Fallback for new-style layers (C,M,N format): TP on M (output channels)
                    output_feature_size = layer_config["M"]
                elif "H" in layer_config:
                    # Attention layers: TP on H (heads)
                    output_feature_size = layer_config["H"]
                else:
                    output_feature_size = 1
            else:
                output_feature_size = layer_config.get("M", 1) # OUTPUT channel
                
            output_tile_channels = []
            for tp_degree in self.tp_degrees:
                output_tile_channels += [int(math.floor(output_feature_size/tp_degree))]
    
            layer_tp_degrees = self.tp_degrees
            
            if output_feature_size == 1: 
                output_tile_channels = [1]
                layer_tp_degrees = [1]
            # disable TP for output channel =1

            # if softmax use vector unit
            temp_arch_targets = self.arch_targets
            if is_softmax_layers(layer_name):
                temp_arch_targets = arch_vec_targets

            # if decode fix sequence length to 1
            #temp_sequence_lengths = self.sequence_lengths_to_run
            temp_sequence_lengths = [1] # default for CNN
            if "prefill" in net:
                temp_sequence_lengths = self.sequence_lengths_to_run
            else:
                if "decode" in net:
                    temp_sequence_lengths = [1]
                if net == 'vit':
                    temp_sequence_lengths = [197]
                if net == 'stable_diffusion':
                    temp_sequence_lengths = [layer_config['P']]
            
            temp_batch_sizes = self.batch_sizes
            # for those operators, running each batch sequentially
            if unifyname(layer_name) in batch_agnostic_ops:
                temp_batch_sizes = [1]
            
            # this config is the same as database builder;
            # it's used only to read the mapper.st
            configs = [
                (net, layer_path, batch_size, sequence_length, mapper_idx, fused_layer_type if not is_softmax_layers(layer_name) else layer_forced_fused_dict[unifyname(layer_name)], output_tile_channel, layer_tp_degrees[output_tile_channel_index], 
                 arch, glb_scale, pe_x_scale, pe_y_scale, dram_config, self.is_transformer, True, 'eq', {})
                for batch_size in temp_batch_sizes
                for sequence_length in temp_sequence_lengths
                for mapper_idx in range(num_mapping_per_arch)
                for fused_layer_type in self.fused_layer_types
                for output_tile_channel_index, output_tile_channel in enumerate(output_tile_channels)

                for arch in temp_arch_targets
                for glb_scale in self.glb_scales
                for pe_x_scale in self.pe_scales
                for pe_y_scale in self.pe_scales
                for dram_config in self.dram_configs  
            ]

            total_configs = len(configs)
            found_configs = 0
            
            # Instead of running the configs, collect the existing results
            # layer_results = []
            with tqdm(total=total_configs, desc=f"Reading {layer_key}") as pbar:
                for config in configs:
                    
                    net, layer_path, batch_size,sequence_length,mapper_idx, fused_layer_type, output_tile_channels, tp_degree, \
                        arch, glb_scale, pe_x_scale, pe_y_scale, dram_config, _,_,_,_ = config
                    
                    # Construct the config_id and problem_id like run_single_config would
                    config_id = f"{arch}@glb{glb_scale}@pe_x_scale{pe_x_scale}@pe_y_scale{pe_y_scale}@{dram_config_to_id(dram_config)}"
                    problem_id = f"{net}@{layer_name}@{batch_size}@{sequence_length}@{mapper_idx}@{fused_layer_type}@{output_tile_channels}@{tp_degree}"

                    # Check if the result exists by trying to process it
                    output_dir = OUTPUT_DIR
                    if self.output_base_dir:
                        output_dir = os.path.join(self.output_base_dir, "outputs")
                    
                    parsed_result = process_mapping_results(config_id, problem_id, output_dir)
                    
                    if parsed_result:
                        #layer_results.append((config_id, problem_id))
                        parsed_result["is_duplicate"] = False
                        parsed_result["master_layer"] = layer_key    
                        found_configs += 1

                    if parsed_result is not None:
                        parsed_results = [parsed_result]
                        # generate multiple fusion results
                        parsed_results += gen_multi_fusion(parsed_result)

                        post_processed_results = []
                        for _parsed_result in parsed_results:
                            post_processed_results.append(post_process_mapping_results(_parsed_result))

                        # For batch-agnostic ops, duplicate batch=1 results to batch=4 and batch=8
                        results_to_add = post_processed_results
                        if unifyname(layer_name) in batch_agnostic_ops:
                            augmented_results = []
                            for _res in post_processed_results:
                                if _res and _res.get("batch_size") == 1:
                                    for _b in [4, 8]:
                                        _dup = _res.copy()
                                        _dup["batch_size"] = _b
                                        augmented_results.append(_dup)
                            if augmented_results:
                                results_to_add = post_processed_results + augmented_results

                        self.parsed_results += results_to_add
                    pbar.update(1)
            
            # Process successful results
            # for config_id, problem_id in layer_results:
            #     parsed_result = process_mapping_results(config_id, problem_id, OUTPUT_DIR)
            #     if parsed_result:
            #         parsed_result["is_duplicate"] = False
            #         parsed_result["master_layer"] = layer_key
            #         self.parsed_results.append(parsed_result)
            
            # Mark this master layer as processed
            self.processed_masters.add(layer_key)
            
            # Log completion and success rate
            if total_configs:
                self.logger.info(
                    f"Processed {layer_key}: "
                    f"Found {found_configs}/{total_configs} existing results "
                    f"({found_configs/total_configs*100:.2f}%)"
                )
            else:
                self.logger.info(
                    f"Skipped {layer_key}: "
                )
        
        # Export the results to CSV
        if shared_csv_path:
            self.export_results_to_csv(shared_csv_path)
        else:
            self.export_results_to_csv()
        
        self.logger.info(f"Export completed with {len(self.parsed_results)} total results")

    def export_results_to_csv(self, shared_csv_path=None):
        """
        Export all results to CSV with power function fitting, keeping static_power and area.
        If shared_csv_path is provided, results will be appended to that file atomically.
        """
        # Create a deep copy of existing parsed results
        all_results = self.parsed_results.copy()

            
        # Now add entries for duplicate layers by copying from their fitted masters
        for layer_key, layer_info in self.all_layers.items():
            # If this is a duplicate layer, create entries for it
            if layer_info["flag"]:
                net, layer_name = layer_key.split("@", 1)
                layer_path = layer_info["yaml_path"]
                master_key = layer_info["master_layer_key"]
                
                # Find all results for the master layer
                master_net, master_layer_name = master_key.split("@", 1)
                master_results = [r for r in self.parsed_results 
                                    if r["net"] == master_net and 
                                        r["layer_name"] == os.path.splitext(master_layer_name)[0]]
                
                # Create copies for this duplicate layer
                for master_result in master_results:
                    # Create a deep copy of the master result
                    duplicate_result = master_result.copy()
                    
                    # Update with this layer's information
                    duplicate_result["net"] = net
                    duplicate_result["layer_name"] = os.path.splitext(layer_name)[0]
                    duplicate_result["is_duplicate"] = True
                    duplicate_result["master_layer"] = master_key
                    
                    # Add to our results
                    all_results.append(duplicate_result)
        
        if shared_csv_path:
            csv_file = shared_csv_path
            write_mode = 'a'  # Append mode
        else:
            csv_file = os.path.join(self.base_dir, f"timeloop_fitted_results.csv")
            write_mode = 'w'  # Write mode (overwrite)
        
        if all_results:
            # Get all field names and exclude specific columns
            all_fieldnames = list(all_results[0].keys())
            fieldnames = [field for field in all_fieldnames 
                        if field not in ['is_duplicate', 'master_layer']]
            
            # Check if file already exists
            file_exists = os.path.isfile(csv_file)
            
            with open(csv_file, write_mode, newline='') as f:
                # If appending to shared file, use file locking
                if shared_csv_path:
                    fcntl.flock(f, fcntl.LOCK_EX)
                
                try:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                    
                    # Write header only if file is new or in write mode
                    if not file_exists or write_mode == 'w':
                        writer.writeheader()
                    
                    writer.writerows(all_results)
                finally:
                    # Always release lock if we acquired one
                    if shared_csv_path:
                        fcntl.flock(f, fcntl.LOCK_UN)
            
            self.logger.info(f"Exported {len(all_results)} results to {csv_file}")
        else:
            self.logger.info("No results to export")

    
    def print_savings_report(self):
        """Print a report of the computational savings from layer deduplication."""
        total_layers = len(self.all_layers)
        unique_layers = sum(1 for info in self.all_layers.values() if not info["flag"])
        duplicate_layers = total_layers - unique_layers
        
        # Calculate average configurations per layer
        config_per_layer = len(self.arch_targets) * len(self.glb_scales) * len(self.pe_scales)**2 * len(tp_degrees) * num_mapping_per_arch * len(fused_layer_types)
        
        total_configs_naive = total_layers * config_per_layer
        total_configs_optimized = unique_layers * config_per_layer
        
        self.logger.info("\n=== Optimization Report ===")
        self.logger.info(f"Total layers: {total_layers}")
        self.logger.info(f"Unique layers: {unique_layers}")
        self.logger.info(f"Duplicate layers: {duplicate_layers}")
        self.logger.info(f"Configurations per layer: {config_per_layer}")
        self.logger.info(f"Total configurations (naive): {total_configs_naive}")
        self.logger.info(f"Total configurations (optimized): {total_configs_optimized}")


    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run Timeloop for neural network layers')
    parser.add_argument('--run-id', type=int, help='Unique ID for this run')
    parser.add_argument('--total-runs', type=int, help='Total number of runs')
    parser.add_argument('--shared-csv', type=str, help='Path to shared CSV file')
    parser.add_argument('--networks', type=str, help='Comma-separated list of networks to process')
    parser.add_argument('--networks-to-run', type=str, help='Comma-separated list of specific networks to run on this machine')
    parser.add_argument('--is-transformer', type=int, help='if current workload to run is transformer')
    parser.add_argument('--dram-configs', type=str,
                    default='[{"I":"LPDDR5","O":"GDDR7"}]',
                    help='JSON string of DRAM configurations (default: [{"I":"LPDDR5","O":"GDDR7"},{"I":"GDDR7","O":"LPDDR5"}])')
    parser.add_argument('--arch', type=str)
    parser.add_argument('--batches-to-run', type=str)
    parser.add_argument('--skip-softmax', type=int)
    parser.add_argument('--softmax-only', type=int, default=0)
    parser.add_argument('--output-base-dir', type=str, default=None, help='Base output directory')
    parser.add_argument('--export-only', action='store_true', help='Skip running timeloop; only export existing results to CSV')
    args = parser.parse_args()
    
    # cannot skip softmax layers and only run softmax layers at the same time
    assert not (args.skip_softmax and args.softmax_only), "skip_softmax and softmax_only cannot both be True"

    # Configuration
    arch_targets = DEFAULT_ARCH_TARGETS
    if args.arch:
        arch_targets = args.arch.split(',')
        print(f"Will only process layers for arch: {arch_targets}")
    
    batches_to_run = [1]
    if args.batches_to_run:
        batches_to_run = [int(x.strip()) for x in args.batches_to_run.split(',')]
    else:
        if args.is_transformer:
            batches_to_run = [1,4,8,16]
        else:
            batches_to_run = [1,32]
    
    # Use provided networks or default
    if args.networks:
        nets = args.networks.split(',')
    else:
        #nets = ["alexnet"]
        nets = ["mobilenet_v3_small","efficientnet_b0","resnet50", "replknet31b"]
    
    # Use provided networks_to_run if available
    networks_to_run = ["mobilenet_v3_small","efficientnet_b0","resnet50", "replknet31b"]
    if args.networks_to_run:
        networks_to_run = args.networks_to_run.split(',')
        print(f"Will only process layers from networks: {networks_to_run}")
    
    net_layers_dict = gen_net_layers_dict(nets)
    
    print(f"Processing networks: {nets}")
    print(f"Progress will be written to {os.path.join(THIS_SCRIPT_DIR, 'progress_<timestamp>.log')}")
    print(f"Any errors will be written to {os.path.join(THIS_SCRIPT_DIR, 'timeloop_errors.log')}")
    
    # Initialize and run
    db_builder = TimeloopDBBuilder(
        arch_targets=arch_targets,
        nets=nets,
        net_layers_dict=net_layers_dict,
        base_dir=THIS_SCRIPT_DIR,
        chunk_size=64,
        cycle_time=cycle_time,  # 1 ns clock cycle time
        batch_sizes_to_run= batches_to_run,
        sequence_lengths_to_run = seq_configs(transformer=args.is_transformer),
        dram_configs = args.dram_configs,
        skip_softmax = args.skip_softmax,
        softmax_only= args.softmax_only,
        output_base_dir = args.output_base_dir,
        is_transformer = args.is_transformer
    )

    if args.export_only:
        # Only export existing results without running timeloop (formerly database_builder_csv.py)
        db_builder.export_only(
            shared_csv_path=args.shared_csv,
            networks_to_run=networks_to_run
        )
    else:
        # Build the database with deduplication
        db_builder.build_database(
            run_id=args.run_id,
            total_runs=args.total_runs,
            shared_csv_path=args.shared_csv,
            networks_to_run=networks_to_run
        )

    # Print savings report
    db_builder.print_savings_report()

#'[{"I":"LPDDR5","O":"LPDDR5"}, {"I":"GDDR7","O":"GDDR7"}, {"I":"LPDDR5","O":"GDDR7"},{"I":"GDDR7","O":"LPDDR5"}, {"I":"HBM3","O":"HBM3"}]'

# python3 database_builder.py --networks=gpt_OPT-66B_prefill --networks-to-run=gpt_OPT-66B_prefill --is-transformer=1 --dram-configs='[{"I":"LPDDR5","O":"LPDDR5"}, {"I":"GDDR7","O":"GDDR7"}, {"I":"LPDDR5","O":"GDDR7"},{"I":"GDDR7","O":"LPDDR5"}, {"I":"HBM3","O":"HBM3"}]'
    
# python3 database_builder.py   --networks=mobilenet_v3_small,efficientnet_b0,resnet50,replknet31b --networks-to-run=mobilenet_v3_small,efficientnet_b0 --is-transformer=0 --dram-configs='[{"I":"LPDDR5","O":"LPDDR5"}, {"I":"GDDR7","O":"GDDR7"}]'

# python3 database_builder.py   --networks=mobilenet_v3_small,efficientnet_b0,resnet50,replknet31b --networks-to-run=resnet50,replknet31b --is-transformer=0 --dram-configs='[{"I":"LPDDR5","O":"LPDDR5"}, {"I":"GDDR7","O":"GDDR7"}]'


# python3 database_builder.py --networks=gpt-1.3B_decode --networks-to-run=gpt-1.3B_decode --is-transformer=1 --dram-configs='[{"I":"LPDDR5","O":"LPDDR5"},{"I":"HBM3","O":"HBM3"}]'
