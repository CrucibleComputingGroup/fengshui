# parser.py
import os
import re
from typing import Dict, List, Optional, Tuple
import yaml
from global_parameter import *

import utility_functions

class MappingResult:
    """Store results for a single mapping configuration
    All units are in SI units:
    - Energy: Joules (J)
    - Power: Watts (W)
    - Time: seconds (s)
    - Area: um^2
    """
    def __init__(self,
                 static_energy: float,  # input in pJ, converted to J
                 dynamic_energy: float,  # input in pJ, converted to J
                 area: float,
                 cycles: int,
                 utilization: float,
                 energy_breakdown: Dict[str, float],
                 dram_metrics: Optional[Dict[str, float]] = None):
        # Convert energies from pJ to J
        self.static_energy = static_energy * 1e-12
        self.dynamic_energy = dynamic_energy * 1e-12
        self.area = area
        self.cycles = cycles
        self.utilization = utilization
        # Convert energy breakdown from fJ/compute to J/compute
        self.energy_breakdown = {k: v * 1e-15 for k, v in energy_breakdown.items()}
        
        # DRAM specific metrics
        self.dram_metrics = dram_metrics or {}
    
    def get_metrics(self, cycle_time: float) -> Tuple[float, float, float]:
        """
        Calculate metrics in SI units.
        Returns:
            Tuple[float, float, float]: (latency in s, static power in W, dynamic energy in J)
        """
        latency = self.cycles * cycle_time
        static_power = self.static_energy / latency if latency > 0 else 0
        #print(static_power,latency,cycle_time)
        return latency, static_power, self.dynamic_energy
    
    def get_dram_metrics(self) -> Dict[str, float]:
        """
        Get DRAM specific metrics.
        Returns:
            Dict[str, float]: Dictionary containing DRAM metrics
        """
        return self.dram_metrics
    
    def __str__(self, cycle_time: float = 1e-9):
        latency, static_power, dynamic_energy = self.get_metrics(cycle_time)
        result = (f"Cycles: {self.cycles}\n"
                f"Latency: {latency:.2e} s\n"
                f"Static Power: {static_power:.2e} W\n"
                f"Dynamic Energy: {dynamic_energy:.2e} J\n"
                f"Utilization: {self.utilization:.2f}%\n"
                f"Area: {self.area:.2f} um^2")
        
        # Add DRAM metrics if available
        if self.dram_metrics:
            result += "\n\nDRAM Metrics:"
            for key, value in self.dram_metrics.items():
                formatted_key = key.replace('_', ' ').title()
                result += f"\n  {formatted_key}: {value:.2f}"
                
        return result

def extract_dram_metrics(content: str) -> Dict[str, float]:
    """Extract detailed metrics from the DRAM_I and DRAM_O sections of the stats file."""
    
    dram_metrics = {
        "i_utilized_capacity": 0,
        "i_scalar_reads": 0,
        "i_scalar_fills": 0,
        "i_scalar_updates": 0,
        "w_utilized_capacity": 0,
        "w_scalar_reads": 0,
        "w_scalar_fills": 0,
        "w_scalar_updates": 0,
        "o_utilized_capacity": 0,
        "o_scalar_reads": 0,
        "o_scalar_fills": 0,
        "o_scalar_updates": 0,
        "i_throttling": 1.0,
        "o_throttling": 1.0
    }
    
    # Process DRAM_I section (for Inputs/Inputs2 and Weights/Inputs1)
    # New LLaMA format uses Inputs1 (weights) and Inputs2 (activations) instead of Weights/Inputs
    dram_i_section = re.search(r"=== DRAM_I ===.*?(?=Networks|Summary Stats|===|\Z)", content, re.DOTALL)
    if dram_i_section:
        dram_i_text = dram_i_section.group(0)

        # Get bandwidth throttling for DRAM_I
        throttling_match = re.search(r"Bandwidth throttling\s*:\s*([\d.]+)", dram_i_text)
        if throttling_match:
            dram_metrics["i_throttling"] = float(throttling_match.group(1))

        # Process Inputs/Inputs2 section (activations → i_ fields)
        inputs_section = re.search(r"(?:Inputs|Inputs2):.*?(?=(?:Weights|Inputs1|Outputs):|Networks|Summary Stats|===|\Z)",
                                  dram_i_text, re.DOTALL)
        if inputs_section:
            inputs_text = inputs_section.group(0)

            utilized_capacity_match = re.search(r"Utilized capacity\s*:\s*([\d.]+)", inputs_text)
            scalar_reads_match = re.search(r"Scalar reads \(per-instance\)\s*:\s*([\d.]+)", inputs_text)
            scalar_fills_match = re.search(r"Scalar fills \(per-instance\)\s*:\s*([\d.]+)", inputs_text)
            scalar_updates_match = re.search(r"Scalar updates \(per-instance\)\s*:\s*([\d.]+)", inputs_text)

            if utilized_capacity_match:
                dram_metrics["i_utilized_capacity"] = float(utilized_capacity_match.group(1))
            if scalar_reads_match:
                dram_metrics["i_scalar_reads"] = float(scalar_reads_match.group(1))
            if scalar_fills_match:
                dram_metrics["i_scalar_fills"] = float(scalar_fills_match.group(1))
            if scalar_updates_match:
                dram_metrics["i_scalar_updates"] = float(scalar_updates_match.group(1))

        # Process Weights/Inputs1 section (weights → w_ fields)
        weights_section = re.search(r"(?:Weights|Inputs1):.*?(?=(?:Inputs|Inputs2|Outputs):|Networks|Summary Stats|===|\Z)",
                                   dram_i_text, re.DOTALL)
        if weights_section:
            weights_text = weights_section.group(0)

            utilized_capacity_match = re.search(r"Utilized capacity\s*:\s*([\d.]+)", weights_text)
            scalar_reads_match = re.search(r"Scalar reads \(per-instance\)\s*:\s*([\d.]+)", weights_text)
            scalar_fills_match = re.search(r"Scalar fills \(per-instance\)\s*:\s*([\d.]+)", weights_text)
            scalar_updates_match = re.search(r"Scalar updates \(per-instance\)\s*:\s*([\d.]+)", weights_text)

            if utilized_capacity_match:
                dram_metrics["w_utilized_capacity"] = float(utilized_capacity_match.group(1))
            if scalar_reads_match:
                dram_metrics["w_scalar_reads"] = float(scalar_reads_match.group(1))
            if scalar_fills_match:
                dram_metrics["w_scalar_fills"] = float(scalar_fills_match.group(1))
            if scalar_updates_match:
                dram_metrics["w_scalar_updates"] = float(scalar_updates_match.group(1))
    
    # Process DRAM_O section (for Outputs)
    dram_o_section = re.search(r"=== DRAM_O ===.*?(?=Networks|Summary Stats|===|\Z)", content, re.DOTALL)
    if dram_o_section:
        dram_o_text = dram_o_section.group(0)
        
        # Get bandwidth throttling for DRAM_O
        throttling_match = re.search(r"Bandwidth throttling\s*:\s*([\d.]+)", dram_o_text)
        if throttling_match:
            dram_metrics["o_throttling"] = float(throttling_match.group(1))
        
        # Process Outputs section
        outputs_section = re.search(r"Outputs:.*?(?=Weights:|Inputs:|Networks|Summary Stats|===|\Z)", 
                                   dram_o_text, re.DOTALL)
        if outputs_section:
            outputs_text = outputs_section.group(0)
            
            utilized_capacity_match = re.search(r"Utilized capacity\s*:\s*([\d.]+)", outputs_text)
            scalar_reads_match = re.search(r"Scalar reads \(per-instance\)\s*:\s*([\d.]+)", outputs_text)
            scalar_fills_match = re.search(r"Scalar fills \(per-instance\)\s*:\s*([\d.]+)", outputs_text)
            scalar_updates_match = re.search(r"Scalar updates \(per-instance\)\s*:\s*([\d.]+)", outputs_text)
            
            if utilized_capacity_match:
                dram_metrics["o_utilized_capacity"] = float(utilized_capacity_match.group(1))
            if scalar_reads_match:
                dram_metrics["o_scalar_reads"] = float(scalar_reads_match.group(1))
            if scalar_fills_match:
                dram_metrics["o_scalar_fills"] = float(scalar_fills_match.group(1))
            if scalar_updates_match:
                dram_metrics["o_scalar_updates"] = float(scalar_updates_match.group(1))
    
    return dram_metrics

def add_vector_unit(pe_x: int):
    # for now assume vector units use the same glb as the 2d array
    # i.e. 2d array idle when executing softmax on vector unit
    from global_parameter import layer_factors
    # area, power
    
    # 2 8bit 39.41 um^2
    # 0.083 per 2 words
    total_area = 0
    total_leak = 0
    for unit in layer_factors:
        total_area += pe_x * layer_factors[unit]["area"]
        total_leak += pe_x * layer_factors[unit]["leak"]*1e-12
        #print(pe_x,layer_factors[unit]["leak"]*1e-12)
    total_area += 16*39.41/2 * pe_x #um2
    total_leak += 1.2646724994722446e-06/2*16 * pe_x #W
    return total_area, total_leak


def parse_stats_file(stats_file: str) -> Optional[MappingResult]:
    """Parse the timeloop-mapper.stats.txt file and corresponding ERT summary."""
    try:
        # Read file
        with open(stats_file, 'r') as f:
            content = f.read()
        # Extract layer name from path (e.g., "layer0_q_proj" → "q_proj", "expert_down_proj" → "expert_down_proj")
        match = re.search(r"layer\d+_([^/\\]+)", stats_file)
        if match:
            layer_name = match.group(1)
        else:
            # For ops without layerN_ prefix (e.g., expert_gate_proj, router, lm_head)
            path_parts = stats_file.replace("\\", "/").split("/")
            # Find the operator directory name (should be after "outputs/{net}/")
            layer_name = ""
            for i, p in enumerate(path_parts):
                if p == "outputs" and i + 2 < len(path_parts):
                    layer_name = path_parts[i + 2]
                    break

        from global_parameter import layer_factors
        # MAC per-instance-cycle leakage (pJ/cycle/instance)
        # Derived from Accelergy ERT: total_energy - (compute_energy * computes) / (instances * cycles)
        # BF16 MAC at 14nm: 0.00334 pJ/cycle (larger than INT8's 0.00144 due to wider datapath)
        mac_leak = 0.00334

        # Get basic stats from Summary section
        # DOTALL: change dot behavior in re The dot . matches any character including newline (\n)
        # This allows the pattern to match across multiple lines
        summary_section = re.search(r"Summary Stats\s*[-]+\s*(.*?)(?:Computes|$)", content, re.DOTALL)

        if summary_section:
            summary_text = summary_section.group(1)
            cycles_match = re.search(r"Cycles:\s*(\d+)", summary_text)
            util_match = re.search(r"Utilization:\s*([\d.]+)%", summary_text)
        else:
            cycles_match = None
            util_match = None

        cycles = int(cycles_match.group(1)) if cycles_match else 0
        utilization = float(util_match.group(1)) if util_match else 0

        # Initialize accumulators
        area = 0
        total_dynamic_energy = 0
        total_static_energy = 0

        # Process MAC section
        mac_section = re.search(r"=== mac ===.*?(?===|\Z)", content, re.DOTALL)
        if mac_section:
            mac_text = mac_section.group(0)

            # Get MAC energy and area
            mac_energy_match = re.search(r"Energy \(total\)\s*:\s*([\d.]+)\s*pJ", mac_text)
            mac_area_match = re.search(r"Area(?:\s\(total\))?\s*:\s*([\d.]+)\s*um\^2", mac_text)

            if mac_energy_match:
                mac_total_energy = float(mac_energy_match.group(1))

                mac_static_energy = mac_leak * cycles  # Calculate static energy using ERT leak value
                mac_dynamic_energy = mac_total_energy - mac_static_energy
                # scale dynamic energy if softmax
                if layer_name in layer_factors:
                    mac_dynamic_energy *= layer_factors[layer_name]["compute"]
                total_dynamic_energy += mac_dynamic_energy
                total_static_energy += mac_static_energy

            if mac_area_match:
                mac_total_area = float(mac_area_match.group(1))
                if layer_name in layer_factors:
                    mac_total_area *= layer_factors[layer_name]["area"]
                area += mac_total_area

        # Process all memory buffer sections
        buffer_sections = re.finditer(r"===\s+(\w+)\s+===\s+SPECS\s+-----.*?(?=(?:===|$))", content, re.DOTALL)
        for section in buffer_sections:
            section_text = section.group(0)
            component_name = section.group(1)

            # Skip MAC and DRAM sections
            if component_name in ['mac']:
                continue

            # Get buffer energy and area
            # dynamic energy is split into multiple sections
            energy_matches = re.finditer(r"Energy \(total\)\s*:\s*([\d.]+)\s*pJ", section_text)
            static_match = re.search(r"Leakage energy \(total\)\s*:\s*([\d.]+)\s*pJ", section_text)
            # Check if area has "total" in it
            area_total_match = re.search(r"Area\s*\(total\)\s*:\s*([\d.]+)\s*um\^2", section_text)
            area_per_instance_match = re.search(r"Area\s*:\s*([\d.]+)\s*um\^2", section_text)
            instances_match = re.search(r"Instances\s*:\s*(\d+)", section_text)
            
            # Add all dynamic energy values found
            for energy_match in energy_matches:
                total_dynamic_energy += float(energy_match.group(1))

            if static_match:
                total_static_energy += float(static_match.group(1))

            # Handle area calculation
            if area_total_match:
                # If "Area (total)" is found, use it directly
                area += float(area_total_match.group(1))
            elif area_per_instance_match:
                # If only "Area" is found (per-instance), multiply by instances
                instances_match = re.search(r"Instances\s*:\s*(\d+)", section_text)
                if instances_match:
                    per_instance_area = float(area_per_instance_match.group(1))
                    num_instances = int(instances_match.group(1))
                    total_area = per_instance_area * num_instances
                    area += total_area

        # Get energy breakdown (in fJ/compute)
        energy_breakdown = {}
        breakdown_section = re.search(r"fJ/Compute\s*(.*?)$", content, re.DOTALL)
        if breakdown_section:
            for line in breakdown_section.group(1).split('\n'):
                if '=' in line and 'Total' not in line:
                    component, energy = line.split('=')
                    energy_breakdown[component.strip()] = float(energy.strip())

        # Extract DRAM metrics
        dram_metrics = extract_dram_metrics(content)

        return MappingResult(
            static_energy=total_static_energy,  # in pJ
            dynamic_energy=total_dynamic_energy,  # in pJ
            area=area,  # in um^2
            cycles=cycles,
            utilization=utilization,  # in percentage
            energy_breakdown=energy_breakdown,  # in fJ/compute
            dram_metrics=dram_metrics  # Various metrics
        )

    except Exception as e:
        print(f"Error parsing stats file {stats_file}: {e}")
        return None

def gen_multi_fusion(res_dict: dict) -> List:
        # default_res = {
        #     "net": net,
        #     "layer_name": utility_functions.get_layer_name_from_yaml(layer),
        #     "batch_size": batch_size,
        #     "sequence_length": sequence_length,
        #     "mapper_idx": mapper_idx,
        #     "fused_layer_type": fused_layer_type,
        #     "output_tile_channels": output_tile_channels,
        #     "tp_degree": tp_degree,
        #     "arch_target": arch_target,
        #     "glb_scale": glb_scale,
        #     "pe_x_scale": pe_x_scale,
        #     "pe_y_scale": pe_y_scale,
        #     "dram_i": dram_i,
        #     "dram_o": dram_o,

        #     "cycles": float('inf'),
        #     "latency": float('inf'),
        #     "static_power": float('inf'),
        #     "dynamic_energy": float('inf'),
        #     "area": float('inf'),
        #     "utilization": 0,
            
        #     "i_utilized_capacity": 0,
        #     "i_scalar_reads": 0,
        #     "i_scalar_fills": 0,
        #     "i_scalar_updates": 0,

        #     "w_utilized_capacity": 0,
        #     "w_scalar_reads": 0,
        #     "w_scalar_fills": 0,
        #     "w_scalar_updates": 0,

        #     "o_utilized_capacity": 0,
        #     "o_scalar_reads": 0,
        #     "o_scalar_fills": 0,
        #     "o_scalar_updates": 0,

        #     "i_throttling": 1,
        #     "o_throttling": 1
        # }
    
    
    existing_fusion_type = res_dict["fused_layer_type"]
    
    if existing_fusion_type !='single':
        # softmax operators
        # force fusion
        return []

    
    i_total_dynamic_e = dram_type_bandwidth_width_dict[res_dict['dram_i']]['timeloop_e']*1e-12*word_size*(res_dict["i_scalar_reads"]+res_dict["i_scalar_fills"]+res_dict["i_scalar_updates"])
    o_total_dynamic_e = dram_type_bandwidth_width_dict[res_dict['dram_o']]['timeloop_e']*1e-12*word_size*(res_dict["o_scalar_reads"]+res_dict["o_scalar_fills"]+res_dict["o_scalar_updates"])
    #print(i_total_dynamic_e)
    # weight energy included

    bare_dynamic_energy = res_dict["dynamic_energy"] \
        - i_total_dynamic_e \
        - o_total_dynamic_e
    assert bare_dynamic_energy > 0, print(res_dict, res_dict['dram_i'],res_dict['dram_o'],res_dict["dynamic_energy"],i_total_dynamic_e,o_total_dynamic_e)    
    
    # start
    start_res = res_dict.copy()
    start_res["fused_layer_type"] = "start"
    start_res["dynamic_energy"] = bare_dynamic_energy + i_total_dynamic_e
    start_res["o_scalar_reads"] = 0
    start_res["o_scalar_fills"] = 0
    start_res["o_scalar_updates"] = 0
    # middle
    middle_res = res_dict.copy()
    middle_res["fused_layer_type"] = "middle"
    middle_res["dynamic_energy"] = bare_dynamic_energy 
    middle_res["i_scalar_reads"] = 0
    middle_res["i_scalar_fills"] = 0
    middle_res["i_scalar_updates"] = 0
    middle_res["o_scalar_reads"] = 0
    middle_res["o_scalar_fills"] = 0
    middle_res["o_scalar_updates"] = 0
    # end
    end_res = res_dict.copy()
    end_res["fused_layer_type"] = "end"
    end_res["dynamic_energy"] = bare_dynamic_energy + o_total_dynamic_e
    end_res["i_scalar_reads"] = 0
    end_res["i_scalar_fills"] = 0
    end_res["i_scalar_updates"] = 0
    
    # TODO calculate latency for non existing fusion type 
    # TODO update parsing to get cycle numbers for every level
    # if one fusion type replace max cycle, update cycle number with the next value
    # for now use pessimistic values (assume worst case latency)
    return [start_res,middle_res,end_res]
    

def process_mapping_results(config_id: str, problem_id: str, output_dir: Optional[str]=None, cycle_time:float=cycle_time) -> Dict:
    """
    Process mapping results for a specific configuration.
    
    Args:
        config_id: The configuration ID in the format 
        problem_id: The problem ID in the format "net_layer_tile_width_mapper_idx_fused_layer_type"
        output_dir: Base directory for outputs
    
    Returns:
        Dictionary with parsed results (using inf values if timeloop failed)
    """

    # Determine base directory if not provided
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    try:
        #config_id = f"{arch}@glb{glb_scale}@pe_x_scale{pe_x_scale}@pe_y_scale{pe_y_scale}@{dram_config_to_id(dram_config)}"
        
        # Parse config_id
        parts = config_id.split('@')
        arch_target = parts[0]
        glb_scale = int(parts[1].replace('glb', ''))
        pe_x_scale = int(parts[2].replace('pe_x_scale', ''))
        pe_y_scale = int(parts[3].replace('pe_y_scale', ''))
        dram_i = parts[4]
        dram_o = parts[5]
        #problem_id = f"{net}@{layer_name}@{batch_size}@{sequence_length}@{mapper_idx}@{fused_layer_type}@{output_tile_channels}@{tp_degree}"
        #print(problem_id)
        
        # Parse problem_id
        problem_parts = problem_id.split('@')
        net = problem_parts[0]
        layer = problem_parts[1]
        batch_size = int(problem_parts[2])
        sequence_length = int(problem_parts[3])
        mapper_idx = int(problem_parts[4])
        fused_layer_type = problem_parts[5] if len(problem_parts) > 4 else ""
        output_tile_channels = int(problem_parts[6])
        tp_degree = int(problem_parts[7])
        #print(fused_layer_type)
        default_res = {
            "net": net,
            "layer_name": utility_functions.get_layer_name_from_yaml(layer),
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "mapper_idx": mapper_idx,
            "fused_layer_type": fused_layer_type,
            "output_tile_channels": output_tile_channels,
            "tp_degree": tp_degree,

            "arch_target": arch_target,
            "glb_scale": glb_scale,
            "pe_x_scale": pe_x_scale,
            "pe_y_scale": pe_y_scale,
            "dram_i": dram_i,
            "dram_o": dram_o,

            "cycles": float('inf'),
            "latency": float('inf'),
            "static_power": float('inf'),
            "dynamic_energy": float('inf'),
            "area": float('inf'),
            "utilization": 0,
            
            "i_utilized_capacity": 0,
            "i_scalar_reads": 0,
            "i_scalar_fills": 0,
            "i_scalar_updates": 0,

            "w_utilized_capacity": 0,
            "w_scalar_reads": 0,
            "w_scalar_fills": 0,
            "w_scalar_updates": 0,

            "o_utilized_capacity": 0,
            "o_scalar_reads": 0,
            "o_scalar_fills": 0,
            "o_scalar_updates": 0,

            "i_throttling": 1,
            "o_throttling": 1
        }
        
        # Construct file path
        file_path = os.path.join(
            output_dir,
            net,
            utility_functions.get_layer_name_from_yaml(layer),
            str(batch_size),
            str(sequence_length),
            str(mapper_idx),
            fused_layer_type,
            str(output_tile_channels),
            str(tp_degree),
            f"arch={arch_target}@glb_scale={glb_scale}@pe_x_scale={pe_x_scale}@pe_y_scale={pe_y_scale}",
            f"{dram_i}@{dram_o}"
        )
        # print(fused_layer_type)
        # print(file_path)
        stats_file = os.path.join(file_path, "timeloop-mapper.stats.txt")
        if os.path.exists(stats_file):
            if os.stat(stats_file).st_size > 0:
                result = parse_stats_file(stats_file)
                # print(result)
                if result:
                    latency, static_power, dynamic_energy = result.get_metrics(cycle_time)
                    #print(static_power)
                    # print(pe_x_base_size,pe_x_scale)
                    # print(int(pe_x_base_size*pe_x_scale))
                    # Vector unit area/leakage is added at chiplet level in compute_area.py
                    # and cal_perf_phy_net.py, not per-layer (to avoid double-counting)
                    vector_area = 0
                    vector_leak = 0
                    assert vector_leak < 1e6, f"{vector_leak}"
                    static_power += vector_leak
                    result.area += vector_area
                    #print(static_power)
                    dram_res = result.get_dram_metrics()

                    return {
                        "net": net,
                        "layer_name": utility_functions.get_layer_name_from_yaml(layer),
                        "batch_size": batch_size,
                        "sequence_length": sequence_length,
                        "mapper_idx": mapper_idx,
                        "fused_layer_type": fused_layer_type,
                        "output_tile_channels": output_tile_channels,
                        "tp_degree": tp_degree,

                        "arch_target": arch_target,
                        "glb_scale": glb_scale,
                        "pe_x_scale": pe_x_scale,
                        "pe_y_scale": pe_y_scale,
                        "dram_i": dram_i,
                        "dram_o": dram_o,

                        "cycles": result.cycles,
                        "latency": latency,
                        "static_power": static_power,
                        "dynamic_energy": dynamic_energy,
                        "area": result.area,
                        "utilization": result.utilization,
                        
                        "i_utilized_capacity": dram_res["i_utilized_capacity"],
                        "i_scalar_reads": dram_res["i_scalar_reads"],
                        "i_scalar_fills": dram_res["i_scalar_fills"],
                        "i_scalar_updates": dram_res["i_scalar_updates"],

                        "w_utilized_capacity": dram_res["w_utilized_capacity"],
                        "w_scalar_reads": dram_res["w_scalar_reads"],
                        "w_scalar_fills": dram_res["w_scalar_fills"],
                        "w_scalar_updates": dram_res["w_scalar_updates"],

                        "o_utilized_capacity": dram_res["o_utilized_capacity"],
                        "o_scalar_reads": dram_res["o_scalar_reads"],
                        "o_scalar_fills":  dram_res["o_scalar_fills"],
                        "o_scalar_updates":  dram_res["o_scalar_updates"],

                        "i_throttling": dram_res["i_throttling"],
                        "o_throttling": dram_res["o_throttling"]
                    }

        
        # If we get here, timeloop ran but we couldn't find or parse the stats file
        return default_res
    
    except Exception as e:
        print(f"Error processing results: {e}")
        return default_res


def post_process_mapping_results(default_res) -> Dict:
    """
    Process mapping results for a specific configuration.
    
    Args:
        default_res: results in raw format
    
    Returns:
        Dictionary with post-processed results (using inf values if timeloop failed)
    """

    try:
        i_access = default_res["i_scalar_reads"] + default_res["i_scalar_fills"] + default_res["i_scalar_updates"]
        w_access = default_res["w_scalar_reads"] + default_res["w_scalar_fills"] + default_res["w_scalar_updates"]
        o_access =  default_res["o_scalar_reads"] + default_res["o_scalar_fills"] + default_res["o_scalar_updates"]
        #print(i_access+w_access, dram_type_bandwidth_width_dict[default_res["dram_i"]]["timeloop_e"])
        # Timeloop's reported dynamic_energy ALREADY embeds DRAM access energy at
        # timeloop_e pJ/bit (parse_stats sums the === DRAM_I/O === sections into the
        # total). Strip that embedded copy, THEN re-add at final_e (a DRAM-type swap).
        # BUGFIX 2026-06-15: the re-add must build on the stripped `dynamic_energy`
        # temp below, NOT re-read default_res["dynamic_energy"] (which still embeds the
        # DRAM copy) -- doing the latter double-counted one full DRAM-access term.
        dynamic_energy = default_res["dynamic_energy"] - (i_access+w_access) *dram_type_bandwidth_width_dict[default_res["dram_i"]]["timeloop_e"]*1e-12*word_size -\
            o_access * dram_type_bandwidth_width_dict[default_res["dram_o"]]["timeloop_e"]*1e-12*word_size

        assert dynamic_energy > 0, "negative dynamic energy"
        dynamic_energy = dynamic_energy + (i_access+w_access) *dram_type_bandwidth_width_dict[default_res["dram_i"]]["final_e"]*1e-12*word_size +\
            o_access * dram_type_bandwidth_width_dict[default_res["dram_o"]]["final_e"]*1e-12*word_size

        processed_res = {
            "net": default_res["net"],
            "layer_name": default_res["layer_name"],
            "batch_size": default_res["batch_size"],
            "sequence_length": default_res["sequence_length"],
            "mapper_idx":default_res["mapper_idx"],
            "fused_layer_type": default_res["fused_layer_type"],
            # tp_degree is enable to select a row
            "tp_degree": default_res["tp_degree"],

            "arch_target": default_res["arch_target"],
            "glb_scale": default_res["glb_scale"],
            "pe_x_scale": default_res["pe_x_scale"],
            "pe_y_scale": default_res["pe_y_scale"],
            "dram_i": default_res["dram_i"],
            "dram_o": default_res["dram_o"],

            "latency": default_res["latency"],
            "static_power": default_res["static_power"],
            "dynamic_energy": dynamic_energy,
            "area": default_res["area"],
            "utilization": default_res["utilization"],
            

            "i_access": i_access,
            "w_access": w_access,
            "o_access": o_access,

            # "i_throttling": 1,
            # "o_throttling": 1,

            "is_duplicate": default_res["is_duplicate"],
            "master_layer":default_res["master_layer"]
        }
    

        
        # If we get here, timeloop ran but we couldn't find or parse the stats file
        return processed_res
    
    except Exception as e:
        print(f"Error processing results: {e}")
        return default_res

if __name__ == "__main__":
    # test purpose
    res = parse_stats_file("outputs/replknet31b/layer3_stem_2_conv/1/1/1/single/128/1/arch=eyeriss_like@glb_scale=16@pe_x_scale=4@pe_y_scale=4/LPDDR5@LPDDR5/timeloop-mapper.stats.txt")
    print(res)
    res = parse_stats_file("outputs/replknet31b/layer3_stem_2_conv/32/1/1/single/128/1/arch=eyeriss_like@glb_scale=16@pe_x_scale=4@pe_y_scale=4/LPDDR5@LPDDR5/timeloop-mapper.stats.txt")
    print(res)