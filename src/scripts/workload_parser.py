import yaml
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import math
from utility_functions import is_softmax_layers

class AcceleratorConfig:
    """Configuration for accelerator hardware specifications."""
    
    def __init__(self, 
                 pe_array_size: int = 16,
                 vector_array_size: int = 16, 
                 pe_frequency_ghz: float = 1.0,
                 dram_bandwidth_gbps: float = 100.0,
                 word_size: int = 8):
        """
        Initialize accelerator configuration.
        
        Args:
            pe_array_size: Size of PE array (X*X matrix)
            vector_array_size: Size of 1D vector array (for softmax operations)
            pe_frequency_ghz: Operating frequency in GHz
            dram_bandwidth_gbps: DRAM bandwidth in GB/s
            word_size: Bits per word
        """
        self.pe_array_size = pe_array_size
        self.vector_array_size = vector_array_size
        self.pe_frequency_ghz = pe_frequency_ghz
        self.dram_bandwidth_gbps = dram_bandwidth_gbps
        self.word_size = word_size
        
        # Compute capabilities
        self.pe_array_ops_per_sec = pe_array_size * pe_array_size * pe_frequency_ghz * 1e9  # MACs/sec, 与 FLOPs 有 2×关系
        self.vector_ops_per_sec = vector_array_size * pe_frequency_ghz * 1e9  # MACs/sec
        
        # Memory bandwidth in bits/sec
        self.dram_bandwidth_bps = dram_bandwidth_gbps * 8 * 1e9
        
    def __str__(self):
        return f"PE Array: {self.pe_array_size}*{self.pe_array_size}, Vector: {self.vector_array_size}, Freq: {self.pe_frequency_ghz}GHz, DRAM: {self.dram_bandwidth_gbps}GB/s"


class RooflineAnalysis:
    """Roofline model analysis for compute vs memory bound determination."""
    
    def __init__(self, accelerator_config: AcceleratorConfig):
        self.config = accelerator_config
        
    def analyze_layer(self, macs: int, memory_bits: int, layer_name: str = "", 
                     layer_type: str = "") -> Dict[str, Any]:
        """
        Perform roofline analysis on a layer.
        
        Args:
            macs: Number of MAC operations
            memory_bits: Memory access in bits
            layer_name: Name of the layer
            layer_type: Type of layer (CNN/Transformer)
            
        Returns:
            Dictionary with roofline analysis results
        """        
        # Calculate operational intensity (OPS per byte) - this should always use actual operations
        memory_bytes = memory_bits // 8
        
        if is_softmax_layers(layer_name):
            # Use vector array for softmax operations
            peak_compute_ops_per_sec = self.config.vector_ops_per_sec
            compute_unit = f"Vector Array ({self.config.vector_array_size})"
            # For softmax, operations are element-wise operations (approximate as memory elements)
            compute_ops = macs
            operational_intensity = compute_ops / memory_bytes if memory_bytes > 0 else 0
        else:
            # Use PE array for matrix operations (MACs)
            peak_compute_ops_per_sec = self.config.pe_array_ops_per_sec
            compute_unit = f"PE Array ({self.config.pe_array_size}*{self.config.pe_array_size})"
            compute_ops = macs
            # Operational intensity for matrix ops is MACs per byte
            operational_intensity = compute_ops / memory_bytes if memory_bytes > 0 else 0
        
        # Calculate peak memory bandwidth utilization
        peak_memory_bytes_per_sec = self.config.dram_bandwidth_bps // 8 # 1 byte = 8 bits
        
        # Calculate roofline intersection point
        # Intersection: peak_compute = peak_memory * intensity_intersection
        # Unit should be OPS/Byte
        intensity_intersection = peak_compute_ops_per_sec / peak_memory_bytes_per_sec
        
        # Determine bottleneck
        if operational_intensity < intensity_intersection:
            bottleneck = "Memory Bound"
            limited_by = "DRAM Bandwidth"
            achievable_ops_per_sec = operational_intensity * peak_memory_bytes_per_sec
        else:
            bottleneck = "Compute Bound" 
            limited_by = compute_unit
            achievable_ops_per_sec = peak_compute_ops_per_sec
        
        # Calculate execution time
        execution_time_sec = compute_ops / achievable_ops_per_sec
        
        # Calculate utilization
        compute_utilization = min(1.0, achievable_ops_per_sec / peak_compute_ops_per_sec)
        memory_utilization = min(1.0, (memory_bits / execution_time_sec) / self.config.dram_bandwidth_bps)
        
        return {
            'layer_name': layer_name,
            'layer_type': layer_type,
            'compute_ops': compute_ops,
            'memory_bytes': memory_bytes,
            'operational_intensity': operational_intensity,
            'intensity_intersection': intensity_intersection,
            'bottleneck': bottleneck,
            'limited_by': limited_by,
            'compute_unit': compute_unit,
            'execution_time_sec': execution_time_sec,
            'execution_time_ms': execution_time_sec * 1000,
            'compute_utilization': compute_utilization,
            'memory_utilization': memory_utilization,
            'achievable_ops_per_sec': achievable_ops_per_sec,
            'peak_compute_ops_per_sec': peak_compute_ops_per_sec,
            'peak_memory_bytes_per_sec': peak_memory_bytes_per_sec
        }

class WorkloadParser:
    """Parse YAML workload files and compute MAC operations and memory access."""
    
    def __init__(self, word_size: int = 8, batch_size: Optional[int] = None):
        """
        Initialize parser with precision and batch settings.
        
        Args:
            word_size: Bits per word for precision calculations (default: 8)
            batch_size: Batch size to use for calculations. If None, uses YAML file's batch size (default: None)
        """
        self.word_size = word_size
        self.batch_size = batch_size
    
    def parse_cnn_layer(self, yaml_data: Dict) -> Tuple[int, int]:
        """
        Parse CNN layer (convolution or fully connected) and compute MACs and memory access.
        
        CNN format uses dimensions: G, C, M, N, P, Q, R, S
        For grouped convolutions:
        - G: Number of groups
        - C: Input channels PER GROUP  
        - M: Output channels PER GROUP
        - N: Batch size
        - P, Q: Output spatial dimensions
        - R, S: Filter spatial dimensions
        
        Total input channels = G × C
        Total output channels = G × M
        
        Returns:
            Tuple of (MACs, memory_access_bits)
        """
        instance = yaml_data['problem']['instance']
        
        # Extract dimensions
        G = instance.get('G', 1)  # Groups
        C = instance.get('C', 1)  # Input channels PER GROUP
        M = instance.get('M', 1)  # Output channels PER GROUP  
        N = self.batch_size if self.batch_size is not None else instance.get('N', 1)  # Batch size (user override)
        P = instance.get('P', 1)  # Output height
        Q = instance.get('Q', 1)  # Output width
        R = instance.get('R', 1)  # Filter height
        S = instance.get('S', 1)  # Filter width
        
        # Calculate total channels
        total_input_channels = G * C
        total_output_channels = G * M
        
        # Calculate MACs for grouped convolution: N * P * Q * G * C * M * R * S
        # Each group processes C input channels to produce M output channels
        macs = N * P * Q * G * C * M * R * S
        
        # Calculate memory access in bits
        # Inputs: N * total_input_channels * P * Q (approximate input spatial size)
        # Weights: G * C * M * R * S (per-group weights across all groups)
        # Outputs: N * total_output_channels * P * Q
        
        # For input size calculation, we need to account for the spatial extent of the convolution
        # Approximate input spatial size considering stride and padding
        input_height = P * 2 if R > 1 else P  # Rough approximation for stride effect
        input_width = Q * 2 if S > 1 else Q
        
        input_elements = N * total_input_channels * input_height * input_width
        weight_elements = G * C * M * R * S  
        output_elements = N * total_output_channels * P * Q
        
        total_memory_access = (input_elements + weight_elements + output_elements) * self.word_size
        
        return macs, total_memory_access
    
    def parse_transformer_layer(self, yaml_data: Dict) -> Tuple[int, int]:
        """
        Parse transformer layer and compute MACs and memory access.
        
        Transformer formats:
        1. Projection (Q/K/V, FFN): B, P, O, D/I
        2. Attention (QK): B, E, H, M, P  
        3. Attention (AV): B, F, H, M, P
        4. Attention (Softmax): B, H, M, P
        Returns:
            Tuple of (MACs, memory_access_bits)
        """
        instance = yaml_data['problem']['instance']
        shape_name = yaml_data['problem']['shape']['name']
        
        if 'Projection' in shape_name or 'FFN' in shape_name:
            # Matrix multiplication: (B, P, O) x (O, D) -> (B, P, D)
            B = self.batch_size if self.batch_size is not None else instance.get('B', 1)  # Batch size (user override)
            P = instance.get('P', 1)  
            O = instance.get('O', 1)
            D = instance.get('D', instance.get('I', 1))  # D for projection, I for FFN
            
            macs = B * P * O * D
            
            input_elements = B * P * O
            weight_elements = O * D
            output_elements = B * P * D
            
        elif shape_name == 'QK':
            # Attention QK: (B,H,P,E) x (B,H,M,E)^T -> (B,H,P,M)
            B = self.batch_size if self.batch_size is not None else instance.get('B', 1)
            E = instance.get('E', 1)
            H = instance.get('H', 1)
            M = instance.get('M', 1)
            P = instance.get('P', 1)
            
            macs = B * H * P * M * E
            
            # Q, K, and Attention Scores
            input_elements = B * H * P * E      # Q
            weight_elements = B * H * M * E     # K
            output_elements = B * H * P * M     # Attention Scores
            
        elif shape_name == 'AV':
            # Attention AV: (B,H,P,M) x (B,H,M,F) -> (B,H,P,F)
            B = self.batch_size if self.batch_size is not None else instance.get('B', 1)
            F = instance.get('F', 1)
            H = instance.get('H', 1)
            M = instance.get('M', 1)
            P = instance.get('P', 1)

            macs = B * H * P * M * F

            # Attention Scores, V, and Output
            input_elements = B * H * P * M      # Attention Scores (input)
            weight_elements = B * H * M * F     # V (weights)
            output_elements = B * H * P * F     # Output
            
        elif shape_name in ['M', 'SN', 'SD', 'A']:
            # Handle element-wise operations from Softmax decomposition
            B = self.batch_size if self.batch_size is not None else instance.get('B', 1)
            H = instance.get('H', 1)
            M = instance.get('M', 1)
            P = instance.get('P', 1)
            
            elements = B * H * P * M
            
            if shape_name == 'M' or shape_name == 'SD': # Max or Sum
                macs = elements
                input_elements = elements
                weight_elements = 0
                output_elements = B * H * P # Output is reduced

            elif shape_name == 'SN': # Subtract and Exp
                macs = int(elements) * 2 # TODO: add real constant
                input_elements = elements
                weight_elements = B * H * P # Reads the max values
                output_elements = elements

            elif shape_name == 'A': # Attention score final division
                macs = int(elements) # TODO: add real constant
                input_elements = elements
                weight_elements = B * H * P # Reads the sum values
                output_elements = elements
        
        total_memory_access = (input_elements + weight_elements + output_elements) * self.word_size
        
        return macs, total_memory_access
    
    def parse_workload_file(self, file_path: str) -> Tuple[int, int, str]:
        """
        Parse single workload YAML file.
        
        Returns:
            Tuple of (MACs, memory_access_bits, layer_type)
        """
        with open(file_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Determine if CNN or transformer based on dimensions
        instance = yaml_data['problem']['instance']
        dims = set(instance.keys())
        
        # CNN layers have G,C,M,R,S dimensions
        cnn_dims = {'G', 'C', 'M', 'R', 'S'}
        # Transformer layers have B,P,O,D or B,E,H dimensions  
        transformer_dims = {'B', 'P', 'O'} 
        transformer_attention_dims = {'B', 'H', 'P'}
        
        if cnn_dims.issubset(dims):
            macs, memory_access = self.parse_cnn_layer(yaml_data)
            layer_type = "CNN"
        elif transformer_dims.issubset(dims) or transformer_attention_dims.issubset(dims):
            macs, memory_access = self.parse_transformer_layer(yaml_data)
            layer_type = "Transformer"
        else:
            raise NotImplementedError("Layer type not supported")
        return macs, memory_access, layer_type
    
    def analyze_network(self, network_path: str) -> Dict[str, Any]:
        """
        Analyze all layers in a network directory.
        
        Args:
            network_path: Path to network directory containing YAML files
            
        Returns:
            Dictionary with analysis results
        """
        network_path_obj = Path(network_path)
        if not network_path_obj.exists():
            raise FileNotFoundError(f"Network path {network_path} not found")
        
        results = {
            'network_name': network_path_obj.name,
            'layers': {},
            'total_macs': 0,
            'total_memory_access_bits': 0,
            'layer_count': 0
        }
        
        # Process all YAML files in directory
        yaml_files = sorted(network_path_obj.glob('*.yaml'))
        
        for yaml_file in yaml_files:
            try:
                macs, memory_access, layer_type = self.parse_workload_file(str(yaml_file))
                
                layer_name = yaml_file.stem
                results['layers'][layer_name] = {
                    'macs': macs,
                    'memory_access_bits': memory_access,
                    'layer_type': layer_type,
                    'compute_memory_ratio': macs / (memory_access / 8) if memory_access > 0 else 0
                }
                
                results['total_macs'] += macs
                results['total_memory_access_bits'] += memory_access
                results['layer_count'] += 1
                
            except Exception as e:
                print(f"Error processing {yaml_file}: {e}")
                continue
        
        # Calculate overall compute-to-memory ratio
        if results['total_memory_access_bits'] > 0:
            results['compute_memory_ratio'] = results['total_macs'] / (results['total_memory_access_bits'] / self.word_size)
        else:
            results['compute_memory_ratio'] = 0
            
        return results


def print_layer_analysis(results: Dict[str, Dict[str, Any]], show_layers: bool = True):
    """
    Print detailed layer-level analysis results.
    
    Args:
        results: Analysis results from analyze_workloads
        show_layers: Whether to show individual layer details
    """
    print("\n" + "="*80)
    print("DETAILED LAYER-LEVEL ANALYSIS")
    print("="*80)
    
    for network_name, data in results.items():
        print(f"\n{'='*20} {network_name.upper()} {'='*20}")
        print(f"Total Layers: {data['layer_count']}")
        print(f"Total MACs: {data['total_macs']:,}")
        print(f"Total Memory Access: {data['total_memory_access_bits']:,} bits")
        print(f"Overall Compute/Memory Ratio: {data['compute_memory_ratio']:.2f}")
        
        if show_layers and data['layers']:
            print(f"\nLayer-by-Layer Breakdown:")
            print(f"{'Layer Name':<25} {'Type':<12} {'MACs':<15} {'Memory (bits)':<15} {'C/M Ratio':<10}")
            print("-" * 85)
            
            # Sort layers by name for consistent output
            sorted_layers = sorted(data['layers'].items())
            
            for layer_name, layer_data in sorted_layers:
                macs = layer_data['macs']
                memory_bits = layer_data['memory_access_bits']
                layer_type = layer_data['layer_type']
                ratio = layer_data['compute_memory_ratio']
                
                print(f"{layer_name:<25} {layer_type:<12} {macs:<15,} {memory_bits:<15,} {ratio:<10.2f}")
        
        print()


def analyze_workloads(workload_dir: str = "workloads", word_size: int = 16, 
                     networks: Optional[List[str]] = None, verbose: bool = True, 
                     batch_size: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """
    Analyze multiple networks and return comprehensive results.
    
    Args:
        workload_dir: Directory containing network subdirectories
        word_size: Bits per word for precision calculations  
        networks: List of network names to analyze. If None, analyze all networks.
        verbose: Whether to print progress messages
        batch_size: Batch size to use for calculations. If None, uses YAML file's batch size
        
    Returns:
        Dictionary with results for each network
    """
    parser = WorkloadParser(word_size=word_size, batch_size=batch_size)
    workload_path = Path(workload_dir)
    
    if networks is None:
        # Find all network directories
        networks = [d.name for d in workload_path.iterdir() if d.is_dir()]
    
    results = {}
    
    for network_name in networks:
        network_path = workload_path / network_name
        if network_path.exists():
            try:
                results[network_name] = parser.analyze_network(str(network_path))
                if verbose:
                    batch_info = f" (batch_size={batch_size})" if batch_size is not None else ""
                    print(f"Analyzed {network_name}{batch_info}: {results[network_name]['layer_count']} layers, "
                          f"{results[network_name]['total_macs']:,} MACs, "
                          f"{results[network_name]['total_memory_access_bits']:,} memory bits")
            except Exception as e:
                print(f"Error analyzing {network_name}: {e}")
        else:
            print(f"Network directory {network_path} not found")
    
    return results


def analyze_individual_layers(layer_files: List[str], word_size: int = 8, 
                             batch_size: Optional[int] = None) -> None:
    """
    Analyze individual layer files and display their MAC/memory access.
    
    Args:
        layer_files: List of paths to individual YAML layer files
        word_size: Bits per word for precision calculations
        batch_size: Batch size override (if None, uses YAML batch size)
    """
    parser = WorkloadParser(word_size=word_size, batch_size=batch_size)
    
    print(f"{'Layer File':<40} {'Type':<12} {'MACs':<15} {'Memory (bits)':<15} {'C/M Ratio':<10}")
    print("-" * 100)
    
    for layer_file in layer_files:
        try:
            layer_path = Path(layer_file)
            macs, memory_bits, layer_type = parser.parse_workload_file(layer_file)
            ratio = macs / (memory_bits / word_size) if memory_bits > 0 else 0
            
            print(f"{layer_path.name:<40} {layer_type:<12} {macs:<15,} {memory_bits:<15,} {ratio:<10.2f}")
            
        except Exception as e:
            print(f"{layer_file:<40} ERROR: {e}")


def analyze_network_with_roofline(network_path: str,
                                  accelerator_config: AcceleratorConfig,
                                  batch_size: Optional[int] = None) -> None:
    """
    Analyze all layers in a network with roofline model.
    
    Args:
        network_path: Path to network directory containing YAML files
        accelerator_config: Hardware accelerator configuration
        batch_size: Batch size override (if None, uses YAML batch size)
    """
    network_path_obj = Path(network_path)
    network_name = network_path_obj.name
    
    parser = WorkloadParser(word_size=accelerator_config.word_size, batch_size=batch_size)
    roofline = RooflineAnalysis(accelerator_config)
    
    # Get all layer files
    layer_files = sorted(network_path_obj.glob('*.yaml'))
    
    batch_info = f" (batch_size={batch_size})" if batch_size is not None else ""
    print(f"\n{'='*15} {network_name.upper()}{batch_info} {'='*15}")
    print(f"Layers: {len(layer_files)}")
    
    print(f"\n{'Layer':<35} {'Type':<12} {'Bottleneck':<12} {'Op.Int.':<8} {'Exec(ms)':<10} {'Compute%':<9} {'Memory%':<8}")
    print("-" * 110)
    
    total_exec_time = 0
    for layer_file in layer_files:
        try:
            macs, memory_bits, layer_type = parser.parse_workload_file(str(layer_file))
            
            analysis = roofline.analyze_layer(
                macs=macs,
                memory_bits=memory_bits, 
                layer_name=layer_file.stem,
                layer_type=layer_type
            )
            
            print(f"{layer_file.stem:<35} {layer_type:<12} {analysis['bottleneck']:<12} "
                  f"{analysis['operational_intensity']:<8.2f} {analysis['execution_time_ms']:<10.3f} "
                  f"{analysis['compute_utilization']*100:<8.1f}% {analysis['memory_utilization']*100:<7.1f}%")
            
            total_exec_time += analysis['execution_time_ms']
            
        except Exception as e:
            print(f"{layer_file.stem:<35} ERROR: {e}")
    
    print(f"\nNetwork Total Execution Time: {total_exec_time:.3f} ms")


def analyze_layers_with_roofline(layer_files: List[str], 
                                accelerator_config: AcceleratorConfig,
                                batch_size: Optional[int] = None) -> None:
    """
    Analyze specific layers with roofline model to determine compute vs memory bound.
    
    Args:
        layer_files: List of paths to individual YAML layer files
        accelerator_config: Hardware accelerator configuration
        batch_size: Batch size override (if None, uses YAML batch size)
    """
    parser = WorkloadParser(word_size=accelerator_config.word_size, batch_size=batch_size)
    roofline = RooflineAnalysis(accelerator_config)
    
    print(f"\n{'Layer':<35} {'Type':<12} {'Bottleneck':<12} {'Op.Int.':<8} {'Exec(ms)':<10} {'Compute%':<9} {'Memory%':<8}")
    print("-" * 110)
    
    for layer_file in layer_files:
        try:
            layer_path = Path(layer_file)
            macs, memory_bits, layer_type = parser.parse_workload_file(layer_file)
            
            analysis = roofline.analyze_layer(
                macs=macs,
                memory_bits=memory_bits, 
                layer_name=layer_path.stem,
                layer_type=layer_type
            )
            
            print(f"{layer_path.stem:<35} {layer_type:<12} {analysis['bottleneck']:<12} "
                  f"{analysis['operational_intensity']:<8.2f} {analysis['execution_time_ms']:<10.3f} "
                  f"{analysis['compute_utilization']*100:<8.1f}% {analysis['memory_utilization']*100:<7.1f}%")
            
        except Exception as e:
            print(f"{layer_file:<35} ERROR: {e}")


def print_roofline_header(accelerator_config: AcceleratorConfig) -> None:
    """Print roofline analysis header with key metrics."""
    print(f"\nRoofline Analysis - {accelerator_config}")
    print(f"Peak Compute: {accelerator_config.pe_array_ops_per_sec/1e9:.1f} GOPS (PE Array), {accelerator_config.vector_ops_per_sec/1e9:.1f} GOPS (Vector)")
    print(f"Peak Memory: {accelerator_config.dram_bandwidth_bps/8/1e9:.1f} GB/s")
    print(f"Roofline Intersection: {accelerator_config.pe_array_ops_per_sec/(accelerator_config.dram_bandwidth_bps/accelerator_config.word_size):.2f} OPS/Byte")
    print(f"\nMetrics Explanation:")
    print(f"- Compute%: Percentage of peak compute capability utilized")
    print(f"- Memory%: Percentage of peak memory bandwidth utilized")
    print(f"- Op.Int.: Operational Intensity (Operations per Byte)")
    print(f"- Bottleneck: 'Compute Bound' if limited by compute, 'Memory Bound' if limited by memory bandwidth")


def analyze_layers_by_pattern(workload_dir: str = "workloads", 
                             network_pattern: str = "*",
                             layer_pattern: str = "*.yaml",
                             word_size: int = 8,
                             batch_size: Optional[int] = None) -> None:
    """
    Analyze layers matching a pattern.
    
    Args:
        workload_dir: Directory containing network subdirectories
        network_pattern: Pattern to match network directories (e.g., "efficientnet*")
        layer_pattern: Pattern to match layer files (e.g., "layer*_conv*.yaml")
        word_size: Bits per word for precision calculations
        batch_size: Batch size override
    """
    workload_path = Path(workload_dir)
    
    # Find matching network directories
    network_dirs = list(workload_path.glob(network_pattern))
    network_dirs = [d for d in network_dirs if d.is_dir()]
    
    if not network_dirs:
        print(f"No network directories found matching pattern: {network_pattern}")
        return
    
    all_layer_files = []
    for network_dir in network_dirs:
        layer_files = list(network_dir.glob(layer_pattern))
        all_layer_files.extend([str(f) for f in layer_files])
    
    if not all_layer_files:
        print(f"No layer files found matching pattern: {layer_pattern}")
        return
    
    print(f"\nFound {len(all_layer_files)} layer files in {len(network_dirs)} networks")
    batch_info = f" (batch_size={batch_size})" if batch_size is not None else ""
    print(f"Analysis settings: word_size={word_size}{batch_info}\n")
    
    analyze_individual_layers(all_layer_files, word_size, batch_size)


if __name__ == "__main__":
    print("=== Network-by-Network Roofline Analysis ===")
    
    # Define accelerator configuration
    config = AcceleratorConfig(pe_array_size=64, vector_array_size=64, pe_frequency_ghz=1.0, dram_bandwidth_gbps=70.4, word_size=8)
    
    # Print roofline configuration header
    print_roofline_header(config)
    
    # Networks to analyze
    networks = [
        "workloads/resnet18",
        "workloads/efficientnet_b0", 
        "workloads/replknet31b",
        "workloads/gpt-1.3B_prefill",
        "workloads/gpt-1.3B_decode"
    ]
    
    print(f"\n{'='*80}")
    print("NETWORK ANALYSIS - BATCH SIZE = 1")
    print(f"{'='*80}")
    
    for network in networks:
        analyze_network_with_roofline(network, config, batch_size=1)
    
    print(f"\n{'='*80}")
    print("BATCH SIZE COMPARISON - GPT-1.3B PREFILL")
    print(f"{'='*80}")
    
    for batch_size in [1, 8, 16]:
        analyze_network_with_roofline("workloads/gpt-1.3B_prefill", config, batch_size=batch_size)
    
    print(f"\n{'='*80}")
    print("ACCELERATOR CONFIGURATION COMPARISON")
    print(f"{'='*80}")
    
    # # Compare different accelerator configurations
    # alt_configs = [
    #     #AcceleratorConfig(pe_array_size=128, vector_array_size=128, pe_frequency_ghz=1.2, dram_bandwidth_gbps=100, word_size=8),
    #     AcceleratorConfig(pe_array_size=256, vector_array_size=256, pe_frequency_ghz=1.0, dram_bandwidth_gbps=320, word_size=8)
    # ]
    
    # for i, alt_config in enumerate(alt_configs):
    #     print(f"\n--- Configuration {i+2}: {alt_config.pe_array_size}×{alt_config.pe_array_size} PE Array ---")
    #     print_roofline_header(alt_config)
        
    #     analyze_network_with_roofline("workloads/resnet18", alt_config, batch_size=1)