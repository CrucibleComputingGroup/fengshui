#!/usr/bin/env python3
"""
Analysis of layer characteristics for EfficientNet, MobileNet, and ResNet50
to understand memory-bound behavior patterns.
"""

import math

class LayerAnalyzer:
    def __init__(self, name, C, M, P, Q, R, S, G=1, stride=1):
        self.name = name
        self.C = C  # Input channels
        self.M = M  # Output channels
        self.P = P  # Output height
        self.Q = Q  # Output width
        self.R = R  # Kernel height
        self.S = S  # Kernel width
        self.G = G  # Groups (for depthwise convolution)
        self.stride = stride
        
        # Calculate derived properties
        self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate key metrics for the layer."""
        # MAC operations
        self.mac_ops = self.C * self.M * self.P * self.Q * self.R * self.S / self.G
        
        # Memory accesses (assuming single precision floats = 4 bytes)
        # Input feature map size (assuming padded input for same output size)
        input_h = (self.P - 1) * self.stride + self.R
        input_w = (self.Q - 1) * self.stride + self.S 
        self.input_memory = self.C * input_h * input_w * 4  # bytes
        
        # Output feature map size
        self.output_memory = self.M * self.P * self.Q * 4  # bytes
        
        # Weight memory
        self.weight_memory = (self.C * self.M * self.R * self.S / self.G) * 4  # bytes
        
        # Total memory
        self.total_memory = self.input_memory + self.output_memory + self.weight_memory
        
        # Compute to memory ratio (operations per byte)
        self.compute_to_memory_ratio = self.mac_ops / self.total_memory if self.total_memory > 0 else 0
        
        # Arithmetic intensity (MACs per byte of memory accessed)
        self.arithmetic_intensity = self.mac_ops / self.total_memory if self.total_memory > 0 else 0
        
        # Channel expansion ratio
        self.channel_expansion = self.M / self.C if self.C > 0 else 0
        
        # Spatial reduction factor
        self.spatial_ops_per_output = self.R * self.S
        
        # Determine layer type
        self.layer_type = self.classify_layer()
    
    def classify_layer(self):
        """Classify the layer type based on its characteristics."""
        if self.G > 1 and self.G == self.C and self.M == self.C:
            return "Depthwise Convolution"
        elif self.R == 1 and self.S == 1:
            return "Pointwise Convolution (1x1)"
        elif self.R == 3 and self.S == 3:
            return "Standard 3x3 Convolution"
        elif self.R == 7 and self.S == 7:
            return "Large Kernel Convolution (7x7)"
        elif self.P == 1 and self.Q == 1:
            return "Global Average Pool / FC-like"
        else:
            return f"Custom Convolution ({self.R}x{self.S})"
    
    def print_analysis(self):
        """Print detailed analysis of the layer."""
        print(f"\n{'='*60}")
        print(f"Layer: {self.name}")
        print(f"Type: {self.layer_type}")
        print(f"{'='*60}")
        print(f"Dimensions: C={self.C}, M={self.M}, P={self.P}, Q={self.Q}, R={self.R}, S={self.S}, G={self.G}")
        print(f"Channel Expansion Ratio: {self.channel_expansion:.2f}")
        print(f"Spatial Ops per Output: {self.spatial_ops_per_output}")
        print(f"\nCompute:")
        print(f"  MAC Operations: {self.mac_ops:,.0f}")
        print(f"\nMemory (bytes):")
        print(f"  Input: {self.input_memory:,.0f}")
        print(f"  Output: {self.output_memory:,.0f}")
        print(f"  Weights: {self.weight_memory:,.0f}")
        print(f"  Total: {self.total_memory:,.0f}")
        print(f"\nRatios:")
        print(f"  Compute-to-Memory Ratio: {self.compute_to_memory_ratio:.3f} MACs/byte")
        print(f"  Arithmetic Intensity: {self.arithmetic_intensity:.3f}")
        
        # Memory-bound analysis
        if self.arithmetic_intensity < 1.0:
            print(f"  -> MEMORY-BOUND (low arithmetic intensity)")
        elif self.arithmetic_intensity > 10.0:
            print(f"  -> COMPUTE-BOUND (high arithmetic intensity)")
        else:
            print(f"  -> BALANCED")

class TransformerBlockAnalyzer:
    def __init__(self, name, N, D, num_heads, mlp_ratio=4, byte_size=4):
        """
        N: Length of the sequence
        D: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio (default 4)
        """
        self.name = name
        self.N = N
        self.D = D
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.byte_size = byte_size
        self.calculate_metrics()

    def calculate_metrics(self):
        # Q, K, V projection: 3 × (N × D × D)
        self.qkv_mac = 3 * self.N * self.D * self.D
        self.qkv_weight = 3 * self.D * self.D * self.byte_size  # float32
        # Attention score: N × N × D (before softmax)
        self.attn_score_mac = self.N * self.N * (self.D + 4)
        # Attention weighted sum: N × N × D
        self.attn_weighted_mac = self.N * self.N * self.D
        # Output projection: N × D × D
        self.proj_mac = self.N * self.D * self.D
        self.proj_weight = self.D * self.D * self.byte_size
        # MLP: two fully connected layers
        self.mlp_fc1_mac = self.N * self.D * (self.D * self.mlp_ratio)
        self.mlp_fc1_weight = self.D * (self.D * self.mlp_ratio) * self.byte_size
        self.mlp_fc2_mac = self.N * (self.D * self.mlp_ratio) * self.D
        self.mlp_fc2_weight = (self.D * self.mlp_ratio) * self.D * self.byte_size
        # Total MACs
        self.total_mac = (
            self.qkv_mac + self.attn_score_mac + self.attn_weighted_mac +
            self.proj_mac + self.mlp_fc1_mac + self.mlp_fc2_mac
        )
        # Total weight memory
        self.total_weight = (
            self.qkv_weight + self.proj_weight + self.mlp_fc1_weight + self.mlp_fc2_weight
        )
        # Input and output memory
        self.input_memory = self.N * self.D * self.byte_size
        self.output_memory = self.N * self.D * self.byte_size
        # ====== Count all intermediate memory accesses ======
        # Q, K, V: write once, read once
        self.qkv_io = 3 * self.N * self.D * self.byte_size * 2
        # QK^T (score matrix): write once, read four times (max, exp, sum, division)
        self.score_io = self.N * self.N * self.byte_size * 5
        # softmax output: write once, read once (for weighted sum)
        self.softmax_io = self.N * self.N * self.byte_size * 2
        # Attention output: write once, read once (for MLP)
        self.attn_out_io = self.N * self.D * self.byte_size * 2
        # MLP activation: write once, read once
        self.mlp_act_io = self.N * self.D * self.mlp_ratio * self.byte_size * 2
        # Total memory access (including all intermediates)
        self.total_memory_full = (
            self.input_memory + self.output_memory + self.total_weight +
            self.qkv_io + self.score_io + self.softmax_io + self.attn_out_io + self.mlp_act_io
        )
        self.total_memory = self.input_memory + self.output_memory + self.total_weight
        self.arithmetic_intensity_full = self.total_mac / self.total_memory_full if self.total_memory_full > 0 else 0
        self.arithmetic_intensity = self.total_mac / self.total_memory if self.total_memory > 0 else 0

    def print_analysis(self):
        print(f"\n{'='*60}")
        print(f"Transformer Block: {self.name}")
        print(f"{'='*60}")
        print(f"Sequence length N: {self.N}")
        print(f"Embedding dimension D: {self.D}")
        print(f"Number of attention heads: {self.num_heads}")
        print(f"MLP expansion ratio: {self.mlp_ratio}")
        print(f"\nMACs (Multiply-Accumulate Operations): {self.total_mac:,.0f}")
        print(f"Weight memory (bytes): {self.total_weight:,.0f}")
        print(f"Input memory (bytes): {self.input_memory:,.0f}")
        print(f"Output memory (bytes): {self.output_memory:,.0f}")
        print(f"Total memory access (main, bytes): {self.input_memory + self.output_memory + self.total_weight:,.0f}")
        print(f"Arithmetic intensity (main, MACs/byte): {self.arithmetic_intensity:.3f}")
        print(f"\n--- Intermediate memory access details (assuming all in DRAM) ---")
        print(f"Q/K/V IO (bytes): {self.qkv_io:,.0f}")
        print(f"QK^T (score matrix) IO (bytes): {self.score_io:,.0f}")
        print(f"Softmax output IO (bytes): {self.softmax_io:,.0f}")
        print(f"Attention output IO (bytes): {self.attn_out_io:,.0f}")
        print(f"MLP activation IO (bytes): {self.mlp_act_io:,.0f}")
        print(f"Total memory access (with all intermediates, bytes): {self.total_memory_full:,.0f}")
        print(f"Arithmetic intensity (with all intermediates, MACs/byte): {self.arithmetic_intensity_full:.3f}")
        if self.arithmetic_intensity_full < 1.0:
            print(f"  -> MEMORY-BOUND (low arithmetic intensity)")
        elif self.arithmetic_intensity_full > 10.0:
            print(f"  -> COMPUTE-BOUND (high arithmetic intensity)")
        else:
            print(f"  -> BALANCED")

def test_transformer_block_print():
    print(f"\n{'#'*80}")
    print("ANALYZING VISION TRANSFORMER BLOCK (ViT-B/16)")
    print(f"{'#'*80}")
    # ViT-B/16: input 224x224, patch=16, N=196, D=768, heads=12, mlp_ratio=4
    vit_block = TransformerBlockAnalyzer(
        name="ViT-B/16 Block",
        N=196,  # 14x14 patches
        D=768,
        num_heads=12,
        mlp_ratio=4,
        byte_size=4
    )
    vit_block.print_analysis()

def test_resnet_layers_print():
    print(f"\n{'#'*80}")
    print(f"ANALYZING ResNet50")
    print(f"{'#'*80}")
    resnet_layers = [
        LayerAnalyzer("ResNet50 - Initial Conv", C=3, M=64, P=112, Q=112, R=7, S=7, stride=2),
        LayerAnalyzer("ResNet50 - Block Conv1", C=64, M=64, P=56, Q=56, R=1, S=1),
        LayerAnalyzer("ResNet50 - Block Conv2", C=64, M=64, P=56, Q=56, R=3, S=3),
        LayerAnalyzer("ResNet50 - Block Conv3", C=64, M=256, P=56, Q=56, R=1, S=1),
        LayerAnalyzer("ResNet50 - Later Block", C=256, M=64, P=56, Q=56, R=1, S=1),
    ]
    all_layers = []
    for layer in resnet_layers:
        layer.print_analysis()
        all_layers.append(("ResNet50", layer))
    return all_layers

def test_efficientnet_layers_print():
    print(f"\n{'#'*80}")
    print(f"ANALYZING EfficientNet-B0")
    print(f"{'#'*80}")
    efficientnet_layers = [
        LayerAnalyzer("EfficientNet - Initial Conv", C=3, M=32, P=112, Q=112, R=3, S=3, stride=2),
        LayerAnalyzer("EfficientNet - Depthwise", C=1, M=1, P=112, Q=112, R=3, S=3, G=32),
        LayerAnalyzer("EfficientNet - SE Reduce", C=32, M=8, P=1, Q=1, R=1, S=1),
        LayerAnalyzer("EfficientNet - SE Expand", C=8, M=32, P=1, Q=1, R=1, S=1),
        LayerAnalyzer("EfficientNet - Project", C=32, M=16, P=112, Q=112, R=1, S=1),
        LayerAnalyzer("EfficientNet - Expand", C=16, M=96, P=112, Q=112, R=1, S=1),
    ]
    all_layers = []
    for layer in efficientnet_layers:
        layer.print_analysis()
        all_layers.append(("EfficientNet-B0", layer))
    return all_layers

def test_mobilenet_layers_print():
    print(f"\n{'#'*80}")
    print(f"ANALYZING MobileNet-V3")
    print(f"{'#'*80}")
    mobilenet_layers = [
        LayerAnalyzer("MobileNet - Initial Conv", C=3, M=16, P=112, Q=112, R=3, S=3, stride=2),
        LayerAnalyzer("MobileNet - Depthwise", C=1, M=1, P=56, Q=56, R=3, S=3, G=16, stride=2),
        LayerAnalyzer("MobileNet - SE Reduce", C=16, M=8, P=1, Q=1, R=1, S=1),
        LayerAnalyzer("MobileNet - SE Expand", C=8, M=16, P=1, Q=1, R=1, S=1),
        LayerAnalyzer("MobileNet - Pointwise", C=16, M=16, P=56, Q=56, R=1, S=1),
    ]
    all_layers = []
    for layer in mobilenet_layers:
        layer.print_analysis()
        all_layers.append(("MobileNet-V3", layer))
    return all_layers

def main():
    print("LAYER CHARACTERISTICS ANALYSIS")
    print("Comparing EfficientNet, MobileNet, and ResNet50")
    all_layers = []
    all_layers += test_resnet_layers_print()
    all_layers += test_efficientnet_layers_print()
    all_layers += test_mobilenet_layers_print()
    # Transformer Block test
    test_transformer_block_print()
    # Comparative analysis
    print(f"\n{'#'*80}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'#'*80}")
    
    # Group by network and calculate averages
    network_stats = {}
    for network_name, layer in all_layers:
        if network_name not in network_stats:
            network_stats[network_name] = {
                'layers': [],
                'arithmetic_intensities': [],
                'compute_to_memory_ratios': [],
                'memory_bound_count': 0
            }
        
        network_stats[network_name]['layers'].append(layer)
        network_stats[network_name]['arithmetic_intensities'].append(layer.arithmetic_intensity)
        network_stats[network_name]['compute_to_memory_ratios'].append(layer.compute_to_memory_ratio)
        
        if layer.arithmetic_intensity < 1.0:
            network_stats[network_name]['memory_bound_count'] += 1
    
    print(f"\n{'Network':<15} {'Avg AI':<10} {'Avg C/M':<12} {'Memory-Bound':<15} {'Total Layers':<12}")
    print(f"{'-'*70}")
    
    for network_name, stats in network_stats.items():
        avg_ai = sum(stats['arithmetic_intensities']) / len(stats['arithmetic_intensities'])
        avg_cm = sum(stats['compute_to_memory_ratios']) / len(stats['compute_to_memory_ratios'])
        mb_percentage = (stats['memory_bound_count'] / len(stats['layers'])) * 100
        
        print(f"{network_name:<15} {avg_ai:<10.3f} {avg_cm:<12.3f} {mb_percentage:<6.1f}% ({stats['memory_bound_count']}/{len(stats['layers'])}) {len(stats['layers']):<12}")
    
    # Key insights
    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}")
    
    print("\n1. DEPTHWISE SEPARABLE CONVOLUTIONS:")
    print("   - Break standard convolution into depthwise + pointwise operations")
    print("   - Depthwise: Very low arithmetic intensity due to minimal compute per spatial location")
    print("   - Pointwise: High memory traffic due to many small 1x1 operations")
    
    print("\n2. SQUEEZE-AND-EXCITATION (SE) BLOCKS:")
    print("   - Global pooling + small FC layers")
    print("   - Very low arithmetic intensity (P=Q=1)")
    print("   - High memory overhead relative to computation")
    
    print("\n3. CHANNEL EXPANSION/REDUCTION PATTERNS:")
    print("   - Frequent channel dimension changes require more memory bandwidth")
    print("   - Smaller spatial dimensions = lower compute reuse")
    
    print("\n4. RESNET50 VS EFFICIENT NETWORKS:")
    print("   - ResNet50: Larger kernels (3x3, 7x7) with consistent channels")
    print("   - EfficientNet/MobileNet: Many 1x1 convs + depthwise separable convs")
    print("   - Efficient networks trade compute for memory bandwidth efficiency")
    
    # Sample calculation
    print(f"\n{'='*80}")
    print("SAMPLE MEMORY BANDWIDTH CALCULATION")
    print(f"{'='*80}")
    
    # Compare a ResNet block vs MobileNet block
    resnet_3x3 = LayerAnalyzer("ResNet 3x3", C=64, M=64, P=56, Q=56, R=3, S=3)
    mobile_dw = LayerAnalyzer("MobileNet DW", C=1, M=1, P=56, Q=56, R=3, S=3, G=64)
    mobile_pw = LayerAnalyzer("MobileNet PW", C=64, M=64, P=56, Q=56, R=1, S=1)
    
    print(f"\nResNet 3x3 Convolution:")
    print(f"  MACs: {resnet_3x3.mac_ops:,.0f}")
    print(f"  Memory: {resnet_3x3.total_memory:,.0f} bytes")
    print(f"  Arithmetic Intensity: {resnet_3x3.arithmetic_intensity:.3f}")
    
    print(f"\nMobileNet Depthwise + Pointwise (equivalent):")
    mobile_total_macs = mobile_dw.mac_ops + mobile_pw.mac_ops
    mobile_total_memory = mobile_dw.total_memory + mobile_pw.total_memory
    mobile_combined_ai = mobile_total_macs / mobile_total_memory
    
    print(f"  Total MACs: {mobile_total_macs:,.0f}")
    print(f"  Total Memory: {mobile_total_memory:,.0f} bytes")
    print(f"  Combined Arithmetic Intensity: {mobile_combined_ai:.3f}")
    
    print(f"\nComparison:")
    print(f"  MAC Reduction: {((resnet_3x3.mac_ops - mobile_total_macs) / resnet_3x3.mac_ops * 100):.1f}%")
    print(f"  Memory Increase: {((mobile_total_memory - resnet_3x3.total_memory) / resnet_3x3.total_memory * 100):.1f}%")
    print(f"  AI Change: {((mobile_combined_ai - resnet_3x3.arithmetic_intensity) / resnet_3x3.arithmetic_intensity * 100):.1f}%")

if __name__ == "__main__":
    main()