#!/usr/bin/env python3
import os
import yaml
import csv
import argparse
import copy
import utility_functions
from utility_functions import unifyname
import global_parameter
# No longer need utility functions since we parse dimensions directly from YAML

def adjust_projection_parameters(data, tp=None, batch_size=None, sequence_length=None, layer_name=None):
    """调整projection层的tp、batch size和sequence length参数"""
    adjusted_data = copy.deepcopy(data)
    
    if 'problem' in adjusted_data and 'instance' in adjusted_data['problem']:
        instance = adjusted_data['problem']['instance']
        
        # 调整tp参数
        if tp is not None and tp > 1 and layer_name is not None:
            # 根据layer_name确定tp维度
            if unifyname(layer_name) in global_parameter.TRANSFORMER_TP_CONFIG:
                tp_dim = global_parameter.TRANSFORMER_TP_CONFIG[unifyname(layer_name)]
                if tp_dim in instance:
                    instance[tp_dim] = instance[tp_dim] // tp
        
        # 调整batch size (B维度)
        if batch_size is not None and 'B' in instance:
            instance['B'] = batch_size
            
        # 调整sequence length (P维度)
        if sequence_length is not None and 'P' in instance:
            instance['P'] = sequence_length
    
    return adjusted_data

def adjust_attention_parameters(data, tp=None, sequence_length=None, layer_name=None):
    """调整attention层的tp和sequence length参数"""
    # 深拷贝数据以避免修改原始数据
    adjusted_data = copy.deepcopy(data)
    
    if 'problem' in adjusted_data and 'instance' in adjusted_data['problem']:
        instance = adjusted_data['problem']['instance']
        
        # 调整tp参数
        if tp is not None and tp > 1 and layer_name is not None:
            # 根据layer_name确定tp维度
            if unifyname(layer_name) in global_parameter.TRANSFORMER_TP_CONFIG:
                tp_dim = global_parameter.TRANSFORMER_TP_CONFIG[unifyname(layer_name)]
                if tp_dim in instance:
                    instance[tp_dim] = instance[tp_dim] // tp
        
        # 只调整sequence length相关的维度
        if sequence_length is not None:
            # 调整P维度 (sequence length)
            if 'P' in instance:
                instance['P'] = sequence_length
            # 调整M维度 (如果存在，通常也是sequence length相关)
            if 'M' in instance:
                instance['M'] = sequence_length
    return adjusted_data

def adjust_cnn_parameters(data, tp=None, batch_size=None, layer_name=None):
    """调整CNN层的tp和batch_size参数"""
    # 深拷贝数据以避免修改原始数据
    adjusted_data = copy.deepcopy(data)
    
    if 'problem' in adjusted_data and 'instance' in adjusted_data['problem']:
        instance = adjusted_data['problem']['instance']
        
        # 调整tp参数
        if tp is not None and tp > 1:
            # CNN层的tp调整逻辑：优先使用M维度，如果M为1则使用C维度
            if 'M' in instance and instance['M'] > 1:
                instance['M'] = instance['M'] // tp
            else:
                print(f"Warning: Cannot apply tp={tp} to CNN layer{layer_name} - M dimension is 1")
        
        # 调整batch size (N维度)
        if batch_size is not None and 'N' in instance:
            instance['N'] = batch_size
    
    return adjusted_data

def calculate_sizes_from_shape(layer_data, bits_per_word, layer_name):
    """Generic size calculator that reads data_spaces from the YAML shape.
    Works for any GEMM/matmul layer regardless of dimension naming convention.
    """
    instance = layer_data['problem']['instance']
    shape = layer_data['problem'].get('shape', {})
    data_spaces = shape.get('data_spaces', [])

    tensor_sizes = {}
    for ds in data_spaces:
        name = ds.get('name', '')
        proj = ds.get('projection', [])
        # Each projection entry is a list of dimension lists, e.g. [[[B],[N],[C]]]
        size = 1
        for dim_group in proj:
            for dim_list in dim_group:
                for dim_name in dim_list:
                    size *= instance.get(dim_name, 1)
        tensor_sizes[name] = size

    # Convention: Inputs1 = weights, Inputs2 = activations, Outputs = output
    weights_size_bits = tensor_sizes.get('Inputs1', 1) * bits_per_word
    input_size_bits = tensor_sizes.get('Inputs2', 1) * bits_per_word
    output_size_bits = tensor_sizes.get('Outputs', 1) * bits_per_word

    # Operations = product of all instance dimensions (GEMM flops)
    operations = 1
    for v in instance.values():
        operations *= v

    return {
        'dimensions': dict(instance),
        'weights_size_gb': weights_size_bits / (8 * 10 ** 9),
        'output_size_gb': output_size_bits / (8 * 10 ** 9),
        'input_size_gb': input_size_bits / (8 * 10 ** 9),
        'operations': operations
    }

def calculate_sizes(layer_data, bits_per_word, layer_name):
    """Calculate sizes of weights, input, and output features in GB based on actual YAML structure."""
    instance = layer_data['problem']['instance']

    # Check which dimensions are present to determine the calculation type
    if utility_functions.is_attention_layers(layer_name):
        return calculate_attention_sizes(instance, bits_per_word, layer_name)
    elif utility_functions.is_projection_layers(layer_name):
        # For new-style projections, try unified name lookup first
        unified = global_parameter.unifyname(layer_name)
        if unified in global_parameter.xy_dict:
            return calculate_projection_sizes(instance, bits_per_word, layer_name)
        else:
            # Generic GEMM fallback (lm_head, router, expert_*, etc.)
            return calculate_sizes_from_shape(layer_data, bits_per_word, layer_name)
    elif utility_functions.is_softmax_layers(layer_name):
        return calculate_transformer_elementwise_sizes(instance, bits_per_word, layer_name)
    elif 'shape' in layer_data.get('problem', {}) and 'data_spaces' in layer_data['problem'].get('shape', {}):
        # Generic fallback for any layer with data_spaces (e.g., lm_head, router)
        return calculate_sizes_from_shape(layer_data, bits_per_word, layer_name)
    else:
        return calculate_conv_sizes(layer_data, bits_per_word, layer_name)

def calculate_conv_sizes(layer_data, bits_per_word, layer_name):
    """Calculate sizes for convolution layers with stride and 'same' padding."""
    instance = layer_data['problem']['instance']
    
    # Extract dimensions
    N = instance.get('N', 1)
    G = instance.get('G', 1)
    M = instance.get('M', 1)
    C = instance.get('C', 1)
    P = instance.get('P', 1)  # Output height
    Q = instance.get('Q', 1)  # Output width
    R = instance.get('R', 1)  # Kernel height
    S = instance.get('S', 1)  # Kernel width

    # Calculate sizes in bits
    weights_size_bits = G * C * M * R * S * bits_per_word  # Weight tensor: [G, C, M, R, S]
    output_size_bits = N * G * M * P * Q * bits_per_word   # Output feature map: [N, G, M, P, Q]
    
    # Extract stride values from YAML coefficients
    H_stride = 1
    W_stride = 1
    
    # Access the coefficients section to get actual stride values
    if 'shape' in layer_data['problem'] and 'coefficients' in layer_data['problem']['shape']:
        coefficients = layer_data['problem']['shape']['coefficients']
        for coef in coefficients:
            if coef.get('name') == 'Hstride':
                H_stride = coef.get('default', 1)
            elif coef.get('name') == 'Wstride':
                W_stride = coef.get('default', 1)
    
    # Calculate actual input spatial dimensions with stride and 'same' padding
    # Formula: Input size = Output size × stride
    input_height = P * H_stride
    input_width = Q * W_stride
    
    input_size_bits = N * G * C * input_height * input_width * bits_per_word
    
    dimension = {
        'N': N, 'G': G, 'M': M, 'C': C, 
        'P': P, 'Q': Q, 'R': R, 'S': S,
        'input_height': input_height, 'input_width': input_width,
        'H_stride': H_stride, 'W_stride': W_stride
    }
    
    # Calculate operations for convolution: N * G * M * P * Q * C * R * S
    operations = N * G * M * P * Q * C * R * S
    
    # Convert to GB (1 GB = 8 * 10^9 bits)
    weights_size_gb = weights_size_bits / (8 * 10 ** 9)
    output_size_gb = output_size_bits / (8 * 10 ** 9)
    input_size_gb = input_size_bits / (8 * 10 ** 9)
    
    return {
        'dimensions': dimension,
        'weights_size_gb': weights_size_gb,
        'output_size_gb': output_size_gb,
        'input_size_gb': input_size_gb,
        'operations': operations
    }

def calculate_projection_sizes(instance, bits_per_word, layer_name):
    """Calculate sizes for projection layers using actual YAML dimensions."""
    dims = {key: instance.get(key, 1) for key in instance.keys()}
    B = dims.get('B', 1)
    # Sequence length: P (legacy) or N (new-style)
    P = dims.get('P', dims.get('N', 1))

    if unifyname(layer_name) in global_parameter.xy_dict:
        x = global_parameter.xy_dict[unifyname(layer_name)]['x']
        y = global_parameter.xy_dict[unifyname(layer_name)]['y']
        if x in dims and y in dims:
            x_dim, y_dim = dims[x], dims[y]
            input_size_bits = B * P * x_dim * bits_per_word
            weights_size_bits = x_dim * y_dim * bits_per_word
            output_size_bits = B * P * y_dim * bits_per_word
            operations = B * P * x_dim * y_dim
        else:
            # New-style dimensions: C=input_channels, M=output_channels, N=seq_len
            C = dims.get('C', 1)
            M = dims.get('M', 1)
            input_size_bits = B * P * C * bits_per_word
            weights_size_bits = M * C * bits_per_word
            output_size_bits = B * P * M * bits_per_word
            operations = B * P * M * C
    else:
        raise ValueError(f"Layer {layer_name} not found in global_parameter.xy_dict")
    # Convert to GB
    weights_size_gb = weights_size_bits / (8 * 10 ** 9)
    output_size_gb = output_size_bits / (8 * 10 ** 9)
    input_size_gb = input_size_bits / (8 * 10 ** 9)

    return {
        'dimensions': dims,
        'weights_size_gb': weights_size_gb,
        'output_size_gb': output_size_gb,
        'input_size_gb': input_size_gb,
        'operations': operations
    }

def calculate_attention_sizes(instance, bits_per_word, layer_name):
    """Calculate sizes for attention layers using actual YAML dimensions."""
    # Extract all available dimensions
    dims = {key: instance.get(key, 1) for key in instance.keys()}
    B = dims.get('B', 1)
    H = dims.get('H', 1)

    # New-style naming (llama/qwen): D, K, Q dimensions
    D = dims.get('D', 1)
    K = dims.get('K', 1)
    Q = dims.get('Q', 1)
    # Legacy naming (OPT/ViT): E, F, M, P dimensions
    E = dims.get('E', 1)
    F = dims.get('F', 1)
    M = dims.get('M', 1)
    P = dims.get('P', 1)

    if D > 1 and K > 1 and 'Q' in dims and 'K' in dims and 'D' in dims:
        # New-style attention with D, K, Q dimensions (llama/qwen)
        if "qk" in layer_name.lower():
            # QK attention: (B,H,Q,D) x (B,H,K,D) -> (B,H,Q,K)
            input_size_bits = B * H * Q * D * bits_per_word
            weights_size_bits = B * H * K * D * bits_per_word
            output_size_bits = B * H * Q * K * bits_per_word
            operations = B * H * Q * K * D
        else:
            # AV attention: (B,H,Q,K) x (B,H,K,D) -> (B,H,Q,D)
            input_size_bits = B * H * Q * K * bits_per_word
            weights_size_bits = B * H * K * D * bits_per_word
            output_size_bits = B * H * Q * D * bits_per_word
            operations = B * H * Q * K * D
    elif E > 1:  # Legacy QK: (B, E, H, P) x (B, E, H, M) -> (B, H, M, P)
        input_size_bits = B * E * H * P * bits_per_word
        weights_size_bits = B * E * H * M * bits_per_word
        output_size_bits = B * H * M * P * bits_per_word
        operations = B * H * E * M * P
    elif F > 1:  # Legacy AV: (B, H, M, P) x (B, F, H, M) -> (B, F, H, P)
        input_size_bits = B * H * M * P * bits_per_word
        weights_size_bits = B * F * H * M * bits_per_word
        output_size_bits = B * F * H * P * bits_per_word
        operations = B * H * M * F * P
    else:
        raise ValueError("Unknown attention layer type")
    
    # Convert to GB
    weights_size_gb = weights_size_bits / (8 * 10 ** 9)
    output_size_gb = output_size_bits / (8 * 10 ** 9)
    input_size_gb = input_size_bits / (8 * 10 ** 9)
    
    return {
        'dimensions': dims,
        'weights_size_gb': weights_size_gb,
        'output_size_gb': output_size_gb,
        'input_size_gb': input_size_gb,
        'operations': operations
    }

def calculate_transformer_elementwise_sizes(instance, bits_per_word, layer_name):
    """Calculate sizes for transformer elementwise operations using actual YAML dimensions."""
    # Extract all available dimensions
    dims = {key: instance.get(key, 1) for key in instance.keys()}
    B = dims.get('B', 1)
    H = dims.get('H', 1)
    # Support both legacy (M, P) and new (Q, K) dimension naming
    M = dims.get('M', dims.get('Q', 1))
    P = dims.get('P', dims.get('K', 1))
    
    # Base input size
    input_size_bits = B * H * M * P * bits_per_word
    
    # Determine operation type from the LAST token of the layer name
    # to avoid false matches (e.g., "softmax_sub_exp" contains "max")
    last_token = layer_name.rsplit('_', 1)[-1]
    reduce_ops = {'max', 'sd', 'sum'}       # reduce one dimension
    keep_ops   = {'sn', 'a', 'exp', 'div'}  # element-wise, keep all dims

    if last_token in reduce_ops:
        # Max, std-dev, and sum operations reduce M dimension
        weights_size_bits = 0  # No weights for elementwise operations
        output_size_bits = B * H * P * bits_per_word
        operations = B * H * M * P  # Process all elements
    elif last_token in keep_ops:
        # Softmax normalization / exp / div keeps all dimensions
        weights_size_bits = B * H * P * bits_per_word # -max
        output_size_bits = B * H * M * P * bits_per_word
        operations = B * H * M * P  # exp or div
    else:
        raise ValueError(f"Unknown transformer elementwise layer type: {layer_name}")
    
    # Convert to GB
    weights_size_gb = weights_size_bits / (8 * 10 ** 9)
    output_size_gb = output_size_bits / (8 * 10 ** 9)
    input_size_gb = input_size_bits / (8 * 10 ** 9)
    
    return {
        'dimensions': dims,
        'weights_size_gb': weights_size_gb,
        'output_size_gb': output_size_gb,
        'input_size_gb': input_size_gb,
        'operations': operations
    }

def generate_fused_layer_results(net_name, layer_name, result, fused_layer_type):
    """生成单个fused layer类型的结果"""
    if fused_layer_type == "start":
        return {
            'net_name': net_name,
            'layer_name': layer_name,
            'fused_layer_type': fused_layer_type,
            'in_mem': result['input_size_gb'],
            'weight_mem': result['weights_size_gb'],
            'out_mem': 0,
            'operations': result['operations']
        }
    elif fused_layer_type == "middle":
        return {
            'net_name': net_name,
            'layer_name': layer_name,
            'fused_layer_type': fused_layer_type,
            'in_mem': 0,  # Input comes from previous layer
            'weight_mem': result['weights_size_gb'],
            'out_mem': 0,
            'operations': result['operations']
        }
    elif fused_layer_type == "end":
        return {
            'net_name': net_name,
            'layer_name': layer_name,
            'fused_layer_type': fused_layer_type,
            'in_mem': 0,  # Input comes from previous layer
            'weight_mem': result['weights_size_gb'],
            'out_mem': result['output_size_gb'],
            'operations': result['operations']
        }
    else:  # single
        return {
            'net_name': net_name,
            'layer_name': layer_name,
            'fused_layer_type': fused_layer_type,
            'in_mem': result['input_size_gb'],
            'weight_mem': result['weights_size_gb'],
            'out_mem': result['output_size_gb'],
            'operations': result['operations']
        }

def generate_all_fused_results(net_name, layer_name, result, tp=None, batch_size=None, sequence_length=None):
    """为单个层生成所有四种fused layer类型的结果"""
    results = []
    
    # 为每种fused layer类型生成结果
    for fused_type in ["start", "middle", "end", "single"]:
        fused_result = generate_fused_layer_results(net_name, layer_name, result, fused_type)
        
        # 添加tp、batch_size和sequence_length参数，没有的设为"N/A"
        fused_result['tp'] = tp if tp is not None else "N/A"
        fused_result['batch_size'] = batch_size if batch_size is not None else 1
        fused_result['sequence_length'] = sequence_length if sequence_length is not None else 1
            
        results.append(fused_result)
    
    return results

def _get_seq_len_from_instance(instance, net_name=""):
    """Extract sequence length from the network name or instance dimensions.
    For workloads with seq baked into the name (e.g., prefill_s512, decode_kv512),
    parse from the name to ensure all layers in the same network use the same value.
    """
    import re
    # Parse from network name: prefill_s512 → 512, decode_kv512 → 1
    if 'decode' in net_name:
        return 1
    m = re.search(r'_s(\d+)', net_name)
    if m:
        return int(m.group(1))
    # Fallback: extract from YAML dimensions
    for key in ['P', 'Q', 'N']:
        if key in instance and instance[key] > 1:
            return instance[key]
    return 1

def _is_transformer_network(net_name):
    """Check if network is a transformer (needs sequence length / batch sweep)."""
    return any(x in net_name for x in ['gpt', 'llama', 'qwen', 'vit', 'stable_diffusion'])

def _is_prefill_network(net_name):
    """Check if network is a prefill phase (sequence length varies)."""
    return 'prefill' in net_name

def analyze_network(network_dir, bits_per_word):
    """Analyze all layers in a single network."""
    results = []

    tp_degrees = global_parameter.tp_degrees
    batch_sizes = [1, 4, 8]
    sequence_lengths = [256, 512, 1024]

    layer_files = [f for f in sorted(os.listdir(network_dir)) if f.endswith('.yaml')]

    for idx, filename in enumerate(layer_files):
        file_path = os.path.join(network_dir, filename)

        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
            if 'problem' not in data:
                continue

            net_name = os.path.basename(network_dir)
            layer_name = os.path.splitext(filename)[0]
            instance = data['problem']['instance']
            default_seq_len = _get_seq_len_from_instance(instance, net_name=net_name)
            is_transformer = _is_transformer_network(net_name)

            if utility_functions.is_attention_layers(layer_name) or utility_functions.is_softmax_layers(layer_name):
                for tp in tp_degrees:
                    for batch_size in batch_sizes:
                        adjusted_data = adjust_attention_parameters(data, tp=tp, sequence_length=None, layer_name=layer_name)
                        result = calculate_sizes(adjusted_data, bits_per_word, layer_name)
                        results.extend(generate_all_fused_results(net_name, layer_name, result, tp, batch_size=batch_size, sequence_length=default_seq_len))
            elif utility_functions.is_projection_layers(layer_name) or utility_functions.is_gemm_layer(layer_name):
                for tp in tp_degrees:
                    for batch_size in batch_sizes:
                        adjusted_data = adjust_projection_parameters(data, tp=tp, batch_size=batch_size, sequence_length=None, layer_name=layer_name)
                        result = calculate_sizes(adjusted_data, bits_per_word, layer_name)
                        results.extend(generate_all_fused_results(net_name, layer_name, result, tp, batch_size, default_seq_len))
            elif is_transformer:
                # Generic transformer layer (fallback)
                for tp in tp_degrees:
                    for batch_size in batch_sizes:
                        result = calculate_sizes(data, bits_per_word, layer_name)
                        results.extend(generate_all_fused_results(net_name, layer_name, result, tp, batch_size, default_seq_len))
            else:
                # CNN layers
                for tp in tp_degrees:
                    for batch_size in [1, 32]:
                        adjusted_data = adjust_cnn_parameters(data, tp=tp, batch_size=batch_size, layer_name=layer_name)
                        result = calculate_sizes(adjusted_data, bits_per_word, layer_name)
                        results.extend(generate_all_fused_results(net_name, layer_name, result, tp, batch_size))

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return results

def main():
    parser = argparse.ArgumentParser(description='Analyze neural networks and output memory requirements')
    parser.add_argument('--workloads', required=True, help='Directory containing network workloads')
    parser.add_argument('--bits', type=int, default=8, help='Bits per word (default: 8)')
    parser.add_argument('--output', default='network_analysis.csv', help='Output CSV file (default: network_analysis.csv)')
    args = parser.parse_args()
    
    all_results = []
    
    # Process each network in the workloads directory
    for network in sorted(os.listdir(args.workloads)):
        network_dir = os.path.join(args.workloads, network)
        
        if os.path.isdir(network_dir):
            print(f"Analyzing network: {network}")
            network_results = analyze_network(network_dir, args.bits)
            all_results.extend(network_results)
    
    # Write results to CSV
    with open(args.output, 'w', newline='') as csvfile:
        fieldnames = ['net_name', 'layer_name', 'fused_layer_type', 'in_mem', 'weight_mem', 'out_mem', 'operations', 'tp', 'batch_size', 'sequence_length']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)
    
    print(f"\nAnalysis complete. Results written to {args.output}")

if __name__ == "__main__":
    main()
