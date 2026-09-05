from typing import Dict, List, Optional, Tuple, Set
import os
import re
import random
from global_parameter import *
from utility_functions import get_layer_name_from_yaml, is_attention_layers
import utility_functions
import yaml
from typing import Any
import math
'''
class Tile:
    """Represents a rectangular tile in a layer."""
    def __init__(self, x, y, width, height):
        self.x = x  # Starting x coordinate
        self.y = y  # Starting y coordinate
        self.width = width
        self.height = height
        
        # 新增属性：支持不同tiling策略
        # CNN channel tiling
        self.channel_start = None
        self.channel_end = None
        
        # Attention head tiling
        self.head_start = None
        self.head_end = None
        
        # Projection P tiling
        self.p_start = None
        self.p_end = None
    
    def __str__(self):
        base_str = f"Tile(x={self.x}, y={self.y}, width={self.width}, height={self.height})"
        
        # 添加tiling策略信息
        if self.channel_start is not None:
            base_str += f", channels[{self.channel_start}:{self.channel_end}]"
        if self.head_start is not None:
            base_str += f", heads[{self.head_start}:{self.head_end}]"
        if self.p_start is not None:
            base_str += f", p[{self.p_start}:{self.p_end}]"
            
        return base_str
    
    def __repr__(self):
        return self.__str__()
'''

class LayerConfig:
    def __init__(self, yaml_path: str, is_attn=False, is_padding=False, batch_size=1, sequence_length=1):
        # Define default values for all expected variables
        self.is_attn = is_attn
        self.is_padding = is_padding
        self.name = get_layer_name_from_yaml(yaml_path)      # Layer name
        self.layer_type = None  # Inferred from name if possible (conv, pooling, etc.)

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        with open(yaml_path, 'r') as f:
            # 加载整个 'problem' 字典，因为 _get_resolved_shapes 需要它
            self.problem_data = yaml.safe_load(f).get("problem", {})
        if not is_attn:
            self.tiling_dimension = "M"  
            self.instance = {
                'N': 1,  # Batch size
                'C': 1,  # Input channels
                'H': 1,  # Input height
                'W': 1,  # Input width
                'G': 1,  # Groups
                'R': 1,  # Weight height
                'S': 1,  # Weight width
                'Hdilation': 1,  # (Conv) Height dilation
                'Hstride': 1,  # (Conv) Height stride
                'Wdilation': 1,  # (Conv) Width dilation
                'Wstride': 1,  # (Conv) Width stride
                'M': 1,  # Output channels
                'P': 1,  # Output height
                'Q': 1  # Output width
            }
        else:
            self.tiling_dimension = TRANSFORMER_TP_CONFIG.get(unifyname(self.name), "M")
            self.instance = {
                'B': 1,  # Batch size
                'H': 1,  # Head
                'E': 1,  # Feature dim per head (Q,K)
                'F': 1,  # Feature dim per head (V)
                'M': 1,  # Key/Value tokens
                'P': 1,  # Query tokens
                'O': 1,  # Original embedding dim (for projection layers)
                'D': 1,  # Projected dim (for projection layers)
                'I': 1   # Intermediate dim (for FFN layers)
            }
        
        if yaml_path:
            self._parse_file(yaml_path)
            self._infer_layer_type()
    
    def _infer_layer_type(self):
        """Infer the layer type from the name if possible."""
        if self.name:
            if "features" in self.name:
                self.layer_type = "conv"
            elif "classifier" in self.name:
                self.layer_type = "fc"
            elif "pool" in self.name:
                self.layer_type = "pool"
            else:
                self.layer_type = "unknown"
                
    def is_attention_layer(self):
        return utility_functions.is_attention_layers(self.name)

    def is_projection_layer(self):
        return utility_functions.is_projection_layers(self.name)
    def is_softmax_layer(self):
        return utility_functions.is_softmax_layers(self.name)

    def is_cnn_layer(self):
        return (not utility_functions.is_attention_layers(self.name)) and (not utility_functions.is_projection_layers(self.name)) and (not utility_functions.is_softmax_layers(self.name))

    
    def _parse_file(self, yaml_path: str):
        """
        Parse the YAML file as plain text to extract layer configuration values.
        """
        try:
            with open(yaml_path, 'r') as file:
                lines = file.readlines()
            
            # Track indentation levels
            in_problem = False
            in_instance = False
            
            for i, line in enumerate(lines):
                stripped_line = line.strip()
                
                # Track section hierarchy
                if stripped_line == "problem:":
                    in_problem = True
                    continue
                    
                if in_problem and stripped_line == "instance:":
                    in_instance = True
                    continue
                    
                if in_problem and stripped_line == "shape:":
                    in_instance = False
                    continue
                
                # Parse instance section
                if in_instance and ":" in stripped_line:
                    # Make sure we're at the right indentation level (2 levels deep)
                    if line.startswith("    "):
                        stripped_line = stripped_line.split("#", 1)[0].strip()
                        parts = stripped_line.split(":", 1)
                        key = parts[0].strip()
                        value = parts[1].strip()
                        
                        if key in self.instance:
                            try:
                                self.instance[key] = int(value)
                            except ValueError:
                                self.instance[key] = value
                
            
            # Parse coefficients for strides and other parameters
            self._extract_shape_parameters(lines)
                    
        except FileNotFoundError:
            print(f"Error: File {yaml_path} not found")
        except Exception as e:
            print(f"Error parsing file: {e}")
                
    def _extract_shape_parameters(self, lines):
        """
        Extract parameters from the coefficients section with the pattern shown in the file.
        """
        in_shape = False
        in_coefficients = False
        current_coefficient = {}
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # Find the shape section
            if stripped_line == "shape:":
                in_shape = True
                continue
                
            # Find the coefficients section within shape
            if in_shape and stripped_line == "coefficients:":
                in_coefficients = True
                continue
                
            # Process lines within the coefficients section
            if in_coefficients:
                # Check if we've moved to another section
                if not line.startswith("    ") and stripped_line and not stripped_line.startswith("-"):
                    in_coefficients = False
                    continue
                    
                # Start of a new coefficient entry
                if stripped_line.startswith("-"):
                    # Save previous coefficient if complete
                    if 'name' in current_coefficient and 'default' in current_coefficient:
                        name = current_coefficient['name']
                        default = current_coefficient['default']
                        
                        if name in self.instance:
                            self.instance[name] = default
                    
                    # Reset for new coefficient
                    current_coefficient = {}
                    
                    # Check if this line also contains a key-value pair (e.g., "- default: 3")
                    # Remove the leading "- " and parse the rest
                    if ":" in stripped_line:
                        key_value = stripped_line[2:].strip()  # Remove "- " prefix
                        if ":" in key_value:
                            parts = key_value.split(":", 1)
                            key = parts[0].strip()
                            value = parts[1].strip()
                            
                            if key == "name":
                                current_coefficient['name'] = value
                            elif key == "default":
                                try:
                                    current_coefficient['default'] = int(value)
                                except ValueError:
                                    current_coefficient['default'] = value
                    
                    continue
                    
                # Parse key-value pairs within a coefficient entry
                if ":" in stripped_line:
                    parts = stripped_line.split(":", 1)
                    key = parts[0].strip()
                    value = parts[1].strip()
                    
                    if key == "name":
                        current_coefficient['name'] = value
                    elif key == "default":
                        try:
                            current_coefficient['default'] = int(value)
                        except ValueError:
                            current_coefficient['default'] = value
        
        # Handle the last coefficient if we exited the loop
        if 'name' in current_coefficient and 'default' in current_coefficient:
            name = current_coefficient['name']
            default = current_coefficient['default']
            
            if name in self.instance:
                self.instance[name] = default
        if self.batch_size:
            if self.is_cnn_layer():
                self.instance['N'] = self.batch_size
            else:
                self.instance['B'] = self.batch_size

        if self.sequence_length:
            self.instance['P'] = self.sequence_length
            # Attention/Softmax 层 M 与 P 一致
            if self.is_attention_layer() or self.is_softmax_layer():
                self.instance['M'] = self.sequence_length
                
    def calculate_input_shape(self):
        """
        Calculate input shape (H, W) from output shape (P, Q) using the repo's formula.
        Returns a tuple of (H, W) representing the input height and width.
        
        Formula from the repo:
        Tile_Width = ((Tile_Width - 1) * Wstride) - (2 * Padding_Width) + (Wdilation * (Kernel_Width - 1)) + 1
        
        Note: Since padding isn't explicitly stored in our instance dict, we'll assume padding = 0
        unless it can be calculated from other parameters.
        """
        # Extract values from instance
        if not self.is_attn:
            P = self.instance['P']
            Q = self.instance['Q']
            R = self.instance['R']
            S = self.instance['S']
            Hstride = self.instance['Hstride']
            Wstride = self.instance['Wstride']
            Hdilation = self.instance['Hdilation']
            Wdilation = self.instance['Wdilation']

            # Assume padding is 0 if not specified
            Padding_H = 0
            Padding_W = 0

            # Calculate input dimensions using the formula from the repo
            H = ((P - 1) * Hstride) - (2 * Padding_H) + (Hdilation * (R - 1)) + 1
            W = ((Q - 1) * Wstride) - (2 * Padding_W) + (Wdilation * (S - 1)) + 1
            return (H, W)
        elif self.is_padding: ## input is the same size with output
            P = self.instance['P']
            Q = self.instance['Q']
            return (P, Q)
        else:
            # Transformer层：根据层类型返回不同的输入形状
            if self.is_attention_layer():
                # Attention层：根据具体类型计算输入形状
                if self.name.endswith("_qk"):
                    # QK层：(B, H, P, E) -> (B, H, P, M)
                    # 输入形状：(P, E)
                    P = self.instance['P']
                    E = self.instance.get('E', 1)
                    return (P, E)
                else:
                    # AV层：(B, H, P, M) × (B, H, M, F) -> (B, H, P, F)
                    # 输入是注意力分数矩阵：(B, H, P, M)
                    # 输入形状：(P, M)
                    M = self.instance.get('M', 1)
                    P = self.instance['P']
                    return (P, M)
            else:
                # Projection层和FFN层：(B, P, O) -> (B, P, D) 或 (B, P, I)
                # 输入形状：(P, O)
                P = self.instance['P']
                if self.name.endswith("_ffn2"):
                    O = self.instance.get('I', 1)
                elif self.name.endswith("_o"):
                    O = self.instance.get('D', 1)
                else:
                    O = self.instance.get('O', 1)
                return (P, O)
    '''                
    # 旧的tile分割方法，已注释掉
    def divide_into_tiles_old(self, max_tile_size):
        """
        Divide the layer into tiles using the specified maximum tile size.
        
        Args:
            max_tile_size: The maximum size (width and height) of a tile
            
        Returns:
            A list of Tile objects representing the tiling of the layer
        """
        # Clear any existing tiles
        raise NotImplementedError("Not implemented")
    '''
        
    def __eq__(self, other):
        if not isinstance(other, LayerConfig):
            return False
        return self.instance == other.instance and self.name == other.name
    
    def __hash__(self):
        return hash((tuple(sorted(self.instance.items())), self.name))
    
    def __str__(self):
        return f"LayerConfig(name={self.name}, type={self.layer_type}, instance={self.instance})"
    
    def __repr__(self):
        return self.__str__()

# functions to handle pooling affect
def infer_pooling_layers(current_layer, next_layer):
    """
    Infer if there might be pooling layers between two consecutive layers.
    This is done by comparing the expected input shape of the next layer
    with the output shape of the current layer.
    
    Args:
        current_layer: The earlier layer in the network
        next_layer: The later layer in the network
    
    Returns:
        tuple: (pooling_detected, h_scale, w_scale) where:
               - pooling_detected is a boolean indicating if pooling was detected
               - h_scale is the vertical scaling factor (e.g., 2 for 2x2 pooling)
               - w_scale is the horizontal scaling factor
    """
    # Get output dimensions of current layer
    current_P = current_layer.instance['P']  # Output height
    current_Q = current_layer.instance['Q']  # Output width
    
    # Calculate the expected input dimensions for the next layer
    # using the next layer's parameters and output dimensions
    expected_H, expected_W = next_layer.calculate_input_shape()
    
    # Check if there's a significant difference that might indicate pooling
    # Allow small differences (< 3) for potential padding variations
    h_diff = abs(current_P - expected_H)
    w_diff = abs(current_Q - expected_W)
    
    # Initialize scaling factors
    h_scale = 1
    w_scale = 1
    pooling_detected = False
    
    # Simple heuristic: check if dimensions differ by common pooling factors (2, 3)
    if h_diff > 1 or w_diff > 1:
        # Calculate potential scaling factors
        if current_P > 0 and expected_H > 0:
            h_ratio = current_P / expected_H
            # Round to nearest common pooling factor (typically 2 or 3)
            if 1.8 <= h_ratio <= 2.5:
                h_scale = 2
                pooling_detected = True
            elif 2.5 < h_ratio <= 3.2:
                h_scale = 3
                pooling_detected = True
            else:
                h_scale = 4 # do not consider larger pooling
                pooling_detected = True
        
        if current_Q > 0 and expected_W > 0:
            w_ratio = current_Q / expected_W
            # Round to nearest common pooling factor
            if 1.8 <= w_ratio <= 2.5:
                w_scale = 2
                pooling_detected = True
            elif 2.5 < w_ratio <= 3.2:
                w_scale = 3
                pooling_detected = True            
            else:
                w_scale = 4 # do not consider larger pooling
                pooling_detected = True
    
    return pooling_detected, h_scale, w_scale
'''
# 旧的tile计算辅助函数，已注释掉
def calculate_input_tiles_normal_old(current_layer, next_layer, next_layer_tiles):
    """
    Calculate input tiles normally without accounting for pooling.
    This is a helper function extracted from your original calculate_tiles method.
    
    Args:
        current_layer: The earlier layer in the network
        next_layer: The later layer in the network
        next_layer_tiles: List of tiles from the next layer
        
    Returns:
        list: Normal tiles for the current layer
    """
    current_layer_tiles = []
    
    # For each tile in the next layer, calculate the corresponding input tile in this layer
    for next_tile in next_layer_tiles:
        # Save original P and Q values
        original_P = next_layer.instance['P']
        if not next_layer.is_attn:
            original_Q = next_layer.instance['Q']
        
        # Set temporary P and Q values to the tile's height and width
        next_layer.instance['P'] = next_tile.height
        if not next_layer.is_attn:
            next_layer.instance['Q'] = next_tile.width
        
        # Calculate input shape for this tile
        input_height, input_width = next_layer.calculate_input_shape()
        
        # Restore original P and Q values
        next_layer.instance['P'] = original_P
        if not next_layer.is_attn:
            next_layer.instance['Q'] = original_Q
        
            # Get stride values to calculate the starting position
            h_stride = next_layer.instance.get('Hstride', 1)
            w_stride = next_layer.instance.get('Wstride', 1)

            # Calculate starting point in input coordinates
            input_x = next_tile.x * w_stride
            input_y = next_tile.y * h_stride
        else:
            input_x = next_tile.x
            input_y = next_tile.y
        # Create a new tile for the current layer
        current_tile = Tile(input_x, input_y, input_width, input_height)
        current_layer_tiles.append(current_tile)
    
    return current_layer_tiles

# 旧的tile计算辅助函数，已注释掉
def scale_input_tiles_for_pooling_old(current_layer, next_layer, next_layer_tiles):
    """
    Scale input tiles based on inferred pooling layers between current_layer and next_layer.
    
    Args:
        current_layer: The earlier layer in the network
        next_layer: The later layer in the network
        next_layer_tiles: List of tiles from the next layer
        
    Returns:
        list: Scaled tiles for the current layer
    """
    if next_layer.is_attn:
        return calculate_input_tiles_normal_old(current_layer, next_layer, next_layer_tiles)

    pooling_detected, h_scale, w_scale = infer_pooling_layers(current_layer, next_layer)
    
    if not pooling_detected:
        # No pooling detected, use normal calculation
        return calculate_input_tiles_normal_old(current_layer, next_layer, next_layer_tiles)
    
    # Pooling detected, scale the tiles appropriately
    current_layer_tiles = []
    
    # For each tile in the next layer, calculate the corresponding input tile in current layer
    for next_tile in next_layer_tiles:
        # Get stride values to calculate the starting position
        h_stride = next_layer.instance.get('Hstride', 1)
        w_stride = next_layer.instance.get('Wstride', 1)
        
        # Calculate starting point in input coordinates (accounting for pooling)
        input_x = next_tile.x * w_stride * w_scale
        input_y = next_tile.y * h_stride * h_scale
        
        # Save original P and Q values
        original_P = next_layer.instance['P']
        original_Q = next_layer.instance['Q']
        
        # Set temporary P and Q values to the tile's height and width
        next_layer.instance['P'] = next_tile.height
        next_layer.instance['Q'] = next_tile.width
        
        # Calculate input shape for this tile 
        input_height, input_width = next_layer.calculate_input_shape()
        
        # Scale the input dimensions according to the pooling factor
        input_height *= h_scale
        input_width *= w_scale
        
        # Restore original P and Q values
        next_layer.instance['P'] = original_P
        next_layer.instance['Q'] = original_Q
        
        # Create a new tile for the current layer
        current_tile = Tile(input_x, input_y, input_width, input_height)
        current_layer_tiles.append(current_tile)
    
    return current_layer_tiles

# 旧的tile可视化函数，已注释掉
def visualize_layer_tiles_old(layer, tiles):
    """
    Visualize the tiles for a specific layer as a grid.
    
    Args:
        layer: LayerConfig object
        tiles: List of Tile objects for this layer
    """
    # Get output dimensions
    P = layer.instance['P']  # Output height
    Q = layer.instance['Q']  # Output width
    
    # Create a matrix to represent the layer
    # -1 means uncovered, non-negative integers represent tile indices
    coverage = [[-1 for _ in range(Q)] for _ in range(P)]
    
    # Fill in the coverage matrix with tile indices
    for i, tile in enumerate(tiles):
        for y in range(tile.y, min(tile.y + tile.height, P)):
            for x in range(tile.x, min(tile.x + tile.width, Q)):
                if 0 <= y < P and 0 <= x < Q:
                    if coverage[y][x] != -1:
                        print(f"Warning: Pixel at ({x}, {y}) is covered by multiple tiles")
                    coverage[y][x] = i
    
    # Count covered and uncovered cells
    total_cells = P * Q
    covered_cells = sum(1 for row in coverage for cell in row if cell != -1)
    coverage_percentage = (covered_cells / total_cells) * 100 if total_cells > 0 else 0
    
    # Print coverage information
    print(f"Layer: {layer.name}")
    print(f"Dimensions: {P}x{Q}")
    print(f"Number of tiles: {len(tiles)}")
    print(f"Coverage: {covered_cells}/{total_cells} cells ({coverage_percentage:.2f}%)")
    
    # Check for uncovered cells
    if covered_cells < total_cells:
        print("WARNING: Some cells are not covered by any tile!")
        # Find and report uncovered regions
        uncovered_regions = []
        for y in range(P):
            for x in range(Q):
                if coverage[y][x] == -1:
                    uncovered_regions.append((x, y))
        
        if len(uncovered_regions) <= 10:
            print(f"Uncovered pixels: {uncovered_regions}")
        else:
            print(f"First 10 uncovered pixels: {uncovered_regions[:10]}...")
    
    # Check for overlaps
    # Create a new matrix to track which tiles cover each cell
    overlap_matrix = [[[] for _ in range(Q)] for _ in range(P)]
    
    for i, tile in enumerate(tiles):
        for y in range(tile.y, min(tile.y + tile.height, P)):
            for x in range(tile.x, min(tile.x + tile.width, Q)):
                if 0 <= y < P and 0 <= x < Q:
                    overlap_matrix[y][x].append(i)
    
    # Count overlapping cells
    overlapping_cells = sum(1 for row in overlap_matrix for cell in row if len(cell) > 1)
    
    if overlapping_cells > 0:
        print(f"WARNING: {overlapping_cells} cells are covered by multiple tiles!")
        # Print the first few overlapping regions
        overlap_regions = []
        for y in range(P):
            for x in range(Q):
                if len(overlap_matrix[y][x]) > 1:
                    overlap_regions.append((x, y, overlap_matrix[y][x]))
                    if len(overlap_regions) >= 5:
                        break
            if len(overlap_regions) >= 5:
                break
        
        print(f"Sample overlapping regions: {overlap_regions}")
    
    # Optional: Visualize the coverage matrix as ASCII art for small matrices
    if P <= 20 and Q <= 50:  # Only visualize small matrices
        print("\nCoverage Visualization:")
        for row in coverage:
            print(''.join([f"{cell:2d}" if cell != -1 else " -" for cell in row]))
    else:
        print("\nMatrix too large to visualize in ASCII")
    
    print("-" * 50)
    
    return coverage_percentage

# 旧的tile覆盖率验证函数，已注释掉
def verify_all_layer_coverage_old(physical_network):
    """
    Verify that all elements in each layer are covered by the calculated tiles.
    
    Args:
        physical_network: PhysicalNetwork object with fusion groups and tiles
        
    Returns:
        dict: Coverage statistics for each layer
    """
    # Calculate tiles for all fusion groups
    all_tiles = physical_network.get_all_tiles()
    
    # Track coverage statistics
    coverage_stats = {}
    
    # Process each fusion group
    for fusion_group in physical_network.fusion_groups:
        print(f"\nFusion Group (Parallelism: {fusion_group.tensor_parallelism}, "
              f"Tile Size: {fusion_group.output_tile_size})")
        print("=" * 50)
        
        # Process each layer in the fusion group
        for layer in fusion_group.layers:
            if layer.name in all_tiles:
                tiles = all_tiles[layer.name]
                coverage_percentage = visualize_layer_tiles_old(layer, tiles)
                coverage_stats[layer.name] = {
                    'tiles': len(tiles),
                    'coverage': coverage_percentage,
                    'dimensions': (layer.instance['P'], layer.instance['Q'])
                }
            else:
                print(f"No tiles found for layer: {layer.name}")
    
    # Print summary
    print("\nCoverage Summary:")
    print("=" * 50)
    
    total_coverage = 0
    layer_count = 0
    
    for layer_name, stats in coverage_stats.items():
        print(f"{layer_name}: {stats['coverage']:.2f}% coverage with {stats['tiles']} tiles")
        total_coverage += stats['coverage']
        layer_count += 1
    
    if layer_count > 0:
        avg_coverage = total_coverage / layer_count
        print(f"\nAverage coverage across all layers: {avg_coverage:.2f}%")
    
    return coverage_stats
'''
class FusionGroup:
    """Represents a group of consecutive layers that can be fused together."""
    def __init__(self, layers=None, is_attn=False):
        self.layers = layers or []
        self.tensor_parallelism = 1  # Default value (can be 1, 2, or 4)
        # shares among all layers in a fusion group
        self.output_tile_size = None
        self.is_attn = is_attn

    def add_layer(self, layer):
        """Add a layer to this fusion group."""
        self.layers.append(layer)
    
    def set_tensor_parallelism(self, parallelism):
        """Set the tensor parallelism degree for this fusion group."""
        if parallelism not in [1, 2, 4]:
            raise ValueError("Tensor parallelism must be 1, 2, or 4")
        self.tensor_parallelism = parallelism
    
    def set_output_tile_size(self, tile_size):
        """Set the output tile size for this fusion group."""
        self.output_tile_size = tile_size
    '''
    # 旧的复杂tile计算方法，已注释掉
    def calculate_tiles_old(self, tensor_parallelism=None, output_tile_size=None, update_self= True):
        """
        Calculate actual tiles for each layer in the fusion group based on tensor parallelism.
        Tiles are calculated in a pyramid shape where output_tile_size defines the tile size 
        for the last layer, and other layers' tiles are derived from their input/output relationships.

        tensor_parallelism: if given, overload self.tensor_parallelism
        output_tile_size: if given, overload self.output_tile_size
        update_self: if update tiles in this fusion groups
        Returns:
            A dictionary mapping layer names to lists of tiles
        """
        if not self.output_tile_size:
            raise ValueError("Output tile size must be set before calculating tiles")
        if not output_tile_size:
            output_tile_size = self.output_tile_size
        if not tensor_parallelism:
            tensor_parallelism = self.tensor_parallelism
        result = {}
        num_layers = len(self.layers)
        
        # First, process the last layer (output layer) of the fusion group
        if num_layers > 0:
            last_layer = self.layers[-1]
            output_layer_tiles = []
            
            # Get output dimensions
            P = last_layer.instance['P']  # Output height
            if not self.is_attn:
                Q = last_layer.instance['Q']  # Output width

                # Handle tensor parallelism for the last layer
                if tensor_parallelism == 1 or P == 1 or Q ==1:
                    # No parallelism, create tiles up to output_tile_size
                    output_layer_tiles = last_layer.divide_into_tiles_old(output_tile_size)

                elif tensor_parallelism == 2:
                    # Divide by output width
                    half_width = Q // 2
                    remainder = Q % 2

                    # Create first tile
                    tile1 = Tile(0, 0, half_width, P)

                    # Create second tile (including remainder if width is odd)
                    tile2 = Tile(half_width, 0, half_width + remainder, P)

                    # Store initial partition tiles
                    partition_tiles = [tile1, tile2]

                    # Further subdivide each partition tile if needed
                    for tile in partition_tiles:
                        if tile.width > output_tile_size or tile.height > output_tile_size:
                            # This tile is too large and needs further subdivision
                            x, y = tile.x, tile.y
                            remaining_height = tile.height

                            while remaining_height > 0:
                                row_height = min(remaining_height, output_tile_size)
                                remaining_width = tile.width
                                curr_x = x

                                while remaining_width > 0:
                                    sub_width = min(remaining_width, output_tile_size)
                                    sub_tile = Tile(curr_x, y, sub_width, row_height)
                                    output_layer_tiles.append(sub_tile)
                                    curr_x += sub_width
                                    remaining_width -= sub_width

                                y += row_height
                                remaining_height -= row_height
                        else:
                            # This tile is already within size limits
                            output_layer_tiles.append(tile)

                elif tensor_parallelism == 4:
                    # Divide by both output width and height
                    half_width = Q // 2
                    half_height = P // 2
                    width_remainder = Q % 2
                    height_remainder = P % 2

                    # Create four quadrant tiles
                    # Top-left
                    tile1 = Tile(0, 0, half_width, half_height)

                    # Top-right
                    tile2 = Tile(half_width, 0, half_width + width_remainder, half_height)

                    # Bottom-left
                    tile3 = Tile(0, half_height, half_width, half_height + height_remainder)

                    # Bottom-right
                    tile4 = Tile(half_width, half_height, half_width + width_remainder, half_height + height_remainder)

                    # Store initial partition tiles
                    partition_tiles = [tile1, tile2, tile3, tile4]

                    # Further subdivide each partition tile if needed
                    for tile in partition_tiles:
                        if tile.width > output_tile_size or tile.height > output_tile_size:
                            # This tile is too large and needs further subdivision
                            x, y = tile.x, tile.y
                            remaining_height = tile.height

                            while remaining_height > 0:
                                row_height = min(remaining_height, output_tile_size)
                                remaining_width = tile.width
                                curr_x = x

                                while remaining_width > 0:
                                    sub_width = min(remaining_width, output_tile_size)
                                    sub_tile = Tile(curr_x, y, sub_width, row_height)
                                    output_layer_tiles.append(sub_tile)
                                    curr_x += sub_width
                                    remaining_width -= sub_width

                                y += row_height
                                remaining_height -= row_height
                        else:
                            # This tile is already within size limits
                            output_layer_tiles.append(tile)
            else:
                if last_layer.instance['F'] > 1: # AV
                    Q = last_layer.instance['F']
                else:
                    if "max" in last_layer.name or "sd" in last_layer.name:
                        Q = 1
                    else: #QK or sn or A
                        Q = last_layer.instance['M']
                if tensor_parallelism == 1 or P == 1 or Q == 1:
                    output_layer_tiles = last_layer.divide_into_tiles_old(output_tile_size)
                else:
                    # 1) 先做TP分大块(行方向)
                    partition_tiles = []
                    base_height = P // tensor_parallelism
                    remainder = P % tensor_parallelism
                    y_start = 0
                    for i in range(tensor_parallelism): #按行分无所谓TP是2还是4
                        # 当前块的行数
                        this_chunk_height = base_height + (1 if i < remainder else 0)
                        big_tile = Tile(0, y_start, Q, this_chunk_height)
                        partition_tiles.append(big_tile)
                        y_start += this_chunk_height

                    # 2) 对每个大块做行切分
                    for t in partition_tiles:
                        if t.height <= output_tile_size:
                            output_layer_tiles.append(t)
                        else:
                            y_curr = t.y
                            remaining_height = t.height
                            while remaining_height > 0:
                                sub_h = min(remaining_height, output_tile_size)
                                sub_tile = Tile(0, y_curr, Q, sub_h)
                                output_layer_tiles.append(sub_tile)

                                y_curr += sub_h
                                remaining_height -= sub_h
            # Store the tiles for the last layer
            if update_self:
                last_layer.tiles = output_layer_tiles
            result[last_layer.name] = output_layer_tiles
            
            # Now work backwards from the last layer to the first layer
            for i in range(num_layers - 2, -1, -1):
                current_layer = self.layers[i]
                next_layer = self.layers[i + 1]
                
                # Scale input tiles based on potential pooling layers
                current_layer_tiles = scale_input_tiles_for_pooling(
                    current_layer, next_layer, result[next_layer.name]
                )
                
                # Store the tiles for the current layer
                if update_self:
                    current_layer.tiles = current_layer_tiles
                result[current_layer.name] = current_layer_tiles
        
        return result
    '''
    
    def calculate_tiles(self, tensor_parallelism=None, output_tile_size=None, update_self=True):
        """
        简化的tile计算方法，只对指定维度进行切分：
        - CNN层：对channel维度(M)进行切分
        - Transformer层：根据tiling_dimension进行切分（H、P）
        
        Args:
            tensor_parallelism: 如果提供，覆盖self.tensor_parallelism
            output_tile_size: 如果提供，覆盖self.output_tile_size
            update_self: 是否更新fusion group中的tiles
            
        Returns:
            字典，映射层名到切分信息列表
        """
        if not self.output_tile_size:
            raise ValueError("Output tile size must be set before calculating tiles")
        if not output_tile_size:
            output_tile_size = self.output_tile_size
        if not tensor_parallelism:
            tensor_parallelism = self.tensor_parallelism
            
        result = {}
        num_layers = len(self.layers)
        
        # 处理每个层
        for i in range(num_layers):
            layer = self.layers[i]
            layer_tiles = []
            
            # 获取切分维度
            tiling_dim = layer.tiling_dimension
            
            # 获取该维度的总大小
            if tiling_dim in layer.instance:
                total_size = layer.instance[tiling_dim]
            else:
                # 如果维度不存在，跳过切分
                layer_tiles.append({"dimension": tiling_dim, "start": 0, "end": 1, "size": 1})
                result[layer.name] = layer_tiles
                continue
            
            # 根据tensor parallelism进行切分
            if tensor_parallelism == 1:
                # 无并行，创建单个tile
                layer_tiles.append({
                    "dimension": tiling_dim,
                    "start": 0,
                    "end": total_size,
                    "size": total_size
                })
            else:
                # 有并行，进行切分
                base_size = total_size // tensor_parallelism
                remainder = total_size % tensor_parallelism
                
                start = 0
                for tp_idx in range(tensor_parallelism):
                    # 当前块的大小（考虑余数）
                    current_size = base_size + (1 if tp_idx < remainder else 0)
                    
                    # 如果当前块太大，进一步切分
                    if current_size > output_tile_size:
                        # 按output_tile_size切分
                        remaining_size = current_size
                        current_start = start
                        
                        while remaining_size > 0:
                            tile_size = min(remaining_size, output_tile_size)
                            layer_tiles.append({
                                "dimension": tiling_dim,
                                "start": current_start,
                                "end": current_start + tile_size,
                                "size": tile_size
                            })
                            current_start += tile_size
                            remaining_size -= tile_size
                    else:
                        # 当前块大小合适
                        layer_tiles.append({
                            "dimension": tiling_dim,
                            "start": start,
                            "end": start + current_size,
                            "size": current_size
                        })
                    
                    start += current_size
            
            # 存储结果
            if update_self:
                layer.tiles = layer_tiles
            result[layer.name] = layer_tiles
        
        return result
    
    def __str__(self):
        layer_names = [layer.name for layer in self.layers]
        return f"FusionGroup(layers={layer_names}parallelism={self.tensor_parallelism}tile_size={self.output_tile_size})"
    
    def __repr__(self):
        return self.__str__()


class PhysicalNetwork:
    """Represents a physical network with fusion groups and tiling."""
    def __init__(self, virtual_network=None):
        self.virtual_network = virtual_network
        self.fusion_groups = []
    
    def add_fusion_group(self, group):
        """Add a fusion group to the physical network."""
        self.fusion_groups.append(group)
    '''
    注释
    def get_all_tiles(self):
        """
        Get all tiles for all layers in all fusion groups.
        
        Returns:
            A dictionary mapping layer names to lists of tiles
        """
        all_tiles = {}
        
        for group in self.fusion_groups:
            group_tiles = group.calculate_tiles()
            all_tiles.update(group_tiles)
        
        return all_tiles
    '''
    def __str__(self):
        return f"PhysicalNetwork(fusion_groups={len(self.fusion_groups)})"
    
    def __repr__(self):
        return self.__str__()


class VirtualNetwork:
    """
    A data structure that stores and manages layers of a neural network.
    """
    def __init__(self, network_name="unknown", is_attn=False, batch_size=1, sequence_length=1):
        self.network_name = network_name
        self.layers = []  # List of LayerConfig objects
        self.layer_dict = {}  # Dictionary mapping layer names to LayerConfig objects
        self.is_attn = is_attn
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.moe_config = None  # Populated by load_from_dir if NETWORK.yaml has moe section
    
    def get_unique_name(self):
        return f"{self.network_name}_b{self.batch_size}_seq{self.sequence_length}"

    def add_layer(self, layer):
        """Add a layer to the network."""
        if not isinstance(layer, LayerConfig):
            raise TypeError("Layer must be a LayerConfig object")
        
        self.layers.append(layer)
        if layer.name:
            self.layer_dict[layer.name] = layer
    
    def load_from_dir(self, directory_path, db_layers=None):
        """
        Load all YAML files from a directory and add them as layers.
        Files are sorted by the layer number in the filename (if present).
        If db_layers is provided, only load layers whose names are in that set.
        """

        # Get all YAML files in the directory (exclude metadata files like NETWORK.yaml)
        yaml_files = [f for f in os.listdir(directory_path) if f.endswith('.yaml') and f != 'NETWORK.yaml']

        # Skip legacy softmax sub-op YAMLs if fused layer0_softmax.yaml exists
        _has_fused_softmax = 'layer0_softmax.yaml' in yaml_files
        if _has_fused_softmax:
            _sub_op_suffixes = ('_softmax_max.yaml', '_softmax_sub_exp.yaml',
                                '_softmax_sum.yaml', '_softmax_div.yaml')
            yaml_files = [f for f in yaml_files if not f.endswith(_sub_op_suffixes)]

        # Filter to only layers present in the database
        if db_layers is not None:
            yaml_files = [f for f in yaml_files if os.path.splitext(f)[0] in db_layers]
        
        # Extract layer numbers if present in filenames
        def get_layer_number(filename):
            # Extract number after layer like layer10
            match = re.search(r'^layer(\d+)', filename)
            if match:
                return int(match.group(1))
            return float('inf')  # Place files without layer numbers at the end
        
        # Sort files by layer number
        yaml_files.sort(key=get_layer_number)
        
        # Process each file
        for filename in yaml_files:
            layername = get_layer_name_from_yaml(filename)
            file_path = os.path.join(directory_path, filename)
            try:
                if utility_functions.is_softmax_layers(layername) or utility_functions.is_projection_layers(layername) or utility_functions.is_attention_layers(layername) or utility_functions.is_gemm_layer(layername):
                    is_attn = True
                else:
                    is_attn = False
                layer = LayerConfig(file_path, is_attn=is_attn, is_padding=(self.network_name == "stable_diffusion"), batch_size=self.batch_size, sequence_length=self.sequence_length)
                self.add_layer(layer)
                #print(f"Added layer: {layer.name} (Type: {layer.layer_type})")
            except Exception as e:
                print(f"Error loading layer from {filename}: {e}")

        # Parse MoE config from NETWORK.yaml if present
        network_yaml_path = os.path.join(directory_path, 'NETWORK.yaml')
        if os.path.exists(network_yaml_path):
            try:
                import yaml
                with open(network_yaml_path) as f:
                    net_desc = yaml.safe_load(f)
                if net_desc and 'architecture' in net_desc and 'moe' in net_desc.get('architecture', {}):
                    moe_section = net_desc['architecture']['moe']
                    self.moe_config = {
                        'num_experts': int(moe_section['num_experts']),
                        'num_experts_per_tok': int(moe_section['num_experts_per_tok']),
                        'moe_intermediate_size': int(moe_section['moe_intermediate_size']),
                        'hidden_size': int(net_desc['architecture'].get('hidden_size', 0)),
                    }
            except Exception as e:
                pass  # Not MoE or malformed NETWORK.yaml
        
    
    def print_network_summary(self):
        """Print a summary of the network architecture."""
        print(f"\nNetwork: {self.network_name}")
        print(f"Total layers: {len(self.layers)}")
        print("\nLayer details:")
        print(f"{'#':<3} {'Name':<15} {'Type':<10} {'Input Shape':<15} {'Output Shape':<15} {'Parameters':<15}")
        print("-" * 70)
        
        for i, layer in enumerate(self.layers):
            # Calculate input shape
            input_shape = layer.calculate_input_shape()
            
            # Estimate parameters (very rough approximation)
            params = 0
            if layer.layer_type == "conv":
                params = layer.instance['C'] * layer.instance['M'] * layer.instance['R'] * layer.instance['S']
            elif layer.layer_type == "fc":
                # For fully connected, input is flattened
                input_size = layer.instance['C'] * input_shape[0] * input_shape[1]
                params = input_size * layer.instance['M']
            
            # Print layer info
            print(f"{i:<3} {layer.name:<15} {layer.layer_type:<10} "
                  f"({layer.instance['C']},{input_shape[0]},{input_shape[1]})  "
                  f"({layer.instance['M']},{layer.instance['P']},{layer.instance['Q']})  "
                  f"{params:,}")
    
    def visualize_network(self):
        """
        Generate a simple ASCII visualization of the network architecture.
        """
        print(f"\nNetwork Architecture: {self.network_name}")
        print("=" * 50)
        
        for i, layer in enumerate(self.layers):
            input_shape = layer.calculate_input_shape()
            # Determine layer type symbol
            if layer.layer_type == "conv":
                symbol = "[ Conv ]"
            elif layer.layer_type == "pool":
                symbol = "[ Pool ]"
            elif layer.layer_type == "fc":
                symbol = "[ FC   ]"
            else:
                symbol = "[ ???? ]"
            
            # Print layer with connections
            if i > 0:
                print("     |")
                print("     V")
            
            print(f"{symbol} {layer.name} - {layer.instance['C']}x{input_shape[0] if 'H' in layer.instance else '?'}x{input_shape[1] if 'W' in layer.instance else '?'} → {layer.instance['M']}x{layer.instance['P']}x{layer.instance['Q']}")
        
        print("=" * 50)
        
    def visualize_physical_network(self, physical_network):
        """
        Generate an ASCII visualization of the physical network with fusion groups.
        
        Args:
            physical_network: A PhysicalNetwork object
        """
        print(f"\nPhysical Network Architecture: {self.network_name}")
        print("=" * 60)
        
        for group_idx, group in enumerate(physical_network.fusion_groups):
            print(f"Fusion Group {group_idx+1}: Parallelism={group.tensor_parallelism}, Tile Size={group.output_tile_size}")
            print("-" * 40)
            
            for layer_idx, layer in enumerate(group.layers):
                input_shape = layer.calculate_input_shape()
                # Determine layer type symbol
                if layer.layer_type == "conv":
                    symbol = "[ Conv ]"
                elif layer.layer_type == "pool":
                    symbol = "[ Pool ]"
                elif layer.layer_type == "fc":
                    symbol = "[ FC   ]"
                else:
                    symbol = "[ ???? ]"
                
                # Print layer with connections
                if layer_idx > 0:
                    print("     |")
                    print("     V")
                if not self.is_attn:
                    print(
                        f"{symbol} {layer.name} - {layer.instance['C']}x{input_shape[0]}x{input_shape[1]} → {layer.instance['M']}x{layer.instance['P']}x{layer.instance['Q']}")
                else:
                    print(f"{symbol} {layer.name} - {layer.instance['C']}x{input_shape[0]}x{input_shape[1]} → {layer.instance['M']}x{layer.instance['P']}x{layer.instance['Q']}")
                
                # Print tile information
                if layer.tiles:
                    print(f"     Tiles: {len(layer.tiles)}")
                    # Print first few tiles as examples
                    for i, tile in enumerate(layer.tiles[:3]):
                        print(f"       Tile {i+1}: ({tile.x}, {tile.y}) → ({tile.x + tile.width}, {tile.y + tile.height})")
                    if len(layer.tiles) > 3:
                        print(f"       ... {len(layer.tiles)-3} more tiles")
            
            print("-" * 40)
            if group_idx < len(physical_network.fusion_groups) - 1:
                print("     ||")
                print("     vv")
        
        print("=" * 60)
    
    def __str__(self):
        return f"VirtualNetwork(name={self.network_name}, layers={len(self.layers)})"
        
def create_fusion_groups_from_binary(binary_string, layers):
    """
    Convert a binary string representation into fusion groups.
    
    Args:
        binary_string: String of 0s and 1s where 1 marks the start of a fusion group
        layers: List of layers to be grouped
        
    Returns:
        List of lists, where each inner list contains the layers for one fusion group
    """
    if not binary_string.startswith('1'):
        binary_string = '1' + binary_string[1:]  # Ensure the first group starts at the beginning
        
    fusion_groups = []
    current_group = []
    
    for i, bit in enumerate(binary_string):
        if i < len(layers):
            if bit == '1' and i > 0:
                # Start of a new group, store the previous one
                if current_group:
                    fusion_groups.append(current_group)
                current_group = [i]
            elif bit == '1' and i == 0:
                # First group starts
                current_group = [i]
            else:
                # Continue current group
                current_group.append(i)
    
    # Add the last group
    if current_group:
        fusion_groups.append(current_group)
        
    return fusion_groups

def create_physical_network_from_gene_binary_string(virtual_network, binary_string):
    """
    Create a physical network from a gene using the new encoding.
    
    Args:
        virtual_network: The virtual network to partition
        binary_string: Dict with binary_string (fusion) in gene
        
    Returns:
        PhysicalNetwork object with fusion groups based on the gene
    """
    
    # Create fusion groups from binary string
    fusion_group_indices = create_fusion_groups_from_binary(binary_string, virtual_network.layers)
    
    # Create physical network
    physical_network = PhysicalNetwork(virtual_network)
    
    # For each fusion group, create a FusionGroup object
    for group_indices in fusion_group_indices:
        # Get parameters from first layer in the group

        # TODO: 把组内第一个层的type当整个组type，如果想融合CNN和Attn，则这里需要修改，若分开融合，则不需要
        group_is_attn = virtual_network.layers[group_indices[0]].is_attn
        # Create new fusion group
        fusion_group = FusionGroup(is_attn=group_is_attn)

        
        # Add layers to the fusion group
        for idx in group_indices:
            if idx < len(virtual_network.layers):
                fusion_group.add_layer(virtual_network.layers[idx])
        
        # Add fusion group to physical network
        physical_network.add_fusion_group(fusion_group)
    
    return physical_network

def create_physical_network_from_gene(virtual_network, gene):
    """
    Create a physical network from a gene using the new encoding.
    
    Args:
        virtual_network: The virtual network to partition
        gene: Dict with binary_string, parallelism, and tile_size
        
    Returns:
        PhysicalNetwork object with fusion groups based on the gene
    """
    binary_string = gene['binary_string']

    return create_physical_network_from_gene_binary_string(virtual_network=virtual_network,binary_string=binary_string)

class LayerParser:
    """This class is used to parse all layers and identify layers that are functionally identical."""

    def __init__(self, nets: List[str], net_layers_dict: Dict[str, List[str]], base_dir: str, transformer: bool):
        self.nets = nets
        self.net_layers_dict = net_layers_dict
        self.base_dir = base_dir
        self.all_layers: Dict[str, Any] = {}
        self.transformer = transformer
    def parse_all_layers(self):
        """
        Parse all layers in all networks, load their configurations, and identify identical layers.
        """
        #print("Step 1: Starting to load configurations for all layers...")
        # Step 1: Load all layers
        for net in self.nets:
            for layer_path in self.net_layers_dict[net]:
                full_path = os.path.join(self.base_dir, layer_path)
                layer_key = f"{net}@{get_layer_name_from_yaml(full_path)}"
                layer_config = LayerConfig(full_path)

                if not layer_config.instance: # Skip if loading fails
                    continue

                self.all_layers[layer_key] = {
                    "layer_config": layer_config,
                    "yaml_path": full_path,
                    "flag": False,          # True if it's a duplicate
                    "master_layer_key": None # Points to its master layer
                }
        #print(f"Loading completed, found {len(self.all_layers)} layers in total.")

        #print("\nStep 2: Starting to identify identical layers...")
        # Step 2: Identify identical layers
        if self.transformer:
            layer_keys = sorted(self.all_layers.keys(), key=lambda k: (0 if k.endswith("_q") else 1, k))
        else:
            layer_keys = list(self.all_layers.keys())
        for i, key1 in enumerate(layer_keys):
            if self.all_layers[key1]["flag"]:
                continue # Skip if this layer has been marked as a duplicate

            layer1_config = self.all_layers[key1]["layer_config"]

            for key2 in layer_keys[i+1:]:
                if self.all_layers[key2]["flag"]:
                    continue # Don't compare with already marked duplicates

                layer2_config = self.all_layers[key2]["layer_config"]

                if self.is_layer_identical(layer1_config, layer2_config):
                    #print(f"  - Found identical layers: '{key2}' is identical to '{key1}'.")
                    if key2.endswith("_q"):
                        self.all_layers[key1]["flag"] = True
                        self.all_layers[key1]["master_layer_key"] = key2
                        for key in self.all_layers.keys():
                            if self.all_layers[key]["master_layer_key"] == key1:
                                self.all_layers[key]["master_layer_key"] = key2
                    elif key2.endswith("_sn") or key2.endswith("_sd"):
                        self.all_layers[key1]["flag"] = True
                        self.all_layers[key1]["master_layer_key"] = key2
                    else:    
                        self.all_layers[key2]["flag"] = True
                        self.all_layers[key2]["master_layer_key"] = key1        
        
        # Print summary
        total_layers = len(self.all_layers)
        unique_layers = sum(1 for info in self.all_layers.values() if not info["flag"])
        duplicate_layers = total_layers - unique_layers
        
        print(f"\nLayer Parsing Summary:")
        print(f"Total layers: {total_layers}")
        print(f"Unique layers: {unique_layers}")
        print(f"Duplicate layers: {duplicate_layers} ({duplicate_layers/total_layers*100:.1f}%)")
            
        return self.all_layers

    def _flatten_projection_symbols(self, nested_list: list, symbols: list):
        """
        Recursively traverse nested lists and collect all symbol names into the 'symbols' list.
        """
        for item in nested_list:
            if isinstance(item, list):
                self._flatten_projection_symbols(item, symbols)
            elif isinstance(item, str):
                symbols.append(item.strip('-'))

    def _get_resolved_shapes(self, problem_data: Dict[str, Any]) -> Dict[str, List[int]]:
        """
        Resolve symbolic data_spaces into numerical shape dictionary.
        Can handle nested projection formats and coefficients.
        """
        instance = problem_data.get("instance", {})
        shape_info = problem_data.get("shape", {})
        coefficients = {coeff['name']: coeff['default'] for coeff in shape_info.get("coefficients", [])}
        data_spaces = shape_info.get("data_spaces", [])
        
        lookup_table = {**coefficients, **instance}
        
        resolved_shapes = {}
        for space in data_spaces:
            name = space['name']
            numerical_dims = []
            
            for dim_projection in space.get('projection', []):
                symbols_for_dim = []
                self._flatten_projection_symbols(dim_projection, symbols_for_dim)
                
                try:
                    values = [lookup_table[symbol] for symbol in symbols_for_dim]
                    dim_size = math.prod(values)
                    numerical_dims.append(dim_size)
                except KeyError as e:
                    print(f"Warning: Symbol {e} definition not found in data_space '{name}'.")
                    return {}
            
            resolved_shapes[name] = numerical_dims
        return resolved_shapes

    def is_layer_identical(self, layer1_config: LayerConfig, layer2_config: LayerConfig) -> bool:
        """
        Check if layers are identical by comparing resolved numerical shapes and instance parameters.
        """
        if layer1_config.instance != layer2_config.instance:
            return False

        try:
            resolved_shapes1 = self._get_resolved_shapes(layer1_config.problem_data)
            resolved_shapes2 = self._get_resolved_shapes(layer2_config.problem_data)

            if not resolved_shapes1 or not resolved_shapes2 or resolved_shapes1 != resolved_shapes2:
                return False

        except Exception as e:
            print(f"Error occurred while comparing layers: {e}")
            return False
            
        return True
    
    def get_all_layers(self):
        """Return the layer data structure."""
        return self.all_layers

    def get_unique_layers(self):
        """Return the layer data structure."""
        new_structure = {}
        
        for layer_key, layer_data in self.all_layers.items():
            # Include all layers (consider every layer as unique)
            if layer_data["flag"] == False:
                new_structure[layer_key] = {
                    "layer_config": layer_data["layer_config"],
                    "yaml_path": layer_data["yaml_path"],
                    "flag": False,
                    "master_layer_key": layer_data["master_layer_key"]
                }
        
        return new_structure
    
    def get_layer_statistics(self):
        """Return statistics about the layers."""
        total_layers = len(self.all_layers)
        unique_layers = sum(1 for info in self.all_layers.values() if not info["flag"])
        duplicate_layers = total_layers - unique_layers
        
        return {
            "total_layers": total_layers,
            "unique_layers": unique_layers,
            "duplicate_layers": duplicate_layers,
            "duplicate_percentage": duplicate_layers/total_layers*100 if total_layers > 0 else 0
        }


def test_calculate_input_shape():
    """Test the calculate_input_shape method"""
    print("=== Testing calculate_input_shape method ===")
    
    # Test CNN layer
    cnn_layer = LayerConfig("", is_attn=False)
    cnn_layer.name = "conv1"
    cnn_layer.instance = {
        'P': 112, 'Q': 112, 'R': 7, 'S': 7,
        'Hstride': 2, 'Wstride': 2, 'Hdilation': 1, 'Wdilation': 1
    }
    input_shape = cnn_layer.calculate_input_shape()
    print(f"CNN layer {cnn_layer.name}: output shape({cnn_layer.instance['P']}, {cnn_layer.instance['Q']}) -> input shape{input_shape}")
    
    # Test Attention layer (QK)
    qk_layer = LayerConfig("", is_attn=True)
    qk_layer.name = "layer1_qk"
    qk_layer.instance = {'E': 128, 'M': 2048, 'P': 1}
    print(f"QK layer is_attention_layer(): {qk_layer.is_attention_layer()}")
    input_shape = qk_layer.calculate_input_shape()
    print(f"Attention layer {qk_layer.name}: output shape({qk_layer.instance['P']}, {qk_layer.instance['M']}) -> input shape{input_shape}")
    
    # Test Attention layer (AV)
    av_layer = LayerConfig("", is_attn=True)
    av_layer.name = "layer6_av"
    av_layer.instance = {'E': 128, 'M': 2048, 'P': 1, 'F': 128}
    print(f"AV layer is_attention_layer(): {av_layer.is_attention_layer()}")
    input_shape = av_layer.calculate_input_shape()
    print(f"Attention layer {av_layer.name}: output shape({av_layer.instance['P']}, {av_layer.instance['F']}) -> input shape{input_shape}")
    
    # Test Projection layer
    proj_layer = LayerConfig("", is_attn=True)
    proj_layer.name = "q"
    proj_layer.instance = {'O': 2048, 'P': 1, 'D': 2048}
    print(f"Projection layer '{proj_layer.name}' is_attention_layer(): {proj_layer.is_attention_layer()}")
    print(f"Projection layer name.lower(): '{proj_layer.name.lower()}'")
    input_shape = proj_layer.calculate_input_shape()
    print(f"Projection layer {proj_layer.name}: output shape({proj_layer.instance['P']}, {proj_layer.instance['D']}) -> input shape{input_shape}")
    
    # Test FFN layer
    ffn_layer = LayerConfig("", is_attn=True)
    ffn_layer.name = "ffn1"
    ffn_layer.instance = {'O': 2048, 'P': 1, 'I': 8192}
    print(f"FFN layer '{ffn_layer.name}' is_attention_layer(): {ffn_layer.is_attention_layer()}")
    print(f"FFN layer name.lower(): '{ffn_layer.name.lower()}'")
    input_shape = ffn_layer.calculate_input_shape()
    print(f"FFN layer {ffn_layer.name}: output shape({ffn_layer.instance['P']}, {ffn_layer.instance['I']}) -> input shape{input_shape}")

# Example usage
if __name__ == "__main__":
    '''
    # Create a simple network
    network = VirtualNetwork("alex")
    network.load_from_dir(os.path.join(NET_DIR,"alexnet"))

    gene = {
        'binary_string':'10011111',
        'parallelism': [1,1,1,4,2,2,2,1],
        'tile_size':[8,8,8,4,8,8,8,1]
    }

    physical_network = create_physical_network_from_gene(network, gene)
    print("\nPhysical Network with random fusion pattern:")
    network.visualize_physical_network(physical_network)
    verify_all_layer_coverage(physical_network)
    '''
    test_calculate_input_shape()