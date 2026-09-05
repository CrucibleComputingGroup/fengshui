import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
import numpy as np
from matplotlib.gridspec import GridSpec
from collections import OrderedDict

def parse_function_id(function_id):
    """
    Parse a function ID like 'simba_likeglb1.0pe_x1.0pe_y2.0@2@32@0'
    Returns (arch_type, glb_scale, pe_x_scale, pe_y_scale, parallel_count)
    
    In this format:
    - The first number after @ is the parallel count (2 in the example)
    - The stage number is determined by the position in the function list
    """
    # Extract the architecture part first
    parts = function_id.split('@')
    if len(parts) < 2:
        print(f"Warning: Invalid function ID format: {function_id}")
        return None, None, None, None, 1
    
    arch_part = parts[0]  # e.g., 'simba_likeglb1.0pe_x1.0pe_y2.0'
    
    # Parse the architecture details
    arch_match = re.match(r'(\w+_like)glb([\d\.]+)pe_x([\d\.]+)pe_y([\d\.]+)', arch_part)
    if not arch_match:
        print(f"Warning: Could not parse architecture from: {arch_part}")
        return None, None, None, None, 1
    
    arch_type = arch_match.group(1)  # e.g., 'simba_like'
    glb_scale = float(arch_match.group(2))
    pe_x_scale = float(arch_match.group(3))
    pe_y_scale = float(arch_match.group(4))
    
    # Get the parallel count (first number after @)
    try:
        parallel_count = int(parts[1])
    except (ValueError, IndexError):
        print(f"Warning: Invalid parallel count in: {function_id}, using default of 1")
        parallel_count = 1
    
    return arch_type, glb_scale, pe_x_scale, pe_y_scale, parallel_count

def extract_net_config_from_string(config_string):
    """
    Extract network configuration from the config string in the CSV data
    
    Parameters:
    - config_string: String representing the configuration data
    
    Returns:
    - Dictionary with 'functions' key containing parsed Function objects
    """
    # Define a simple Function class to match the data
    class Function:
        def __init__(self, x1, a, b, id):
            self.x1 = x1
            self.a = a
            self.b = b
            self.id = id
        
        def __repr__(self):
            return f"Function(x1={self.x1}, a={self.a}, b={self.b}, id='{self.id}')"
    
    # Parse the functions string
    functions = []
    
    # Use regex to extract Function patterns - flexible to handle different formats
    pattern = r"Function\s*\(\s*x1\s*=\s*(?:np\.float64\s*\()?([\d\.\-e]+)(?:\))?\s*,\s*a\s*=\s*(?:np\.float64\s*\()?([\d\.\-e]+)(?:\))?\s*,\s*b\s*=\s*(?:(?:np\.float64\s*\()?([\d\.\-e]+)(?:\))?|[\d\.\-e]+)\s*,\s*id\s*=\s*['\"]([^'\"]+)['\"]\s*\)"
    
    matches = re.finditer(pattern, config_string)
    
    for match in matches:
        x1 = float(match.group(1))
        a = float(match.group(2))
        try:
            b = float(match.group(3))
        except ValueError:
            # In case b is not correctly matched as a float
            b = 0.0
        id = match.group(4)
        functions.append(Function(x1=x1, a=a, b=b, id=id))
    
    return {'functions': functions}

def combine_similar_stages(stages):
    """
    Combine stages with the same architecture into one stage with a count.
    
    Parameters:
    - stages: List of stage dictionaries
    
    Returns:
    - List of combined stages
    """
    # Group stages by architecture
    grouped_stages = {}
    for stage in stages:
        arch_key = stage['arch_key']
        if arch_key in grouped_stages:
            grouped_stages[arch_key]['count'] += 1
        else:
            grouped_stages[arch_key] = stage.copy()
    
    # Convert back to list
    return list(grouped_stages.values())

def visualize_accelerator_architecture(config_string, output_file='accelerator_architecture.png'):
    """
    Visualize the accelerator architecture based on the configuration string.
    Each function in the list represents a sequential stage in the pipeline.
    The parallel count is determined by the first number after the @ symbol.
    
    Parameters:
    - config_string: Configuration string containing function definitions
    - output_file: Filename to save the visualization
    """
    # Parse the configuration string
    net_config = extract_net_config_from_string(config_string)
    
    if 'functions' not in net_config or not net_config['functions']:
        raise ValueError("No functions found in configuration")
    
    # Extract stages with their architecture details - keep original order
    stages = []
    for i, func in enumerate(net_config['functions']):
        # Parse the architecture details from the function ID
        arch_type, glb_scale, pe_x_scale, pe_y_scale, parallel_count = parse_function_id(func.id)
        
        if arch_type is None:
            print(f"Warning: Skipping function with invalid ID: {func.id}")
            continue
            
        # Create a key for the architecture type
        arch_key = f"{arch_type}glb{glb_scale}pe_x{pe_x_scale}pe_y{pe_y_scale}"
        
        # Add to stages list with stage number based on position
        stages.append({
            'arch_type': arch_type,
            'glb_scale': glb_scale,
            'pe_x_scale': pe_x_scale,
            'pe_y_scale': pe_y_scale,
            'arch_key': arch_key,
            'parallel_count': parallel_count,
            'stage_num': i + 1  # Stage number is position in list
        })
    
    # Create the visualization
    fig = plt.figure(figsize=(16, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Colors for different architecture types
    color_map = {
        'eyeriss': 'lightblue',
        'simba': 'lightgreen',
        'simple': 'lightsalmon'
    }
    
    # Calculate layout dimensions
    num_stages = len(stages)
    stage_width = min(0.8 / (2 * num_stages), 0.08)  # Width of each stage
    buffer_width = stage_width * 0.4  # Width of buffers
    stage_spacing = stage_width * 0.2  # Spacing between elements
    
    # Calculate total width needed
    total_width = 0
    for stage in stages:
        # Add space for this stage and a buffer (if not the last stage)
        total_width += stage_width + (buffer_width + stage_spacing if stage != stages[-1] else 0)
    
    # Starting position
    start_x = (1.0 - total_width) / 2
    
    # Input block
    input_width = stage_width * 0.5
    input_x = start_x - input_width - stage_spacing
    input_box = plt.Rectangle((input_x, 0.4), input_width, 0.2, 
                            facecolor='lightgray', 
                            edgecolor='black', linewidth=1)
    ax.add_patch(input_box)
    ax.text(input_x + input_width/2, 0.5, 'Input', 
            ha='center', va='center', fontsize=10)
    
    # Draw arrow from input to first stage
    ax.arrow(input_x + input_width, 0.5, stage_spacing, 0,
            head_width=0.02, head_length=0.01, fc='black', ec='black')
    
    # Current x position for drawing
    current_x = start_x
    
    # Draw stages and buffers in the exact order from the functions list
    for i, stage in enumerate(stages):
        arch_type = stage['arch_type']
        # Determine color - use first part of architecture name to match color_map
        for color_key in color_map:
            if color_key in arch_type.lower():
                color = color_map[color_key]
                break
        else:
            color = 'white'  # Default color if no match
            
        parallel_count = stage['parallel_count']
        
        # Format architecture name for display
        glb_scale = stage['glb_scale']
        pe_x_scale = stage['pe_x_scale']
        pe_y_scale = stage['pe_y_scale']
        
        # Create a shorter display text that will fit in blocks
        display_text = f"{arch_type}\nglb{glb_scale}\npe_x{pe_x_scale}\npe_y{pe_y_scale}"
        
        # Draw multiple parallel blocks if parallel_count > 1
        if parallel_count > 1:
            # Calculate block height to fit all parallel instances
            block_height = 0.5 / parallel_count
            start_y = 0.25 + (0.5 - (block_height * parallel_count)) / 2
            
            # Draw each parallel block
            for j in range(parallel_count):
                y_pos = start_y + j * block_height
                block = plt.Rectangle((current_x, y_pos), stage_width, block_height * 0.9, 
                                      facecolor=color, 
                                      edgecolor='black', linewidth=1, alpha=0.8)
                ax.add_patch(block)
                
                # Add text to each block - make font smaller to fit
                ax.text(current_x + stage_width/2, y_pos + block_height * 0.45, 
                        display_text, ha='center', va='center', fontsize=5)
        else:
            # Draw a single block for this stage
            block = plt.Rectangle((current_x, 0.3), stage_width, 0.4, 
                                 facecolor=color, 
                                 edgecolor='black', linewidth=1, alpha=0.8)
            ax.add_patch(block)
            
            # Add text to the block
            ax.text(current_x + stage_width/2, 0.5, display_text, 
                    ha='center', va='center', fontsize=6)
        
        # Move to next position
        current_x += stage_width
        
        # Add buffer between stages (except after the last stage)
        if i < len(stages) - 1:
            buffer_box = plt.Rectangle((current_x, 0.4), buffer_width, 0.2, 
                                      facecolor='lightyellow', 
                                      edgecolor='black', linewidth=1)
            ax.add_patch(buffer_box)
            ax.text(current_x + buffer_width/2, 0.5, 'Buffer', 
                    ha='center', va='center', fontsize=7)
            
            # Draw arrow to next stage
            ax.arrow(current_x + buffer_width, 0.5, stage_spacing, 0,
                    head_width=0.02, head_length=0.01, fc='black', ec='black')
            
            current_x += buffer_width + stage_spacing
    
    # Output block
    output_width = stage_width * 0.5
    output_x = current_x
    output_box = plt.Rectangle((output_x, 0.4), output_width, 0.2, 
                             facecolor='lightgray', 
                             edgecolor='black', linewidth=1)
    ax.add_patch(output_box)
    ax.text(output_x + output_width/2, 0.5, 'Output', 
            ha='center', va='center', fontsize=10)
    
    # Draw arrow from last stage to output
    ax.arrow(output_x - stage_spacing, 0.5, stage_spacing, 0,
            head_width=0.02, head_length=0.01, fc='black', ec='black')
    
    # Set axis limits with some padding
    ax.set_xlim(input_x - 0.05, output_x + output_width + 0.05)
    ax.set_ylim(0.1, 0.9)
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return fig

def extract_config_from_csv(csv_file, n_chiplets, network_name):
    """
    Extract the configuration string for a specific network and chiplet count
    from a CSV file.
    
    Parameters:
    - csv_file: Path to the CSV file
    - n_chiplets: Number of chiplets
    - network_name: Name of the network (e.g., 'alexnet', 'googlenet')
    
    Returns:
    - Configuration string
    """
    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)
            
            # Find the network configuration column
            config_col = None
            for i, col in enumerate(header):
                if network_name.lower() in col.lower() and 'config' in col.lower():
                    config_col = i
                    break
            
            if config_col is None:
                print("Available columns:", header)
                raise ValueError(f"No configuration column found for network: {network_name}")
            
            # Find the row with the specified number of chiplets
            for row in reader:
                if len(row) > 0 and row[0].strip() == str(n_chiplets):
                    return row[config_col]
            
            raise ValueError(f"No data found for n_chiplets={n_chiplets}")
    
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file '{csv_file}' not found")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize accelerator architecture from CSV data.')
    parser.add_argument('csv_file', help='Path to the CSV file containing configurations')
    parser.add_argument('n_chiplets', type=int, help='Number of chiplets to use')
    parser.add_argument('network_name', help='Network name (e.g., alexnet, googlenet)')
    parser.add_argument('--output', '-o', help='Output file name', default=None)
    parser.add_argument('--config', '-c', help='Use this configuration string directly instead of reading from CSV')
    
    args = parser.parse_args()
    
    try:
        if args.config:
            config_string = args.config
        else:
            config_string = extract_config_from_csv(args.csv_file, args.n_chiplets, args.network_name)
        
        print(f"Configuration string length: {len(config_string)}")
        print(f"First 100 characters: {config_string[:100]}...")
        
        output_file = args.output
        if output_file is None:
            output_file = f'accelerator_{args.network_name}_{args.n_chiplets}.png'
        
        visualize_accelerator_architecture(config_string, output_file)
        print(f"Visualization saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)