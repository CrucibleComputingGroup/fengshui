import pandas as pd
import math
import os
import sys
# Add the parent directory of the current directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from get_cost import calculate_die_cost
from global_parameter import *

def find_min_cache_size(mem):
    """
    Find the minimum cache sizes needed for input and output memory requirements separately.
    
    Parameters:
    in_mem (float): Input memory size in GB
    
    Returns:
    tuple: (in_cache_row, out_cache_row) - rows from the DataFrame representing the minimum 
           suitable cache sizes, or None for either if no cache is large enough
    """
    # Find the smallest cache for input memory
    in_suitable_caches = mem_specs_df[mem_specs_df['size_gb'] >= mem]
    in_cache = None if in_suitable_caches.empty else in_suitable_caches.iloc[0].to_dict()
    
    return in_cache

def find_min_cache_sizes(in_mem, out_mem):
    """
    Find the minimum cache sizes needed for input and output memory requirements separately.
    
    Parameters:
    in_mem (float): Input memory size in GB
    out_mem (float): Output memory size in GB
    cache_df (DataFrame): DataFrame containing memory configurations
    
    Returns:
    tuple: (in_cache_row, out_cache_row) - rows from the DataFrame representing the minimum 
           suitable cache sizes, or None for either if no cache is large enough
    """    
    return find_min_cache_size(in_mem), find_min_cache_size(out_mem)

# mem_specs = {
#     # TODO fix here
#     'LPDDR5': {
#         'shared_bw': 70.4,
#         'cost_per_GB': 2.31,
#         'area_per_GB': 29.3,
#         'ctrl_area': 0.07,
#         'phy_area': 7.5,
#         'leakage_power': 0.03,
#         'module_caps': [4]
#     },
#     'DDR5': {
#         'shared_bw': 70.4,
#         'cost_per_GB': 4.38,
#         'area_per_GB': 18.0,
#         'ctrl_area': 0.07,
#         'phy_area': 7.5,
#         'leakage_power': 0.025,
#         # https://datasheet.lcsc.com/lcsc/2204251615_Samsung-K4Z80325BC-HC14_C2920181.pdf
#         'module_caps': [2]
#     },
#     'GDDR7': {
#         'shared_bw': 320,
#         'cost_per_GB': 12.0, 
#         'area_per_GB': 84.0,
#         'ctrl_area': 0.07,
#         'phy_area': 8.0,
#         'leakage_power': 0.06,
#         # Graphics Double Data Rate 7 SGRAM Standard (GDDR7) 
#         'module_caps': [2]
#     },
#     'HBM3': {
#         'shared_bw': 819,
#         'cost_per_GB': 110.0,
#         'area_per_GB': 62.5,
#         'ctrl_area': 1.00,
#         'phy_area': 19.28,
#         'leakage_power': 0.04,
#         # High Bandwidth Memory (HBM3) DRAM
#         'module_caps': [8]
#     },
#     'HBM3E': {
#         'shared_bw': 1229,
#         'cost_per_GB': 171.9,
#         'area_per_GB': 41.7,
#         'ctrl_area': 1.00,
#         'phy_area': 19.28,
#         'leakage_power': 0.04,
#         'module_caps': [8]
#     }
# }
mem_specs = {
    'LPDDR5': {
        'shared_bw': 25.6,
        'cost_per_GB': 2.31,
        'area_per_GB': 29.3,
        'ctrl_area': 0.07,
        'phy_area': 7.5,
        'leakage_power': 0.03,
        'module_caps': [4]
        # https://en.wikipedia.org/wiki/LPDDR
        # https://semiconductor.samsung.com/dram/lpddr/lpddr5
        # https://www.synopsys.com/articles/key-features-about-lpddr5.html
    },
    'DDR5': {
        'shared_bw': 70.4,
        'cost_per_GB': 4.38,
        'area_per_GB': 18.0,
        'ctrl_area': 0.07,
        'phy_area': 7.5,
        'leakage_power': 0.025,
        'module_caps': [8]
        # https://en.wikipedia.org/wiki/DDR5_SDRAM
        # https://datasheet.lcsc.com/lcsc/2204251615_Samsung-K4Z80325BC-HC14_C2920181.pdf
        # https://www.crucial.com/articles/about-memory/everything-about-ddr5-ram
    },
    'GDDR7': {
        'shared_bw': 192,
        'cost_per_GB': 12.0, 
        'area_per_GB': 84.0,
        'ctrl_area': 0.07,
        'phy_area': 8.0,
        'leakage_power': 0.06,
        'module_caps': [2]
        # Graphics Double Data Rate 7 SGRAM Standard (GDDR7) 
        # https://hothardware.com/news/jedec-gddr7-spec-bandwidth-upgrade-next-gen-gpus
        # https://en.wikipedia.org/wiki/GDDR7_SDRAM
    },
    'HBM3': {
        'shared_bw': 819,
        'cost_per_GB': 110.0,
        'area_per_GB': 62.5,        # per-layer die area density (mm²/GB)
        'stack_layers': 12,          # 12-high 3D TSV stack
        'ctrl_area': 1.00,
        'phy_area': 19.28,
        'leakage_power': 0.04,
        'module_caps': [24]          # 24 GB per stack (12 layers × 2 GB/layer)
        # High Bandwidth Memory (HBM3) DRAM
        # https://www.rambus.com/blogs/hbm3-everything-you-need-to-know
        # https://www.mouser.com/new/micron-technology/micron-hbm3-gen2-memory
        # https://en.wikipedia.org/wiki/High_Bandwidth_Memory
    },
    'HBM3E': {
        'shared_bw': 1229,
        'cost_per_GB': 171.9,
        'area_per_GB': 41.7,        # per-layer die area density (mm²/GB)
        'stack_layers': 12,          # 12-high 3D TSV stack
        'ctrl_area': 1.00,
        'phy_area': 19.28,
        'leakage_power': 0.04,
        'module_caps': [48]          # 48 GB per stack (12 layers × 4 GB/layer)
        # https://www.rambus.com/blogs/hbm3-everything-you-need-to-know
        # https://en.wikipedia.org/wiki/High_Bandwidth_Memory
        # https://wccftech.com/nvidia-blackwell-gpu-architecture-official-208-billion-transistors-5x-ai-performance-192-gb-hbm3e-memory
    }
}

def get_memory_spec(mem, dram_type):
    # TODO
    # given a type of dram and the required size
    # return the dict objective containing attributes like cost, area 
    if dram_type not in mem_specs:
        raise KeyError(f"Unknown DRAM type: {dram_type}")
    spec = mem_specs[dram_type]

    if mem <= 0:
        return {
            'type': dram_type,
            'requested_gb': 0.0,
            'provisioned_gb': 0.0,
            'num_modules': 0,
            'module_size_gb': None,
            'stack_layers': spec.get('stack_layers', 1),
            'bandwidth_GBps': spec['shared_bw'],
            'cost': 0.0,
            'area': 0.0,
            'footprint_area': 0.0,
            'ctrl_area': spec['ctrl_area'],
            'phy_area': spec['phy_area'],
            'leakage_power': 0
        }

    best = None
    for caps in spec.get('module_caps', []) or [mem]:
        if caps <= 0:
            continue
        num = math.ceil(mem / caps)
        prov = num * caps
        cand = (prov, num, caps)
        if best is None or cand < best:
            best = cand

    provisioned_gB, num_modules, module_size = best

    # Total silicon area across all stacked layers (for cost calculation)
    total_silicon_area = float(provisioned_gB * spec['area_per_GB'])

    # Footprint area on interposer/package (accounts for 3D stacking)
    stack_layers = spec.get('stack_layers', 1)
    cap_per_layer = module_size / stack_layers
    footprint_per_module = cap_per_layer * spec['area_per_GB']
    footprint_area = footprint_per_module * num_modules

    return {
        'type': dram_type,
        'requested_gb': float(mem),
        'provisioned_gb': float(provisioned_gB),
        'num_modules': int(num_modules),
        'module_size_gb': float(module_size),
        'stack_layers': stack_layers,
        'bandwidth_GBps': spec['shared_bw'],
        'cost': float(provisioned_gB * spec['cost_per_GB']),
        'area': total_silicon_area,
        'footprint_area': float(footprint_area),
        'ctrl_area': spec['ctrl_area'],
        'phy_area': spec['phy_area'],
        'leakage_power': spec['leakage_power'] * int(num_modules)
    }
