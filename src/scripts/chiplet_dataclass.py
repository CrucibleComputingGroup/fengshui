from typing import List, Tuple, Optional, Set, Dict
import numpy as np
import random
from global_parameter import *

class ChipletConfig:
    def __init__(self,
                 arch_target: str,
                 global_buffer_size_scale: int,
                 pe_x_scale: int,
                 pe_y_scale: int,
                 dram_type: str = 'HBM3'
                 ):
        self.arch_target = arch_target
        self.global_buffer_size_scale = global_buffer_size_scale
        self.pe_x_scale = pe_x_scale
        self.pe_y_scale = pe_y_scale
        self.dram_type = dram_type
        self.arch_para_dict = {"eyeriss_like":[glb_base_word, 64, 64],
                               "simba_like":[glb_base_word, 64, 16], # do not consider reg_mac level
                               "gemmini_like":[glb_base_word, 64, 64],
                               "switch_8port":[0, 0, 0],  # switch chiplet (no compute)
                               "PIM":[0, 1, 1],  # PIM chiplet (memory is compute, no separate GLB/PE)
                              }
    
    def get_rounded_config(self) -> tuple:
        """Get the configuration after rounding, matching run_mapper's rounding."""
        rounded_glb = round(self.arch_para_dict[self.arch_target][0] * self.global_buffer_size_scale)
        rounded_pe_x = round(self.arch_para_dict[self.arch_target][1] * self.pe_x_scale)
        rounded_pe_y = round(self.arch_para_dict[self.arch_target][2] * self.pe_y_scale)
        return (self.arch_target, rounded_glb, rounded_pe_x, rounded_pe_y)
    
    def get_identifier(self) -> str:
        """
        Get a unique string identifier for this chiplet configuration.
        Uses rounded values to ensure stability.
        
        Args:
            base_glb_depth: Base GLB depth before scaling
            base_pe_x: Base PE X dimension before scaling
            base_pe_y: Base PE Y dimension before scaling
            
        Returns:
            String identifier in format: archglbXXXpe_xXXXpe_yYYY
        """
        return f"{self.arch_target}@glb{self.global_buffer_size_scale}@pe_x_scale{self.pe_x_scale}@pe_y_scale{self.pe_y_scale}"
    
    @classmethod
    def from_identifier(cls, identifier: str) -> 'ChipletConfig':
        """
        Create a ChipletConfig from a string identifier.
        
        Args:
            identifier: String identifier in format: arch@glbX@pe_x_scaleY@pe_y_scaleZ
            
        Returns:
            ChipletConfig object
            
        Raises:
            ValueError: If the identifier format is invalid
        """
        try:
            # Parse the identifier format: arch@glbX@pe_x_scaleY@pe_y_scaleZ
            # (legacy format with trailing @dram_type is also accepted)
            parts = identifier.split('@')
            if len(parts) not in (4, 5):
                raise ValueError(f"Invalid identifier format: {identifier}")

            arch_target = parts[0]
            glb_scale = int(parts[1].replace('glb', ''))
            pe_x_scale = int(parts[2].replace('pe_x_scale', ''))
            pe_y_scale = int(parts[3].replace('pe_y_scale', ''))

            return cls(
                arch_target=arch_target,
                global_buffer_size_scale=glb_scale,
                pe_x_scale=pe_x_scale,
                pe_y_scale=pe_y_scale,
            )
        except (ValueError, IndexError) as e:
            raise ValueError(f"Failed to parse identifier '{identifier}': {e}")
    
    @classmethod
    def from_csv_for_network(cls, network_name: str, batch_size: int, seq_length: int, csv_filename: str) -> 'ChipletConfig':
        """
        Construct a ChipletConfig from CSV data for a specific network configuration.
        
        Args:
            network_name: Name of the network (e.g., 'resnet50', 'mobilenet_v3_small')
            batch_size: Batch size (e.g., 1, 32)
            seq_length: Sequence length (e.g., 1, 256)
            csv_filename: Path to the CSV file
        
        Returns:
            ChipletConfig object constructed from the CSV data
        
        Raises:
            ValueError: If the network configuration is not found in the CSV
        """
        import pandas as pd
        
        # Read the CSV file
        df = pd.read_csv(csv_filename)
        
        # Construct the network identifier to match against the 'network' column
        # Based on your CSV, the format appears to be: network_bBATCH_seqSEQ
        network_id = f"{network_name}_b{batch_size}_seq{seq_length}"
        
        # Find the row matching the network configuration
        matching_rows = df[df['network'] == network_id]
        
        if matching_rows.empty:
            raise ValueError(f"Network configuration '{network_id}' not found in {csv_filename}")
        
        # Get the first matching row (assuming unique network configurations)
        row = matching_rows.iloc[0]
        
        # Extract the configuration parameters
        arch_target = row['arch_target']
        glb_scale = int(row['glb_scale'])
        pe_x_scale = int(row['pe_x_scale'])
        pe_y_scale = int(row['pe_y_scale'])
        
        # Create and return the ChipletConfig
        return cls(
            arch_target=arch_target,
            global_buffer_size_scale=glb_scale,
            pe_x_scale=pe_x_scale,
            pe_y_scale=pe_y_scale
        )
    
    @classmethod
    def from_csv_for_n_chiplets(cls, n_chiplets: int, csv_filename: str) -> List['ChipletConfig']:
        """
        Parse a CSV file and return a list of ChipletConfig objects for a given n_chiplets value.
        
        Args:
            n_chiplets: The number of chiplets to look for in the CSV
            csv_filename: Path to the CSV file
        
        Returns:
            List of ChipletConfig objects parsed from that row
        
        Raises:
            ValueError: If n_chiplets value is not found in the CSV
        """
        import pandas as pd
        
        # Read the CSV file
        df = pd.read_csv(csv_filename)
        
        # Find the row with matching n_chiplets
        matching_rows = df[df['n_chiplets'] == n_chiplets]
        
        if matching_rows.empty:
            raise ValueError(f"n_chiplets={n_chiplets} not found in {csv_filename}")
        
        # Get the first matching row
        row = matching_rows.iloc[0]
        
        # Parse chiplets from the row
        chiplets = []
        chiplet_num = 1
        
        # Keep looking for chiplet columns until we don't find them
        while f'chiplet_{chiplet_num}_arch' in row and pd.notna(row[f'chiplet_{chiplet_num}_arch']):
            arch = row[f'chiplet_{chiplet_num}_arch']
            glb_scale = int(row[f'chiplet_{chiplet_num}_glb_scale'])
            pe_x_scale = int(row[f'chiplet_{chiplet_num}_pe_x_scale'])
            pe_y_scale = int(row[f'chiplet_{chiplet_num}_pe_y_scale'])
            
            chiplet = cls(
                arch_target=arch,
                global_buffer_size_scale=glb_scale,
                pe_x_scale=pe_x_scale,
                pe_y_scale=pe_y_scale
            )
            chiplets.append(chiplet)
            chiplet_num += 1
        
        return chiplets
    
    @staticmethod
    def add_unique_chiplet(chiplet_list: List['ChipletConfig'], new_chiplet: 'ChipletConfig') -> List['ChipletConfig']:
        """
        Add a chiplet to the list only if an equivalent chiplet doesn't already exist.
        
        Args:
            chiplet_list: Existing list of ChipletConfig objects
            new_chiplet: The ChipletConfig object to potentially add
        
        Returns:
            Updated list with the new chiplet added if it was unique, 
            or the original list if the chiplet already existed
        """
        chiplet_list_final = chiplet_list.copy()
        # Check if an equivalent chiplet already exists in the list
        for existing_chiplet in chiplet_list:
            if (existing_chiplet.arch_target == new_chiplet.arch_target and
                existing_chiplet.global_buffer_size_scale == new_chiplet.global_buffer_size_scale and
                existing_chiplet.pe_x_scale == new_chiplet.pe_x_scale and
                existing_chiplet.pe_y_scale == new_chiplet.pe_y_scale):
                # Chiplet already exists, return original list
                return chiplet_list_final
        
        return chiplet_list_final + [new_chiplet]

def create_switch_chiplet() -> ChipletConfig:
    """Create an 8-port switch chiplet for MoE expert parallelism."""
    return ChipletConfig(
        arch_target=SWITCH_ARCH_TARGET,
        global_buffer_size_scale=1,
        pe_x_scale=1,
        pe_y_scale=1,
        dram_type='HBM3',
    )


def generate_unique_chiplet(
    existing_configs: Set[tuple],
    arch_targets: List[str],
    glb_scale_options: List[int],
    pe_scale_options: List[int],
    dram_type_options: Optional[List[str]] = None,
    max_attempts: int = 100
) -> Optional[ChipletConfig]:
    """Generate a chiplet with a unique rounded configuration.
    DRAM type is not randomized — every compute chiplet supports all DRAM types;
    the inner GA decides which DRAM to use per layer."""
    for _ in range(max_attempts):
        arch = random.choice(arch_targets)
        glb_scale = random.choice(glb_scale_options)
        pe_x_scale = random.choice(pe_scale_options)
        pe_y_scale = random.choice(pe_scale_options)
        chiplet = ChipletConfig(
            arch_target=arch,
            global_buffer_size_scale=glb_scale,
            pe_x_scale=pe_x_scale,
            pe_y_scale=pe_y_scale,
        )

        rounded_config = chiplet.get_rounded_config()
        if rounded_config not in existing_configs:
            existing_configs.add(rounded_config)
            return chiplet

    return None
    
def generate_chiplet_group(
    n_chiplets: int = 20,
    arch_targets: Optional[List[str]] = None,
    glb_scale_options: Optional[List[int]] = None,
    pe_scale_options: Optional[List[int]] = None,
    dram_type_options: Optional[List[str]] = None,
    seed: Optional[int] = None
) -> List[ChipletConfig]:
    """
    Generate a group of chiplets with unique rounded configurations.
    
    Args:
        n_chiplets: Number of chiplets to generate
        arch_targets: List of possible architecture targets
        glb_scale_options: List of possible values for global buffer size scaling
        pe_scale_options: List of possible values for PE array scaling
        base_glb_depth: Base GLB depth before scaling
        base_pe_x: Base PE X dimension before scaling
        base_pe_y: Base PE Y dimension before scaling
        seed: Random seed for reproducible generation (default: None)
    
    Returns:
        List of ChipletConfig objects with unique rounded configurations
    """
    # Set default values if None
    if arch_targets is None:
        arch_targets = ["eyeriss_like", "simba_like", "gemmini_like"]
    if glb_scale_options is None:
        glb_scale_options = glb_scales
    if pe_scale_options is None:
        pe_scale_options = pe_scales
    if dram_type_options is None:
        dram_type_options = dram_options

    # Set the random seed if provided
    if seed is not None:
        random.seed(seed)

    existing_configs = set()
    chiplets = []

    for _ in range(n_chiplets):
        chiplet = generate_unique_chiplet(
            existing_configs,
            arch_targets,
            glb_scale_options,
            pe_scale_options,
            dram_type_options
        )
        
        if chiplet is None:
            raise ValueError(f"Could not generate {n_chiplets} unique configurations with current parameters")
        
        chiplets.append(chiplet)
    
    return chiplets
    
def generate_all_chiplet_configs(
    arch_targets: Optional[List[str]] = None,
    glb_scale_options: List[int] = glb_scales,
    pe_scale_options: List[int] = pe_scales
    #dram_options: Optional[List[str]] = None
) -> List[ChipletConfig]:
    """
    Generate all possible unique chiplet configurations from the given parameter space.
    
    Args:
        arch_targets: List of possible architecture targets
        glb_scale_options: List of possible values for global buffer size scaling
        pe_scale_options: List of possible values for PE array scaling
    
    Returns:
        List of ChipletConfig objects with all unique rounded configurations
    """
    if arch_targets is None:
        arch_targets = ["eyeriss_like", "simba_like", "gemmini_like"]
    
    # if dram_options is None:
    #     dram_options = dram_options

    # Set to track unique rounded configurations
    existing_configs = set()
    all_chiplets = []
    
    # Generate all possible combinations
    for arch in arch_targets:
        for glb_scale in glb_scale_options:
            for pe_x_scale in pe_scale_options:
                for pe_y_scale in pe_scale_options:
                    # Create chiplet config
                    # for dram_i in dram_options:
                    #     for dram_o in dram_options:
                    chiplet = ChipletConfig(
                        arch_target=arch,
                        global_buffer_size_scale=glb_scale,
                        pe_x_scale=pe_x_scale,
                        pe_y_scale=pe_y_scale
                        # dram_i = dram_i,
                        # dram_o = dram_o
                    )
                    
                    # Check if the rounded configuration is unique
                    rounded_config = chiplet.get_rounded_config()
                    if rounded_config not in existing_configs:
                        existing_configs.add(rounded_config)
                        all_chiplets.append(chiplet)
    
    return all_chiplets
    
def get_max_pes(chiplet_group):
    max_pes = 0
    for chiplet in chiplet_group:
        max_pes = max(max_pes, int(chiplet.pe_x_scale * chiplet.pe_y_scale *pe_x_base_size))
    return max_pes


# Example usage
if __name__ == "__main__":
    chiplet_single= ChipletConfig.from_csv_for_network(network_name="resnet50",batch_size=1,seq_length=1,csv_filename="optimal_single_chiplet_edp_False_20250819_065323.csv")
    print(chiplet_single.get_identifier())
    chiplet_test = ChipletConfig.from_csv_for_n_chiplets(n_chiplets=8,csv_filename="incremental_chiplet_sweep_edp_True.csv")
    
    for chiplet in chiplet_test:
        print(chiplet.get_identifier())
    chiplet_test = ChipletConfig.add_unique_chiplet(chiplet_test,chiplet_single)
    print(len(chiplet_test))
    for chiplet in chiplet_test:
        print(chiplet.get_identifier())