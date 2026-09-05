#!/usr/bin/env python3
import argparse
import logging
import sys
import csv
from datetime import datetime
import math

# Import the SRAM estimator class
# Assuming the provided code is in a file called cacti_estimator.py
from cacti_wrapper import CactiSRAM

def setup_logger():
    """Set up and return a logger for the application."""
    logger = logging.getLogger("sram_energy_estimator")
    logger.setLevel(logging.INFO)
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    return logger

def calculate_depth_from_size(size_kb, width_bits):
    """
    Calculate the SRAM depth based on desired size in KB and width in bits.
    
    Args:
        size_kb: Size in kilobytes
        width_bits: Width in bits
    
    Returns:
        Depth in entries
    """
    size_bytes = size_kb * 1024
    bytes_per_entry = width_bits / 8
    depth = math.ceil(size_bytes / bytes_per_entry)
    return depth

def estimate_sram_at_1ghz(
    technology: int = 14,
    width: int = 64,
    size_kb: int = 64,
    n_rw_ports: int = 2,
    n_banks: int = 1,
    logger = None
):
    """
    Estimate energy consumption for SRAM at specified parameters with 1 GHz frequency.
    
    Args:
        technology: Technology node in nm (default: 14nm)
        width: Width of the SRAM in bits (default: 64 bits)
        size_kb: Size of the SRAM in kilobytes
        n_rw_ports: Number of read/write ports (default: 2)
        n_banks: Number of banks (default: 1)
        logger: Logger to use (optional)
    
    Returns:
        Dictionary containing energy consumption estimates
    """
    if logger is None:
        logger = setup_logger()
    
    # Calculate depth based on size and width
    depth = calculate_depth_from_size(size_kb, width)
    
    # Create the SRAM estimator
    sram = CactiSRAM(
        technology=technology,
        width=width,
        depth=depth,
        n_rw_ports=n_rw_ports,
        n_banks=n_banks
    )
    
    # Attach the logger to the SRAM object
    sram.logger = logger
    
    # Get energy estimates for different operations
    read_energy = sram.read()
    write_energy = sram.write()
    update_energy = sram.update()
    
    # Get leakage power for 1 cycle with a cycle time of 1ns (1 GHz)
    cycle_time_seconds = 1e-9  # 1 nanosecond = 1 GHz
    leakage_energy = sram.leak(cycle_time_seconds)
    
    # Get area
    area = sram.get_area()
    
    # Calculate bandwidth at 1 GHz
    bytes_per_cycle = (width * n_rw_ports * n_banks) / 8
    bandwidth_gbps = bytes_per_cycle * 1  # 1 GB/s per GB/cycle at 1 GHz
    
    # Return results
    return {
        "technology_nm": technology,
        "width_bits": width,
        "depth_entries": depth,
        "size_kb": size_kb,
        "n_rw_ports": n_rw_ports,
        "n_banks": n_banks,
        "read_energy_joules": read_energy,
        "write_energy_joules": write_energy,
        "update_energy_joules": update_energy,
        "leakage_energy_per_cycle_joules": leakage_energy,
        "area_m2": area,
        "bytes_per_cycle": bytes_per_cycle,
        "bandwidth_GBps_at_1GHz": bandwidth_gbps
    }

def run_size_sweep(
    technology: int = 14,
    width: int = 64,
    n_rw_ports: int = 2,
    n_banks: int = 1,
    sizes_kb: list = [64, 128, 256, 512, 1024, 2048],
    output_filename: str = None,
    logger = None
):
    """
    Run a sweep across different SRAM sizes.
    
    Args:
        technology: Technology node in nm
        width: Width in bits
        n_rw_ports: Number of read/write ports
        n_banks: Number of banks
        sizes_kb: List of sizes in KB to analyze
        output_filename: Output CSV filename
        logger: Logger instance
    
    Returns:
        Path to the output CSV file
    """
    if logger is None:
        logger = setup_logger()
    
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"sram_size_sweep_{timestamp}.csv"
    
    # Setup CSV
    with open(output_filename, 'w', newline='') as csvfile:
        fieldnames = [
            'Size (KB)', 
            'Depth (entries)',
            'Read Energy per Bit (pJ)', 
            'Write Energy per Bit (pJ)', 
            'Update Energy per Bit (pJ)',
            'Leakage Energy per 1ns Cycle (pJ)', 
            'Area (mm²)',
            'Bytes per Cycle',
            'Bandwidth at 1 GHz (GB/s)'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Add configuration info as a header
        csvfile.write(f"# SRAM Configuration: {technology}nm, {width}-bit width, {n_rw_ports} read/write ports, {n_banks} banks, 1 GHz\n")
        
        for size_kb in sizes_kb:
            logger.info(f"Estimating for {size_kb} KB SRAM")
            
            try:
                results = estimate_sram_at_1ghz(
                    technology=technology,
                    width=width,
                    size_kb=size_kb,
                    n_rw_ports=n_rw_ports,
                    n_banks=n_banks,
                    logger=logger
                )
                
                # Prepare row for CSV - converting read/write/update to per-bit energy in pJ
                row = {
                    'Size (KB)': size_kb,
                    'Depth (entries)': results['depth_entries'],
                    'Read Energy per Bit (pJ)': (results['read_energy_joules'] * 1e12) / width,
                    'Write Energy per Bit (pJ)': (results['write_energy_joules'] * 1e12) / width,
                    'Update Energy per Bit (pJ)': (results['update_energy_joules'] * 1e12) / width,
                    'Leakage Energy per 1ns Cycle (pJ)': results['leakage_energy_per_cycle_joules'] * 1e12,
                    'Area (mm²)': results['area_m2'] * 1e6,
                    'Bytes per Cycle': results['bytes_per_cycle'],
                    'Bandwidth at 1 GHz (GB/s)': results['bandwidth_GBps_at_1GHz']
                }
                
                writer.writerow(row)
                
                # Also log a summary
                logger.info(f"  Size: {size_kb} KB, Depth: {results['depth_entries']} entries")
                logger.info(f"  Read: {row['Read Energy per Bit (pJ)']:.4f} pJ/bit, Write: {row['Write Energy per Bit (pJ)']:.4f} pJ/bit")
                logger.info(f"  Update: {row['Update Energy per Bit (pJ)']:.4f} pJ/bit")
                logger.info(f"  Leakage: {row['Leakage Energy per 1ns Cycle (pJ)']:.4f} pJ/cycle")
                logger.info(f"  Area: {row['Area (mm²)']:.4f} mm²")
                logger.info(f"  Bandwidth: {row['Bandwidth at 1 GHz (GB/s)']:.2f} GB/s @ 1 GHz")
                
            except Exception as e:
                logger.error(f"Error processing {size_kb} KB configuration: {e}")
    
    logger.info(f"Size sweep completed. Results saved to {output_filename}")
    return output_filename

def main():
    """Main function to run SRAM size sweep."""
    parser = argparse.ArgumentParser(description='Analyze SRAM configurations at 1 GHz')
    parser.add_argument('--technology', type=int, default=14, help='Technology node in nm (default: 14)')
    parser.add_argument('--width', type=int, default=64, help='Width in bits (default: 64)')
    parser.add_argument('--rw-ports', type=int, default=2, help='Number of read/write ports (default: 2)')
    parser.add_argument('--banks', type=int, default=1, help='Number of banks (default: 1)')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--sizes', type=str, default='64,128,256,512,1024,2048', 
                        help='Comma-separated list of sizes in KB to analyze (default: 64,128,256,512,1024,2048)')
    
    args = parser.parse_args()
    
    # Parse sizes
    sizes_kb = [int(size) for size in args.sizes.split(',')]
    
    # Setup logger
    logger = setup_logger()
    
    logger.info(f"Running SRAM size sweep at 1 GHz for {args.technology}nm technology")
    logger.info(f"Configuration: {args.width}-bit width, {args.rw_ports} read/write ports, {args.banks} banks")
    logger.info(f"Analyzing sizes: {sizes_kb} KB")
    
    output_file = run_size_sweep(
        technology=args.technology,
        width=args.width,
        n_rw_ports=args.rw_ports,
        n_banks=args.banks,
        sizes_kb=sizes_kb,
        output_filename=args.output,
        logger=logger
    )
    
    logger.info(f"Analysis complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
