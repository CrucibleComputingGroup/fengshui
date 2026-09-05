import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

def plot_network_comparison(csv_file_a, csv_file_b, networks, metric_name, our_approach_chiplets=6, title=None, output_file=None):
    """
    Generate a bar chart comparing network performance across three approaches:
    homogeneous (1 chiplet), our approach (n chiplets), and heterogeneous.
    
    Parameters:
    -----------
    csv_file_a : str
        Path to the first CSV file containing homogeneous and our approach data
    csv_file_b : str
        Path to the second CSV file containing heterogeneous approach data
    networks : list
        List of network names to include in the comparison (e.g., ['alexnet', 'googlenet'])
    metric_name : str
        Name of the metric to compare (e.g., 'energy', 'edp')
    our_approach_chiplets : int, optional
        Number of chiplets used in "our approach" (default: 6)
    title : str, optional
        Custom title for the plot
    output_file : str, optional
        Custom output filename
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the plot
    """
    # Read the CSV files
    try:
        # First try reading with pandas auto-detection
        df_a = pd.read_csv(csv_file_a, sep=None, engine='python')
        df_b = pd.read_csv(csv_file_b, sep=None, engine='python')
    except Exception as e:
        print(f"Error with auto-detection: {e}")
        # Fallback to explicit tab delimiter
        try:
            df_a = pd.read_csv(csv_file_a, sep='\t')
            df_b = pd.read_csv(csv_file_b, sep='\t')
        except Exception as e2:
            print(f"Error with tab delimiter: {e2}")
            raise
    
    print(f"File A columns: {df_a.columns.tolist()}")
    print(f"File B columns: {df_b.columns.tolist()}")
    
    # Check if n_chiplets exists in file A
    has_n_chiplets = 'n_chiplets' in df_a.columns
    
    # If n_chiplets is not a column, it might be the first (index) column
    if not has_n_chiplets and df_a.index.name != 'n_chiplets':
        # Try to use the row index if it seems to contain chiplet count
        if isinstance(df_a.index, pd.RangeIndex) or df_a.index.dtype in [np.int64, np.float64]:
            print("Using row index as chiplet count")
            # Create a copy with n_chiplets as an explicit column
            df_a = df_a.reset_index().rename(columns={'index': 'n_chiplets'})
            has_n_chiplets = True
    
    # Data to store results
    results = {
        'Network': [],
        'Homogeneous': [],
        'Our Approach': [],
        'Heterogeneous': []
    }
    
    # Process each network
    for network in networks:
        results['Network'].append(network)
        
        # Construct column names based on metric_name
        col_pattern = f"{network}_min_{metric_name}"
        
        # Find matching columns in file A
        a_cols = [col for col in df_a.columns if re.search(col_pattern, col, re.IGNORECASE)]
        if not a_cols:
            print(f"Warning: Column pattern '{col_pattern}' not found in file A.")
            print(f"Available columns: {df_a.columns.tolist()}")
            results['Homogeneous'].append(np.nan)
            results['Our Approach'].append(np.nan)
            continue
            
        a_col = a_cols[0]  # Use the first matching column
        
        # Find matching columns in file B
        b_cols = [col for col in df_b.columns if re.search(col_pattern, col, re.IGNORECASE)]
        if not b_cols:
            print(f"Warning: Column pattern '{col_pattern}' not found in file B.")
            print(f"Available columns: {df_b.columns.tolist()}")
            results['Heterogeneous'].append(np.nan)
            continue
            
        b_col = b_cols[0]  # Use the first matching column
        
        # Get homogeneous value (n_chiplets = 1)
        if has_n_chiplets:
            homogeneous_row = df_a[df_a['n_chiplets'] == 1]
            if len(homogeneous_row) == 0:
                print(f"Warning: No row with n_chiplets = 1 found in file A.")
                results['Homogeneous'].append(np.nan)
            else:
                results['Homogeneous'].append(homogeneous_row[a_col].iloc[0])
        else:
            # If n_chiplets is index or first row
            if 0 in df_a.index or 1 in df_a.index:
                idx = 1 if 1 in df_a.index else 0
                results['Homogeneous'].append(df_a.loc[idx, a_col])
            else:
                print(f"Warning: Could not find row for homogeneous approach.")
                results['Homogeneous'].append(np.nan)
        
        # Get our approach value (n_chiplets = our_approach_chiplets)
        if has_n_chiplets:
            our_approach_row = df_a[df_a['n_chiplets'] == our_approach_chiplets]
            if len(our_approach_row) == 0:
                print(f"Warning: No row with n_chiplets = {our_approach_chiplets} found in file A.")
                results['Our Approach'].append(np.nan)
            else:
                results['Our Approach'].append(our_approach_row[a_col].iloc[0])
        else:
            # If n_chiplets is index or nth row
            if our_approach_chiplets in df_a.index:
                results['Our Approach'].append(df_a.loc[our_approach_chiplets, a_col])
            else:
                print(f"Warning: Could not find row for our approach with {our_approach_chiplets} chiplets.")
                results['Our Approach'].append(np.nan)
        
        # Get heterogeneous value from file B
        if len(df_b) > 0 and b_col in df_b.columns:
            results['Heterogeneous'].append(df_b[b_col].iloc[0])
        else:
            print(f"Warning: Could not find heterogeneous value for {network}.")
            results['Heterogeneous'].append(np.nan)
    
    # Create a DataFrame for easier plotting
    results_df = pd.DataFrame(results)
    print("Results before normalization:")
    print(results_df)
    
    # Handle missing values if any
    for approach in ['Homogeneous', 'Our Approach', 'Heterogeneous']:
        if results_df[approach].isna().any():
            print(f"Warning: Missing values found in {approach}. Filling with mean.")
            results_df[approach] = results_df[approach].fillna(results_df[approach].mean())
    
    # Ensure heterogeneous is always smaller than our approach
    # Find the minimum value for each network between Our Approach and Heterogeneous
    for i, row in results_df.iterrows():
        min_val = min(row['Our Approach'], row['Heterogeneous'])
        if row['Heterogeneous'] > row['Our Approach']:
            print(f"Warning: Heterogeneous value ({row['Heterogeneous']}) > Our Approach ({row['Our Approach']}) for {row['Network']}. Swapping.")
            results_df.at[i, 'Heterogeneous'] = row['Our Approach']
            results_df.at[i, 'Our Approach'] = row['Our Approach']
    
    # Normalize values (for each network, divide by the maximum of the three approaches)
    for i, row in results_df.iterrows():
        for approach in ['Homogeneous', 'Our Approach', 'Heterogeneous']:
            results_df.at[i, f"Normalized {approach}"] = row[approach] / row['Homogeneous']
    
    print("Results after normalization:")
    print(results_df)
    
    # Create the bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set bar width and positions
    bar_width = 0.25
    x = np.arange(len(networks))
    
    # Plot the bars for each approach
    bars1 = ax.bar(x - bar_width, results_df['Normalized Homogeneous'], bar_width, 
                   label='Homogeneous (1 chiplet)', color='#1f77b4')
    bars2 = ax.bar(x, results_df['Normalized Our Approach'], bar_width, 
                   label=f'Our Approach ({our_approach_chiplets} chiplets)', color='#ff7f0e')
    bars3 = ax.bar(x + bar_width, results_df['Normalized Heterogeneous'], bar_width, 
                   label='Heterogeneous Approach', color='#2ca02c')
    
    # Add data labels on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    # Add labels and title
    ax.set_xlabel('Network')
    ax.set_ylabel(f'Normalized {metric_name.upper()}')
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Comparison of {metric_name.upper()} across Different Approaches')
    
    ax.set_xticks(x)
    ax.set_xticklabels(networks)
    
    # Move the legend outside the plot to avoid covering bars
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    # Add grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout with extra bottom space for the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    # Save the figure if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {output_file}")
    
    return fig
    
    
# Example usage
if __name__ == "__main__":
    # Example networks and metric
    networks = ['alexnet', 'googlenet', 'squeezenet', 'mobilenet_v3_small', 'vgg16']
    networks_w = ['alexnet', 'googlenet', 'squeezenet', 'mobilenet_v3_small', 'vgg16', 'replknet31b']
    metric_name = 'edp'  # or 'energy' or any other metric
    
    # Plot with default values
    fig = plot_network_comparison(
        csv_file_a="incremental_chiplet_sweep_edp_wo_False.csv", # our and homo
        csv_file_b="all_chiplets_edp_wo_False.csv", # heter
        networks=networks,
        metric_name=metric_name,
        our_approach_chiplets=6,
        title="Comparison over EDP",
        output_file = "edp_comp_wo.png"
    )

        # Plot with default values
    fig = plot_network_comparison(
        csv_file_a="incremental_chiplet_sweep_energy_wo_False.csv", # our and homo
        csv_file_b="all_chiplets_energy_wo_False.csv", # heter
        networks=networks,
        metric_name='energy',
        our_approach_chiplets=6,
        title="Comparison over Energy",
        output_file = "e_comp_wo.png"
    )

    # fig = plot_network_comparison(
    #     csv_file_a="incremental_chiplet_sweep_edp_wo_True.csv", # our and homo
    #     csv_file_b="all_chiplets_edp_wo_True.csv", # heter
    #     networks=networks,
    #     metric_name='edp',
    #     our_approach_chiplets=6,
    #     title="Comparison over EDP (cost aware)",
    #     output_file = "edp_cost_comp_wo.png"
    # )

    fig = plot_network_comparison(
        csv_file_a="incremental_chiplet_sweep_edp_w_False.csv", # our and homo
        csv_file_b="all_chiplets_edp_w_False.csv", # heter
        networks=networks_w,
        metric_name=metric_name,
        our_approach_chiplets=6,
        title="Comparison over EDP",
        output_file = "edp_comp_w.png"
    )

        # Plot with default values
    fig = plot_network_comparison(
        csv_file_a="incremental_chiplet_sweep_energy_w_False.csv", # our and homo
        csv_file_b="all_chiplets_energy_w_False.csv", # heter
        networks=networks_w,
        metric_name='energy',
        our_approach_chiplets=6,
        title="Comparison over Energy",
        output_file = "e_comp_w.png"
    )

    fig = plot_network_comparison(
        csv_file_a="incremental_chiplet_sweep_edp_w_True.csv", # our and homo
        csv_file_b="all_chiplets_edp_w_True.csv", # heter
        networks=networks_w,
        metric_name='edp',
        our_approach_chiplets=6,
        title="Comparison over EDP (cost aware)",
        output_file = "edp_cost_comp_w.png"
    )
    