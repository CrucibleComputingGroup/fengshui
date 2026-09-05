import pandas as pd
import glob
import os
from pathlib import Path

def parse_csv_files_and_find_min(file_pattern, file_type_name):
    """
    Parse CSV files matching the pattern and find rows with minimum e2e_best_value
    for each combination of net, objective, and cost_aware.
    
    Args:
        file_pattern (str): Pattern to match CSV files
        file_type_name (str): Name of the file type (e.g., 'our', 'homo')
    
    Returns:
        pandas.DataFrame: DataFrame with minimum e2e_best_value rows
    """
    
    # Find all CSV files matching the pattern
    csv_files = glob.glob(file_pattern)
    
    # Filter out unwanted files based on file type
    if file_type_name == 'our':
        csv_files = [f for f in csv_files 
                     if not any(x in f for x in ['test.csv', 'test_debug.csv'])]
    elif file_type_name == 'homo':
        csv_files = [f for f in csv_files 
                     if not any(x in f for x in ['_13_', 'sd_only'])]
    
    print(f"\nProcessing {file_type_name.upper()} files:")
    print(f"Found {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"  - {file}")
    
    if not csv_files:
        print(f"No valid {file_type_name} CSV files found!")
        return pd.DataFrame()
    
    # Read and combine all CSV files
    all_data = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Add source file column for tracking
            df['source_file'] = file
            all_data.append(df)
            print(f"Successfully read {file}: {len(df)} rows")
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not all_data:
        print(f"No valid data found in {file_type_name} files!")
        return pd.DataFrame()
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total combined rows for {file_type_name}: {len(combined_df)}")
    
    # Group by net, objective, and cost_aware, then find row with minimum e2e_best_value
    grouping_columns = ['net', 'objective', 'cost_aware']
    
    # Find the index of minimum e2e_best_value for each group
    min_indices = combined_df.groupby(grouping_columns)['e2e_best_value'].idxmin()
    
    # Select rows with minimum e2e_best_value
    result_df = combined_df.loc[min_indices].copy()
    
    # Sort by grouping columns for better readability
    result_df = result_df.sort_values(grouping_columns).reset_index(drop=True)
    
    # Drop the source_file column from final result
    result_df = result_df.drop('source_file', axis=1)
    
    print(f"Final result for {file_type_name}: {len(result_df)} unique combinations")
    
    return result_df

def save_result_csv(df, output_filename):
    """Save the result DataFrame to a CSV file."""
    df.to_csv(output_filename, index=False)
    print(f"Results saved to: {output_filename}")

def print_summary_stats(df, file_type_name):
    """Print summary statistics for the results."""
    if df.empty:
        print(f"No data to summarize for {file_type_name}")
        return
        
    print(f"\n{file_type_name.upper()} Summary:")
    print(f"Unique networks: {df['net'].nunique()}")
    print(f"Unique objectives: {df['objective'].nunique()}")
    print(f"Cost-aware configurations: {df['cost_aware'].sum()}")
    print(f"Non-cost-aware configurations: {(~df['cost_aware']).sum()}")
    
    print(f"\nE2E Best Value Statistics for {file_type_name}:")
    print(f"Min: {df['e2e_best_value'].min():.6f}")
    print(f"Max: {df['e2e_best_value'].max():.6f}")
    print(f"Mean: {df['e2e_best_value'].mean():.6f}")

# Main execution
if __name__ == "__main__":
    # Process OUR files
    print("="*60)
    print("PROCESSING OUR FILES")
    print("="*60)
    our_results = parse_csv_files_and_find_min("case_study1_results_our_*.csv", "our")
    
    if not our_results.empty:
        # Display the results
        print("\nOUR Results:")
        print(our_results.to_string(index=False))
        
        # Save to file
        save_result_csv(our_results, "case_study1_results_final_our.csv")
        
        # Show summary statistics
        print_summary_stats(our_results, "our")
    
    # Process HOMO files
    print("\n" + "="*60)
    print("PROCESSING HOMO FILES")
    print("="*60)
    homo_results = parse_csv_files_and_find_min("case_study1_results_homo_*.csv", "homo")
    
    if not homo_results.empty:
        # Display the results
        print("\nHOMO Results:")
        print(homo_results.to_string(index=False))
        
        # Save to file
        save_result_csv(homo_results, "case_study1_results_final_homo.csv")
        
        # Show summary statistics
        print_summary_stats(homo_results, "homo")
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    if not our_results.empty and not homo_results.empty:
        print("Both OUR and HOMO files processed successfully!")
        print(f"OUR file: case_study1_results_final_our.csv ({len(our_results)} rows)")
        print(f"HOMO file: case_study1_results_final_homo.csv ({len(homo_results)} rows)")
    elif not our_results.empty:
        print("Only OUR files processed successfully!")
        print(f"OUR file: case_study1_results_final_our.csv ({len(our_results)} rows)")
    elif not homo_results.empty:
        print("Only HOMO files processed successfully!")
        print(f"HOMO file: case_study1_results_final_homo.csv ({len(homo_results)} rows)")
    else:
        print("No valid data found in either file type!")