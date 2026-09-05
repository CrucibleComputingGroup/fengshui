#!/usr/bin/env python3
import os
import sys

def find_leaf_directories_missing_file(root_folder, target_file="timeloop-mapper.stats.txt"):
    """
    Find all leaf directories (no subdirectories) that don't contain the target file.
    Skips directories with '.ipynb_checkpoints' in their path.
    
    Args:
        root_folder (str): Path to the root folder to search
        target_file (str): Name of the file to look for
    
    Returns:
        list: List of absolute paths to leaf directories missing the target file
    """
    missing_file_dirs = []
    
    for root, dirs, files in os.walk(root_folder):
        # Check if this is a leaf directory (no subdirectories)
        if not dirs:  # This directory has no subdirectories
            # Skip directories with .ipynb_checkpoints in the path
            if '.ipynb_checkpoints' in root:
                continue
            
            # Check if the target file exists in this directory
            if target_file not in files:
                missing_file_dirs.append(os.path.abspath(root))
    
    return missing_file_dirs

def main():
    # Get the folder path from command line argument or use current directory
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = input("Enter the folder path to search (or press Enter for current directory): ").strip()
        if not folder_path:
            folder_path = "."
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a directory.")
        return
    
    # Find leaf directories missing the target file
    target_file = "timeloop-mapper.stats.txt"
    missing_dirs = find_leaf_directories_missing_file(folder_path, target_file)
    
    print(f"Searching for leaf directories missing '{target_file}' in: {os.path.abspath(folder_path)}")
    print("(Skipping directories with '.ipynb_checkpoints' in path)")
    print("=" * 60)
    
    if missing_dirs:
        print(f"Found {len(missing_dirs)} leaf directories missing '{target_file}':")
        print()
        for directory in sorted(missing_dirs):
            print(directory)
    else:
        print(f"All leaf directories contain '{target_file}' or no leaf directories found.")

if __name__ == "__main__":
    main()