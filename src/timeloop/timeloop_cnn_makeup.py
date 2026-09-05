#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

def replace_word_in_file(file_path, old_word, new_word):
    """
    Replace all occurrences of old_word with new_word in the given file.
    
    Args:
        file_path (str): Path to the file
        old_word (str): Word to replace
        new_word (str): Replacement word
    
    Returns:
        bool: True if file was modified, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Check if the old word exists in the file
        if old_word not in content:
            print(f"    No occurrences of '{old_word}' found in {file_path}")
            return False
        
        # Replace the word
        modified_content = content.replace(old_word, new_word)
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(modified_content)
        
        print(f"    Replaced '{old_word}' with '{new_word}' in {file_path}")
        return True
        
    except FileNotFoundError:
        print(f"    ERROR: File {file_path} not found")
        return False
    except Exception as e:
        print(f"    ERROR: Failed to process {file_path}: {e}")
        return False

def run_timeloop_mapper(directory, yaml_file):
    """
    Run timeloop-mapper command in the specified directory.
    
    Args:
        directory (str): Directory to run the command in
        yaml_file (str): Path to the YAML file
    
    Returns:
        bool: True if command succeeded, False otherwise
    """
    try:
        # Change to the directory
        original_cwd = os.getcwd()
        os.chdir(directory)
        
        # Run timeloop-mapper
        cmd = ['timeloop-mapper', yaml_file]
        print(f"    Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print(f"    SUCCESS: timeloop-mapper completed")
            return True
        else:
            print(f"    ERROR: timeloop-mapper failed with return code {result.returncode}")
            if result.stderr:
                print(f"    STDERR: {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"    ERROR: timeloop-mapper timed out (5 minutes)")
        return False
    except FileNotFoundError:
        print(f"    ERROR: timeloop-mapper command not found. Make sure it's in your PATH")
        return False
    except Exception as e:
        print(f"    ERROR: Failed to run timeloop-mapper: {e}")
        return False
    finally:
        # Always change back to original directory
        os.chdir(original_cwd)

def process_directory(directory):
    """
    Process a single directory: replace word in YAML file and run timeloop-mapper.
    
    Args:
        directory (str): Directory to process
    
    Returns:
        bool: True if all operations succeeded, False otherwise
    """
    print(f"\nProcessing: {directory}")
    
    # Check if directory exists
    if not os.path.isdir(directory):
        print(f"    ERROR: Directory {directory} does not exist")
        return False
    
    # Define the YAML file path
    yaml_file = os.path.join(directory, "parsed-processed-input.yaml")
    
    # Step 1: Replace "hybrid" with "random" in the YAML file
    success = replace_word_in_file(yaml_file, "hybrid", "random")
    if not success and not os.path.exists(yaml_file):
        print(f"    ERROR: Required file {yaml_file} does not exist")
        return False
    
    # Step 2: Run timeloop-mapper
    success = run_timeloop_mapper(directory, yaml_file)
    
    return success

def main():
    # List of directories to process
    directories = [
        "/home/workspace/chiplet_timeloop/outputs/efficientnet_b0/layer82_classifier_1/1/1/0/single/500/2/arch=simple_output_stationary@glb_scale=16@pe_x_scale=2@pe_y_scale=3/LPDDR5@GDDR7",
        "/home/workspace/chiplet_timeloop/outputs/efficientnet_b0/layer82_classifier_1/1/1/0/single/500/2/arch=simple_output_stationary@glb_scale=16@pe_x_scale=2@pe_y_scale=3/LPDDR5@HBM3",
        "/home/workspace/chiplet_timeloop/outputs/efficientnet_b0/layer82_classifier_1/1/1/0/single/500/2/arch=simple_output_stationary@glb_scale=4@pe_x_scale=2@pe_y_scale=4/LPDDR5@GDDR7",
        "/home/workspace/chiplet_timeloop/outputs/efficientnet_b0/layer82_classifier_1/1/1/0/single/500/2/arch=simple_output_stationary@glb_scale=4@pe_x_scale=2@pe_y_scale=4/LPDDR5@HBM3",
        "/home/workspace/chiplet_timeloop/outputs/efficientnet_b0/layer82_classifier_1/1/1/0/single/500/2/arch=simple_output_stationary@glb_scale=4@pe_x_scale=2@pe_y_scale=4/LPDDR5@LPDDR5"
    ]
    
    print("Starting batch processing of timeloop directories...")
    print("=" * 80)
    
    successful = 0
    failed = 0
    
    for directory in directories:
        try:
            success = process_directory(directory)
            if success:
                successful += 1
            else:
                failed += 1
        except KeyboardInterrupt:
            print(f"\nProcess interrupted by user")
            break
        except Exception as e:
            print(f"    UNEXPECTED ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"Batch processing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(directories)}")
    
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()