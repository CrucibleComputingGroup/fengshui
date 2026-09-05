#!/bin/bash

# Script to run 32 instances of the vision case study with different run indices

echo "Starting 32 instances of vision case study..."

# Loop through run indices 0 to 31
for i in {0..31}; do
    echo "Starting run $i..."
    python3 case_study/vision_case_study_add_backup.py --out=vision_summary_run_${i}.csv &
done

echo "All 32 instances started in background."
echo "Output files will be: vision_summary_run_0.csv through vision_summary_run_31.csv"

# Wait for all background processes to complete
wait

echo "All instances completed!"