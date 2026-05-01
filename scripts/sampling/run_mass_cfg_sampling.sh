#!/bin/bash

# Define the datasets
DATASETS=("adult")

# Loop through each dataset
for dataset in "${DATASETS[@]}"; do
    echo "========================================"
    echo "Processing dataset: $dataset"
    echo "========================================"
    
    # Ensure the target directory exists before saving files there
    mkdir -p "data/${dataset}/synthetic/"

    # Loop through gamma values from 0.60 to 0.98 in steps of 0.02
    # The -f "%.2f" flag ensures the gamma value always has two decimal places (e.g., 0.60)
    for gamma in $(seq -f "%.2f" 0.6 0.02 1.2); do
        
        echo "Running Minority Sampling for $dataset with gamma=$gamma..."
        python cfg-ctabsyn/main.py \
            --dataname "$dataset" \
            --method tabsyn \
            --mode sample \
            --save_path "data/${dataset}/synthetic/synthetic_minority_cfg_gamma=${gamma}.csv" \
            --condition_by 1 \
            --gamma "$gamma"

        echo "Running Majority Sampling for $dataset with gamma=$gamma..."
        python cfg-ctabsyn/main.py \
            --dataname "$dataset" \
            --method tabsyn \
            --mode sample \
            --save_path "data/${dataset}/synthetic/synthetic_majority_cfg_gamma=${gamma}.csv" \
            --condition_by 0 \
            --gamma "$gamma"
            
    done
done

echo "All tasks completed!"