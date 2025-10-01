#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Configuration
SIM_SCRIPT="C:/Users/sg802/Documents/git_clone/hybrid_sim_project/scripts/sim_v2.py"
ANALYSIS_SCRIPT="C:/Users/sg802/Documents/git_clone/hybrid_sim_project/scripts/python_helper_scripts/match_hybrid_to_parent_het.py"
NUM_REPLICATES=20
BASE_OUTPUT_DIR="C:/Users/sg802/Documents/git_clone/hybrid_sim_project/simulation_outputs"
ANALYSIS_OUTPUT_FILE="${BASE_OUTPUT_DIR}/combined_matching_generations.csv"

# Remove the old consolidated file before starting a new run
rm -f "${ANALYSIS_OUTPUT_FILE}"

# Use a loop to handle each replicate
for ((i=1; i<=$NUM_REPLICATES; i++)); do
    # Create a dynamic output directory for the current replicate
    REPLICATE_DIR="${BASE_OUTPUT_DIR}/replicate_${i}"
    
    # Define a dynamic output file for the locus data
    LOCUS_OUTPUT_FILE="${REPLICATE_DIR}/results/locus_data.csv"
    
    echo "Starting simulation for replicate $i"
    
    # Step 0: Ensure the output directory exists
    mkdir -p "${REPLICATE_DIR}/results"

    # Step 1: Run the Genetic Simulation
    py "$SIM_SCRIPT" \
        --output_dir "$REPLICATE_DIR" \
        --replicate_id "$i" \
        --file "C:/Users/sg802/Documents/git_clone/hybrid_sim_project/input_data/beetle_input.csv" \
        -npa 100 \
        -npb 100 \
        -HG 2500 \
        -nc 1 \
        -oh

    echo "Simulation for replicate $i complete."

    # Step 2: Run the Analysis on the Simulation Outputs
    # The analysis script now appends to the single master file on each run
    echo "Starting the analysis of outputs for replicate $i"
    py "$ANALYSIS_SCRIPT" --input_dir "$REPLICATE_DIR" --output_file "$ANALYSIS_OUTPUT_FILE" --replicate_id "$i"
    echo "Matching complete for replicate $i."
done

echo "Workflow complete. Consolidated results are in: ${ANALYSIS_OUTPUT_FILE}"