#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Check if both a start and end replicate ID were provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Both a start and end replicate ID are required."
    echo "Usage: $0 <start_replicate_id> <end_replicate_id>"
    exit 1
fi

# Configuration
# Use the direct path to the python executable in the virtual environment
PYTHON_EXE="C:/Users/sg802/Documents/git_clone/hybrid_sim_project/.venv/Scripts/python.exe"

SIM_SCRIPT="C:/Users/sg802/Documents/git_clone/hybrid_sim_project/scripts/sim_v2.py"
ANALYSIS_SCRIPT="C:/Users/sg802/Documents/git_clone/hybrid_sim_project/scripts/python_helper_scripts/match_hybrid_to_parent_het.py"
BASE_OUTPUT_DIR="C:/Users/sg802/Documents/git_clone/hybrid_sim_project/simulation_outputs"
ANALYSIS_OUTPUT_FILE="${BASE_OUTPUT_DIR}/combined_matching_generations_run3.csv"

# Use a loop to handle the specified range of replicates
# The loop now runs from the provided start ID ($1) to the end ID ($2)
for ((i=$1; i<=$2; i++)); do
    
    # Create a dynamic output directory for the current replicate
    REPLICATE_DIR="${BASE_OUTPUT_DIR}/replicate_${i}"
    
    # Define a dynamic output file for the locus data
    LOCUS_OUTPUT_FILE="${REPLICATE_DIR}/results/locus_data.csv"
    
    echo " "
    echo "--------------------------------------------------------"
    echo "Starting simulation for replicate $i"
    echo "--------------------------------------------------------"
    
    # Step 0: Ensure the output directory exists
    mkdir -p "${REPLICATE_DIR}/results"

    # Step 1: Run the Genetic Simulation
    "$PYTHON_EXE" "$SIM_SCRIPT" \
        --output_dir "$REPLICATE_DIR" \
        --replicate_id "$i" \
        --file "C:/Users/sg802/Documents/git_clone/hybrid_sim_project/input_data/beetle_input.csv" \
        -npa 100 \
        -npb 100 \
        -HG 3000 \
        -nc 58 \
        -oh \
        -gmap

    echo "Simulation for replicate $i complete."

    # Step 2: Run the Analysis on the Simulation Outputs
    echo "Starting the analysis of outputs for replicate $i"
    "$PYTHON_EXE" "$ANALYSIS_SCRIPT" --input_dir "$REPLICATE_DIR" --output_file "$ANALYSIS_OUTPUT_FILE" --replicate_id "$i"
    echo "Matching complete for replicate $i."
done

echo " "
echo "Workflow for replicate range $1 to $2 complete."