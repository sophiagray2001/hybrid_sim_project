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
PYTHON_EXE=/usr/bin/python

SIM_SCRIPT="/mnt/nfs2/bioenv/sg802/hybrid_sim_project/scripts/sim_v2.py"
ANALYSIS_SCRIPT="/mnt/nfs2/bioenv/sg802/hybrid_sim_project/scripts/python_helper_scripts/match_hybrid_to_parent_het.py"
PLOTTING_SCRIPT="/mnt/nfs2/bioenv/sg802/hybrid_sim_project/scripts/python_helper_scripts/visualisations/triangle_plot_grey_line.py"

BASE_OUTPUT_DIR="/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs/"
ANALYSIS_OUTPUT_FILE="${BASE_OUTPUT_DIR}/combined_matching_generations.csv"

for ((i=$1; i<=$2; i++)); do
    REPLICATE_DIR="${BASE_OUTPUT_DIR}/replicate_${i}"
    
    echo " "
    echo "--------------------------------------------------------"
    echo "Starting simulation for replicate $i"
    echo "--------------------------------------------------------"
    
    mkdir -p "${REPLICATE_DIR}/results"

    # Step 1: Run the Genetic Simulation
    "$PYTHON_EXE" "$SIM_SCRIPT" \
        --output_dir "$REPLICATE_DIR" \
        --replicate_id "$i" \
        --file "/mnt/nfs2/bioenv/sg802/hybrid_sim_project/beetle_input.csv" \
        -npa 100 \
        -npb 100 \
        -HG 100 \
        -nc 58 \
        -oh \
        -gmap

    echo "Simulation for replicate $i complete."

    # Step 2: Run the Analysis on the Simulation Outputs
    echo "Starting the analysis of outputs for replicate $i"
    "$PYTHON_EXE" "$ANALYSIS_SCRIPT" \
        --input_dir "$REPLICATE_DIR" \
        --output_file "$ANALYSIS_OUTPUT_FILE" \
        --replicate_id "$i"
    echo "Matching complete for replicate $i."
    
    # Step 3: Generate the triangle plot
    echo "Generating triangle plot for replicate $i"

    # Extract the matching generation from the analysis file
    # Get the last row of the CSV and extract the 'generation' column value
    MATCHING_GEN=$(tail -n 1 "$ANALYSIS_OUTPUT_FILE" | awk -F, '{print $2}')
    
    TRIANGLE_PLOT_OUTPUT="${REPLICATE_DIR}/results/triangle_plot_rep_${i}.png"

    "$PYTHON_EXE" "$PLOTTING_SCRIPT" \
        --input_file "${REPLICATE_DIR}/results/results_rep_${i}_individual_hi_het.csv" \
        --output_file "$TRIANGLE_PLOT_OUTPUT" \
        --highlight_gen "$MATCHING_GEN"
        
    echo "Plotting complete for replicate $i."
done

echo " "
echo "Workflow for replicate range $1 to $2 complete."