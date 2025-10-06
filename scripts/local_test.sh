#!/bin/bash

# =========================================================
# LOCAL TEST SCRIPT - DO NOT SUBMIT TO ARTEMIS
# This script bypasses all cluster-specific commands.
# =========================================================

# Check if both a start and end replicate ID were provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Both a start and end replicate ID are required."
    echo "Usage: bash $0 <start_replicate_id> <end_replicate_id>"
    exit 1
fi

# --- LOCAL CONFIGURATION ---
# Points to the Python executable inside my_sim_env, assuming my_sim_env is in the parent directory.
# (Note: Using 'Scripts/python' for Windows Venvs)
VENV_PYTHON_EXE="$(dirname "$0")/../my_sim_env/Scripts/python" 

# Define a local scratch directory for testing
SCRATCH_OUTPUT_DIR="$(pwd)/local_sim_test_scratch"
mkdir -p "${SCRATCH_OUTPUT_DIR}"

# --- REMAINING CONFIGURATION ---
# These paths must match the absolute paths used in your cluster script
SIM_SCRIPT="./sim_v2.py"
ANALYSIS_SCRIPT="./python_helper_scripts/match_hybrid_to_parent_het.py"
PLOTTING_SCRIPT="./python_helper_scripts/visualisations/triangle_plot_grey_line.py"

# Persistent output directory is now local too
PERSISTENT_OUTPUT_DIR="$(pwd)/local_simulation_outputs/"

# Combined analysis file
ANALYSIS_OUTPUT_FILE="${PERSISTENT_OUTPUT_DIR}/combined_matching_generations.csv"

# Ensure local persistent directory exists
mkdir -p "${PERSISTENT_OUTPUT_DIR}"


for ((i=$1; i<=$2; i++)); do
    
    # Define the directory for the current replicate's output
    REPLICATE_DIR_SCRATCH="${SCRATCH_OUTPUT_DIR}/replicate_${i}"
    REPLICATE_DIR_PERSISTENT="${PERSISTENT_OUTPUT_DIR}/replicate_${i}"
    
    echo " "
    echo "--------------------------------------------------------"
    echo "Starting simulation for replicate $i (Outputting to $REPLICATE_DIR_SCRATCH)"
    echo "--------------------------------------------------------"
    
    mkdir -p "${REPLICATE_DIR_SCRATCH}/results"

    # Step 1: Run the Genetic Simulation (using LOCAL VENV Python, added -u for consistency)
    "$VENV_PYTHON_EXE" -u "$SIM_SCRIPT" \
        --output_dir "$REPLICATE_DIR_SCRATCH" \
        --replicate_id "$i" \
        --file "./beetle_input.csv" \
        -npa 100 \
        -npb 100 \
        -HG 3000 \
        -nc 58 \
        -oh \
        -gmap

    echo "Simulation for replicate $i complete."

    # Step 2: Run the Analysis on the Simulation Outputs (using LOCAL VENV Python, added -u)
    echo "Starting the analysis of outputs for replicate $i"
    "$VENV_PYTHON_EXE" -u "$ANALYSIS_SCRIPT" \
        --input_dir "$REPLICATE_DIR_SCRATCH" \
        --output_file "$ANALYSIS_OUTPUT_FILE" \
        --replicate_id "$i"
    echo "Matching complete for replicate $i."
    
    # Step 3: Generate the triangle plot (using LOCAL VENV Python, added -u and fix for arg type)
    echo "Generating triangle plot for replicate $i"

    # FIX: Extract the raw generation label and strip the 'HG' prefix to pass an integer to Python.
    RAW_GEN_LABEL=$(tail -n 1 "$ANALYSIS_OUTPUT_FILE" | awk -F, '{print $2}' | tr -d '"')
    MATCHING_GEN_NUMBER=${RAW_GEN_LABEL/HG/}
    
    TRIANGLE_PLOT_OUTPUT="${REPLICATE_DIR_SCRATCH}/results/triangle_plot_rep_${i}.png"

    "$VENV_PYTHON_EXE" -u "$PLOTTING_SCRIPT" \
        --input_file "${REPLICATE_DIR_SCRATCH}/results/results_rep_${i}_individual_hi_het.csv" \
        --output_file "$TRIANGLE_PLOT_OUTPUT" \
        --highlight_gen "$MATCHING_GEN_NUMBER" # <-- FIX APPLIED
        
    echo "Plotting complete for replicate $i."

    # --- 3. CLEANUP: Copy all temporary files back to the persistent storage ---
    echo "Copying replicate $i results from SCRATCH to PERSISTENT storage..."
    mkdir -p "${REPLICATE_DIR_PERSISTENT}" # Ensure the final destination exists
    cp -r "${REPLICATE_DIR_SCRATCH}/." "${REPLICATE_DIR_PERSISTENT}/"

done

echo " "
echo "Workflow for replicate range $1 to $2 complete."