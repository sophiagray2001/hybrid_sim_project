#!/bin/bash

# =========================================================
# HPC CLUSTER SUBMISSION SCRIPT (run_sim_beetle.sh)
# =========================================================

# --- SLURM DIRECTIVES ---
# Time set for 7 hours
#SBATCH --time=7:00:00
#SBATCH --mem=128G
#SBATCH --output=slurm-rep_%j.out

# Exit immediately if a command exits with a non-zero status.
set -e

# Check if both a start and end replicate ID were provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Both a start and end replicate ID are required."
    echo "Usage: $0 <start_replicate_id> <end_replicate_id>"
    exit 1
fi

# --- VIRTUAL ENVIRONMENT SETUP ---
# Explicitly load the system dependency needed by Python (fixes libbz2.so.1.0 error)
module load bzip2

# Define the Venv Python executable path (The only Python we use!)
VENV_PYTHON_EXE="/mnt/nfs2/bioenv/sg802/hybrid_sim_project/sim_env/bin/python"

# --- REMAINING CONFIGURATION ---
SIM_SCRIPT="/mnt/nfs2/bioenv/sg802/hybrid_sim_project/scripts/sim_v2.py"
ANALYSIS_SCRIPT="/mnt/nfs2/bioenv/sg802/hybrid_sim_project/scripts/python_helper_scripts/match_hybrid_to_parent_het.py"
PLOTTING_SCRIPT="/mnt/nfs2/bioenv/sg802/hybrid_sim_project/scripts/python_helper_scripts/visualisations/triangle_plot_grey_line.py"

BASE_OUTPUT_DIR="/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs_no_recombination_50/"
ANALYSIS_OUTPUT_FILE="${BASE_OUTPUT_DIR}/combined_matching_generations_no_recombination.csv"


for ((i=$1; i<=$2; i++)); do
    REPLICATE_DIR="${BASE_OUTPUT_DIR}/replicate_${i}"

    echo " "
    echo "--------------------------------------------------------"
    echo "Starting simulation for replicate $i"
    echo "--------------------------------------------------------"

    mkdir -p "${REPLICATE_DIR}/results"

    # Step 1: Run the Genetic Simulation
    # Store the entire command and all arguments in an array to prevent word-splitting issues.
    SIM_COMMAND_ARRAY=(
        "$VENV_PYTHON_EXE" -u "$SIM_SCRIPT" 
        --output_dir "$REPLICATE_DIR" --replicate_id "$i" 
        --file "/mnt/nfs2/bioenv/sg802/hybrid_sim_project/beetle_input.csv" 
        -npa 100 -npb 100 -HG 3000 -nc 1 -oh -gmap 
        -cd '{"0": 1.0}' -tp
    )

    # Log the command for checking
    echo "Command: ${SIM_COMMAND_ARRAY[@]}"

    # Execute the command using the array notation ("${array[@]}") 
    # to ensure each element is passed as a single, correctly quoted argument.
    "${SIM_COMMAND_ARRAY[@]}"

    echo "Simulation for replicate $i complete."

    # ---------------------------------------------------------------------

    # Step 2: Run the Analysis on the Simulation Outputs
    echo "Starting the analysis of outputs for replicate $i"
    ANALYSIS_COMMAND="$VENV_PYTHON_EXE -u $ANALYSIS_SCRIPT --input_dir $REPLICATE_DIR --output_file $ANALYSIS_OUTPUT_FILE --replicate_id $i"
    echo "Command: $ANALYSIS_COMMAND"
    $ANALYSIS_COMMAND
    
    echo "Matching complete for replicate $i."
    
    # ---------------------------------------------------------------------
    
    # Step 3: Generate the triangle plot
    echo "Generating triangle plot for replicate $i"

    # Define the required file paths
    ANALYSIS_INPUT_FILE="${REPLICATE_DIR}/results/results_rep_${i}_individual_hi_het.csv" 
    TRIANGLE_PLOT_OUTPUT="${REPLICATE_DIR}/results/triangle_plot_rep_${i}.png"

    # Extract the generation number
    if [ -f "$ANALYSIS_OUTPUT_FILE" ]; then
        # This extracts the generation label (e.g., "HG1525") and strips the quotes/HG
        RAW_GEN_LABEL=$(tail -n 1 "$ANALYSIS_OUTPUT_FILE" | awk -F, '{print $2}' | tr -d '"')
        MATCHING_GEN_NUMBER=${RAW_GEN_LABEL/HG/} # Strips the "HG" prefix
    else
        echo "Warning: Analysis output file not found. Setting highlight generation to 0."
        MATCHING_GEN_NUMBER=0
    fi

    PLOTTING_COMMAND="$VENV_PYTHON_EXE -u $PLOTTING_SCRIPT --input_file $ANALYSIS_INPUT_FILE --output_file $TRIANGLE_PLOT_OUTPUT --highlight_gen $MATCHING_GEN_NUMBER"
    echo "Command: $PLOTTING_COMMAND"
    $PLOTTING_COMMAND
        
    echo "Plotting complete for replicate $i."
done

echo " "
echo "Workflow for replicate range $1 to $2 complete."