#!/bin/bash

# =========================================================
# HPC CLUSTER SUBMISSION SCRIPT (run_sim_beetle.sh)
# =========================================================

# --- SLURM DIRECTIVES ---
# Time set for 5.5 hours (safe for a 10-replicate batch)
#SBATCH --time=05:30:00 
#SBATCH --mem=32G
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

BASE_OUTPUT_DIR="/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs_second_batch/"
ANALYSIS_OUTPUT_FILE="${BASE_OUTPUT_DIR}/combined_matching_generations_second_batch.csv"


for ((i=$1; i<=$2; i++)); do
    REPLICATE_DIR="${BASE_OUTPUT_DIR}/replicate_${i}"
    
    echo " "
    echo "--------------------------------------------------------"
    echo "Starting simulation for replicate $i"
    echo "--------------------------------------------------------"
    
    mkdir -p "${REPLICATE_DIR}/results"

    # Step 1: Run the Genetic Simulation
    SIM_COMMAND="$VENV_PYTHON_EXE -u $SIM_SCRIPT --output_dir $REPLICATE_DIR --replicate_id $i --file "/mnt/nfs2/bioenv/sg802/hybrid_sim_project/beetle_input.csv" -npa 100 -npb 100 -HG 3000 -nc 58 -oh -gmap"
    echo "Command: $SIM_COMMAND"
    $SIM_COMMAND 

    echo "Simulation for replicate $i complete."

    # Step 2: Run the Analysis on the Simulation Outputs
    echo "Starting the analysis of outputs for replicate $i"
    ANALYSIS_COMMAND="$VENV_PYTHON_EXE -u $ANALYSIS_SCRIPT --input_dir $REPLICATE_DIR --output_file $ANALYSIS_OUTPUT_FILE --replicate_id $i"
    echo "Command: $ANALYSIS_COMMAND"
    $ANALYSIS_COMMAND
    
    echo "Matching complete for replicate $i."
    
    # Step 3: Generate the triangle plot
    echo "Generating triangle plot for replicate $i"

    # FIX: Extract the generation label (e.g., "HG1525") and convert it to a number ("1525") 
    RAW_GEN_LABEL=$(tail -n 1 "$ANALYSIS_OUTPUT_FILE" | awk -F, '{print $2}' | tr -d '"')
    MATCHING_GEN_NUMBER=${RAW_GEN_LABEL/HG/} # Strips the "HG" prefix
    
    TRIANGLE_PLOT_OUTPUT="${REPLICATE_DIR}/results/triangle_plot_rep_${i}.png"

    PLOTTING_COMMAND="$VENV_PYTHON_EXE -u $PLOTTING_SCRIPT --input_file ${REPLICATE_DIR}/results/results_rep_${i}_individual_hi_het.csv --output_file $TRIANGLE_PLOT_OUTPUT --highlight_gen $MATCHING_GEN_NUMBER"
    echo "Command: $PLOTTING_COMMAND"
    $PLOTTING_COMMAND
        
    echo "Plotting complete for replicate $i."
done

echo " "
echo "Workflow for replicate range $1 to $2 complete."