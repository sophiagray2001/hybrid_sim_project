#!/bin/bash
#SBATCH --job-name=beetle_sim
#SBATCH --output=sim_%j.out
#SBATCH --error=sim_%j.err
#SBATCH --time=24:00:00          # adjust runtime if needed
#SBATCH --cpus-per-task=16       # match --threads in Python
#SBATCH --mem=128G                # adjust if needed
#SBATCH -p long           # e.g., normal, shared
#SBATCH -A bioenv       # your project/account

# Load Anaconda and activate your environment
module load Anaconda3/2024.06-1
conda activate sim_env

# Optional: limit threads for BLAS/OpenMP (reduces memory per process)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Run your simulation
python combined_RW_sim.py \
    -f input_file_beetle.csv \
    -nc 85 \
    -npa 500 \
    -npb 500 \
    -HG 1000 \
    -no "{0: 0.0, 1: 0.0, 2: 1.0}" \
    --threads 16 \
    -tp \
    --seed 30 \
    -on 'Beetle_Run_1'
