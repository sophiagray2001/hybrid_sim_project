import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator 
import re

# --- REVISED FUNCTION: FULLY SCALED BY NE ---
def plot_crossing_time_distribution_combined(crossing_df: pd.DataFrame, ne_value: float, save_filename: str):
    """
    Plots the distribution of generations required for HET to decrease to 
    the parental mean level across ALL replicates, with the X-axis fully 
    scaled by the Effective Population Size (Ne).
    """
    
    if crossing_df.empty:
        print("Error: Input data DataFrame is empty.")
        return
    
    # 1. Prepare Data
    # Extract original absolute time (G)
    crossing_df['crossing_time'] = crossing_df['matching_hybrid_gen'].astype(str).str.extract(r'HG(\d+)').astype(float)
    crossing_times_g = crossing_df['crossing_time'].dropna().tolist()

    if not crossing_times_g:
        print("Warning: No valid crossing times found in the combined data.")
        return
        
    # CRUCIAL STEP 1: CONVERT DATA POINTS TO Ne GENERATIONS (T)
    crossing_times_ne = [t / ne_value for t in crossing_times_g]
    
    # Define the new maximum X-axis limit in Ne units (3000 / 200 = 15)
    MAX_NE_TIME = 3000 / ne_value 

    # 2. Plot the Distribution (Histogram)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # CHANGE: DOUBLE THE NUMBER OF BINS FOR TWICE THE RESOLUTION
    # 15 * 2 = 30 bins
    #num_bins = int(MAX_NE_TIME) * 2
    num_bins = 16
            
    # Plotting the histogram using the NE-SCALED DATA and NE-SCALED RANGE
    ax.hist(
        crossing_times_ne, 
        bins=num_bins, 
        range=(0, MAX_NE_TIME), 
        edgecolor='black', 
        color='darkgray', 
        alpha=0.8
    )
    
    # 3. Calculate and Convert Statistics
    crossing_series_g = pd.Series(crossing_times_g)
    
    # Absolute Generations (G) - Used only for calculating Ne stats
    mean_time_g = crossing_series_g.mean()
    ci_lower_g = crossing_series_g.quantile(0.025)
    ci_upper_g = crossing_series_g.quantile(0.975)
    
    # Ne Generations (G / Ne)
    mean_time_ne = mean_time_g / ne_value
    ci_lower_ne = ci_lower_g / ne_value
    ci_upper_ne = ci_upper_g / ne_value
    
    # 4. Add Statistics Lines and Labels
    
    # Mean line (Plotted at NE-SCALED X-coordinate, label shows Ne G)
    ax.axvline(
        mean_time_ne, 
        color='red', 
        linestyle='--', 
        linewidth=2, 
        label=f'Mean Time: {mean_time_ne:.2f} Ne Gens (N={len(crossing_times_g)})'
    )
    
    # 95% CI lines (Plotted at NE-SCALED X-coordinate, label shows Ne G)
    ax.axvline(ci_lower_ne, color='blue', linestyle=':', linewidth=1.5, label=f'95% CI: ({ci_lower_ne:.2f}-{ci_upper_ne:.2f}) Ne Gens')
    ax.axvline(ci_upper_ne, color='blue', linestyle=':', linewidth=1.5)

    # 5. Set up labels and title
    
    # CRUCIAL CHANGE: Update X-axis label to reflect the scaling
    ax.set_xlabel(f"Time (Ne Generations)", fontsize=12)
    
    ax.set_ylabel("Number of Replicates", fontsize=12)
    
    # Enforce X-axis limits and ticks (0 to 15, ticks every 1 unit)
    ax.set_xlim(0, MAX_NE_TIME)
    ax.set_xticks(range(int(MAX_NE_TIME) + 1)) 
    
    # Enforce Y-axis ticks to be integers
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right')
    
    plt.savefig(save_filename, bbox_inches='tight')
    plt.close()
    print(f"\nTime distribution plot saved to: {save_filename}")


if __name__ == "__main__":
    
    # --- 1. DEFINE FLEXIBLE BATCH CONFIGURATION ---
    
    # ----------------------------------------------------
    # Configuration - EDIT THIS SECTION FOR SINGLE/MULTIPLE BATCHES
    # ----------------------------------------------------
    
    # Define a list of crossing files to load.
    CROSSING_FILE_CONFIGS = [
        # Batch 1 Crossing File (Replicates 1-50)
        {
            "BASE_DIR": "/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs_immigrationboth/",
            "FILENAME": "combined_matching_generations_immigrationboth.csv"
        }
    ]
    
    # Define your effective population size (Ne)
    N_E_VALUE = 200.0 
    
    # ----------------------------------------------------
    
    # Setup for Output Paths
    
    # Use the first BASE_DIR to determine the parent directory for the results folder
    if not CROSSING_FILE_CONFIGS:
        print("FATAL ERROR: No crossing file configurations defined. Aborting.")
        exit(1)
        
    INPUT_DATA_BASE = os.path.dirname(CROSSING_FILE_CONFIGS[0]["BASE_DIR"].rstrip(os.sep))
    RESULTS_BASE_DIR = os.path.join(INPUT_DATA_BASE, "results")

    # Define the output path for the final combined plot
    COMBINED_PLOT_OUTPUT = os.path.join(
        RESULTS_BASE_DIR, 
        "time_to_parent_het_distribution_combined_Ne_scaled_both.png" 
    )
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(COMBINED_PLOT_OUTPUT), exist_ok=True)
    
    # --- 2. LOAD AND CONCATENATE THE CROSSING DATA ---
    all_dfs = []
    
    print("Loading and combining crossing data from all configured directories...")
    
    for i, config in enumerate(CROSSING_FILE_CONFIGS):
        crossing_path = os.path.join(config["BASE_DIR"], config["FILENAME"])
        
        print(f"  -> Loading Batch {i+1} from: {crossing_path}")
        
        try:
            df = pd.read_csv(crossing_path)
            all_dfs.append(df)
            
        except FileNotFoundError:
            print(f"WARNING: Crossing file not found at: {crossing_path}")
            print(f"Skipping Batch {i+1}.")
            continue
        except pd.errors.EmptyDataError:
            print(f"WARNING: File is empty or not a valid CSV: {crossing_path}")
            print(f"Skipping Batch {i+1}.")
            continue
            
    # Check if any data was loaded
    if not all_dfs:
        print("FATAL ERROR: No crossing data files were successfully loaded. Aborting.")
        exit(1)
        
    # Concatenate all loaded DataFrames
    combined_crossing_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Loaded a total of {len(combined_crossing_df)} replicates.")
    
    # --- 3. RUN THE PLOTTING FUNCTION ---
    plot_crossing_time_distribution_combined(
        crossing_df=combined_crossing_df, 
        ne_value=N_E_VALUE,         
        save_filename=COMBINED_PLOT_OUTPUT
    )