import os
import pandas as pd
import matplotlib.pyplot as plt

# CHECK THIS FUNCTION DEFINITION AT THE TOP OF YOUR SCRIPT!
def plot_hi_at_crossing_distribution(crossing_df: pd.DataFrame, rep_dir_map: dict, save_filename: str):
    """
    Plots the distribution of Hybrid Indices (HI) at the moment each replicate's 
    Heterozygosity (HET) dropped to the mean parental level.
    Accepts the combined DataFrame directly as 'crossing_df'
    and the base directory map as 'rep_dir_map'.
    """
    
    hi_at_crossing = []
    
    # 1. Use the combined DataFrame passed directly to the function
    print(f"Loaded crossing data for {len(crossing_df)} total replicates.")
    print("Extracting HI values at HET crossing points for each replicate...")
    
    # 2. Iterate through each row (replicate) of the combined crossing data
    for index, row in crossing_df.iterrows():
        rep_id = row['replicate_id']
        matching_gen = row['matching_hybrid_gen']
        
        # Find the correct base directory using the map
        base_dir = rep_dir_map.get(rep_id)  # <--- This is where 'base_dir' is defined
        if not base_dir:
            print(f"Warning: Replicate {rep_id} not found in the directory map. Skipping.")
            continue
            
        # Construct the path to this replicate's full data file
        # The line below must use 'base_dir'
        rep_data_path = os.path.join(base_dir, f'replicate_{rep_id}', 'results', f'results_rep_{rep_id}_individual_hi_het.csv')
        
        try:
            # Load the full HI/HET data for this replicate
            full_rep_df = pd.read_csv(rep_data_path)
            
            # Calculate the mean HI for each generation in this replicate's data
            mean_rep_hi = full_rep_df.groupby('generation')['HI'].mean()
            
            # Extract the specific HI value at the matching generation
            if matching_gen in mean_rep_hi.index:
                hi_value = mean_rep_hi.loc[matching_gen]
                hi_at_crossing.append(hi_value)
            else:
                print(f"Warning: Replicate {rep_id} - Matching generation '{matching_gen}' not found in data file.")

        except FileNotFoundError:
            print(f"Warning: Could not find full data file for Replicate {rep_id} at {rep_data_path}")
            continue

    if not hi_at_crossing:
        print("Error: No HI values could be extracted. Cannot create plot.")
        return

    # 3. Plot the Distribution (Histogram) of the collected HI values
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(hi_at_crossing, bins=20, edgecolor='black', color='gray', alpha=0.7, rwidth=1.0)
    
    # Calculate and plot Mean, Median, and 95% CI
    hi_series = pd.Series(hi_at_crossing)
    mean_hi = hi_series.mean()
    #median_hi = hi_series.median()
    ci_lower = hi_series.quantile(0.025)
    ci_upper = hi_series.quantile(0.975)
    
    ax.axvline(mean_hi, color='red', linestyle='--', linewidth=2, label=f'Mean HI: {mean_hi:.3f}')
    #ax.axvline(median_hi, color='orange', linestyle='-', linewidth=2, label=f'Median HI: {median_hi:.3f}')
    ax.axvline(ci_lower, color='gray', linestyle=':', linewidth=1.5, label=f'95% CI: ({ci_lower:.3f}-{ci_upper:.3f})')
    ax.axvline(ci_upper, color='gray', linestyle=':', linewidth=1.5)
    
    # Set up labels and title
    ax.set_xlabel("Mean Hybrid Index (HI) at Parental HET Intercept", fontsize=10)
    ax.set_ylabel("Number of Replicates", fontsize=10)
    #ax.set_title("Distribution of Hybrid Index when HET Reaches Parental Level", fontsize=14)
    ax.set_xlim(0, 1) # HI is always between 0 and 1
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right')
    
    plt.savefig(save_filename, bbox_inches='tight')
    plt.close()
    print(f"\nHI distribution plot saved to: {save_filename}")


if __name__ == "__main__":
    # --- 1. DEFINE FILE PATHS (FIXED PATH 1) ---
    
    # Base path for Replicates 1-50
    BASE_DIR_1 = r"C:\Users\sg802\Documents\git_clone\hybrid_sim_project\input_data\simulation_outputs"
    REPLICATE_IDS_1 = list(range(1, 51)) 
    
    # CORRECTED LOGIC: The 'results' folder is a sibling of 'simulation_outputs' 
    # within the 'input_data' directory.
    INPUT_DATA_BASE = os.path.dirname(BASE_DIR_1) 
    RESULTS_BASE_DIR = os.path.join(INPUT_DATA_BASE, "results")
    
    CROSSING_PATH_1 = os.path.join(
        BASE_DIR_1, 
        "combined_matching_generations.csv" 
    )
    
    # Base path for Replicates 51-100 (No Change Needed Here)
    BASE_DIR_2 = r"C:\Users\sg802\Documents\git_clone\hybrid_sim_project\input_data\simulation_outputs_second_batch"
    REPLICATE_IDS_2 = list(range(51, 101))
    
    # Determine the crossing file path for Batch 2 (using your specific file name)
    CROSSING_PATH_2 = os.path.join(
        BASE_DIR_2, 
        "combined_matching_generations_second_batch.csv" 
    )
    
    # --- 2. LOAD AND CONCATENATE THE CROSSING DATA ---
    print("Loading and combining crossing data from both directories...")
    try:
        df1 = pd.read_csv(CROSSING_PATH_1)
        df2 = pd.read_csv(CROSSING_PATH_2)
        combined_crossing_df = pd.concat([df1, df2], ignore_index=True)
    except FileNotFoundError as e:
        print(f"Error: One of the crossing files was not found. Please check paths:")
        print(f"Path 1: {CROSSING_PATH_1}")
        print(f"Path 2: {CROSSING_PATH_2}")
        print(f"Original Error: {e}")
        exit(1)
    
    # --- 3. CREATE REPLICATE-TO-DIRECTORY MAP ---
    # This map is used by the plotting function to find the individual rep_X_hi_het files
    rep_dir_map = {}
    for rep_id in REPLICATE_IDS_1:
        rep_dir_map[rep_id] = BASE_DIR_1
    for rep_id in REPLICATE_IDS_2:
        rep_dir_map[rep_id] = BASE_DIR_2
    
    # Define the output path for the final plot (using the centralized results folder)
    HI_DISTRIBUTION_PLOT_OUTPUT = os.path.join(
        RESULTS_BASE_DIR, 
        "hi_at_het_crossing_distribution.png" 
    )

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(HI_DISTRIBUTION_PLOT_OUTPUT), exist_ok=True)
    
    # --- 4. RUN THE PLOTTING FUNCTION ---
    # The function signature (plot_hi_at_crossing_distribution) must be the 
    # one provided in the previous turn to accept the DataFrame and the map.
    plot_hi_at_crossing_distribution(
        crossing_df=combined_crossing_df, 
        rep_dir_map=rep_dir_map,         
        save_filename=HI_DISTRIBUTION_PLOT_OUTPUT
    )