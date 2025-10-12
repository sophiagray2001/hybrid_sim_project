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
        # Ensure 'replicate_id' exists in your CSV and is the correct column name
        rep_id = row['replicate_id'] 
        matching_gen = row['matching_hybrid_gen']
        
        # Find the correct base directory using the map
        base_dir = rep_dir_map.get(rep_id)  # <--- This is where 'base_dir' is defined
        if not base_dir:
            # This warning should now be fixed by the correct range in __main__
            print(f"Warning: Replicate {rep_id} not found in the directory map. Skipping.")
            continue
            
        # Construct the path to this replicate's full data file
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
    
    ax.hist(hi_at_crossing, bins=4, edgecolor='black', color='gray', alpha=0.7, rwidth=1.0)
    
    # Calculate and plot Mean, Median, and 95% CI
    hi_series = pd.Series(hi_at_crossing)
    mean_hi = hi_series.mean()
    ci_lower = hi_series.quantile(0.025)
    ci_upper = hi_series.quantile(0.975)
    
    ax.axvline(mean_hi, color='red', linestyle='--', linewidth=2, label=f'Mean HI: {mean_hi:.3f}')
    ax.axvline(ci_lower, color='blue', linestyle=':', linewidth=1.5, label=f'95% CI: ({ci_lower:.3f}-{ci_upper:.3f})')
    ax.axvline(ci_upper, color='blue', linestyle=':', linewidth=1.5)
    
    # Set up labels and title
    ax.set_xlabel("Mean Hybrid Index (HI) at Parental HET Intercept", fontsize=10)
    ax.set_ylabel("Number of Replicates", fontsize=10)
    ax.set_xlim(0, 1) # HI is always between 0 and 1
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right')
    
    plt.savefig(save_filename, bbox_inches='tight')
    plt.close()
    print(f"\nHI distribution plot saved to: {save_filename}")


if __name__ == "__main__":
    # --- 1. DEFINE FLEXIBLE BATCH CONFIGURATION ---
    
    # ----------------------------------------------------
    # Configuration - EDIT THIS SECTION FOR SINGLE/MULTIPLE BATCHES
    # ----------------------------------------------------
    
    # Define a list of configurations. 
    # Current setup is for a single directory containing Replicates 1-100.
    BATCH_CONFIGS = [
        # Batch 1: Replicates 1-100 (CORRECTED RANGE)
        {
            "BASE_DIR": r"/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs_immigrationboth/",
            "REPLICATE_IDS": list(range(1, 51)), # <-- FIX: Now includes 1 through 100
            "CROSSING_FILENAME": "combined_matching_generations_immigrationboth.csv"
        }
    ]
    
    # EXAMPLE: To use two separate batches, comment the above and uncomment below:
    # BATCH_CONFIGS = [
    #     {
    #         "BASE_DIR": r"/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs/",
    #         "REPLICATE_IDS": list(range(1, 51)),
    #         "CROSSING_FILENAME": "combined_matching_generations.csv"
    #     },
    #     {
    #         "BASE_DIR": r"/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs_linked_closed/",
    #         "REPLICATE_IDS": list(range(51, 101)),
    #         "CROSSING_FILENAME": "combined_matching_generations_second_batch.csv"
    #     }
    # ]
    
    # ----------------------------------------------------
    
    # Setup for Output Paths
    
    # Use the first BASE_DIR to determine the parent directory for the results folder
    INPUT_DATA_BASE = os.path.dirname(BATCH_CONFIGS[0]["BASE_DIR"].rstrip(os.sep))
    RESULTS_BASE_DIR = os.path.join(INPUT_DATA_BASE, "results")
    
    # --- 2. LOAD AND CONCATENATE THE CROSSING DATA & BUILD MAP ---
    all_dfs = []
    rep_dir_map = {}
    
    print("Loading and combining crossing data from all configured directories...")

    for i, config in enumerate(BATCH_CONFIGS):
        base_dir = config["BASE_DIR"]
        rep_ids = config["REPLICATE_IDS"]
        crossing_file = config["CROSSING_FILENAME"]
        
        crossing_path = os.path.join(base_dir, crossing_file)
        
        print(f"  -> Loading Batch {i+1} from: {crossing_path}")
        
        try:
            # Load the CSV
            df = pd.read_csv(crossing_path)
            all_dfs.append(df)
            
            # Build the Replicate-to-Directory map for this batch
            for rep_id in rep_ids:
                rep_dir_map[rep_id] = base_dir
                
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
    print(f"Successfully combined data with {len(combined_crossing_df)} total rows.")
    
    # Define the output path for the final plot
    HI_DISTRIBUTION_PLOT_OUTPUT = os.path.join(
        RESULTS_BASE_DIR, 
        "hi_at_het_crossing_distribution_immigrateboth.png" 
    )

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(HI_DISTRIBUTION_PLOT_OUTPUT), exist_ok=True)
    
    # --- 3. RUN THE PLOTTING FUNCTION ---
    plot_hi_at_crossing_distribution(
        crossing_df=combined_crossing_df, 
        rep_dir_map=rep_dir_map,         
        save_filename=HI_DISTRIBUTION_PLOT_OUTPUT
    )