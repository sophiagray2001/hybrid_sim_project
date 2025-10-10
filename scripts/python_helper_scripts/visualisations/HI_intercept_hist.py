import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_hi_at_crossing_distribution(crossing_data_path: str, base_output_dir: str, save_filename: str):
    """
    Plots the distribution of Hybrid Indices (HI) at the moment each replicate's 
    Heterozygosity (HET) dropped to the mean parental level.
    """
    
    # 1. Load the pre-calculated crossing data
    try:
        crossing_df = pd.read_csv(crossing_data_path)
    except FileNotFoundError:
        print(f"Error: Crossing data file not found at {crossing_data_path}")
        return
        
    hi_at_crossing = []
    
    print("Extracting HI values at HET crossing points for each replicate...")
    # 2. Iterate through each replicate to find its HI at the crossing generation
    for index, row in crossing_df.iterrows():
        rep_id = row['replicate_id']
        matching_gen = row['matching_hybrid_gen']
        
        # Construct the path to this replicate's full data file
        rep_data_path = os.path.join(base_output_dir, f'replicate_{rep_id}', 'results', f'results_rep_{rep_id}_individual_hi_het.csv')
        
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
    
    ax.hist(hi_at_crossing, bins=20, edgecolor='black', color='darkcyan', alpha=0.7, rwidth=1.0)
    
    # Calculate and plot Mean, Median, and 95% CI
    hi_series = pd.Series(hi_at_crossing)
    mean_hi = hi_series.mean()
    median_hi = hi_series.median()
    ci_lower = hi_series.quantile(0.025)
    ci_upper = hi_series.quantile(0.975)
    
    ax.axvline(mean_hi, color='red', linestyle='--', linewidth=2, label=f'Mean HI: {mean_hi:.3f}')
    ax.axvline(median_hi, color='orange', linestyle='-', linewidth=2, label=f'Median HI: {median_hi:.3f}')
    ax.axvline(ci_lower, color='gray', linestyle=':', linewidth=1.5, label=f'95% CI: ({ci_lower:.3f}-{ci_upper:.3f})')
    ax.axvline(ci_upper, color='gray', linestyle=':', linewidth=1.5)
    
    # Set up labels and title
    ax.set_xlabel("Mean Hybrid Index (HI) at HET Crossing Point", fontsize=12)
    ax.set_ylabel("Number of Replicates", fontsize=12)
    ax.set_title("Distribution of Hybrid Index when HET Reaches Parental Level", fontsize=14)
    ax.set_xlim(0, 1) # HI is always between 0 and 1
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right')
    
    plt.savefig(save_filename, bbox_inches='tight')
    plt.close()
    print(f"\nHI distribution plot saved to: {save_filename}")


if __name__ == "__main__":
    # --- 1. DEFINE FILE PATHS ---
    
    # This is the directory containing your 'replicate_X' folders
    BASE_OUTPUT_DIR = "/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs/"
    
    # This is the directory for your combined results files (like the CSV)
    # NOTE: Adjust this path if your CSV is in a different location
    RESULTS_DIR = os.path.join(os.path.dirname(BASE_OUTPUT_DIR.rstrip('/')), "results")
    
    # Path to your pre-calculated crossing data CSV
    CROSSING_INPUT_PATH = os.path.join(
        RESULTS_DIR, 
        "combined_matching_generations.csv" 
    )
    
    # Path for the final output plot
    HI_DISTRIBUTION_PLOT_OUTPUT = os.path.join(
        RESULTS_DIR, 
        "hi_at_het_crossing_distribution.png" 
    )

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(HI_DISTRIBUTION_PLOT_OUTPUT), exist_ok=True)
    
    # --- 2. RUN THE PLOTTING FUNCTION ---
    plot_hi_at_crossing_distribution(
        crossing_data_path=CROSSING_INPUT_PATH,
        base_output_dir=BASE_OUTPUT_DIR, # Function needs this to find individual replicate data
        save_filename=HI_DISTRIBUTION_PLOT_OUTPUT
    )