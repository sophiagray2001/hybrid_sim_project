import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde

# --- Configuration for all three datasets (UPDATED with Hex Codes) ---
# The order in this dictionary determines the style applied (i=0, i=1, i=2)
DATASET_CONFIGS = {
    "Tight Linkage": {
        "BASE_DIR": r"/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs_extreme_linkage_0.05/",
        "REPLICATE_IDS": list(range(1, 51)),
        "CROSSING_FILENAME": "combined_matching_generations_extreme_linkage_0.05.csv",
        "color": "#1f77b4" # Blue
    }#,
    #"Linked": {
    #    "BASE_DIR": r"/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs_linked_closed/",
    #    "REPLICATE_IDS": list(range(1, 51)),
    #    "CROSSING_FILENAME": "combined_matching_generations_linked_closed.csv",
    #    "color": "#ff7f0e" # Orange
    #},
    #"20 Chromosomes": {
    #    "BASE_DIR": r"/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs_closed_20chr/",
    #    "REPLICATE_IDS": list(range(1, 51)),
    #    "CROSSING_FILENAME": "combined_matching_generations_closed_20chr.csv",
    #    "color": "#d62728" # Red
    #}
}

# --- Function to Extract HI at the Crossing Point for a single dataset (UNCHANGED) ---

def extract_hi_at_crossing(config_name: str, config: dict) -> pd.DataFrame:
    """
    Loads the HET crossing file and extracts the Mean HI value from the full
    replicate data file for the specific 'matching_hybrid_gen'.
    Returns a DataFrame of ['Dataset', 'Mean_Hybrid_Index'] for plotting.
    """
    base_dir = config["BASE_DIR"]
    crossing_path = os.path.join(base_dir, config["CROSSING_FILENAME"])
    rep_ids = config["REPLICATE_IDS"]
    
    hi_at_crossing_data = []

    try:
        crossing_df = pd.read_csv(crossing_path)
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        print(f"Warning: Could not load crossing file for {config_name}: {e}. Skipping.")
        return pd.DataFrame()

    print(f"Processing {config_name} with {len(crossing_df)} replicates in crossing file.")
    
    # Filter the crossing_df to only include the replicate IDs we are interested in
    crossing_df = crossing_df[crossing_df['replicate_id'].isin(rep_ids)].reset_index(drop=True)

    for index, row in crossing_df.iterrows():
        rep_id = row['replicate_id']
        matching_gen = row['matching_hybrid_gen']
        
        # Construct the path to this replicate's full data file
        rep_data_path = os.path.join(
            base_dir, 
            f'replicate_{rep_id}', 
            'results', 
            f'results_rep_{rep_id}_individual_hi_het.csv'
        )
        
        try:
            full_rep_df = pd.read_csv(rep_data_path)
            
            # Calculate the mean HI for each generation
            mean_rep_hi = full_rep_df.groupby('generation')['HI'].mean()
            
            # Extract the specific HI value at the matching generation
            if matching_gen in mean_rep_hi.index:
                hi_value = mean_rep_hi.loc[matching_gen]
                
                hi_at_crossing_data.append({
                    'Dataset': config_name,
                    'Mean_Hybrid_Index': hi_value
                })

        except FileNotFoundError:
            continue
        except KeyError:
             print(f"Error: Missing 'HI' or 'generation' column in {rep_data_path}. Skipping replicate.")
             continue

    return pd.DataFrame(hi_at_crossing_data)


# --- Function to Plot the Distribution (KDE) with Distinct Statistics Lines (MODIFIED) ---

def plot_hi_crossing_kde(
    combined_df: pd.DataFrame, 
    dataset_configs: dict, 
    save_filename: str,
    bw_adjustment: float = 1.0 # NEW parameter for bandwidth (smoothness) adjustment
):
    """
    Generates a combined KDE plot for all datasets, showing HI distribution
    at the HET crossing point, using SOLID KDE lines and DISTINCT Mean/CI line styles.
    
    The bw_adjustment parameter controls KDE smoothness: < 1.0 is less smooth, > 1.0 is smoother.
    """
    print("\n--- Generating Custom Styled HI KDE Plot (Fixed Alignment) ---")
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    line_handles = []
    line_labels = []
    dataset_names = list(dataset_configs.keys())

    # 1. Plot the KDE Curves using Seaborn
    palette = {name: config['color'] for name, config in dataset_configs.items()}
    
    sns.kdeplot(
        data=combined_df,
        x='Mean_Hybrid_Index',
        hue='Dataset',
        palette=palette,
        fill=False,      # No shading
        linewidth=2.5,
        ax=ax,
        cut=0,
        legend=False,
        bw_adjust=bw_adjustment # <-- Use the new parameter here
    )
    
    # 2. Iterate through each dataset to calculate and plot statistics
    for i, name in enumerate(dataset_names):
        config = dataset_configs[name]
        subset = combined_df[combined_df['Dataset'] == name]['Mean_Hybrid_Index']
        
        if subset.empty:
            continue

        plot_color = config['color']
        
        # --- Determine unique styles for the statistics lines ---
        if i == 0: # First dataset (Unlinked)
            mean_ls = '--'
            ci_ls = ':'
        elif i == 1: # Second dataset (Linked)
            mean_ls = '-.'
            ci_ls = (0, (1, 5)) # Sparse dot pattern
        else: # i >= 2 (20 Chromosomes)
            mean_ls = (0, (3, 1, 1, 1)) # Dash-dot-dot pattern
            ci_ls = (0, (1, 1)) # Very dense dot pattern

        # Calculate statistics
        mean_hi = subset.mean()
        ci_lower = subset.quantile(0.025)
        ci_upper = subset.quantile(0.975)
        
        # Plot Mean Line (using distinct linestyle)
        ax.axvline(
            mean_hi, 
            color=plot_color, 
            linestyle=mean_ls, 
            linewidth=2, 
            zorder=3
        )
        # Plot CI Lines (using distinct linestyle)
        ax.axvline(ci_lower, color=plot_color, linestyle=ci_ls, linewidth=1.5, zorder=3)
        ax.axvline(ci_upper, color=plot_color, linestyle=ci_ls, linewidth=1.5, zorder=3)
        
        # --- Build Custom Legend Handles/Labels (FIXED ALIGNMENT) ---
        # 1. KDE Curve Handle: Manually create a Line2D object with the correct color and solid style.
        kde_handle = plt.Line2D([0], [0], color=plot_color, linestyle='-', linewidth=2.5)
        line_handles.append(kde_handle)
        line_labels.append(f'{name} (KDE)')
        
        # 2. Mean Handle (Using the specific linestyle)
        mean_handle = plt.Line2D([0], [0], color=plot_color, linestyle=mean_ls, linewidth=2)
        line_handles.append(mean_handle)
        line_labels.append(f'{name} Mean HI: {mean_hi:.3f}')
        
        # 3. CI Handle (Using the specific linestyle)
        ci_handle = plt.Line2D([0], [0], color=plot_color, linestyle=ci_ls, linewidth=1.5)
        line_handles.append(ci_handle)
        line_labels.append(f'{name} 95% CI: ({ci_lower:.3f}-{ci_upper:.3f})')

    # 3. Final Plot Cleanup and Legend
    ax.set_xlabel("Mean Hybrid Index (HI) at Parental HET Intercept", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    # Title removed per previous request
    ax.set_xlim(0, 1) # HI is always between 0 and 1
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Use the custom handles and labels for the legend
    ax.legend(
        line_handles, 
        line_labels, 
        loc='upper left', 
        bbox_to_anchor=(1.01, 1.0),
        #title="Dataset and Statistics",
        frameon=True,
        shadow=False,
        fontsize='small'
    )
    
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(save_filename, bbox_inches='tight')
    plt.close()
    print(f"\nFinal KDE distribution plot saved to: {save_filename}")


# --- Main Execution Block (MODIFIED to use bw_adjustment) ---

if __name__ == "__main__":
    
    # --- 1. SET UP OUTPUT PATH ---
    first_base_dir = list(DATASET_CONFIGS.values())[0]["BASE_DIR"].rstrip(os.sep)
    INPUT_DATA_BASE = os.path.dirname(first_base_dir)
    RESULTS_BASE_DIR = os.path.join(INPUT_DATA_BASE, "results")
    
    # Define the final PDF output path
    HI_DISTRIBUTION_PLOT_OUTPUT = os.path.join(
        RESULTS_BASE_DIR, 
        "hi_at_het_crossing_distribution_extreme_linkage.png"
    )

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(HI_DISTRIBUTION_PLOT_OUTPUT), exist_ok=True)
    
    # --- 2. LOAD AND COMBINE DATA FROM ALL DATASETS ---
    full_data_list = []
    
    for name, config in DATASET_CONFIGS.items():
        processed_df = extract_hi_at_crossing(name, config)
        if not processed_df.empty:
            full_data_list.append(processed_df)

    if not full_data_list:
        print("\nFATAL: No HI data could be extracted for any dataset. Aborting.")
        exit(1)
    
    combined_df = pd.concat(full_data_list, ignore_index=True)
    print(f"\nSuccessfully combined data for {len(combined_df)} total replicate-generations.")
    
    # --- 3. RUN THE PLOTTING FUNCTION ---
    # Set bw_adjustment to a value < 1.0 to reduce smoothness (0.5 is a good starting point)
    plot_hi_crossing_kde(
        combined_df=combined_df, 
        dataset_configs=DATASET_CONFIGS, 
        save_filename=HI_DISTRIBUTION_PLOT_OUTPUT,
        bw_adjustment=0.4 # Change this value to adjust the smoothness
    )