import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde

# --- Configuration for all three datasets (UPDATED with Hex Codes and bw_adjust) ---
# The order in this dictionary determines the style applied (i=0, i=1, i=2, i=3)
DATASET_CONFIGS = {
    "Tight Linkage": {
        "BASE_DIR": r"/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs_extreme_linkage_0.05/",
        "REPLICATE_IDS": list(range(1, 51)),
        "CROSSING_FILENAME": "combined_matching_generations_extreme_linkage_0.05.csv",
        "color": "#9467bd", # Blue
        "bw_adjust": 0.5 # Less Smooth (More detail)
    },
    "Linked": {
        "BASE_DIR": r"/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs_linked_closed/",
        "REPLICATE_IDS": list(range(1, 51)),
        "CROSSING_FILENAME": "combined_matching_generations_linked_closed.csv",
        "color": "#ff7f0e", # Orange
        "bw_adjust": 0.9 # Moderately less smooth
    },
    "20 Chromosomes": {
        "BASE_DIR": r"/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs_closed_20chr/",
        "REPLICATE_IDS": list(range(1, 51)),
        "CROSSING_FILENAME": "combined_matching_generations_closed_20chr.csv",
        "color": "#d62728", # Red
        "bw_adjust": 0.9 # Moderately more smooth
    },
    # --- FOURTH DATASET ADDED HERE ---
    "Unlinked": {
        "BASE_DIR": r"/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs_unlinked_closed/",
        "REPLICATE_IDS": list(range(1, 51)),
        "CROSSING_FILENAME": "combined_matching_generations.csv",
        "color": "#2ca02c", # Green
        "bw_adjust": 0.9 # Default smoothness (Most smooth)
    },
    # --- Fifth DATASET ADDED HERE ---
    "No Recombination": {
        "BASE_DIR": r"/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs_no_recombination_50/",
        "REPLICATE_IDS": list(range(1, 51)),
        "CROSSING_FILENAME": "combined_matching_generations_no_recombination.csv",
        "color": "#ffd700", # yellow
        "bw_adjust": 0.5 
    }
    
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
    save_filename: str
    # Removed global bw_adjustment parameter
):
    """
    Generates a combined KDE plot for all datasets, showing HI distribution
    at the HET crossing point, using SOLID KDE lines and DISTINCT Mean lines.
    KDE smoothness is now controlled individually per dataset via DATASET_CONFIGS.
    """
    print("\n--- Generating Custom Styled HI KDE Plot (Individual BW Adjust) ---")
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    line_handles = []
    line_labels = []
    dataset_names = list(dataset_configs.keys())
    
    # 1. Iterate through each dataset to calculate statistics AND plot KDE
    for i, name in enumerate(dataset_names):
        config = dataset_configs[name]
        subset_df = combined_df[combined_df['Dataset'] == name].copy()
        
        if subset_df.empty:
            continue
        
        subset = subset_df['Mean_Hybrid_Index']
        plot_color = config['color']
        bw_adjust_val = config['bw_adjust'] # <-- Retrieve the individual BW adjust

        # --- Determine unique styles for the mean lines ---
        if i == 0: # First dataset (Tight Linkage)
            mean_ls = '--'
        elif i == 1: # Second dataset (Linked)
            mean_ls = '-.'
        elif i == 2: # Third dataset (20 Chromosomes)
            mean_ls = (0, (3, 1, 1, 1)) # Dash-dot-dot pattern
        elif i == 3: # FOURTH DATASET (Unlinked)
            mean_ls = (0, (5, 5)) # Long dash pattern
        else: # For any additional datasets
            mean_ls = '-'

        # 1a. Plot the KDE Curve for this subset using its individual bw_adjust
        sns.kdeplot(
            data=subset_df,
            x='Mean_Hybrid_Index',
            color=plot_color, # Plotting one by one, so use 'color' not 'hue'
            fill=False,
            linewidth=2.5,
            ax=ax,
            cut=0,
            legend=False,
            bw_adjust=bw_adjust_val # <-- Apply individual adjustment
        )

        # 1b. Calculate and plot Mean
        mean_hi = subset.mean()
        
        # Plot Mean Line (using distinct linestyle)
        ax.axvline(
            mean_hi, 
            color=plot_color, 
            linestyle=mean_ls, 
            linewidth=2, 
            zorder=3
        )
        
        # --- Build Custom Legend Handles/Labels (Mean and KDE only) ---
        # 1. KDE Curve Handle: 
        kde_handle = plt.Line2D([0], [0], color=plot_color, linestyle='-', linewidth=2.5)
        line_handles.append(kde_handle)
        line_labels.append(f'{name} (KDE, BW={bw_adjust_val})')
        
        # 2. Mean Handle (Using the specific linestyle)
        mean_handle = plt.Line2D([0], [0], color=plot_color, linestyle=mean_ls, linewidth=2)
        line_handles.append(mean_handle)
        line_labels.append(f'{name} Mean HI: {mean_hi:.3f}')


    # 2. Final Plot Cleanup and Legend
    ax.set_xlabel("Mean Hybrid Index (HI) at Parental HET Intercept", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_xlim(0, 1) # HI is always between 0 and 1
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Use the custom handles and labels for the legend
    ax.legend(
        line_handles, 
        line_labels, 
        loc='upper left', 
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        shadow=False,
        fontsize='small'
    )
    
    plt.tight_layout()
    plt.savefig(save_filename, bbox_inches='tight')
    plt.close()
    print(f"\nFinal KDE distribution plot saved to: {save_filename}")


# --- Main Execution Block (MODIFIED to remove bw_adjustment parameter) ---

if __name__ == "__main__":
    
    # --- 1. SET UP OUTPUT PATH ---
    first_base_dir = list(DATASET_CONFIGS.values())[0]["BASE_DIR"].rstrip(os.sep)
    INPUT_DATA_BASE = os.path.dirname(first_base_dir)
    RESULTS_BASE_DIR = os.path.join(INPUT_DATA_BASE, "results")
    
    # Define the final PDF output path
    HI_DISTRIBUTION_PLOT_OUTPUT = os.path.join(
        RESULTS_BASE_DIR, 
        "hi_admixture_norecomb.pdf" # Updated filename
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
    # The bw_adjustment is now read from the DATASET_CONFIGS inside the function
    plot_hi_crossing_kde(
        combined_df=combined_df, 
        dataset_configs=DATASET_CONFIGS, 
        save_filename=HI_DISTRIBUTION_PLOT_OUTPUT
    )