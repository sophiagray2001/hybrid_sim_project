import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import re
import scipy.stats
import numpy as np

# REMOVED: Global constant KDE_SMOOTHNESS_FACTOR has been removed.
# Smoothness is now defined per-dataset in the configuration.

# --- REVISED FUNCTION: INDIVIDUAL SMOOTHNESS, NO CI, NO LEGEND ---
def plot_crossing_time_distribution_multiple(
    data_list: list,
    ne_value: float,
    save_filename: str,
    plot_title: str = None
):
    """
    Plots the distribution of generations required for HET to decrease to
    the parental mean level for multiple datasets. Now supports individual
    KDE smoothness, removes 95% CI lines, and suppresses the legend.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    MAX_NE_TIME = 15.0

    # Lists to hold handles and labels (Only used to track KDE and Mean for logic/debugging)
    line_handles = []
    line_labels = []

    # 1. Loop through each dataset and plot the KDE curve and its statistics
    for i, data_config in enumerate(data_list):
        # NOTE: data_config must have 'df', 'label', 'color', and 'kde_smoothness_factor' (optional) keys
        crossing_df = data_config['df']
        label = data_config['label']
        kde_color = data_config['color']
        
        # NEW: Read individual smoothness factor, defaulting to 0.5 if not provided
        # (0.5 is used as the default since that was the value of the old global constant)
        kde_smoothness = data_config.get('kde_smoothness_factor', 0.5) 

        # --- Determine unique styles for the mean lines ---
        mean_color = kde_color
        
        if i == 0:
            mean_ls = '--'
        elif i == 1:
            mean_ls = '-.'
        elif i == 2:
            mean_ls = (0, (3, 5, 1, 5))
        else: # i >= 3
            mean_ls = (0, (3, 1, 1, 1))

        if crossing_df.empty:
            print(f"Warning: DataFrame for '{label}' is empty. Skipping.")
            continue

        # Data preparation (remains the same)
        crossing_df['crossing_time'] = crossing_df['matching_hybrid_gen'].astype(str).str.extract(r'HG(\d+)').astype(float)
        crossing_times_g = crossing_df['crossing_time'].dropna().tolist()
        if not crossing_times_g:
            print(f"Warning: No valid crossing times found for '{label}'. Skipping.")
            continue

        crossing_times_ne = [t / ne_value for t in crossing_times_g]

        # 2. MANUALLY CALCULATE and PLOT the KDE curve (Clipped Tails)
        data_min = np.min(crossing_times_ne)
        data_max = np.max(crossing_times_ne)

        x_range = np.linspace(
            max(0, data_min - 0.5),
            min(MAX_NE_TIME, data_max + 0.5),
            500
        )

        kde = scipy.stats.gaussian_kde(crossing_times_ne)

        # --- ADJUST KDE SMOOTHNESS (Bandwidth) using individual factor ---
        kde.set_bandwidth(bw_method=kde.factor * kde_smoothness)

        density = kde(x_range)

        kde_handle = ax.plot(
            x_range,
            density,
            color=kde_color,
            linewidth=3,
            zorder=2,
            label=f'{label}'
        )
        # Only add the KDE line
        line_handles.append(kde_handle[0])
        line_labels.append(f'{label}')

        # 3. Calculate and plot MEAN only for THIS dataset
        crossing_series_g = pd.Series(crossing_times_g)
        mean_time_ne = crossing_series_g.mean() / ne_value
        
        # 95% CI lines and calculations REMOVED as requested.
        
        # Mean line
        ax.axvline(
            mean_time_ne,
            color=mean_color,
            linestyle=mean_ls,
            linewidth=2,
            zorder=3,
        )

        # --- Add MEAN STATS to custom legend lists ---
        mean_handle = plt.Line2D([0], [0], color=mean_color, linestyle=mean_ls, linewidth=2)
        line_handles.append(mean_handle)
        line_labels.append(f'{label} Mean: {mean_time_ne:.2f} Ne Gens')

    # 4. Set up labels and plot limits
    ax.set_xlabel(f"Time (Ne Generations, $N_e={int(ne_value)}$)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)

    ax.set_xlim(0, MAX_NE_TIME)
    ax.set_xticks(range(int(MAX_NE_TIME) + 1))

    # Tidy up the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # LEGEND REMOVED: The ax.legend(...) block has been removed as requested.

    # Save the figure.
    plt.savefig(save_filename, bbox_inches='tight')
    plt.close()
    print(f"\nMultiple time distribution (KDE) plot saved to: {save_filename}")


# ----------------------------------------------------------------------------------
# --- MAIN EXECUTION BLOCK (Updated with individual smoothness factors and original file paths) ---
# ----------------------------------------------------------------------------------
if __name__ == "__main__":

    # --- 1. DEFINE CONFIGURATIONS ---
    N_E_VALUE = 200.0

    # A. CONFIGS FOR THE PRIMARY DATASET (i=0)
    UNLINKED_BATCH_CONFIGS = [
        {
            "BASE_DIR": "/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs_unlinked_closed/",
            "FILENAME": "combined_matching_generations.csv",
            # Add individual smoothness factor (e.g., set to a smoother value)
            "kde_smoothness_factor": 1.0 
        }
    ]

    # B. CONFIGS FOR SECONDARY DATASETS (i=1, i=2, i=3)
    # NOTE: These configs use UPPERCASE keys ("LABEL", "COLOR")
    SECONDARY_PLOTTING_CONFIGS = [
        {
            # Dataset 2 (i=1): Linked Loci (less smooth/peaked)
            "BASE_DIR": "/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs_linked_closed/",
            "FILENAME": "combined_matching_generations_linked_closed.csv",
            "LABEL": "Linked Loci",
            "COLOR": "#ff7f0e",
            "REPLICATE_LIMIT": 50,
            "kde_smoothness_factor": 0.9 # Individual Smoothness Factor
        },
        {
            # Dataset 3 (i=2): 20 Chrs (uses default 0.5 smoothness)
            "BASE_DIR": "/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs_closed_20chr/", 
            "FILENAME": "combined_matching_generations_closed_20chr.csv",
            "LABEL": "20 Chrs",
            "COLOR": "#d62728", # Red
            "REPLICATE_LIMIT": None,
            # kde_smoothness_factor is omitted; it will default to 0.5
        },
        {
            # Dataset 4 (i=3): Extreme Linkage Loci (slightly smoother than default)
            "BASE_DIR": "/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs_extreme_linkage_0.05/", 
            "FILENAME": "combined_matching_generations_extreme_linkage_0.05.csv",
            "LABEL": "Extreme Linkage Loci",
            "COLOR": "#9467bd", # Purple
            "REPLICATE_LIMIT": None,
            "kde_smoothness_factor": 0.4 # Individual Smoothness Factor
        },
        {
            # Dataset 4 (i=3): No recomb (slightly smoother than default)
            "BASE_DIR": "/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs_no_recombination_50/", 
            "FILENAME": "combined_matching_generations_no_recombination.csv",
            "LABEL": "No Recombination",
            "COLOR": "#ffd700", # Yellow
            "REPLICATE_LIMIT": None,
            "kde_smoothness_factor": 0.4 # Individual Smoothness Factor
        }
    ]

    # --- Setup Output Paths ---
    if not UNLINKED_BATCH_CONFIGS:
        print("FATAL ERROR: No unlinked batch configurations defined. Aborting.")
        exit(1)

    # Note: Logic for setting INPUT_DATA_BASE is complex due to structure, 
    # but the paths themselves were not changed.
    INPUT_DATA_BASE = os.path.dirname(UNLINKED_BATCH_CONFIGS[0]["BASE_DIR"].rstrip(os.sep))
    RESULTS_BASE_DIR = os.path.join(INPUT_DATA_BASE, "results")

    # The output filename has been updated to reflect the new plot style
    COMBINED_PLOT_OUTPUT = os.path.join(
        RESULTS_BASE_DIR,
        "KDE_HET_drop_4_Datasets_CustomSmooth_NoStats.pdf"
    )
    os.makedirs(os.path.dirname(COMBINED_PLOT_OUTPUT), exist_ok=True)

    # --- 2. LOAD AND COMBINE THE PRIMARY DATASET ---
    unlinked_dfs = []
    print("Loading and combining primary data batches...")

    # Extract smoothness factor for primary config
    primary_config = UNLINKED_BATCH_CONFIGS[0]
    primary_kde_smoothness = primary_config.get('kde_smoothness_factor', 0.8)
    
    # Load Primary Data
    for i, config in enumerate(UNLINKED_BATCH_CONFIGS):
        crossing_path = os.path.join(config["BASE_DIR"], config["FILENAME"])
        try:
            df = pd.read_csv(crossing_path)
            unlinked_dfs.append(df)
            print(f"  -> Successfully loaded Primary Batch {i+1} ({len(df)} rows)")
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f"WARNING: Primary file not found or empty at: {crossing_path}. Skipping.")

    if not unlinked_dfs:
        print("FATAL ERROR: No primary data files were successfully loaded. Aborting.")
        exit(1)

    combined_unlinked_df = pd.concat(unlinked_dfs, ignore_index=True)
    print(f"Combined Primary Data has {len(combined_unlinked_df)} rows.")

    # --- 3. BUILD THE FINAL PLOTTING LIST ---
    plotting_data_list = []

    # Add the Combined Primary Data as the FIRST entry (i=0) - Keys are already correct (lowercase)
    plotting_data_list.append({
        'df': combined_unlinked_df,
        'label': "Unlinked Loci",
        'color': "#2ca02c", 
        'kde_smoothness_factor': primary_kde_smoothness # Re-add the factor
    })

    # Load and add the Secondary Datasets (i=1, 2, 3...)
    for config in SECONDARY_PLOTTING_CONFIGS:
        crossing_path = os.path.join(config["BASE_DIR"], config["FILENAME"])
        limit = config.get("REPLICATE_LIMIT")

        try:
            df = pd.read_csv(crossing_path)

            # --- APPLY SLICING FOR REPLICATE_LIMIT ---
            if limit is not None:
                # Assuming 'replicate_id' extraction logic is correct for slicing
                df['replicate_id'] = df['matching_hybrid_gen'].astype(str).str.extract(r'(HG\d+)').iloc[:, 0]
                first_replicates = df['replicate_id'].dropna().unique()[:limit]
                df = df[df['replicate_id'].isin(first_replicates)]
                print(f"  -> Sliced data for {config['LABEL']} to {len(first_replicates)} unique replicates.")

            # --- FIX: Standardize keys to lowercase for the plotting function ---
            secondary_data_dict = {
                'df': df,
                # Map from the uppercase keys in the config to the required lowercase keys
                'label': config['LABEL'],     
                'color': config['COLOR'],     
                # Use .get() for optional keys like smoothness
                'kde_smoothness_factor': config.get('kde_smoothness_factor', 0.9) 
            }

            plotting_data_list.append(secondary_data_dict)
            print(f"  -> Loaded secondary dataset: {config['LABEL']} ({len(df)} rows)")

        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            print(f"WARNING: Secondary file error for {config['LABEL']} at {crossing_path}: {e}. Skipping.")

    # --- 4. RUN THE PLOTTING FUNCTION ---
    plot_crossing_time_distribution_multiple(
        data_list=plotting_data_list,
        ne_value=N_E_VALUE,
        save_filename=COMBINED_PLOT_OUTPUT
    )