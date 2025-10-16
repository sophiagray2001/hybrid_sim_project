import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import re
import scipy
import numpy as np 

# --- REVISED FUNCTION: PLOTTING MULTIPLE KDE LINES (with color harmonization and no title) ---
def plot_crossing_time_distribution_multiple(
    data_list: list,  
    ne_value: float, 
    save_filename: str,
    plot_title: str = None # Removed from the argument list, but kept for function signature consistency if needed later
):
    """
    Plots the distribution of generations required for HET to decrease to 
    the parental mean level for multiple datasets, with all statistical 
    lines harmonized to the color of their respective KDE curve, and no plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    MAX_NE_TIME = 3000 / ne_value 
    
    # Lists to hold handles and labels for the final, custom legend
    line_handles = []
    line_labels = []

    # 1. Loop through each dataset and plot the KDE curve and its statistics
    for i, data_config in enumerate(data_list):
        crossing_df = data_config['df']
        label = data_config['label']
        kde_color = data_config['color'] # The color of the KDE curve

        # --- Determine unique styles/colors for the statistics lines ---
        # ALL statistical lines will now use the KDE color, only the style changes
        mean_color = kde_color
        ci_color = kde_color
        
        if i == 0:
            # Styles for the Primary Line (Unlinked)
            mean_ls = '--' # Redefining the Mean to be the blue KDE color, but dashed
            ci_ls = ':'    # Redefining the CI to be the blue KDE color, but dotted
        else:
            # Styles for Secondary Lines (e.g., Linked)
            mean_ls = '-.'
            ci_ls = (0, (1, 5)) 
        
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
        crossing_series_ne = pd.Series(crossing_times_ne)
        
        # 2. Plot the main KDE curve (Density Plot) for THIS dataset
        kde_handle = crossing_series_ne.plot(
            kind='kde',
            ax=ax,
            color=kde_color, # Use the KDE color
            linewidth=3,
            zorder=2,
            label=f'{label}'
        )
        line_handles.append(kde_handle.lines[-1])
        line_labels.append(f'{label}')

        # 3. Calculate and plot statistics for THIS dataset
        crossing_series_g = pd.Series(crossing_times_g)
        
        mean_time_ne = crossing_series_g.mean() / ne_value
        ci_lower_ne = crossing_series_g.quantile(0.025) / ne_value
        ci_upper_ne = crossing_series_g.quantile(0.975) / ne_value
        
        # Mean line
        ax.axvline(
            mean_time_ne, 
            color=mean_color, # Harmonized color
            linestyle=mean_ls, 
            linewidth=2, 
            zorder=3,
        )
        # CI lines
        ax.axvline(ci_lower_ne, color=ci_color, linestyle=ci_ls, linewidth=1.5, zorder=3)
        ax.axvline(ci_upper_ne, color=ci_color, linestyle=ci_ls, linewidth=1.5, zorder=3)
        
        # --- Add STATS to custom legend lists ---
        
        # Add Mean
        mean_handle = plt.Line2D([0], [0], color=mean_color, linestyle=mean_ls, linewidth=2)
        line_handles.append(mean_handle)
        line_labels.append(f'{label} Mean: {mean_time_ne:.2f} Ne Gens')
        
        # Add 95% CI
        ci_handle = plt.Line2D([0], [0], color=ci_color, linestyle=ci_ls, linewidth=1.5)
        line_handles.append(ci_handle)
        line_labels.append(f'{label} 95% CI: ({ci_lower_ne:.2f}-{ci_upper_ne:.2f})')


    # 4. Set up labels and legend
    ax.set_xlabel(f"Time (Ne Generations, $N_e={int(ne_value)}$)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    
    # Enforce X-axis limits and ticks
    ax.set_xlim(0, MAX_NE_TIME)
    ax.set_xticks(range(int(MAX_NE_TIME) + 1))
    
    # Tidy up the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Final Legend
    ax.legend(line_handles, line_labels, loc='upper right')
    
    # --- REMOVED PLOT TITLE HERE ---
    if plot_title:
        plt.title(plot_title, fontsize=14)
    
    plt.savefig(save_filename, bbox_inches='tight')
    plt.close()
    print(f"\nMultiple time distribution (KDE) plot saved to: {save_filename}")


# ----------------------------------------------------------------------------------
# --- MAIN EXECUTION BLOCK (Configuration remains the same) ---
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    
    # --- 1. DEFINE CONFIGURATIONS ---
    N_E_VALUE = 200.0
    
    # A. CONFIGS FOR THE PRIMARY, COMBINED UNLINKED DATASET
    UNLINKED_BATCH_CONFIGS = [
        {
            "BASE_DIR": "/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs_unlinked_closed/",
            "FILENAME": "combined_matching_generations.csv",
        },
        {
            "BASE_DIR": "/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs_unlinked_closed/simulation_outputs_closed_unlinked_51_100/", 
            "FILENAME": "combined_matching_generations_second_batch.csv",
        }
    ]
    
    # B. CONFIGS FOR ANY *OTHER* DATASETS TO BE PLOTTED SEPARATELY
    SECONDARY_PLOTTING_CONFIGS = [
        # Linked Loci
        {
             "BASE_DIR": "/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs_linked_closed/", 
             "FILENAME": "combined_matching_generations_linked_closed.csv",
             "LABEL": "Linked Loci",
             "COLOR": "#ff7f0e" # Distinct Orange
        }
    ]
    
    # --- Setup Output Paths (remains unchanged) ---
    if not UNLINKED_BATCH_CONFIGS:
        print("FATAL ERROR: No unlinked batch configurations defined. Aborting.")
        exit(1)
        
    INPUT_DATA_BASE = os.path.dirname(UNLINKED_BATCH_CONFIGS[0]["BASE_DIR"].rstrip(os.sep))
    RESULTS_BASE_DIR = os.path.join(INPUT_DATA_BASE, "results")

    COMBINED_PLOT_OUTPUT = os.path.join(
        RESULTS_BASE_DIR, 
        "time_to_parent_het_distribution_MULTIPLE_KDE_Ne_scaled_v5_clean.png" # Updated filename
    )
    os.makedirs(os.path.dirname(COMBINED_PLOT_OUTPUT), exist_ok=True)
    
    # --- 2. LOAD AND COMBINE THE PRIMARY DATASET (UNLINKED) ---
    unlinked_dfs = []
    print("Loading and combining primary UNLINKED data batches...")
    
    for i, config in enumerate(UNLINKED_BATCH_CONFIGS):
        crossing_path = os.path.join(config["BASE_DIR"], config["FILENAME"])
        try:
            df = pd.read_csv(crossing_path)
            unlinked_dfs.append(df)
            print(f"  -> Successfully loaded Unlinked Batch {i+1} ({len(df)} replicates)")
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f"WARNING: Unlinked file not found or empty at: {crossing_path}. Skipping.")
            
    if not unlinked_dfs:
        print("FATAL ERROR: No unlinked data files were successfully loaded. Aborting.")
        exit(1)

    combined_unlinked_df = pd.concat(unlinked_dfs, ignore_index=True)
    
    # --- 3. BUILD THE FINAL PLOTTING LIST ---
    plotting_data_list = []
    
    # Add the Combined Unlinked Data as the FIRST entry (i=0)
    plotting_data_list.append({
        'df': combined_unlinked_df,
        'label': "Unlinked Loci",
        'color': "#1f77b4" # Blue
    })
    
    # Load and add any Secondary Datasets
    for config in SECONDARY_PLOTTING_CONFIGS:
        crossing_path = os.path.join(config["BASE_DIR"], config["FILENAME"])
        try:
            df = pd.read_csv(crossing_path)
            plotting_data_list.append({
                'df': df,
                'label': config['LABEL'],
                'color': config['COLOR']
            })
            print(f"  -> Loaded secondary dataset: {config['LABEL']} ({len(df)} replicates)")
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            print(f"WARNING: Secondary file error for {config['LABEL']} at {crossing_path}: {e}. Skipping.")

    # --- 4. RUN THE PLOTTING FUNCTION ---
    plot_crossing_time_distribution_multiple(
        data_list=plotting_data_list,
        ne_value=N_E_VALUE,
        save_filename=COMBINED_PLOT_OUTPUT
    )