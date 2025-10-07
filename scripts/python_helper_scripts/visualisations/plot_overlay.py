import os
import pandas as pd
import matplotlib.pyplot as plt
import re

# =========================================================
# HELPER FUNCTION
# =========================================================

def sort_key(label):
    """
    Custom sort key to handle mixed alphanumeric generation labels (F1, HG1, HG1000, BC1A, etc.).
    """
    if label == 'PA': return (0, 0, '')
    if label == 'PB': return (0, 1, '')
    
    # Check for Fx labels
    if label.startswith('F'): 
        try: return (1, int(label[1:]), '')
        except ValueError: pass
        
    # Check for BCx labels
    if label.startswith('BC'): 
        try: return (2, int(label[2:-1]), label[-1])
        except ValueError: pass
    
    # Handle HGx labels (Hybrid Generations)
    if label.startswith('HG'):
        try: return (3, int(label[2:]), '')
        except ValueError: pass
        
    return (4, 0, label) # Default for anything else

# =========================================================
# PLOTTING FUNCTION
# =========================================================

def plot_simulation_overlay(base_output_dir: str, save_filename: str, replicate_ids: list = None):
    """
    Plots the HI vs. HET paths for all simulation replicates on a single graph.
    """
    all_replicate_dfs = {}
    all_gen_labels = set()
    
    # 1. Identify and process all replicate folders
    if replicate_ids:
        target_items = [f'replicate_{i}' for i in replicate_ids]
    else:
        target_items = os.listdir(base_output_dir)
    
    for item in target_items:
        rep_dir = os.path.join(base_output_dir, item)
        
        if os.path.isdir(rep_dir) and item.startswith('replicate_'):
            
            rep_id = item.split('_')[-1]
            input_file = os.path.join(rep_dir, 'results', f'results_rep_{rep_id}_individual_hi_het.csv')
            
            if os.path.exists(input_file):
                try:
                    hi_het_df = pd.read_csv(input_file)
                    
                    # Calculate mean HI and HET per generation for this replicate
                    mean_hi_het_rep = hi_het_df.groupby('generation').agg(
                        mean_HI=('HI', 'mean'),
                        mean_HET=('HET', 'mean')
                    )
                    
                    # Sort the dataframe by the custom sort key
                    sorted_gen_labels = sorted(mean_hi_het_rep.index, key=sort_key)
                    sorted_df = mean_hi_het_rep.loc[sorted_gen_labels]
                    
                    all_replicate_dfs[rep_id] = sorted_df
                    all_gen_labels.update(sorted_df.index)
                    
                except Exception as e:
                    print(f"Warning: Could not process file {input_file}. Error: {e}")


    if not all_replicate_dfs:
        print("Error: No complete replicate data found to plot.")
        return

    # 2. Calculate the Grand Mean DF
    all_means_combined = pd.concat(all_replicate_dfs.values(), keys=all_replicate_dfs.keys(), names=['replicate', 'generation'])
    grand_mean_df = all_means_combined.groupby('generation').mean()
    
    # Sort the grand mean (LABEL-BASED SORTING: Use .loc)
    all_sorted_gen_labels = sorted(grand_mean_df.index, key=sort_key)
    grand_mean_df = grand_mean_df.loc[all_sorted_gen_labels]
    
    # 3. Setup Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Mean Hybrid Index (HI)", fontsize=14)
    ax.set_ylabel("Mean Heterozygosity (HET)", fontsize=14)

    # 4. Plot Stochastic Paths (Spaghetti Plot)
    path_start_gen = 'HG1' 
    
    for rep_id, df in all_replicate_dfs.items():
        if path_start_gen in df.index:
            # SLICING BY POSITION: Use .iloc
            path_df = df.iloc[df.index.get_loc(path_start_gen):]
            ax.plot(path_df['mean_HI'], path_df['mean_HET'],
                    color='blue', linestyle='-', linewidth=1, alpha=0.15, zorder=2) 

    # 5. Plot the Grand Mean Path (Thick Line)
    if path_start_gen in grand_mean_df.index:
        # SLICING BY POSITION: Use .iloc (FIXED)
        grand_path_df = grand_mean_df.iloc[grand_mean_df.index.get_loc(path_start_gen):] 
        ax.plot(grand_path_df['mean_HI'], grand_path_df['mean_HET'],
                color='red', linestyle='-', linewidth=3, alpha=0.8, zorder=3, label='Grand Mean Path')
                
    # 6. Highlight Key Points (PA, PB, Grand Mean Last Gen)
    
    # Plot Triangle Edges
    triangle_edges = [
        [(0.0, 0.0), (0.5, 1.0)], [(0.5, 1.0), (1.0, 0.0)], [(0.0, 0.0), (1.0, 0.0)]
    ]
    for (x0, y0), (x1, y1) in triangle_edges:
        ax.plot([x0, x1], [y0, y1], linestyle='-', color='black', alpha=0.5, linewidth=1.5, zorder=1)

    # Highlight PA, PB, and the final generation
    points_to_label = ['PA', 'PB', 'HG1', all_sorted_gen_labels[-1]] 
    
    for gen_name in points_to_label:
        if gen_name in grand_mean_df.index:
            mean_data = grand_mean_df.loc[gen_name]
            
            color = 'black'
            if gen_name == 'PA': color = 'black'
            elif gen_name == 'PB': color = 'gray'
            elif gen_name == 'HG1': color = 'blue'
            elif gen_name == all_sorted_gen_labels[-1]: color = 'red' # Final generation point

            ax.scatter(mean_data['mean_HI'], mean_data['mean_HET'],
                        color=color, s=100, edgecolors='black', linewidth=1.5, zorder=4)
            
            # Label the point
            ax.text(mean_data['mean_HI'] + 0.01, mean_data['mean_HET'] + 0.01, gen_name,
                    fontsize=10, color='black', ha='left', va='bottom', zorder=5)

    # Final settings
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    
    plt.savefig(save_filename, bbox_inches='tight')
    plt.close()
    print(f"\nOverlay plot saved to: {save_filename}")


if __name__ == "__main__":
    # Define the persistent directory where all 'replicate_X' folders are saved
    PERSISTENT_OUTPUT_DIR = "/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs/" 
    
    # Define the specific replicates to use for this test run (1 to 9)
    TEST_REPLICATES = list(range(1, 10)) # Creates the list [1, 2, ... 9]
    
    # Define a unique output file name for the test
    OVERLAY_PLOT_OUTPUT = os.path.join(
        os.path.dirname(PERSISTENT_OUTPUT_DIR.rstrip('/')), 
        "results", 
        "stochasticity_overlay_TEST_1_to_9.png" # Unique name for testing
    )

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OVERLAY_PLOT_OUTPUT), exist_ok=True)

    # Run the function, passing the list of test IDs
    plot_simulation_overlay(
        base_output_dir=PERSISTENT_OUTPUT_DIR, 
        save_filename=OVERLAY_PLOT_OUTPUT,
        replicate_ids=TEST_REPLICATES # <-- The list of 1 to 9
    )