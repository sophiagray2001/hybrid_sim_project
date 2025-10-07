import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# Sorting Key (Reused from your simulation script) 
def sort_key(label: str):
    """Custom sort key for generation labels."""
    if label == 'PA': return (0, label)
    if label == 'PB': return (1, label)
    match_hg = re.match(r'HG(\d+)', label)
    if match_hg: return (2, int(match_hg.group(1)))
    match_f = re.match(r'F(\d+)', label)
    if match_f: return (3, int(match_f.group(1)))
    match_bc = re.match(r'BC(\d+)([A-Z]?)', label)
    if match_bc: return (4, int(match_bc.group(1)), match_bc.group(2))
    return (5, label)


def plot_simulation_overlay(base_output_dir: str, save_filename: str):
    """
    Plots the HI vs. HET paths for all simulation replicates on a single graph.
    """
    all_replicate_dfs = {}
    all_gen_labels = set()
    
    # 1. Identify and process all replicate folders
    for item in os.listdir(base_output_dir):
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

    # 2. Calculate the Grand Mean DF (for the thick line and overall highlight)
    
    # Concatenate all mean dataframes into one, indexed by generation
    all_means_combined = pd.concat(all_replicate_dfs.values(), keys=all_replicate_dfs.keys(), names=['replicate', 'generation'])
    
    # Calculate the mean across all replicates for each generation
    grand_mean_df = all_means_combined.groupby('generation').mean()
    
    # Sort the grand mean
    all_sorted_gen_labels = sorted(grand_mean_df.index, key=sort_key)
    grand_mean_df = grand_mean_df.loc[all_sorted_gen_labels]
    
    # 3. Setup Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Mean Hybrid Index (HI)", fontsize=14)
    ax.set_ylabel("Mean Heterozygosity (HET)", fontsize=14)
    ax.set_title(f"Simulation Stochasticity: HI vs. HET Paths ({len(all_replicate_dfs)} Replicates)", fontsize=16)

    # 4. Plot Stochastic Paths (Spaghetti Plot)
    path_start_gen = 'HG1' # Assume HG1 is the start of the line path
    
    for rep_id, df in all_replicate_dfs.items():
        # Find the starting point for the line (e.g., HG1 or F1)
        if path_start_gen in df.index:
            path_df = df.loc[df.index.get_loc(path_start_gen):]
            ax.plot(path_df['mean_HI'], path_df['mean_HET'],
                    color='blue', linestyle='-', linewidth=1, alpha=0.15, zorder=2) # Light, thin line

    # 5. Plot the Grand Mean Path (Thick Line)
    if path_start_gen in grand_mean_df.index:
        grand_path_df = grand_mean_df.loc[grand_mean_df.index.get_loc(path_start_gen):]
        ax.plot(grand_path_df['mean_HI'], grand_path_df['mean_HET'],
                color='red', linestyle='-', linewidth=3, alpha=0.8, zorder=3, label='Grand Mean Path')
                
    # 6. Highlight Key Points (PA, PB, Grand Mean Last Gen)
    
    # Plot Triangle Edges
    triangle_edges = [
        [(0.0, 0.0), (0.5, 1.0)], [(0.5, 1.0), (1.0, 0.0)], [(0.0, 0.0), (1.0, 0.0)]
    ]
    for (x0, y0), (x1, y1) in triangle_edges:
        ax.plot([x0, x1], [y0, y1], linestyle='--', color='lightgray', linewidth=1.5, zorder=1)

    # Highlight PA, PB, and the final generation
    points_to_label = ['PA', 'PB', all_sorted_gen_labels[-1]] 
    
    for gen_name in points_to_label:
        if gen_name in grand_mean_df.index:
            mean_data = grand_mean_df.loc[gen_name]
            
            color = 'black'
            if gen_name == 'PA': color = 'black'
            elif gen_name == 'PB': color = 'gray'
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
    
    # Define the output file name
    OVERLAY_PLOT_OUTPUT = os.path.join(
        os.path.dirname(PERSISTENT_OUTPUT_DIR.rstrip('/')), 
        "results", 
        "stochasticity_overlay_plot.png"
    )

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OVERLAY_PLOT_OUTPUT), exist_ok=True)

    # Run the function
    plot_simulation_overlay(PERSISTENT_OUTPUT_DIR, OVERLAY_PLOT_OUTPUT)