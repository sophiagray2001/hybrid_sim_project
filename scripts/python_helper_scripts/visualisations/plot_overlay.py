import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import random

# --- HELPER FUNCTION ---
def sort_key(label):
    """
    Custom sort key to handle mixed alphanumeric generation labels (F1, HG1, BC1A, etc.).
    """
    if label == 'PA': return (0, 0, '')
    if label == 'PB': return (0, 1, '')
    
    # Check for Fx labels
    if label.startswith('F'): 
        try: return (1, int(label[1:]), '')
        except ValueError: pass
        
    # Check for BCx labels
    if label.startswith('BC'): 
        # Check if the last character is a non-digit (A or B), then parse the number
        if label[-1].isalpha():
            try: return (2, int(label[2:-1]), label[-1])
            except ValueError: pass
        # Handle cases like BC1 without A/B suffix
        try: return (2, int(label[2:]), '')
        except ValueError: pass
        
    # Handle HGx labels (Hybrid Generations)
    if label.startswith('HG'):
        try: return (3, int(label[2:]), '')
        except ValueError: pass
        
    return (4, 0, label) # Default for anything else

# --- DATA LOADING FUNCTION ---
def load_replicate_data(base_output_dir: str, replicate_ids: list):
    """
    Identifies, reads, and processes individual replicate results from a single directory.
    Returns: (all_replicate_dfs, all_gen_labels)
    """
    all_replicate_dfs = {}
    all_gen_labels = set()
    
    for rep_id in replicate_ids:
        item = f'replicate_{rep_id}'
        rep_dir = os.path.join(base_output_dir, item)
        
        if os.path.isdir(rep_dir):
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
                    
                    # Use the replicate ID (as an integer/string) as the key
                    all_replicate_dfs[str(rep_id)] = sorted_df 
                    all_gen_labels.update(sorted_df.index)
                    
                except Exception as e:
                    print(f"Warning: Could not process file {input_file}. Error: {e}")
            
    return all_replicate_dfs, all_gen_labels


# --- PLOTTING FUNCTION (Consolidated, Improved, and Curated) ---
def plot_hi_het_overlay(all_replicate_dfs: dict, all_gen_labels: set, save_filename: str):
    """
    Plots the combined HI vs. HET paths, including all replicates and the grand mean.
    Uses deviation scores to select curated highlight paths (Random, Follower, Outlier).
    """
    if not all_replicate_dfs:
        print("Error: No complete replicate data found to plot.")
        return

    # 1. Calculate the Grand Mean DF
    all_means_combined = pd.concat(all_replicate_dfs.values(), keys=all_replicate_dfs.keys(), names=['replicate', 'generation'])
    grand_mean_df = all_means_combined.groupby('generation').mean()
    
    # Sort the grand mean
    all_sorted_gen_labels = sorted(grand_mean_df.index, key=sort_key)
    grand_mean_df = grand_mean_df.loc[all_sorted_gen_labels]
    
    # ----------------------------------------------------------------------
    # --- CURATED REPLICATE SELECTION (New Logic) ---
    # ----------------------------------------------------------------------
    
    # Calculate Deviation Scores for Selection
    deviation_scores = {}
    common_gens = grand_mean_df.index 
    available_reps = list(all_replicate_dfs.keys())

    for rep_id, df in all_replicate_dfs.items():
        aligned_df = df.reindex(common_gens).dropna()
        aligned_mean = grand_mean_df.reindex(aligned_df.index)
        
        # Calculate Sum of Squared Errors (SSE) for both HI and HET coordinates
        hi_diff = (aligned_df['mean_HI'] - aligned_mean['mean_HI']) ** 2
        het_diff = (aligned_df['mean_HET'] - aligned_mean['mean_HET']) ** 2
        
        # Total deviation score: sum of squared differences
        deviation_scores[rep_id] = (hi_diff + het_diff).sum()

    if len(available_reps) < 3:
        # Fallback to simple random if fewer than 3 available
        curated_reps = random.sample(available_reps, len(available_reps))
        
    else:
        # 1. Select the Mean Follower (Lowest Deviation Score)
        mean_follower_id = min(deviation_scores, key=deviation_scores.get)
        available_reps.remove(mean_follower_id)
        
        # Re-create scores dictionary without the follower ID for max selection
        remaining_scores = {k: v for k, v in deviation_scores.items() if k != mean_follower_id}

        # 2. Select the Outlier (Highest Deviation Score from remaining pool)
        outlier_id = max(remaining_scores, key=remaining_scores.get)
        available_reps.remove(outlier_id)
        
        # 3. Select a Random Replicate (from the rest of the available pool)
        random_id = random.choice(available_reps)
        
        # Final Curated List: [Random, Follower, Outlier]
        curated_reps = [random_id, mean_follower_id, outlier_id]
        
    # Assign specific colors to the roles
    HIGHLIGHT_COLORS = {}
    if len(curated_reps) >= 3:
        HIGHLIGHT_COLORS[curated_reps[0]] = 'blue'    # Random Path
        HIGHLIGHT_COLORS[curated_reps[1]] = 'red'      # Mean Follower Path
        HIGHLIGHT_COLORS[curated_reps[2]] = 'orange' # Outlier Path
    # ----------------------------------------------------------------------
    
    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Mean Hybrid Index (HI)", fontsize=14)
    ax.set_ylabel("Mean Heterozygosity (HET)", fontsize=14)
    
    num_reps = len(all_replicate_dfs) # Define num_reps here for the title later

    # 3. Plot Stochastic Paths (remains the same)
    path_start_gen = 'HG1' 
    
    for rep_id, df in all_replicate_dfs.items():
        if path_start_gen in df.index:
            path_df = df.iloc[df.index.get_loc(path_start_gen):]
            
            # Default style for background paths
            color = 'gray' 
            linewidth = 1
            alpha = 0.15 
            label = None
            zorder = 2
            
            # Apply highlight style for selected paths
            if rep_id in HIGHLIGHT_COLORS:
                color = HIGHLIGHT_COLORS[rep_id]
                linewidth = 2.0 
                alpha = 1.0
                zorder = 4 
                
                # Use a descriptive label based on the color assignment
                if color == 'blue': label = f"Random Replicate {rep_id}"
                elif color == 'red': label = f"Mean Follower {rep_id}"
                elif color == 'orange': label = f"Outlier Replicate {rep_id}"

            ax.plot(path_df['mean_HI'], path_df['mean_HET'],
                    color=color, linestyle='-', linewidth=linewidth, 
                    alpha=alpha, zorder=zorder, label=label) 

    # 4. Plot the Mean Path (remains the same) 
    if path_start_gen in grand_mean_df.index:
        grand_path_df = grand_mean_df.iloc[grand_mean_df.index.get_loc(path_start_gen):] 
        ax.plot(grand_path_df['mean_HI'], grand_path_df['mean_HET'],
                color='black', linestyle='--', linewidth=3, alpha=1.0, zorder=5, label='Mean Path') 
                
    # 5. Highlight Key Points (PA, PB, HG1, Grand Mean Last Gen)

    # Plot Triangle Edges (remains the same)
    triangle_edges = [
        [(0.0, 0.0), (0.5, 1.0)], [(0.5, 1.0), (1.0, 0.0)], [(0.0, 0.0), (1.0, 0.0)]
    ]
    for (x0, y0), (x1, y1) in triangle_edges:
        ax.plot([x0, x1], [y0, y1], linestyle='-', color='black', alpha=0.5, linewidth=1.5, zorder=1)

    # IMPROVED: Define specific offsets for non-overlapping labels
    LABEL_OFFSETS = {
        'PA': (-0.03, -0.01), # Left and Down
        'PB': (0.01, -0.01),  # Right and Down
        'HG1': (0.01, 0.01),  # Right and Up
        'Final': (0.01, -0.03)# Right and Down
    }
    
    # ----------------------------------------------------------------------
    # --- HARD-CODED POINT HIGHLIGHT (NEW SECTION) ---
    # ----------------------------------------------------------------------
    TARGET_GEN = 'HG1468'
    
    if TARGET_GEN in grand_mean_df.index:
        mean_data = grand_mean_df.loc[TARGET_GEN]
        
        # Plot the point
        ax.scatter(mean_data['mean_HI'], mean_data['mean_HET'],
                    color='green', s=150, edgecolors='black', linewidth=1.5, zorder=7, 
                    label=f'{TARGET_GEN} Mean Point') 
        
        # Label the point
        ax.annotate(TARGET_GEN, 
                    (mean_data['mean_HI'], mean_data['mean_HET']), 
                    xytext=(mean_data['mean_HI'] - 0.01, mean_data['mean_HET'] + 0.03), # Custom offset for F2
                    fontsize=12, color='green', ha='right', va='bottom', zorder=8)
    # ----------------------------------------------------------------------

    # Highlight PA, PB, HG1, and the final generation (Original logic adapted)
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
                        color=color, s=100, edgecolors='black', linewidth=1.5, zorder=6)
            
            # Determine offset for clean labeling
            offset_key = 'Final' if gen_name == all_sorted_gen_labels[-1] else gen_name
            dx, dy = LABEL_OFFSETS.get(offset_key, (0.01, 0.01))
            
            # Label the point
            ax.annotate(gen_name, 
                        (mean_data['mean_HI'], mean_data['mean_HET']), 
                        xytext=(mean_data['mean_HI'] + dx, mean_data['mean_HET'] + dy),
                        fontsize=10, color='black', ha='left', va='bottom', zorder=7)

    # Final settings (remains the same)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)

    # Add Legend (remains the same)
    ax.legend(loc='upper right', frameon=True, fontsize=10)

    plt.savefig(save_filename, bbox_inches='tight')
    plt.close()
    print(f"\nOverlay plot saved to: {save_filename}")


if __name__ == "__main__":
    # --- 1. DEFINE DIRECTORIES AND REPLICATE IDs FLEXIBLY ---
    
    # ----------------------------------------------------
    # Configuration - EDIT THIS SECTION FOR SINGLE/DOUBLE BATCH
    # ----------------------------------------------------
    
    # Define the batch(es) to load. 
    # Option B: Two Separate Batches (Configured for Replicates 1-50 and 51-100)
    BATCH_CONFIGS = [
        {
            # This is the folder containing replicates 1 through 50
            "BASE_DIR": "/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs_extreme_linkage_0.05/",
            "REPLICATE_IDS": list(range(1, 51)) # Replicates 1 through 50 (exclusive end)
        }
    ]
        # {
            # This is the folder containing replicates 51 through 100
            # Assuming the path you provided for 51-100 is a subfolder of the first path's parent
            #"BASE_DIR": "/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs_unlinked_closed/simulation_outputs_closed_unlinked_51_100/",
            #"REPLICATE_IDS": list(range(51, 101)) # Replicates 51 through 100 (exclusive end)
        #}
    # ]
    
    # ----------------------------------------------------
    
    # Determine the overall replicate range for the output filename
    # Assumes BATCH_CONFIGS is not empty
    all_start_id = min(c["REPLICATE_IDS"][0] for c in BATCH_CONFIGS if c["REPLICATE_IDS"])
    all_end_id = max(c["REPLICATE_IDS"][-1] for c in BATCH_CONFIGS if c["REPLICATE_IDS"])
    
    # Define the unique output file name using the first config's base directory
    # NOTE: The parent directory is derived from the first path.

    # Define the unique output file name using the first config's base directory
    # NOTE: The parent directory is derived from the first path.
    parent_dir = os.path.dirname(BATCH_CONFIGS[0]["BASE_DIR"].rstrip('/'))
    OVERLAY_PLOT_OUTPUT = os.path.join(
        parent_dir,
        "results", 
        # UPDATED FILENAME to reflect the full 1_100 range
        # Change the extension from .png to .pdf here:
        f"overlay_{all_start_id}_{all_end_id}_extreme_linkage_0.05.png" 
    )


    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OVERLAY_PLOT_OUTPUT), exist_ok=True)
    
    # --- 2. LOAD AND MERGE DATA FROM ALL BATCHES ---
    all_replicate_dfs = {}
    all_gen_labels = set()
    total_replicates_loaded = 0
    
    for i, config in enumerate(BATCH_CONFIGS):
        base_dir = config["BASE_DIR"]
        rep_ids = config["REPLICATE_IDS"]
        
        # Skip if the replicate list is empty
        if not rep_ids:
            print(f"Skipping Batch {i+1}: No replicates defined.")
            continue
            
        print(f"Loading data from Batch {i+1} ({len(rep_ids)} replicates) in: {base_dir}")
        
        # Load the data for the current batch
        # NOTE: load_replicate_data must return a dict and a set
        dfs_batch, labels_batch = load_replicate_data(base_dir, rep_ids)
        
        # Merge the dictionaries and update the labels set
        all_replicate_dfs.update(dfs_batch)
        all_gen_labels.update(labels_batch)
        total_replicates_loaded += len(dfs_batch)
    
    # --- 3. PLOT COMBINED DATA ---
    print(f"Plotting combined data for {total_replicates_loaded} total replicates.")
    
    if total_replicates_loaded == 0:
        print("ERROR: No replicate data was loaded. Aborting plot generation.")
    else:
        plot_hi_het_overlay(
            all_replicate_dfs=all_replicate_dfs, 
            all_gen_labels=all_gen_labels,
            save_filename=OVERLAY_PLOT_OUTPUT
        )