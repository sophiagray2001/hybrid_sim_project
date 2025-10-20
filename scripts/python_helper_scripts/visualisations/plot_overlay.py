import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import random

# --- NEW FUNCTION: CALCULATE AVERAGE CROSSING GENERATION FROM CSV ---
def calculate_average_crossing_gen(analysis_output_file: str):
    """
    Loads the pre-calculated crossing generations file, converts HG labels 
    to numbers from the 'matching_hybrid_gen' column, and computes the 
    average crossing generation number.
    
    Target Column: 'matching_hybrid_gen'
    Returns: The average HG label (e.g., 'HG1525') and the numerical average.
    """
    try:
        crossing_df = pd.read_csv(analysis_output_file)
        
        HG_LABEL_COLUMN = 'matching_hybrid_gen'
        
        if HG_LABEL_COLUMN not in crossing_df.columns:
             print(f"Error: Column '{HG_LABEL_COLUMN}' not found in the CSV.")
             return None
        
        # 1. Extract the generation label and strip 'HG' prefix to get a number
        # Example: 'HG1525' -> 1525
        crossing_df['HG_NUMBER'] = crossing_df[HG_LABEL_COLUMN].astype(str).str.replace('HG', '').astype(int)
        
        # 2. Calculate the mean of the numbers
        avg_hg_number_float = crossing_df['HG_NUMBER'].mean()
        
        # 3. Round to the nearest integer generation
        avg_hg_number = int(round(avg_hg_number_float))
        
        # 4. Convert back to the label format
        avg_hg_label = f'HG{avg_hg_number}'
        
        print(f"\nCalculated Average Crossing Generation Number: {avg_hg_number_float:.2f} (Rounded: {avg_hg_label})")
        return avg_hg_label
        
    except FileNotFoundError:
        print(f"Error: Matching generations file not found at {analysis_output_file}. Skipping target point calculation.")
        return None
    except Exception as e:
        print(f"Error calculating average crossing generation from CSV: {e}")
        return None

#The resulting black mean line plots this final grand_mean_df, which represents the average evolutionary trajectory computed by taking the mean of all the individual replicate means.

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
# UPDATED: Added target_hg_label argument
def plot_hi_het_overlay(all_replicate_dfs: dict, all_gen_labels: set, save_filename: str, target_hg_label: str = None):
    """
    Plots the combined HI vs. HET paths, including all replicates and the grand mean.
    Uses the provided target_hg_label (from the external CSV) to mark the average crossing generation.
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
    # --- Set Target Generation from External CSV ---
    # ----------------------------------------------------------------------
    # Use the externally calculated average generation number
    TARGET_HET_GEN = target_hg_label 

    if TARGET_HET_GEN:
        print(f"Identified Target Generation (Average from Replicates CSV): {TARGET_HET_GEN}")
        
    # ----------------------------------------------------------------------
    # --- CURATED REPLICATE SELECTION ---
    # ----------------------------------------------------------------------
    
    # Calculate Deviation Scores for Selection
    deviation_scores = {}
    common_gens = grand_mean_df.index  
    available_reps = list(all_replicate_dfs.keys())

    for rep_id, df in all_replicate_dfs.items():
        aligned_df = df.reindex(common_gens).dropna()
        aligned_mean = grand_mean_df.reindex(aligned_df.index)
        hi_diff = (aligned_df['mean_HI'] - aligned_mean['mean_HI']) ** 2
        het_diff = (aligned_df['mean_HET'] - aligned_mean['mean_HET']) ** 2
        deviation_scores[rep_id] = (hi_diff + het_diff).sum()

    # Determine the three statistical highlights
    curated_reps_statistical = []
    
    if len(available_reps) >= 3:
        # 1. Select the Mean Follower (Lowest Deviation Score)
        mean_follower_id = min(deviation_scores, key=deviation_scores.get)
        
        # Remove the follower from selection pool for the outlier choice
        remaining_scores = {k: v for k, v in deviation_scores.items() if k != mean_follower_id}

        # 2. Select the Outlier (Highest Deviation Score from remaining pool)
        outlier_id = max(remaining_scores, key=remaining_scores.get)
        
        # 3. Select a Random Replicate (from the original pool, excluding follower/outlier)
        temp_available_for_random = [r for r in available_reps if r not in [mean_follower_id, outlier_id]]
        # Check if there are still reps left for random choice
        if temp_available_for_random:
             random_id = random.choice(temp_available_for_random)
        else:
             # Fallback if somehow only 2 replicates exist (unlikely given >=3 check)
             random_id = mean_follower_id # Arbitrarily set to follower, though this block should not run
        
        # Final Curated List: [Random, Follower, Outlier]
        curated_reps_statistical = [random_id, mean_follower_id, outlier_id]
        
    elif len(available_reps) > 0:
        # Fallback to simple random if fewer than 3 available
        curated_reps_statistical = random.sample(available_reps, len(available_reps))
        
    
    # ----------------------------------------------------------------------
    # --- COLOR ASSIGNMENT (FIXED NAME ERROR HERE) ---
    # ----------------------------------------------------------------------
    
    # FIX: Initialize the dictionary outside the conditional logic to prevent NameError
    HIGHLIGHT_COLORS = {} 
    
    # Assign colors to the three statistical roles
    if len(curated_reps_statistical) >= 3:
        HIGHLIGHT_COLORS[curated_reps_statistical[0]] = 'blue'    # Random Path
        HIGHLIGHT_COLORS[curated_reps_statistical[1]] = 'red'      # Mean Follower Path
        HIGHLIGHT_COLORS[curated_reps_statistical[2]] = 'orange' # Outlier Path
    elif len(curated_reps_statistical) == 2:
        HIGHLIGHT_COLORS[curated_reps_statistical[0]] = 'blue'
        HIGHLIGHT_COLORS[curated_reps_statistical[1]] = 'red'
    elif len(curated_reps_statistical) == 1:
        HIGHLIGHT_COLORS[curated_reps_statistical[0]] = 'blue'

    # The block below for FORCED_REP_ID ('19') is correctly commented out and skipped.
    
    # ----------------------------------------------------------------------
    
    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Mean Hybrid Index (HI)", fontsize=14)
    ax.set_ylabel("Mean Heterozygosity (HET)", fontsize=14)
    
    num_reps = len(all_replicate_dfs) # Define num_reps here for the title later

    # 3. Plot Stochastic Paths 
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
                if color == 'blue': 
                    label = f"Random/Statistical Path {rep_id}"
                elif color == 'red': 
                    label = f"Mean Follower {rep_id}"
                elif color == 'orange': 
                    label = f"Outlier Path {rep_id}"
                # The purple label logic is skipped since the color 'purple' isn't assigned

            ax.plot(path_df['mean_HI'], path_df['mean_HET'],
                    color=color, linestyle='-', linewidth=linewidth, 
                    alpha=alpha, zorder=zorder, label=label) 

    # 4. Plot the Mean Path
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
        'Final': (0.01, -0.03) # Right and Down
    }
    
    # ----------------------------------------------------------------------
    # --- ADD NEW TARGET POINT TO HIGHLIGHT LIST (RESTORED RED POINT) ---
    # ----------------------------------------------------------------------
    
    # List the fixed points (PA, PB, HG1) and the final generation
    points_to_label = ['PA', 'PB', 'HG1'] 

    # If the CSV average was successfully calculated, add it
    if TARGET_HET_GEN:
        points_to_label.append(TARGET_HET_GEN)
        # Define a new offset for this point to keep it clean
        LABEL_OFFSETS[TARGET_HET_GEN] = (-0.01, 0.03) # Left and Up

    # Add the final generation to the list to ensure it's plotted in red
    final_gen = all_sorted_gen_labels[-1]
    if final_gen not in points_to_label:
        points_to_label.append(final_gen)
    
    # Highlight PA, PB, HG1, the final generation, AND the new Target HET generation
    for gen_name in points_to_label:
        if gen_name in grand_mean_df.index:
            mean_data = grand_mean_df.loc[gen_name]
            
            color = 'black'
            
            # Assign colors based on point type
            if gen_name == 'PA': color = 'black'
            elif gen_name == 'PB': color = 'gray'
            elif gen_name == 'HG1': color = 'blue'
            elif gen_name == TARGET_HET_GEN: 
                color = 'green' # The average crossing point
            elif gen_name == final_gen:
                color = 'red' # The final generation point

            ax.scatter(mean_data['mean_HI'], mean_data['mean_HET'],
                       color=color, s=100, edgecolors='black', linewidth=1.5, zorder=6)
            
            # Determine offset for clean labeling
            # Use 'Final' offset key only if the point is the last generation
            offset_key = 'Final' if gen_name == final_gen else gen_name
            dx, dy = LABEL_OFFSETS.get(offset_key, (0.01, 0.01))
            
            # Label the point
            ax.annotate(gen_name, 
                        (mean_data['mean_HI'], mean_data['mean_HET']), 
                        xytext=(mean_data['mean_HI'] + dx, mean_data['mean_HET'] + dy),
                        fontsize=10, color=color, ha='left', va='bottom', zorder=7)

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
    BATCH_CONFIGS = [
        {
            # This is the folder containing replicates 1 through 50
            "BASE_DIR": "/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs_no_recombination_50/",
            "REPLICATE_IDS": list(range(1, 51)) # Replicates 1 through 50 (exclusive end)
        }
    ]
    # ----------------------------------------------------
    
    # Determine the overall replicate range for the output filename
    all_start_id = min(c["REPLICATE_IDS"][0] for c in BATCH_CONFIGS if c["REPLICATE_IDS"])
    all_end_id = max(c["REPLICATE_IDS"][-1] for c in BATCH_CONFIGS if c["REPLICATE_IDS"])
    
    # Define the parent directory for output files
    parent_dir = os.path.dirname(BATCH_CONFIGS[0]["BASE_DIR"].rstrip('/'))
    
    # --- NEW: DEFINE THE PATH TO THE MATCHING GENERATIONS CSV ---
    # CORRECTION: Do NOT join an absolute path with parent_dir. Use the path relative to parent_dir or correct the join.
    ANALYSIS_OUTPUT_FILE = os.path.join(
        parent_dir,
        "/mnt/nfs2/bioenv/sg802/hybrid_sim_project/simulation_outputs_no_recombination_50/combined_matching_generations_no_recombination.csv"
    )
    
    # Define the unique output plot file name
    OVERLAY_PLOT_OUTPUT = os.path.join(
        parent_dir,
        "results", 
        f"tp_{all_start_id}_{all_end_id}_norecomb_50_NEW.pdf" 
    )

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OVERLAY_PLOT_OUTPUT), exist_ok=True)
    
    # --- NEW STEP: CALCULATE AVERAGE CROSSING GENERATION ---
    AVG_HG_LABEL = calculate_average_crossing_gen(ANALYSIS_OUTPUT_FILE) 

    
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
            save_filename=OVERLAY_PLOT_OUTPUT,
            target_hg_label=AVG_HG_LABEL # Pass the new average HG label
        )