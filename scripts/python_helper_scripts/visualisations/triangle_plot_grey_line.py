import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from typing import Optional

def plot_simulation_path(
    mean_hi_het_df: pd.DataFrame, 
    save_filename: Optional[str] = None,
    highlight_gen: Optional[int] = None # Added a new optional parameter
):
    """
    Plots the full path of the mean HI vs. HET from the first generation onwards,
    with no background points. Highlights key generations with larger points and labels.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set up plot aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Mean Hybrid Index (HI)", fontsize=12)
    ax.set_ylabel("Mean Heterozygosity (HET)", fontsize=12)

    # --- Step 1: Sort all generations to ensure correct plotting order ---
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

    # Sort the dataframe by the custom sort key
    all_sorted_gen_labels = sorted(mean_hi_het_df.index, key=sort_key)
    sorted_df_all = mean_hi_het_df.loc[all_sorted_gen_labels]

    # --- Step 2: Plot the full path from the first numeric generation onwards ---
    # Find the starting point for the line (e.g., HG1 or F1)
    path_start_index = next((i for i, label in enumerate(all_sorted_gen_labels) 
                             if not (label.startswith('PA') or label.startswith('PB'))), 
                             None)
    
    if path_start_index is not None:
        path_df = sorted_df_all.iloc[path_start_index:]
        ax.plot(path_df['mean_HI'], path_df['mean_HET'],
                color='gray', linestyle='-', linewidth=2, alpha=0.5, zorder=2)

    # --- Step 3: Identify and plot key generations as larger, labeled points ---
    generations_to_highlight = ['PA', 'PB']
    
    # Find the first and last numeric generations to highlight
    numeric_gens = [g for g in all_sorted_gen_labels if g.startswith('HG') or g.startswith('F') or g.startswith('BC')]
    if numeric_gens:
        first_numeric_gen = numeric_gens[0]
        last_numeric_gen = numeric_gens[-1]
        generations_to_highlight.extend([first_numeric_gen, last_numeric_gen])

    # Add the specific, user-requested generation to the highlight list
    if highlight_gen is not None:
        gen_to_add = f'HG{highlight_gen}'
        if gen_to_add in mean_hi_het_df.index:
            generations_to_highlight.append(gen_to_add)
            print(f"User-specified generation '{gen_to_add}' will be highlighted.")
        else:
            print(f"Warning: Generation '{gen_to_add}' not found in the data.")
            
    print(f"Detected and highlighting first generation: {first_numeric_gen}")
    print(f"Detected and highlighting last generation: {last_numeric_gen}")
    
    # Plot the specific highlighted points on top
    for gen_name in generations_to_highlight:
        if gen_name in mean_hi_het_df.index:
            mean_data = mean_hi_het_df.loc[gen_name]
            
            # Use fixed colors for PA, PB, first and last gen, and the user-specified gen
            color = 'black'
            label_text = gen_name
            
            if gen_name == 'PA':
                color = 'black'
            elif gen_name == 'PB':
                color = 'gray'
            elif gen_name == first_numeric_gen:
                color = 'purple'
            elif gen_name == last_numeric_gen:
                color = 'green'
            elif highlight_gen is not None and gen_name == f'HG{highlight_gen}':
                color = 'red'  # A distinct color for the specified generation
                
            ax.scatter(mean_data['mean_HI'], mean_data['mean_HET'],
                       color=color, s=80, edgecolors='black', linewidth=1.5, zorder=3, label=label_text)
            
            ax.text(mean_data['mean_HI'] + 0.01, mean_data['mean_HET'] + 0.01, label_text,
                    fontsize=10, color='black', ha='left', va='bottom', zorder=4)

    # Plot the triangle edges
    triangle_edges = [
        [(0.0, 0.0), (0.5, 1.0)],
        [(0.5, 1.0), (1.0, 0.0)],
        [(0.0, 0.0), (1.0, 0.0)]
    ]
    for (x0, y0), (x1, y1) in triangle_edges:
        ax.plot([x0, x1], [y0, y1], linestyle='-', color='gray', linewidth=1.5, alpha=0.7, zorder=0)

    # Final plot settings
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)

    if save_filename:
        plt.savefig(save_filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# Main script execution
if __name__ == "__main__":
    input_file = r"C:\Users\sg802\Documents\git_clone\hybrid_sim_project\simulation_outputs\replicates\results\results_rep_10_individual_hi_het.csv"
    output_filename = "simulation_outputs/results/triangle_plot_line_path.png"

    # Specify the generation number you want to highlight here
    # For example, to highlight generation 100:
    highlight_generation_number = 1539

    try:
        if not os.path.exists(input_file):
            print(f"Error: The input file '{input_file}' was not found.")
            print("Please ensure the path is correct or run your simulation script.")
        else:
            hi_het_df = pd.read_csv(input_file)
            mean_hi_het_df = hi_het_df.groupby('generation').agg(
                mean_HI=('HI', 'mean'),
                mean_HET=('HET', 'mean')
            )
            
            # Pass the highlight_generation_number to the plotting function
            plot_simulation_path(mean_hi_het_df, output_filename, highlight_gen=highlight_generation_number)
            print(f"Plot of simulation path saved to: {output_filename}")
            
    except Exception as e:
        print(f"An error occurred during script execution: {e}")