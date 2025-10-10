import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from typing import Optional

def plot_individual_generation(data_df: pd.DataFrame, target_gen: str, save_filename: str):
    """
    Plots individual HI and HET scores for a specified generation, always including
    PA, PB, and HG1 for comparison, with a legend.
    
    Args:
        data_df: The DataFrame containing individual HI and HET scores.
        target_gen: The name of the generation to plot (e.g., 'HG100').
        save_filename: The path to save the plot image.
    """
    generations_to_plot = ['PA', 'PB', 'HG1', target_gen]
    
    # Filter the DataFrame to include only the specified generations
    filtered_df = data_df[data_df['generation'].isin(generations_to_plot)].copy()
    
    # Check if the target generation exists in the data
    if target_gen not in filtered_df['generation'].unique():
        print(f"Error: The specified generation '{target_gen}' was not found in the data.")
        return

    # Set up plot aesthetics
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Hybrid Index (HI)", fontsize=12)
    ax.set_ylabel("Heterozygosity (HET)", fontsize=12)
    
    # Define colors for the generations to be plotted
    gen_colors = {
        'PA': 'black',
        'PB': 'gray',
        'HG1': 'purple',
        target_gen: 'green'
    }
    
    # Plot each selected generation
    for gen_name, color in gen_colors.items():
        if gen_name in filtered_df['generation'].unique():
            gen_data = filtered_df[filtered_df['generation'] == gen_name]
            ax.scatter(gen_data['HI'], gen_data['HET'],
                        s=30, alpha=0.6, color=color, label=gen_name, zorder=2)
            
    ax.legend(loc='upper right', title="Generations")

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
    
    # Save and close
    plt.savefig(save_filename, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_filename}")

# --- Main script execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot individual HI and HET scores for a specified generation.")
    parser.add_argument("generation", type=str, help="The generation to plot (e.g., 'HG500', 'F2').")
    
    args = parser.parse_args()
    
    # Using relative paths for better portability
    input_file = r"C:\Users\sg802\Documents\git_clone\hybrid_sim_project\scripts\simulation_outputs\results\results_rep_100_individual_hi_het.csv"
    output_directory = r"C:\Users\sg802\Documents\git_clone\hybrid_sim_project\scripts\simulation_outputs\results\individual_point_tps"
    output_filename = os.path.join(output_directory, f"triangle_plot_{args.generation}.png")

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    if not os.path.exists(input_file):
        print(f"Error: The input file '{input_file}' was not found.")
        print("Please run your main simulation script first to generate the data.")
    else:
        # Read the individual data
        try:
            hi_het_df = pd.read_csv(input_file)
            plot_individual_generation(hi_het_df, args.generation, output_filename)
        except Exception as e:
            print(f"An error occurred while reading the data or plotting: {e}")
