import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import scipy.stats as stats

def calculate_confidence_interval(data, confidence=0.95):
    """Calculates the 95% confidence interval for a given dataset."""
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def create_matching_generations_histogram(input_file: str, output_file: str):
    """
    Reads a consolidated CSV, extracts hybrid generation numbers,
    and creates a histogram of their distribution in numerical order,
    with an overlay of the average heterozygosity and 95% CI.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'")
        return

    # Read the consolidated data
    df = pd.read_csv(input_file)
    
    # Filter for only hybrid generations (HG) as requested
    hybrid_df = df[df['matching_hybrid_gen'].str.startswith('HG', na=False)].copy()

    if hybrid_df.empty:
        print("No hybrid generations found in the input data. Cannot create plot.")
        return

    # Extract the generation number and convert to a categorical type for sorting
    hybrid_df['gen_num'] = hybrid_df['matching_hybrid_gen'].str.extract(r'HG(\d+)').astype(int)
    
    # Sort the dataframe to get the correct order for plotting
    sorted_generations = hybrid_df.sort_values(by='gen_num')['matching_hybrid_gen'].unique()
    
    # Count the frequency of each generation
    counts = hybrid_df['matching_hybrid_gen'].value_counts().reindex(sorted_generations, fill_value=0)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(counts.index, counts.values, color='skyblue', edgecolor='black')

    # Add labels and a title
    ax.set_xlabel("Hybrid Generation")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Hybrid Generations Matching Parental Heterozygosity")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # --- New code to plot the average and CI ---
    # Calculate the average and 95% CI of the matching heterozygosity
    avg_het, ci_95 = calculate_confidence_interval(hybrid_df['matching_hybrid_het'])
    
    # Plot the average line
    ax.axhline(y=avg_het, color='red', linestyle='--', label=f'Mean Matching Het: {avg_het:.2f}')
    
    # Plot the 95% CI as a shaded area
    ax.axhspan(avg_het - ci_95, avg_het + ci_95, color='red', alpha=0.1, label='95% Confidence Interval')
    
    # Add a legend
    ax.legend()
    
    # Ensure the y-axis is large enough to show the CI
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(min(y_min, avg_het - ci_95 * 1.1), max(y_max, avg_het + ci_95 * 1.1))

    # Save the plot to a file
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(output_file)
    print(f"Plot saved successfully to '{output_file}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a histogram of matching hybrid generations.")
    parser.add_argument("--input_file", required=True, help="Path to the consolidated results CSV file.")
    parser.add_argument("--output_file", required=True, help="Path and filename for the output histogram plot (e.g., 'histogram.png').")

    args = parser.parse_args()
    create_matching_generations_histogram(args.input_file, args.output_file)