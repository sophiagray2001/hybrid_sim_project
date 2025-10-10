# MODIFIED DISTRIBUTION PLOTTING FUNCTION

def plot_crossing_time_distribution(input_filepath: str, save_filename: str):
    """
    Reads the pre-calculated crossing time data and plots the distribution 
    of generations required for HET to decrease to the parental mean level,
    including the 95% confidence interval.
    """
    
    # 1. Load Data
    try:
        data = pd.read_csv(input_filepath)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        return
    
    if data.empty:
        print("Error: Input data file is empty.")
        return
    
    # 2. Extract Generation Time (Convert 'HGx' to integer x)
    data['crossing_time'] = data['matching_hybrid_gen'].str.extract(r'HG(\d+)').astype(float)
    
    crossing_times = data['crossing_time'].dropna().tolist()

    if not crossing_times:
        print(f"Warning: No valid crossing times found in the data.")
        return

    # 3. Plot the Distribution (Histogram)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plotting the histogram
    ax.hist(crossing_times, bins=20, edgecolor='black', color='teal', alpha=0.7)
    
    # Calculate Mean, Median, and CI
    crossing_series = pd.Series(crossing_times)
    mean_time = crossing_series.mean()
    median_time = crossing_series.median()
    
    # Calculate 95% CI using Percentiles
    ci_lower = crossing_series.quantile(0.025)
    ci_upper = crossing_series.quantile(0.975)
    
    # Add Mean and Median lines
    ax.axvline(mean_time, color='red', linestyle='--', linewidth=2, label=f'Mean Time: {mean_time:.1f} Gens')
    ax.axvline(median_time, color='orange', linestyle='-', linewidth=2, label=f'Median Time: {median_time:.1f} Gens')
    
    # Add 95% CI lines
    ax.axvline(ci_lower, color='gray', linestyle=':', linewidth=1.5, label=f'95% CI: ({ci_lower:.1f}-{ci_upper:.1f}) Gens')
    ax.axvline(ci_upper, color='gray', linestyle=':', linewidth=1.5)

    # Set up labels and title
    ax.set_xlabel("Time (Generations, HGx) to Reach Parent HET Threshold", fontsize=12)
    ax.set_ylabel("Number of Replicates", fontsize=12)
    ax.set_title("Distribution of Time Required to Purge Heterozygosity", fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right')
    
    plt.savefig(save_filename, bbox_inches='tight')
    plt.close()
    print(f"\nTime distribution plot saved to: {save_filename}")