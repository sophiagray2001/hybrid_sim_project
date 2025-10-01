import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_allele_frequencies(df, population_label):
    """
    Calculates allele frequencies for allele '0' for a specific population.
    
    Args:
        df (pd.DataFrame): The full dataframe containing all individuals.
        population_label (str): The label for the population to analyze (e.g., 'PA').
        
    Returns:
        pd.Series: A Series with marker_id as the index and allele frequency as the value.
    """
    # Filter the dataframe to only include the specified population
    pop_df = df[df['individual_id'].str.startswith(population_label)].copy()

    # Split the genotype column into two allele columns
    pop_df['allele1'] = pop_df['genotype'].str[0]
    pop_df['allele2'] = pop_df['genotype'].str[2]

    # Explicitly replace '.' with NaN and then drop those rows
    pop_df.replace('.', np.nan, inplace=True)
    pop_df.dropna(subset=['allele1', 'allele2'], inplace=True)
    
    # Convert allele columns to numeric
    pop_df['allele1'] = pd.to_numeric(pop_df['allele1'])
    pop_df['allele2'] = pd.to_numeric(pop_df['allele2'])
    
    # Count the number of '0' alleles
    total_zeros_per_marker = (pop_df[['allele1', 'allele2']] == 0).groupby(pop_df['marker_id']).sum().sum(axis=1)

    # Count the total number of non-missing alleles per marker
    total_alleles_per_marker = pop_df.groupby('marker_id')[['allele1', 'allele2']].count().sum(axis=1)
    
    # Calculate the frequency
    allele_frequencies = total_zeros_per_marker / total_alleles_per_marker
    
    return allele_frequencies

def plot_allele_frequency_comparison(input_file_path, output_file_path):
    """
    Compares input allele frequencies to calculated output frequencies for PA and PB.
    """
    # 1. Load the known allele frequencies from the input file
    # Ensure you are using the correct column name for marker IDs
    input_df = pd.read_csv(input_file_path, index_col='marker_id')
    
    # 2. Load the full output data
    output_df = pd.read_csv(output_file_path)

    # 3. Handle potential column name discrepancies in the output file
    # Replace these with the actual column names from your output file if they are different
    output_df.rename(columns={
        'individual_id': 'individual_id',
        'marker_id': 'marker_id',
        'genotype': 'genotype' 
    }, inplace=True)

    # 4. Calculate frequencies for both populations
    calculated_freq_pa = calculate_allele_frequencies(output_df, 'PA')
    calculated_freq_pb = calculate_allele_frequencies(output_df, 'PB')
    
    # 5. Prepare data for plotting
    combined_pa = pd.DataFrame({
        'Known_Freq': input_df['allele_freq_A'],
        'Calculated_Freq': calculated_freq_pa
    }).dropna()
    
    combined_pb = pd.DataFrame({
        'Known_Freq': input_df['allele_freq_B'],
        'Calculated_Freq': calculated_freq_pb
    }).dropna()

    # 6. Create the scatterplot
    plt.figure(figsize=(8, 8))
    
    plt.scatter(combined_pa['Known_Freq'], combined_pa['Calculated_Freq'], color='blue', label='Population A')
    plt.scatter(combined_pb['Known_Freq'], combined_pb['Calculated_Freq'], color='red', label='Population B')
    
    plt.plot([0, 1], [0, 1], 'k--', label='y=x line') # A black dashed line
    
    plt.title('Simulated vs. Known Allele Frequencies (Allele 0)')
    plt.xlabel('Known Allele Frequency (from Input File)')
    plt.ylabel('Calculated Allele Frequency (from Output Data)')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.show()

# Example Usage:
# You must change these file paths to match your specific files
input_file = r"C:\Users\sg802\Documents\git_clone\hybrid_sim_project\beetle_input_1.csv"
output_file = r"C:\Users\sg802\Documents\git_clone\hybrid_sim_project\parent_genotypes.csv"
plot_allele_frequency_comparison(input_file, output_file)