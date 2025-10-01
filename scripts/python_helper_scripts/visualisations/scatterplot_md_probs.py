import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_missing_data_comparison(input_file_path, output_file_path):
    """
    Compares input missing data probability to calculated proportion of
    missing data in the output file for PA and PB.
    """
    # 1. Load the known missing data probabilities from the input file
    input_df = pd.read_csv(input_file_path, index_col='marker_id')
    
    # 2. Load the full output data
    output_df = pd.read_csv(output_file_path)

    # 3. Create a helper column to flag missing genotypes
    output_df['is_missing'] = output_df['genotype'] == './.'
    
    # 4. Filter and calculate missing data proportion for each population
    # Population A
    pop_a_df = output_df[output_df['individual_id'].str.startswith('PA')]
    missing_prop_a = pop_a_df.groupby('marker_id')['is_missing'].mean()

    # Population B
    pop_b_df = output_df[output_df['individual_id'].str.startswith('PB')]
    missing_prop_b = pop_b_df.groupby('marker_id')['is_missing'].mean()
    
    # 5. Prepare data for plotting
    combined_a = pd.DataFrame({
        'Known_Prob': input_df['md_prob'],
        'Calculated_Prop': missing_prop_a
    }).dropna()

    combined_b = pd.DataFrame({
        'Known_Prob': input_df['md_prob'],
        'Calculated_Prop': missing_prop_b
    }).dropna()

    # 6. Create the scatterplot
    plt.figure(figsize=(8, 8))
    
    plt.scatter(combined_a['Known_Prob'], combined_a['Calculated_Prop'], color='blue', label='Population A')
    plt.scatter(combined_b['Known_Prob'], combined_b['Calculated_Prop'], color='red', label='Population B')
    
    plt.plot([0, 1], [0, 1], 'k--', label='y=x line') # A black dashed line
    
    plt.title('Simulated Missing Data vs. Known Probability')
    plt.xlabel('Known Missing Data Probability (from Input File)')
    plt.ylabel('Calculated Missing Data Proportion (from Output Data)')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.show()

# Example Usage:
# You must change these file paths to match your specific files
input_file = r"C:\Users\sg802\Documents\git_clone\hybrid_sim_project\beetle_input_1.csv"
output_file = r"C:\Users\sg802\Documents\git_clone\hybrid_sim_project\parent_genotypes.csv"
plot_missing_data_comparison(input_file, output_file)