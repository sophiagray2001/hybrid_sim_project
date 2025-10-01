import os
import pandas as pd

def analyze_hybridization_data(input_filename, Ne_value=100):
    """
    Analyzes hybridization data to find the first generation with a mean HET
    greater than or equal to a specified threshold.

    Args:
        input_filename (str): The path to the input CSV file. The file should
                              have 'generation', 'HI', and 'HET' columns.
        Ne_value (int): The effective population size (Ne) used for the final
                        calculation. This is a configurable parameter.
    """
    # Check if the input file exists
    if not os.path.exists(input_filename):
        print(f"Error: The input file '{input_filename}' was not found.")
        print("Please ensure the file is in the correct directory.")
        return

    try:
        # Read the individual data into a pandas DataFrame
        hi_het_df = pd.read_csv(input_filename)

        # --- Step 1: Calculate the mean HI and HET per generation ---
        mean_hi_het_df = hi_het_df.groupby('generation').agg(
            mean_HI=('HI', 'mean'),
            mean_HET=('HET', 'mean')
        ).reset_index()

        print("Analysis Results:")
        print("-" * 30)

        # --- Step 2: Find the first generation with mean HET >= 0.02 ---
        target_het_threshold = 0.02

        # Filter for generations that meet the HET threshold
        filtered_generations = mean_hi_het_df[mean_hi_het_df['mean_HET'] >= target_het_threshold].copy()
        
        if filtered_generations.empty:
            print(f"No generation found with a mean HET greater than or equal to {target_het_threshold}.")
            return

        # Find the first generation that meets the condition
        # This is typically the one with the lowest numeric part of its name (e.g., F1, BC1A, etc.)
        closest_generation_row = filtered_generations.sort_values(by='mean_HET').iloc[0]
        
        closest_generation = closest_generation_row['generation']
        closest_het = closest_generation_row['mean_HET']

        print(f"The first generation with a mean HET ({closest_het:.4f}) >= {target_het_threshold} is: {closest_generation}")

        # --- Step 3: Calculate the number of Ne generations it took ---
        try:
            # Extract the numeric part of the generation string.
            gen_number_str = ''.join(filter(str.isdigit, closest_generation))
            if gen_number_str:
                gen_number = int(gen_number_str)
            else:
                print(f"Warning: Could not extract a generation number from '{closest_generation}'.")
                print("Skipping the Ne generations calculation.")
                return
        except ValueError:
            print(f"Error: Could not convert '{gen_number_str}' to an integer.")
            print("Skipping the Ne generations calculation.")
            return

        # Assuming 'F1' is generation 1 for this calculation.
        # A backcross generation like 'BC1A' is the result of one cross,
        # so we count it as one step from the parental cross (F1).
        num_generations_passed = gen_number
        if closest_generation.startswith('BC'):
             num_generations_passed += 1
        
        ne_generations = num_generations_passed / Ne_value
        
        print(f"Number of generations passed since F1: {num_generations_passed}")
        print(f"Number of Ne generations required (with Ne={Ne_value}): {ne_generations:.2f}")

    except FileNotFoundError:
        print(f"Error: The input file '{input_filename}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Specify your input file here.
    input_file = "C:/Users/sg802/Documents/git_clone/hybrid_sim_project/simulation_outputs/results/For presentation_2_individual_hi_het.csv"
    
    # You can also specify the effective population size (Ne) here.
    # The default is Ne = 100.
    analyze_hybridization_data(input_file, Ne_value=200)
