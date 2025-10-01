import pandas as pd
import os
import glob
import argparse
from typing import Optional

def find_matching_hybrid_gen(input_df: pd.DataFrame) -> dict:
    """
    Calculates the mean parental and hybrid heterozygosity scores
    for a single replicate and finds the closest hybrid generation.
    Returns a dictionary of the results.
    """
    # Calculate average parental heterozygosity (baseline)
    parental_df = input_df[input_df['generation'].isin(['PA', 'PB'])]
    if parental_df.empty:
        raise ValueError("Parental generations 'PA' and 'PB' not found in the data.")
    
    avg_parental_het = parental_df['HET'].mean()

    # Find the closest hybrid generation
    hybrid_df = input_df[~input_df['generation'].isin(['PA', 'PB'])]
    if hybrid_df.empty:
        raise ValueError("No hybrid generations found in the data.")
        
    mean_hybrid_het_df = hybrid_df.groupby('generation')['HET'].mean().reset_index()
    
    # Calculate the difference and find the minimum
    mean_hybrid_het_df['het_diff'] = abs(mean_hybrid_het_df['HET'] - avg_parental_het)
    closest_hybrid_gen = mean_hybrid_het_df.loc[mean_hybrid_het_df['het_diff'].idxmin()]
    
    return {
        'avg_parental_het': avg_parental_het,
        'matching_hybrid_gen': closest_hybrid_gen['generation'],
        'matching_hybrid_het': closest_hybrid_gen['HET'],
        'difference': closest_hybrid_gen['het_diff']
    }

def run_analysis_and_append(input_directory: str, output_file: str, replicate_id: int):
    """
    Processes a single replicate's files and appends the result to a master file.
    """
    file_pattern = "results_rep_*individual_hi_het.csv"
    
    all_file_paths = glob.glob(os.path.join(input_directory, '**', file_pattern), recursive=True)
    if not all_file_paths:
        print(f"No files found for replicate {replicate_id} in '{input_directory}'.")
        return

    # Process the first file found (there should only be one per replicate)
    file_path = all_file_paths[0]
    print(f"Processing file: {file_path}")
    
    replicate_df = pd.read_csv(file_path)
    
    try:
        result_dict = find_matching_hybrid_gen(replicate_df)
        result_dict['replicate_id'] = replicate_id
        result_df = pd.DataFrame([result_dict])
        
        # Check if the file exists to decide on writing the header
        write_header = not os.path.exists(output_file)
        
        # Append the result to the master file
        result_df.to_csv(output_file, mode='a', header=write_header, index=False)
        
        print(f"Result for replicate {replicate_id} appended to: {output_file}")
        
    except ValueError as e:
        print(f"Skipping replicate {replicate_id} due to an error: {e}")
    except Exception as e:
        print(f"An error occurred during script execution for replicate {replicate_id}: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze simulation results and append to a single file.")
    parser.add_argument("--input_dir", required=True, help="Base directory for the current replicate.")
    parser.add_argument("--output_file", required=True, help="Path to the single master output file.")
    parser.add_argument("--replicate_id", type=int, required=True, help="ID of the current replicate.")
    
    args = parser.parse_args()
    run_analysis_and_append(args.input_dir, args.output_file, args.replicate_id)