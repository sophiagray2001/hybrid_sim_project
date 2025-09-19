import pandas as pd

# IMPORTANT: Change 'your_input_file.csv' to the name of your file
file_path = r'C:\Users\sophi\Jupyter_projects\Hybrid_Code\real_world_sim\allele_freq_test.csv'

print(f"Attempting to read file: {file_path}")

try:
    # Use the 'on_bad_lines' parameter to skip any corrupted rows
    df = pd.read_csv(file_path, on_bad_lines='skip')
    
    # Check if the dataframe is empty
    if df.empty:
        print("Error: File was read successfully, but it contains no data.")
    else:
        # Print the exact column names pandas read from the file
        print("File read successfully!")
        print("Columns found:")
        print(list(df.columns))
        
        # Print the first few rows to check for data issues
        print("\nFirst 5 rows of data:")
        print(df.head())
        
except Exception as e:
    # This will catch any error and give you a detailed message
    print(f"\nAn error occurred while reading the file:")
    print(e)