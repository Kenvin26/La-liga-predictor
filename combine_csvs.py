import pandas as pd
import glob

# Get all CSV files in the current directory matching the pattern
csv_files = glob.glob('SP1*.csv')

# Load and concatenate all CSVs
all_dfs = [pd.read_csv(f) for f in csv_files]
combined_df = pd.concat(all_dfs, ignore_index=True)

# Clean and normalize columns
combined_df.columns = [col.strip().lower() for col in combined_df.columns]
combined_df = combined_df.drop_duplicates()

# Save the cleaned DataFrame to a new CSV
combined_df.to_csv('combined_la_liga_cleaned.csv', index=False)

print(f"Combined and cleaned {len(csv_files)} CSV files into 'combined_la_liga_cleaned.csv' with {len(combined_df)} rows.") 