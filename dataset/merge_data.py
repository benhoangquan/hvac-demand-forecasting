import pandas as pd
import os

# Define file paths
test_features_file = 'dataset/test_features.csv'
submission_file = 'results/quan_submission_09b30ae419144d8a901d993de8277877.csv'
output_file = 'dataset/visualisation_trial_8/merged_test_data.csv'
output_dir = 'dataset/visualisation_trial_8'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the datasets
try:
    test_features_df = pd.read_csv(test_features_file)
    submission_df = pd.read_csv(submission_file)
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit()

# Check for 'ID' column in both dataframes
if 'ID' not in test_features_df.columns or 'ID' not in submission_df.columns:
    print("Error: 'ID' column not found in one or both files. Cannot merge.")
    exit()

# Merge the dataframes on the 'ID' column
merged_df = pd.merge(test_features_df, submission_df, on='ID', how='inner')

# Save the merged dataframe
merged_df.to_csv(output_file, index=False)

print(f"Merged data saved to {output_file}") 