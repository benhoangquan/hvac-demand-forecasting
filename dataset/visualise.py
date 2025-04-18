import pandas as pd
import matplotlib.pyplot as plt
import os

# Define paths
# data_file = 'dataset/train_dataset.csv' # Old path
data_file = 'dataset/visualisation_trial_8/merged_test_data.csv' # New path for merged test data
output_dir = 'dataset/visualisation_trial_8'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
try:
    df = pd.read_csv(data_file)
except FileNotFoundError:
    print(f"Error: Data file not found at {data_file}")
    exit()

# Identify features and target variables
features = [
    'temperature_exterieure',
    'humidite',
    'ensoleillement',
    'vitesse_vent',
    'direction_vent',
    'temperature_interieure'
]
# targets = ['puissance_cvac', 'puissance_cvac_future']
targets = ['puissance_cvac_future']
# Check if all features and targets exist in the dataframe
missing_cols = [col for col in features + targets if col not in df.columns]
if missing_cols:
    print(f"Error: The following columns are missing in the CSV: {', '.join(missing_cols)}")
    exit()

# Convert target columns to numeric, coercing errors
for target in targets:
    df[target] = pd.to_numeric(df[target], errors='coerce')

# Drop rows with NaN values in target columns that resulted from coercion
df.dropna(subset=targets, inplace=True)


# Generate and save scatter plots
for feature in features:
    for target in targets:
        plt.figure(figsize=(10, 6))
        plt.scatter(df[feature], df[target], alpha=0.5, s=5) # Added s=5 for smaller points
        plt.title(f'{feature.replace("_", " ").title()} vs {target.replace("_", " ").title()}')
        plt.xlabel(feature.replace("_", " ").title())
        plt.ylabel(target.replace("_", " ").title())
        plt.grid(True)

        # Construct filename and save plot
        filename = f'trial_8_{feature}_vs_{target}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close() # Close the plot to free memory

print(f"Scatter plots saved to {output_dir}")
