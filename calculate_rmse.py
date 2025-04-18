import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load the data
csv_path = 'results/test_set_eval.csv'
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: '{csv_path}' not found.")
    exit()

# Check if required columns exist
if 'y_true' not in df.columns or 'y_pred' not in df.columns:
    print(f"Error: CSV file must contain 'y_true' and 'y_pred' columns.")
    exit()

# Calculate RMSE
# Ensure data is numeric and handle potential NaNs
y_true = pd.to_numeric(df['y_true'], errors='coerce')
y_pred = pd.to_numeric(df['y_pred'], errors='coerce')

# Drop rows where conversion failed (NaNs)
df_clean = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).dropna()

if df_clean.empty:
    print("Error: No valid numeric data found in 'y_true' or 'y_pred' columns after cleaning.")
    exit()

rmse = np.sqrt(mean_squared_error(df_clean['y_true'], df_clean['y_pred']))
mse = mean_squared_error(df_clean['y_true'], df_clean['y_pred'])
mae = mean_absolute_error(df_clean['y_true'], df_clean['y_pred'])
r2 = r2_score(df_clean['y_true'], df_clean['y_pred'])

# Print the result
print(f"RMSE on {csv_path}: {rmse:.4f}") 
print(f"MSE on {csv_path}: {mse:.4f}") 
print(f"MAE on {csv_path}: {mae:.4f}") 
print(f"R2 on {csv_path}: {r2:.4f}") 
