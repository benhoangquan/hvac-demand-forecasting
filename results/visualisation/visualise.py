# calculate rmse of trial 8 over timestep 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load the test evaluation data
df = pd.read_csv('results/test_set_eval_trial_8.csv')

# Calculate the squared error for each row
df['squared_error'] = (df['y_true'] - df['y_pred'])**2

# Calculate the rolling RMSE with a window size
window_size = 100  # Adjust as needed
df['rolling_rmse'] = np.sqrt(df['squared_error'].rolling(window=window_size).mean())

# Plot RMSE over row number
plt.figure(figsize=(12, 6))
plt.plot(df.index, np.sqrt(df['squared_error']), 'b-', alpha=0.3, label='Point-wise Error')
plt.plot(df.index, df['rolling_rmse'], 'r-', linewidth=2, label=f'Rolling RMSE (window={window_size})')

# Calculate overall RMSE for reference
overall_rmse = np.sqrt(mean_squared_error(df['y_true'], df['y_pred']))
plt.axhline(y=overall_rmse, color='g', linestyle='--', label=f'Overall RMSE: {overall_rmse:.4f}')

plt.xlabel('Row Number')
plt.ylabel('RMSE')
plt.title('RMSE Over Row Number')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/visualisation/rmse_over_rows_trial_8.png')
plt.show()

# Print overall metrics
print(f"Overall RMSE: {overall_rmse:.4f}")




