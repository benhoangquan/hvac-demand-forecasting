import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
try:
    df = pd.read_csv('results/xgb_test_set_eval.csv')
except FileNotFoundError:
    print("Error: 'results/xgb_test_set_eval.csv' not found.")
    exit()

# Create the scatter plot
plt.figure(figsize=(10, 10))
sns.scatterplot(data=df, x='y_true', y='y_pred', alpha=0.6, label='Predictions')

# Add a line for perfect prediction (y=x)
max_val = max(df['y_true'].max(), df['y_pred'].max())
min_val = min(df['y_true'].min(), df['y_pred'].min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction (y=x)')

# Add labels and title
plt.xlabel('Ground Truth (y_true)')
plt.ylabel('Predictions (y_pred)')
plt.title('Ground Truth vs. Predictions')
plt.legend()
plt.grid(True)
plt.axis('equal') # Ensure the scale is the same for both axes
plt.tight_layout()

# Save the plot
output_path = 'results/results_comparison.png'
plt.savefig(output_path)
print(f"Plot saved to {output_path}")

# Show the plot (optional, comment out if running in a non-GUI environment)
# plt.show() 