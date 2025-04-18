#!/usr/bin/env python3
"""
predict_hvac.py

Load a trained Hybrid model (LSTM+TCN) and scalers saved by train.py.
Predict the single 'puissance_cvac_future' (16 steps ahead) value for each ID
in test_features.csv and save results in Kaggle submission format.
"""
import os
import argparse
import pandas as pd
import numpy as np
import torch
import joblib

# Import necessary components
from hvac_lstm_prototype import LSTMModel
from tcn_model import TCNModel
from hybrid_model import HybridModel

# Copy the add_time_features function from train.py
def add_time_features(df):
    """Adds time-based features to the DataFrame."""
    # Ensure 'date' column is datetime type
    if 'date' not in df.columns:
        raise ValueError("'date' column is required for time features but not found in test data.")
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12.0)
    return df

def main():
    parser = argparse.ArgumentParser(description='Predict future HVAC consumption using Hybrid Model (LSTM+TCN)')
    parser.add_argument('--test_csv', default=os.path.join('dataset', 'test_features.csv'), help='Path to test features CSV file')
    parser.add_argument('--model_dir', default='models', help='Directory containing the saved model and scalers')
    parser.add_argument('--results_dir', default='results', help='Directory to save prediction results')
    parser.add_argument('--model_filename', default='lstm_hvac_best.pth', help='Filename of the saved model state dict')
    parser.add_argument('--feature_scaler_filename', default='feature_scaler.save', help='Filename of the saved feature scaler')
    parser.add_argument('--target_scaler_filename', default='target_scaler.save', help='Filename of the saved target scaler')
    parser.add_argument('--output_filename', default='submission.csv', help='Filename for the Kaggle submission CSV')
    parser.add_argument('--seq_len', type=int, default=16, help='Sequence length used during training')

    # Add arguments required to reconstruct model architecture
    # Feature Handling
    parser.add_argument('--use_time_features', action='store_true', default=True, help='Model was trained WITH time features (requires test data with date)')
    parser.add_argument('--no_time_features', action='store_false', dest='use_time_features', help='Model was trained WITHOUT time features')
    # LSTM Params (needed to init LSTM part of Hybrid)
    parser.add_argument('--hidden_size', type=int, default=128, help='Number of hidden units in LSTM (must match trained model)')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers (must match trained model)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for LSTM/FC (must match trained model)')
    # TCN Params (needed to init TCN part of Hybrid)
    parser.add_argument('--tcn_num_channels', type=int, nargs="+", default=[64, 64], help="Number of channels in each TCN layer (must match trained model)")
    parser.add_argument('--tcn_kernel_size', type=int, default=3, help="Kernel size for TCN convolutions (must match trained model)")
    parser.add_argument('--tcn_dropout', type=float, default=0.2, help="Dropout rate for TCN layers (must match trained model)")

    args = parser.parse_args()

    # Construct full paths
    model_path = os.path.join(args.model_dir, args.model_filename)
    feature_scaler_path = os.path.join(args.model_dir, args.feature_scaler_filename)
    target_scaler_path = os.path.join(args.model_dir, args.target_scaler_filename)
    output_path = os.path.join(args.results_dir, args.output_filename)

    # Load test features
    print(f"Loading test features from {args.test_csv}...")
    # Parse 'date' if time features are expected
    parse_dates = ['date'] if args.use_time_features else None
    df_test = pd.read_csv(args.test_csv, parse_dates=parse_dates)

    if 'ID' not in df_test.columns:
        raise ValueError("Test CSV must contain an 'ID' column.")
    ids = df_test['ID'].values

    # Define feature columns based on whether time features are used
    base_feature_cols = [
        'temperature_exterieure', 'humidite', 'ensoleillement',
        'vitesse_vent', 'direction_vent', 'temperature_interieure'
    ]
    if args.use_time_features:
        print("Adding time features...")
        df_test = add_time_features(df_test)
        time_feature_cols = ['hour_sin', 'hour_cos', 'dayofweek', 'month_sin', 'month_cos']
        all_feature_cols = base_feature_cols + time_feature_cols
    else:
        print("Not using time features.")
        all_feature_cols = base_feature_cols

    # Check if all expected columns are present
    if not set(all_feature_cols).issubset(df_test.columns):
        missing_cols = set(all_feature_cols) - set(df_test.columns)
        raise ValueError(f'Test CSV missing required feature columns: {missing_cols}')
    features = df_test[all_feature_cols].values
    input_size = len(all_feature_cols)
    print(f"Using {input_size} features: {all_feature_cols}")

    # Load feature scaler and scale features
    print(f"Loading feature scaler from {feature_scaler_path}...")
    if not os.path.isfile(feature_scaler_path):
        raise FileNotFoundError(f'Feature scaler not found at {feature_scaler_path}. Run train.py first.')
    feat_scaler = joblib.load(feature_scaler_path)

    # Validate scaler compatibility
    num_features_expected_by_scaler = feat_scaler.n_features_in_
    if input_size != num_features_expected_by_scaler:
         raise ValueError(
             f"Feature mismatch: Prepared {input_size} features based on script arguments, "
             f"but the loaded scaler expects {num_features_expected_by_scaler} features. "
             f"Ensure the --[no_]time_features flag matches how the scaler was trained."
         )

    print(f"Scaling {input_size} features...")
    features_scaled = feat_scaler.transform(features)

    # --- Input Sequence Preparation --- #
    print(f"Preparing input sequences by replicating the last step {args.seq_len} times...")
    num_samples = features_scaled.shape[0]
    num_features = features_scaled.shape[1]
    X_test_seq = np.zeros((num_samples, args.seq_len, num_features), dtype=np.float32)
    for i in range(num_samples):
        X_test_seq[i, :, :] = np.tile(features_scaled[i, :], (args.seq_len, 1))
    print(f"Input shape for model: {X_test_seq.shape}")

    # Load model state
    print(f"Loading model state from {model_path}...")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f'Model state not found at {model_path}. Run train.py first.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model structure based on args, then load state
    print("Initializing Hybrid Model structure...")
    lstm_model_instance = LSTMModel(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    tcn_model_instance = TCNModel(
        input_size=input_size,
        output_size=1, # TCN needs explicit output size
        num_channels=args.tcn_num_channels,
        kernel_size=args.tcn_kernel_size,
        dropout=args.tcn_dropout
    )
    model = HybridModel(lstm_model=lstm_model_instance, tcn_model=tcn_model_instance)

    # Load the state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Hybrid Model state loaded successfully.")

    # Run predictions
    print("Running predictions...")
    preds_scaled_list = []
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_test_seq).float().to(device)
        # Use the HybridModel for prediction
        batch_preds_scaled = model(X_tensor).cpu().numpy() # Shape: (num_samples, 1)
        preds_scaled_list.append(batch_preds_scaled)

    preds_scaled = np.concatenate(preds_scaled_list) # Shape: (num_samples, 1)

    # Load target scaler and inverse transform predictions
    print(f"Loading target scaler from {target_scaler_path}...")
    if not os.path.isfile(target_scaler_path):
        raise FileNotFoundError(f'Target scaler not found at {target_scaler_path}. Run train.py first.')
    target_scaler = joblib.load(target_scaler_path)
    preds = target_scaler.inverse_transform(preds_scaled).flatten() # Shape: (num_samples,)

    # Format predictions for submission
    print("Formatting predictions for submission...")
    os.makedirs(args.results_dir, exist_ok=True)
    df_out = pd.DataFrame({
        'ID': ids,
        'puissance_cvac_future': preds
    })

    df_out.to_csv(output_path, index=False)
    print(f'Submission file saved to {output_path}')

if __name__ == '__main__':
    main() 