#!/usr/bin/env python3
"""
predict_hvac.py

Load a trained LSTM model and scalers saved by train.py.
Predict the single 'puissance_cvac_future' (16 steps ahead) value for each ID
in test_features.csv and save results in Kaggle submission format.
"""
import os
import argparse
import pandas as pd
import numpy as np
import torch
import joblib

from hvac_lstm_prototype import LSTMModel

def main():
    parser = argparse.ArgumentParser(description='Predict future HVAC consumption (single value)')
    parser.add_argument('--test_csv', default=os.path.join('dataset', 'test_features.csv'), help='Path to test features CSV file')
    parser.add_argument('--model_dir', default='models', help='Directory containing the saved model and scalers')
    parser.add_argument('--results_dir', default='results', help='Directory to save prediction results')
    parser.add_argument('--model_filename', default='lstm_hvac.pth', help='Filename of the saved model state dict')
    parser.add_argument('--feature_scaler_filename', default='feature_scaler.save', help='Filename of the saved feature scaler')
    parser.add_argument('--target_scaler_filename', default='target_scaler.save', help='Filename of the saved target scaler')
    parser.add_argument('--output_filename', default='submission.csv', help='Filename for the Kaggle submission CSV')
    parser.add_argument('--seq_len', type=int, default=16, help='Sequence length used during training')

    args = parser.parse_args()

    # Construct full paths
    model_path = os.path.join(args.model_dir, args.model_filename)
    feature_scaler_path = os.path.join(args.model_dir, args.feature_scaler_filename)
    target_scaler_path = os.path.join(args.model_dir, args.target_scaler_filename)
    output_path = os.path.join(args.results_dir, args.output_filename)

    # Load test features
    print(f"Loading test features from {args.test_csv}...")
    df_test = pd.read_csv(args.test_csv)
    if 'ID' not in df_test.columns:
        raise ValueError("Test CSV must contain an 'ID' column.")
    ids = df_test['ID'].values
    feature_cols = [
        'temperature_exterieure', 'humidite', 'ensoleillement',
        'vitesse_vent', 'direction_vent', 'temperature_interieure'
    ]
    if not set(feature_cols).issubset(df_test.columns):
        missing_cols = set(feature_cols) - set(df_test.columns)
        raise ValueError(f'Test CSV missing required feature columns: {missing_cols}')
    features = df_test[feature_cols].values

    # Load feature scaler and scale features
    print(f"Loading feature scaler from {feature_scaler_path}...")
    if not os.path.isfile(feature_scaler_path):
        raise FileNotFoundError(f'Feature scaler not found at {feature_scaler_path}. Run train.py first.')
    feat_scaler = joblib.load(feature_scaler_path)
    features_scaled = feat_scaler.transform(features)

    # --- Input Sequence Preparation --- #
    # IMPORTANT: This assumes each row in test_features.csv is the *last* known time step.
    # To feed the LSTM which expects `seq_len` steps, we replicate the last step.
    # This might not be the optimal way if historical data relevant to the test IDs is available elsewhere.
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

    # Infer model dimensions from state dict
    state = torch.load(model_path, map_location=device)
    try:
        ih_keys = [k for k in state.keys() if k.startswith('lstm.weight_ih_l')]
        layers = [int(k.split('_l')[-1].split('_')[0]) for k in ih_keys]
        num_layers = max(layers) + 1 if layers else 1
        wih0 = state.get('lstm.weight_ih_l0')
        if wih0 is None: raise KeyError('Missing lstm.weight_ih_l0')
        hidden_size = wih0.shape[0] // 4
        input_size = wih0.shape[1]
        # Check output layer size (should be 1)
        fc_weight = state.get('fc.weight')
        if fc_weight is None: raise KeyError('Missing fc.weight')
        output_size = fc_weight.shape[0]
        if output_size != 1:
            print(f"Warning: Loaded model expects output size {output_size}, but should be 1 for this task.")

    except Exception as e:
        raise ValueError(f"Could not determine model parameters from state dict: {e}")

    # Initialize model and load state
    print(f"Initializing model (input={input_size}, hidden={hidden_size}, layers={num_layers}, output=1)...")
    # Ensure model is initialized with output_size=1 regardless of loaded state check
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Run predictions
    print("Running predictions...")
    preds_scaled_list = []
    with torch.no_grad():
        # Process in batches if needed (simple approach for now)
        X_tensor = torch.from_numpy(X_test_seq).float().to(device)
        batch_preds_scaled = model(X_tensor).cpu().numpy() # Shape: (num_samples, 1)
        preds_scaled_list.append(batch_preds_scaled)

    preds_scaled = np.concatenate(preds_scaled_list) # Shape: (num_samples, 1)

    # Load target scaler and inverse transform predictions
    print(f"Loading target scaler from {target_scaler_path}...")
    if not os.path.isfile(target_scaler_path):
        raise FileNotFoundError(f'Target scaler not found at {target_scaler_path}. Run train.py first.')
    target_scaler = joblib.load(target_scaler_path)
    # Inverse transform expects shape (n_samples, n_features), here (n_samples, 1)
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