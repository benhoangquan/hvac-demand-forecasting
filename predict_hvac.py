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
import xgboost as xgb

from hvac_lstm_prototype import LSTMModel

def main():
    parser = argparse.ArgumentParser(description='Predict future HVAC consumption (single value)')
    parser.add_argument('--test_csv', default=os.path.join('dataset', 'test_features.csv'), help='Path to test features CSV file')
    parser.add_argument('--model_dir', default='models', help='Directory containing the saved model and scalers')
    parser.add_argument('--results_dir', default='results', help='Directory to save prediction results')
    parser.add_argument('--model_filename', default='lstm_hvac_best.pth', help='Filename of the saved model state dict')
    parser.add_argument('--feature_scaler_filename', default='feature_scaler.save', help='Filename of the saved feature scaler')
    parser.add_argument('--target_scaler_filename', default='target_scaler.save', help='Filename of the saved target scaler')
    parser.add_argument('--output_filename', default='submission.csv', help='Filename for the Kaggle submission CSV')
    parser.add_argument('--seq_len', type=int, default=16, help='Sequence length used during training')
    parser.add_argument('--xgb_model_filename', default='xgb_hvac_best.json', help='Filename of the saved XGBoost model')

    args = parser.parse_args()

    # Construct full paths
    model_path = os.path.join(args.model_dir, args.model_filename)
    feature_scaler_path = os.path.join(args.model_dir, args.feature_scaler_filename)
    target_scaler_path = os.path.join(args.model_dir, args.target_scaler_filename)
    output_path = os.path.join(args.results_dir, args.output_filename)
    xgb_model_path = os.path.join(args.model_dir, args.xgb_model_filename)

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
    
    # --- Feature Count Check --- #
    # Check if the number of features in the test data matches what the scaler expects
    num_features_in_test_csv = features.shape[1]
    num_features_expected_by_scaler = feat_scaler.n_features_in_
    
    if num_features_in_test_csv != num_features_expected_by_scaler:
        # Handle the mismatch based on whether time features were likely used
        base_feature_count = 6 # Known base features
        if num_features_expected_by_scaler > base_feature_count and num_features_in_test_csv == base_feature_count:
            # This likely means the model was trained WITH time features, but test CSV lacks them
            raise ValueError(
                f"Feature mismatch: Scaler expects {num_features_expected_by_scaler} features (likely including time features), "
                f"but test CSV provides only {num_features_in_test_csv}. "
                f"Please re-train the model using 'train.py --no_time_features' "
                f"and use the resulting model/scalers for prediction, or provide a test CSV with appropriate date information."
            )
        else:
            # General mismatch
            raise ValueError(
                f"Feature mismatch: Scaler expects {num_features_expected_by_scaler} features, "
                f"but test CSV provides {num_features_in_test_csv}. Check data loading and feature preparation."
            )
            
    print(f"Feature count ({num_features_in_test_csv}) matches scaler expectation.")
    # Scale the features from the test CSV
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
            print(f"Warning: Loaded LSTM model expects output size {output_size}, but should be 1.")

    except Exception as e:
        raise ValueError(f"Could not determine LSTM parameters from state dict: {e}")

    # Check if LSTM input size matches scaled feature size
    if input_size != features_scaled.shape[1]:
        raise ValueError(f"LSTM input size mismatch: Model expects {input_size} features, but data has {features_scaled.shape[1]} after scaling.")

    # Initialize LSTM model and load state
    print(f"Initializing LSTM model (input={input_size}, hidden={hidden_size}, layers={num_layers}, output=1)...")
    lstm_model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    lstm_model.load_state_dict(state)
    lstm_model.to(device)
    lstm_model.eval()

    # Load XGBoost model
    print(f"Loading XGBoost model from {xgb_model_path}...")
    if not os.path.isfile(xgb_model_path):
        raise FileNotFoundError(f'XGBoost model not found at {xgb_model_path}. Run train.py first.')
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(xgb_model_path)
    
    # Verify XGBoost feature count matches expected input
    # Expected features: last_step_features + lstm_prediction (1)
    expected_xgb_features = features_scaled.shape[1] + 1
    # Note: xgb_model.n_features_in_ is available after fitting or loading
    if hasattr(xgb_model, 'n_features_in_') and xgb_model.n_features_in_ != expected_xgb_features:
         raise ValueError(
             f"XGBoost feature mismatch: Model expects {xgb_model.n_features_in_} features, "
             f"but generated data has {expected_xgb_features} (original features + LSTM prediction)."
         )
    else:
        print(f"XGBoost model expects {expected_xgb_features} features.")

    # --- Run Prediction Pipeline ---
    print("Running hybrid prediction pipeline (LSTM -> XGBoost)...")
    
    # 1. Get LSTM Predictions (Scaled)
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_test_seq).float().to(device)
        lstm_preds_scaled = lstm_model(X_tensor).cpu().numpy() # Shape: (num_samples, 1)

    # 2. Prepare XGBoost Input
    # Combine the original scaled features (last time step) with the LSTM's scaled prediction
    # features_scaled shape: (num_samples, num_original_features)
    # lstm_preds_scaled shape: (num_samples, 1)
    xgb_input_features = np.concatenate((features_scaled, lstm_preds_scaled), axis=1)
    print(f"XGBoost input shape: {xgb_input_features.shape}")

    # 3. Get XGBoost Predictions (Scaled)
    final_preds_scaled = xgb_model.predict(xgb_input_features).reshape(-1, 1) # Ensure shape (num_samples, 1)

    # 4. Inverse Transform Final Predictions
    print(f"Loading target scaler from {target_scaler_path}...")
    if not os.path.isfile(target_scaler_path):
        raise FileNotFoundError(f'Target scaler not found at {target_scaler_path}. Run train.py first.')
    target_scaler = joblib.load(target_scaler_path)
    final_preds = target_scaler.inverse_transform(final_preds_scaled).flatten() # Shape: (num_samples,)

    # Format predictions for submission
    print("Formatting predictions for submission...")
    os.makedirs(args.results_dir, exist_ok=True)
    df_out = pd.DataFrame({
        'ID': ids,
        'puissance_cvac_future': final_preds
    })

    df_out.to_csv(output_path, index=False)
    print(f'Submission file saved to {output_path}')

if __name__ == '__main__':
    main() 