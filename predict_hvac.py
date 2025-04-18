#!/usr/bin/env python3
"""
predict_hvac.py

Load a trained LSTM model and feature scaler, predict HVAC consumption on test_features.csv,
and save results to results/test1.csv (ID, prediction).
"""
import os
import pandas as pd
import numpy as np
import torch
import joblib

from hvac_lstm_prototype import LSTMModel

def main():
    # Directories and file paths
    model_dir = 'models'
    test_csv = os.path.join('dataset', 'test_features.csv')
    output_dir = 'results'
    output_csv = os.path.join(output_dir, 'test2.csv')

    # Load test features
    df_test = pd.read_csv(test_csv)
    ids = df_test['ID'].values
    feature_cols = [
        'temperature_exterieure', 'humidite', 'ensoleillement',
        'vitesse_vent', 'direction_vent', 'temperature_interieure'
    ]
    if not set(feature_cols).issubset(df_test.columns):
        raise ValueError(f'Test CSV missing required feature columns: {feature_cols}')
    features = df_test[feature_cols].values

    # Load scaler and scale features
    scaler_path = os.path.join(model_dir, 'feature_scaler.save')
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(f'Feature scaler not found at {scaler_path}. Please train model first.')
    scaler = joblib.load(scaler_path)
    features_scaled = scaler.transform(features)

    # Prepare input for LSTM: use each row as a single time step sequence
    # shape: (num_samples, seq_len=1, num_features)
    X = features_scaled[:, None, :]

    # Load model state
    model_path = os.path.join(model_dir, 'lstm_hvac_3.pth')
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f'Model state not found at {model_path}. Please train model first.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Dynamically infer model dimensions from state dict
    state = torch.load(model_path, map_location=device)
    # Determine number of layers
    ih_keys = [k for k in state.keys() if k.startswith('lstm.weight_ih_l')]
    layers = [int(k.split('_l')[-1].split('_')[0]) for k in ih_keys]
    num_layers = max(layers) + 1 if layers else 1
    # Determine hidden_size and input_size
    wih0 = state.get('lstm.weight_ih_l0')
    if wih0 is None:
        raise KeyError('Unexpected state dict format: missing lstm.weight_ih_l0')
    hidden_size = wih0.shape[0] // 4
    input_size = wih0.shape[1]

    # Initialize model and load state
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Run predictions
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(device)
        preds = model(X_tensor).cpu().numpy()
    # Inverse scale predictions if scaled in [0,1]
    # Try loading a saved target scaler, else fallback to train data min/max
    target_scaler_path = os.path.join(model_dir, 'target_scaler_3.save')
    if os.path.isfile(target_scaler_path):
        ts = joblib.load(target_scaler_path)
        preds = ts.inverse_transform(preds.reshape(-1, 1)).flatten()
    else:
        df_train = pd.read_csv(os.path.join('dataset', 'test_split.csv'))
        y_min = df_train['puissance_cvac'].min()
        y_max = df_train['puissance_cvac'].max()
        preds = preds * (y_max - y_min) + y_min

    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    df_out = pd.DataFrame(preds)
    df_out['ID'] = ids
    df_out = df_out.melt(id_vars='ID', value_name='prediction')
    df_out = df_out.drop(columns='variable')
    df_out.to_csv(output_csv, index=False)
    print(f'Predictions saved to {output_csv}')

if __name__ == '__main__':
    main()