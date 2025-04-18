#!/usr/bin/env python3
"""
train.py

Train an LSTM model to predict 'puissance_cvac_future' (16 steps ahead).
Evaluates on dataset/test_split.csv and saves the trained model, scalers,
and evaluation results.
"""
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import joblib
from tqdm import tqdm

# Import core components from hvac_lstm_prototype
from hvac_lstm_prototype import HVACDataset, LSTMModel, create_sequences

def load_series(csv_path, feature_cols, target_col):
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    features = df[feature_cols].values
    target = df[target_col].values
    return features, target, df

def main():
    parser = argparse.ArgumentParser(description='Train LSTM for single-step future prediction')
    parser.add_argument('--train_csv', default='dataset/train_split.csv', help='Path to training data CSV')
    parser.add_argument('--test_csv', default='dataset/test_split.csv', help='Path to test data CSV')
    parser.add_argument('--output_dir', default='models', help='Directory to save model and scalers')
    parser.add_argument('--results_dir', default='results', help='Directory to save evaluation results')
    parser.add_argument('--model_filename', default='lstm_hvac.pth', help='Filename for saved model state dict')
    parser.add_argument('--feature_scaler_filename', default='feature_scaler.save', help='Filename for saved feature scaler')
    parser.add_argument('--target_scaler_filename', default='target_scaler.save', help='Filename for saved target scaler')
    parser.add_argument('--seq_len', type=int, default=16, help='Sequence length for LSTM input')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and evaluation')
    parser.add_argument('--hidden_size', type=int, default=64, help='Number of hidden units in LSTM')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for Adam optimizer')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define feature and target columns
    feature_cols = [
        'temperature_exterieure', 'humidite', 'ensoleillement',
        'vitesse_vent', 'direction_vent', 'temperature_interieure'
    ]
    target_col = 'puissance_cvac' # The target for create_sequences is based on this

    # Create output directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # --- Data Loading and Preparation ---
    print("Loading data...")
    feat_train, targ_train, df_train = load_series(args.train_csv, feature_cols, target_col)
    feat_test, targ_test, df_test = load_series(args.test_csv, feature_cols, target_col)
    print(f"Loaded {len(df_train)} training samples and {len(df_test)} test samples.")

    # Scale features
    print("Scaling features...")
    feat_scaler = MinMaxScaler()
    feat_train_s = feat_scaler.fit_transform(feat_train)
    feat_test_s = feat_scaler.transform(feat_test)

    # Scale targets
    print("Scaling targets...")
    targ_scaler = MinMaxScaler()
    targ_train_s = targ_scaler.fit_transform(targ_train.reshape(-1,1)).flatten()
    targ_test_s = targ_scaler.transform(targ_test.reshape(-1,1)).flatten()

    # Create sequences (using the updated function)
    print("Creating sequences...")
    X_train, y_train = create_sequences((feat_train_s, targ_train_s), seq_len=args.seq_len)
    X_test, y_test = create_sequences((feat_test_s, targ_test_s), seq_len=args.seq_len)
    print(f"Created {len(X_train)} training sequences (y shape: {y_train.shape}) and {len(X_test)} test sequences (y shape: {y_test.shape}).")

    # DataLoaders (HVACDataset now handles y reshape)
    train_ds = HVACDataset(X_train, y_train)
    test_ds = HVACDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # --- Model Training ---
    print("Initializing model...")
    model = LSTMModel(input_size=len(feature_cols), hidden_size=args.hidden_size, num_layers=args.num_layers)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Starting training...")
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss = 0.0
        for Xb, yb in tqdm(train_loader, desc=f'Train Epoch {epoch}', leave=False):
            Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(Xb) # Shape: (batch, 1)
            loss = criterion(out, yb) # yb shape: (batch, 1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * Xb.size(0)
        avg_train_loss = total_loss / len(train_ds)
        print(f'Epoch {epoch}/{args.num_epochs} - Train MSE: {avg_train_loss:.6f}')
    print("Training finished.")

    # --- Model Evaluation ---
    print("Evaluating model on test set...")
    model.eval()
    preds_scaled_list = []
    trues_scaled_list = []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device, non_blocking=True)
            out_scaled = model(Xb).cpu().numpy() # Shape: (batch, 1)
            preds_scaled_list.append(out_scaled)
            trues_scaled_list.append(yb.numpy()) # Shape: (batch, 1)

    preds_scaled = np.concatenate(preds_scaled_list) # Shape: (n_samples, 1)
    trues_scaled = np.concatenate(trues_scaled_list) # Shape: (n_samples, 1)

    # Inverse transform predictions and true values to original scale
    preds = targ_scaler.inverse_transform(preds_scaled).flatten() # Shape: (n_samples,)
    trues = targ_scaler.inverse_transform(trues_scaled).flatten() # Shape: (n_samples,)

    # Calculate metrics
    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    print(f'Test MSE: {mse:.4f}, MAE: {mae:.4f}')

    # --- Saving Artifacts ---
    # Save model state dictionary
    model_path = os.path.join(args.output_dir, args.model_filename)
    torch.save(model.state_dict(), model_path)
    print(f'Saved model state dict to {model_path}')

    # Save scalers
    feature_scaler_path = os.path.join(args.output_dir, args.feature_scaler_filename)
    target_scaler_path = os.path.join(args.output_dir, args.target_scaler_filename)
    joblib.dump(feat_scaler, feature_scaler_path)
    joblib.dump(targ_scaler, target_scaler_path)
    print(f'Saved feature scaler to {feature_scaler_path}')
    print(f'Saved target scaler to {target_scaler_path}')

    # Save evaluation results (predictions vs true values)
    results_df = pd.DataFrame({'y_true': trues, 'y_pred': preds})
    eval_path = os.path.join(args.results_dir, 'train_test_eval.csv')
    results_df.to_csv(eval_path, index=False)
    print(f'Saved evaluation results to {eval_path}')

if __name__ == '__main__':
    main() 