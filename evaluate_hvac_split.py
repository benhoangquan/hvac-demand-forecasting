#!/usr/bin/env python3
"""
evaluate_hvac_split.py

Train an LSTM model on dataset/train_split.csv and evaluate on dataset/test_split.csv.
Prints MSE and MAE, and saves per-sample true vs. predicted values.
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

from hvac_lstm_prototype import HVACDataset, LSTMModel, create_sequences

def load_series(csv_path, feature_cols, target_col):
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    features = df[feature_cols].values
    target = df[target_col].values
    return features, target, df

def main():
    parser = argparse.ArgumentParser(description='Train/evaluate LSTM on HVAC splits')
    parser.add_argument('--train_csv', default='dataset/train_split.csv')
    parser.add_argument('--test_csv', default='dataset/test_split.csv')
    parser.add_argument('--seq_len', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feature_cols = [
        'temperature_exterieure', 'humidite', 'ensoleillement',
        'vitesse_vent', 'direction_vent', 'temperature_interieure'
    ]
    target_col = 'puissance_cvac'

    # Load train/test series
    feat_train, targ_train, df_train = load_series(args.train_csv, feature_cols, target_col)
    feat_test, targ_test, df_test = load_series(args.test_csv, feature_cols, target_col)

    # Scale features
    feat_scaler = MinMaxScaler()
    feat_train_s = feat_scaler.fit_transform(feat_train)
    feat_test_s = feat_scaler.transform(feat_test)

    # Scale targets
    targ_scaler = MinMaxScaler()
    targ_train_s = targ_scaler.fit_transform(targ_train.reshape(-1,1)).flatten()
    targ_test_s = targ_scaler.transform(targ_test.reshape(-1,1)).flatten()

    # Create sequences
    X_train, y_train = create_sequences((feat_train_s, targ_train_s), seq_len=args.seq_len)
    X_test, y_test = create_sequences((feat_test_s, targ_test_s), seq_len=args.seq_len)

    # DataLoaders
    train_ds = HVACDataset(X_train, y_train)
    test_ds = HVACDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Model, loss, optimizer
    model = LSTMModel(input_size=len(feature_cols), hidden_size=args.hidden_size, num_layers=args.num_layers)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * Xb.size(0)
        avg_train = total_loss / len(train_ds)
        print(f'Epoch {epoch}/{args.num_epochs} - Train MSE: {avg_train:.4f}')

    # Evaluation
    model.eval()
    preds_scaled = []
    trues_scaled = []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            out = model(Xb).cpu().numpy()
            preds_scaled.append(out)
            trues_scaled.append(yb.numpy())
    preds_scaled = np.concatenate(preds_scaled)
    trues_scaled = np.concatenate(trues_scaled)

    # Inverse scale
    preds = targ_scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()
    trues = targ_scaler.inverse_transform(trues_scaled.reshape(-1,1)).flatten()

    # Metrics
    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    print(f'Test MSE: {mse:.4f}, MAE: {mae:.4f}')

    # Save results
    os.makedirs('results', exist_ok=True)
    df_out = pd.DataFrame({'y_true': trues, 'y_pred': preds})
    out_path = os.path.join('results', 'train_test_eval.csv')
    df_out.to_csv(out_path, index=False)
    print(f'Saved evaluation results to {out_path}')

if __name__ == '__main__':
    main()