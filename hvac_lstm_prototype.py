#!/usr/bin/env python3
"""
hvac_lstm_prototype.py

A simple LSTM-based prototype for predicting HVAC consumption 4 hours (16 time steps) ahead.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class HVACDataset(Dataset):
    def __init__(self, sequences, targets):
        self.X = sequences.astype(np.float32)
        self.y = targets.astype(np.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (hn, cn) = self.lstm(x)
        hn = hn[-1]  # take last layer's hidden state
        return self.fc(hn).squeeze(1)

def create_sequences(data, seq_len=16):
    """
    Create input sequences and targets for time series forecasting.
    data: tuple (features, consumption),
          features: np.array shape (N, n_features)
          consumption: np.array shape (N,)
    Returns:
      X: np.array shape (num_samples, seq_len, n_features)
      y: np.array shape (num_samples,)
    Target is consumption at time index i+seq_len.
    """
    features, consumption = data
    X, y = [], []
    N = len(consumption)
    for i in range(N - seq_len):
        X.append(features[i:i+seq_len])
        y.append(consumption[i+seq_len])
    return np.array(X), np.array(y)

def main():
    # Configuration
    seq_len = 16
    batch_size = 64
    hidden_size = 64
    num_layers = 1
    num_epochs = 10
    learning_rate = 1e-3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and prepare data
    df = pd.read_csv('dataset/train_dataset.csv', parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    feature_cols = [
        'temperature_exterieure', 'humidite', 'ensoleillement',
        'vitesse_vent', 'direction_vent', 'temperature_interieure'
    ]
    target_col = 'puissance_cvac'

    features = df[feature_cols].values
    consumption = df[target_col].values

    # Scale features to [0,1]
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Create sequences for modeling
    X, y = create_sequences((features_scaled, consumption), seq_len=seq_len)

    # Train/validation split (no shuffle for time series)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Create PyTorch datasets and loaders
    train_dataset = HVACDataset(X_train, y_train)
    val_dataset = HVACDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, optimizer
    model = LSTMModel(input_size=len(feature_cols), hidden_size=hidden_size, num_layers=num_layers)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in tqdm(train_loader, desc=f'Train Epoch {epoch}'):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')

    # Save trained model and scaler
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/lstm_hvac.pth')
    # Save scaler for inference
    import joblib
    joblib.dump(scaler, 'models/feature_scaler.save')
    print('Training complete. Model and scaler saved in models/')

if __name__ == '__main__':
    main()