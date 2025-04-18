#!/usr/bin/env python3
"""
hvac_lstm_prototype.py

Defines core components (Dataset, Model, sequence creation) for HVAC LSTM prediction.
"""
import numpy as np
import pandas as pd
# Removed MinMaxScaler as scaling is handled in training script
import torch
import torch.nn as nn
from torch.utils.data import Dataset
# Removed DataLoader, tqdm, os, train_test_split, joblib as they are not needed here

class HVACDataset(Dataset):
    def __init__(self, sequences, targets):
        self.X = sequences.astype(np.float32)
        # Ensure target is float32 and potentially reshape for loss function compatibility
        self.y = targets.astype(np.float32).reshape(-1, 1)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Change output layer to predict a single value
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (hn, cn) = self.lstm(x)
        hn = hn[-1]  # take last layer's hidden state
        return self.fc(hn)

def create_sequences(data, seq_len=16):
    """
    Create input sequences and targets for time series forecasting.
    data: tuple (features, consumption),
          features: np.array shape (N, n_features)
          consumption: np.array shape (N,)
    Returns:
      X: np.array shape (num_samples, seq_len, n_features)
      y: np.array shape (num_samples, 1) - Target is consumption at step t + seq_len + 15
    Target is a single consumption value 16 time steps after the *end* of the input sequence.
    """
    features, consumption = data
    X, y = [], []
    N = len(consumption)
    # Adjust loop range to ensure target index i + seq_len + 15 is valid
    for i in range(N - seq_len - 15):
        X.append(features[i:i+seq_len])
        # Target is the single value at step i + seq_len + 15
        y.append(consumption[i+seq_len+15])
    # Return y as a column vector
    return np.array(X), np.array(y)