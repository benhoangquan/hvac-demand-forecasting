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
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Apply dropout between LSTM layers only if num_layers > 1
        lstm_dropout = dropout if num_layers > 1 else 0.0
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=lstm_dropout
        )
        
        # Batch normalization after LSTM output (applied to hidden_size features)
        self.bn = nn.BatchNorm1d(hidden_size)
        
        # Dropout layer before final FC layer
        self.dropout = nn.Dropout(dropout)
        
        # Final fully connected layer
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (hn, cn) = self.lstm(x)
        
        # Use the output of the last time step
        # out shape: (batch, seq_len, hidden_size)
        # We want the features from the last time step: (batch, hidden_size)
        last_time_step_out = out[:, -1, :]
        
        # Apply batch normalization
        # BatchNorm1d expects input of shape (N, C) or (N, C, L)
        # Here N=batch_size, C=hidden_size
        bn_out = self.bn(last_time_step_out)
        
        # Apply dropout
        dropout_out = self.dropout(bn_out)
        
        # Final prediction
        final_out = self.fc(dropout_out)
        
        return final_out

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