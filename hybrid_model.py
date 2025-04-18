import torch
import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self, lstm_model, tcn_model):
        super(HybridModel, self).__init__()
        self.lstm_model = lstm_model
        self.tcn_model = tcn_model
        
        # Learnable weight for combining the outputs
        # Initialized to 0.5, after sigmoid it will be ~0.62
        # We use a single scalar weight parameter
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # Pass input through LSTM
        lstm_out = self.lstm_model(x)
        
        # Pass input through TCN
        tcn_out = self.tcn_model(x)
        
        # Calculate the weight using sigmoid to keep it between 0 and 1
        weight = torch.sigmoid(self.alpha)
        
        # Combine outputs using the weighted average
        combined_out = weight * lstm_out + (1 - weight) * tcn_out
        
        return combined_out 