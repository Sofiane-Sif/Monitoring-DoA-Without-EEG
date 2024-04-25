"""
Trains a MLP for BIS prediction
"""

import numpy as np
import torch
import torch.nn as nn

class MLPBisPredictor(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # input flattened contrary to LSTMs
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            # nn.Dropout(dropout),
            # nn.ReLU(),
            # nn.BatchNorm1d(num_features=hidden_dim),
            # nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)

