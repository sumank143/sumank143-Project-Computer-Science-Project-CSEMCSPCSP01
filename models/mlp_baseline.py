"""
MLP Baseline Model
===================
Simple fully-connected baseline that ignores graph structure.
Used for comparison to quantify the benefit of spatial modelling.
"""

import torch
import torch.nn as nn


class MLP_Baseline(nn.Module):
    """MLP baseline — no spatial graph structure.
    
    Flattens (stations × window) into a single vector and
    predicts all station values simultaneously.
    
    Args:
        window: Lookback window size (timesteps).
        n_nodes: Number of monitoring stations.
        hidden: Hidden layer dimension.
    """
    
    def __init__(self, window: int = 6, n_nodes: int = 10, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_nodes * window, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_nodes),
        )
    
    def forward(self, x, edge_index=None, edge_weight=None):
        """Forward pass. edge_index and edge_weight are ignored."""
        return self.net(x)
