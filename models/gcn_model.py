"""
GCN-based Air Quality Forecasting Model
=========================================
Combines GRU temporal encoding with Graph Convolutional Network
layers for spatio-temporal air quality prediction.

Reference:
    Kipf & Welling, "Semi-Supervised Classification with Graph
    Convolutional Networks", ICLR 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN_AQ(nn.Module):
    """GCN-based air quality forecasting model.
    
    Architecture:
        Input (batch, stations, window)
          → GRU temporal encoder per station
          → GCNConv layer 1 (with ReLU + Dropout)
          → GCNConv layer 2
          → Linear prediction head
        Output (batch, stations) — 1-step-ahead forecast
    
    Args:
        in_channels: Input feature dimension per timestep (default: 1).
        hidden_channels: Hidden dimension for GRU and GCN layers.
        out_channels: Output dimension of second GCN layer.
        dropout: Dropout rate between GCN layers.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        out_channels: int = 16,
        dropout: float = 0.2
    ):
        super().__init__()
        self.dropout = dropout
        
        # Temporal encoder
        self.temporal = nn.GRU(in_channels, hidden_channels, batch_first=True)
        
        # Spatial graph convolutions
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
        # Prediction head
        self.fc = nn.Linear(out_channels, 1)
    
    def forward(self, x, edge_index, edge_weight=None):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, n_nodes, window).
            edge_index: Graph connectivity [2, num_edges].
            edge_weight: Optional edge weights [num_edges].
        
        Returns:
            Predictions of shape (batch, n_nodes).
        """
        batch_size, n_nodes, T = x.shape
        
        # Step 1: Temporal encoding — GRU processes each station's window
        x_flat = x.reshape(batch_size * n_nodes, T, 1)
        _, h = self.temporal(x_flat)
        h = h.squeeze(0).reshape(batch_size, n_nodes, -1)
        
        # Step 2: Spatial aggregation — GCN layers per sample
        outputs = []
        for b in range(batch_size):
            hb = h[b]  # (n_nodes, hidden_channels)
            hb = F.relu(self.conv1(hb, edge_index, edge_weight))
            hb = F.dropout(hb, p=self.dropout, training=self.training)
            hb = self.conv2(hb, edge_index, edge_weight)
            out = self.fc(hb).squeeze(-1)  # (n_nodes,)
            outputs.append(out)
        
        return torch.stack(outputs)  # (batch, n_nodes)
