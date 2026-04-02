"""
GraphSAGE-based Air Quality Forecasting Model
===============================================
Combines GRU temporal encoding with GraphSAGE layers for
inductive spatio-temporal air quality prediction.

Reference:
    Hamilton et al., "Inductive Representation Learning on
    Large Graphs", NeurIPS 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SAGE_AQ(nn.Module):
    """GraphSAGE-based air quality forecasting model.
    
    Architecture:
        Input (batch, stations, window)
          → GRU temporal encoder per station
          → SAGEConv layer 1 (with ReLU + Dropout)
          → SAGEConv layer 2
          → Linear prediction head
        Output (batch, stations) — 1-step-ahead forecast
    
    Note:
        GraphSAGE uses mean aggregation by default and does not
        use edge weights. This makes it suitable for inductive
        learning (predicting at unseen stations).
    
    Args:
        in_channels: Input feature dimension per timestep (default: 1).
        hidden_channels: Hidden dimension for GRU and SAGE layers.
        out_channels: Output dimension of second SAGE layer.
        dropout: Dropout rate between SAGE layers.
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
        
        self.temporal = nn.GRU(in_channels, hidden_channels, batch_first=True)
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.fc = nn.Linear(out_channels, 1)
    
    def forward(self, x, edge_index, edge_weight=None):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, n_nodes, window).
            edge_index: Graph connectivity [2, num_edges].
            edge_weight: Ignored (SAGE uses unweighted aggregation).
        
        Returns:
            Predictions of shape (batch, n_nodes).
        """
        batch_size, n_nodes, T = x.shape
        
        x_flat = x.reshape(batch_size * n_nodes, T, 1)
        _, h = self.temporal(x_flat)
        h = h.squeeze(0).reshape(batch_size, n_nodes, -1)
        
        outputs = []
        for b in range(batch_size):
            hb = h[b]
            hb = F.relu(self.conv1(hb, edge_index))
            hb = F.dropout(hb, p=self.dropout, training=self.training)
            hb = self.conv2(hb, edge_index)
            out = self.fc(hb).squeeze(-1)
            outputs.append(out)
        
        return torch.stack(outputs)
