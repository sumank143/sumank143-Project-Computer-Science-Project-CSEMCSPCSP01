"""
Spatial Graph Construction
===========================
Builds k-NN graphs from monitoring station coordinates.
Edge weights are inversely proportional to geographic distance.
"""

import numpy as np
import torch
import networkx as nx
from scipy.spatial.distance import cdist
from typing import Tuple, Dict


def build_knn_graph(
    station_coords: Dict[str, Tuple[float, float]],
    k: int = 4,
    epsilon: float = 0.01
) -> Tuple[torch.Tensor, torch.Tensor, nx.Graph]:
    """Build a k-nearest neighbour spatial graph.
    
    Args:
        station_coords: Dict mapping station names to (lat, lon) tuples.
        k: Number of nearest neighbours per node.
        epsilon: Small constant to prevent division by zero in weights.
    
    Returns:
        Tuple of (edge_index, edge_weight, networkx_graph).
        - edge_index: [2, num_edges] tensor of (src, dst) pairs.
        - edge_weight: [num_edges] tensor of inverse-distance weights.
        - G: NetworkX graph for visualization.
    """
    names = list(station_coords.keys())
    coords = np.array(list(station_coords.values()))
    n = len(names)
    
    dist_matrix = cdist(coords, coords)
    
    edges_src, edges_dst, weights = [], [], []
    G = nx.Graph()
    
    for i in range(n):
        G.add_node(i, pos=(coords[i, 1], coords[i, 0]), label=names[i])
        neighbors = np.argsort(dist_matrix[i])[1:k + 1]
        for j in neighbors:
            w = 1.0 / (dist_matrix[i, j] + epsilon)
            edges_src.append(i)
            edges_dst.append(j)
            weights.append(w)
            G.add_edge(i, j, weight=min(w, 50))
    
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    
    return edge_index, edge_weight, G


def create_sequences(
    data: np.ndarray,
    window: int = 6,
    horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding window sequences for temporal input.
    
    Args:
        data: Normalized data array of shape (timesteps, stations).
        window: Lookback window in timesteps.
        horizon: Forecast horizon in timesteps.
    
    Returns:
        Tuple of (X, Y):
        - X: shape (samples, stations, window)
        - Y: shape (samples, stations)
    """
    X, Y = [], []
    for t in range(len(data) - window - horizon):
        X.append(data[t:t + window].T)
        Y.append(data[t + window + horizon - 1])
    return np.array(X), np.array(Y)
