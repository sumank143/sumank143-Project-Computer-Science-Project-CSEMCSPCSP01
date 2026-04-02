"""
UAFN Model Evaluation
======================
Computes regression metrics on test set predictions.
"""

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict


def evaluate_model(
    model,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    scaler=None
) -> Dict[str, float]:
    """Evaluate a trained model on the test set.
    
    Args:
        model: Trained PyTorch model.
        X_test: Test input tensor (samples, stations, window).
        Y_test: Test target tensor (samples, stations).
        edge_index: Graph edge index.
        edge_weight: Graph edge weights.
        scaler: Optional StandardScaler for inverse-transforming predictions
                back to original µg/m³ scale.
    
    Returns:
        Dictionary with RMSE, MAE, and R² metrics.
    """
    model.eval()
    with torch.no_grad():
        pred = model(X_test, edge_index, edge_weight).cpu().numpy()
    actual = Y_test.cpu().numpy()
    
    if scaler is not None:
        pred = scaler.inverse_transform(pred)
        actual = scaler.inverse_transform(actual)
    
    rmse = np.sqrt(mean_squared_error(actual.flatten(), pred.flatten()))
    mae = mean_absolute_error(actual.flatten(), pred.flatten())
    r2 = r2_score(actual.flatten(), pred.flatten())
    
    return {"RMSE": rmse, "MAE": mae, "R2": r2}
