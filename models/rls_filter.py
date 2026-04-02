"""
Recursive Least Squares (RLS) Filter for Data Assimilation
============================================================
Adaptively combines sensor observations with numerical model
predictions (e.g., SILAM) to produce improved estimates,
especially when sensor data is missing.

Reference:
    Haykin, S. "Adaptive Filter Theory", 4th ed., Prentice Hall, 2002.
"""

import numpy as np
from typing import Tuple


class RLSFilter:
    """Recursive Least Squares adaptive filter.
    
    Combines two input sources (observation + model prediction) by
    learning optimal weights that minimize squared estimation error.
    The forgetting factor controls how quickly the filter adapts
    to changing conditions.
    
    Args:
        n_features: Number of input features (default: 2 for obs + model).
        forgetting_factor: λ ∈ (0, 1]. Smaller = faster adaptation.
            Typical range: 0.95–0.99.
        init_cov: Initial diagonal value for inverse correlation matrix.
    """
    
    def __init__(
        self,
        n_features: int = 2,
        forgetting_factor: float = 0.98,
        init_cov: float = 100.0
    ):
        self.lam = forgetting_factor
        self.n = n_features
        self.w = np.zeros(n_features)
        self.P = np.eye(n_features) * init_cov
    
    def update(self, x: np.ndarray, y: float) -> float:
        """Update filter weights and return filtered estimate.
        
        Args:
            x: Input vector [observation, model_prediction].
            y: True value (ground truth or best available).
        
        Returns:
            Filtered estimate (weighted combination of inputs).
        """
        # Gain vector
        Px = self.P @ x
        K = Px / (self.lam + x @ Px)
        
        # Prediction error
        e = y - self.w @ x
        
        # Update weights
        self.w = self.w + K * e
        
        # Update inverse correlation matrix
        self.P = (self.P - np.outer(K, x @ self.P)) / self.lam
        
        return self.w @ x
    
    def predict(self, x: np.ndarray) -> float:
        """Predict without updating weights."""
        return self.w @ x
    
    def get_weights(self) -> np.ndarray:
        """Return current filter weights."""
        return self.w.copy()
    
    def reset(self, init_cov: float = 100.0):
        """Reset filter to initial state."""
        self.w = np.zeros(self.n)
        self.P = np.eye(self.n) * init_cov


def assimilate_timeseries(
    observations: np.ndarray,
    model_predictions: np.ndarray,
    ground_truth: np.ndarray = None,
    forgetting_factor: float = 0.98
) -> Tuple[np.ndarray, np.ndarray]:
    """Run RLS assimilation on a full time series.
    
    When observations are NaN (missing), the filter uses only
    the model prediction. Otherwise it fuses both sources.
    
    Args:
        observations: Sensor observations with NaN for missing.
        model_predictions: Numerical model outputs (always available).
        ground_truth: Optional ground truth for weight updates.
            If None, uses observations where available.
        forgetting_factor: RLS forgetting factor.
    
    Returns:
        Tuple of (estimates, weights_history).
    """
    n = len(observations)
    rls = RLSFilter(n_features=2, forgetting_factor=forgetting_factor)
    
    estimates = np.zeros(n)
    weights_history = np.zeros((n, 2))
    
    for t in range(n):
        if np.isnan(observations[t]):
            # Missing observation: rely on model only
            x = np.array([0.0, model_predictions[t]])
            estimates[t] = rls.predict(x)
        else:
            x = np.array([observations[t], model_predictions[t]])
            target = ground_truth[t] if ground_truth is not None else observations[t]
            estimates[t] = rls.update(x, target)
        
        weights_history[t] = rls.get_weights()
    
    return estimates, weights_history
