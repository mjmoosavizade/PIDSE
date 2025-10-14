"""
Physics-Informed Differentiable State Estimator (PIDSE)

A framework for learning dynamics and state estimation that combines:
- Physics-Informed Neural Networks (PINNs)
- Differentiable Extended Kalman Filters
- Hybrid loss functions for physical consistency
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core.pidse import PIDSE, PIDSEConfig
from .core.state_space import StateSpaceModel
from .models.pinn import DynamicsNetwork, MeasurementNetwork
from .filters.differentiable_ekf import DifferentiableEKF
from .losses.hybrid_loss import HybridLoss

__all__ = [
    "PIDSE",
    "PIDSEConfig",
    "StateSpaceModel", 
    "DynamicsNetwork",
    "MeasurementNetwork",
    "DifferentiableEKF",
    "HybridLoss"
]