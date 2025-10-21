"""
Main PIDSE class and configuration.

This module contains the core Physics-Informed Differentiable State Estimator
implementation that combines PINN dynamics learning with differentiable EKF.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.pinn import DynamicsNetwork, MeasurementNetwork
from ..filters.differentiable_ekf import DifferentiableEKF
from ..losses.hybrid_loss import HybridLoss
from .state_space import StateSpaceModel


@dataclass
class PIDSEConfig:
    """Configuration for PIDSE model."""
    
    # State space dimensions
    state_dim: int = 12  # [position(3), velocity(3), orientation(3), angular_vel(3)]
    control_dim: int = 4  # control inputs
    measurement_dim: int = 9  # sensor measurements
    
    # Network architecture
    pinn_hidden_layers: List[int] = None
    dynamics_activation: str = "tanh"
    measurement_activation: str = "relu"
    
    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 32
    sequence_length: int = 50  # length of training sequences
    
    # Loss function weights
    physics_weight: float = 0.1  # λ_P
    regularization_weight: float = 0.01  # λ_D
    
    # EKF parameters
    initial_process_noise: float = 1e-3  # initial Q diagonal
    initial_measurement_noise: float = 1e-2  # initial R diagonal
    learn_noise_matrices: bool = True
    
    # Physics constraints
    enforce_energy_conservation: bool = True
    enforce_momentum_conservation: bool = True
    mass: Optional[float] = None  # system mass (if known)
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.pinn_hidden_layers is None:
            self.pinn_hidden_layers = [64, 64, 32]


class PIDSE(nn.Module):
    """
    Physics-Informed Differentiable State Estimator.
    
    Combines:
    1. Physics-Informed Neural Network for dynamics learning
    2. Differentiable Extended Kalman Filter for state estimation
    3. Hybrid loss function for end-to-end training
    """
    
    def __init__(self, config: PIDSEConfig):
        super().__init__()
        self.config = config
        
        # Initialize state space model with known physics
        self.state_space = StateSpaceModel(
            state_dim=config.state_dim,
            control_dim=config.control_dim,
            measurement_dim=config.measurement_dim,
            mass=config.mass
        )
        
        # Initialize PINN for learning residual dynamics
        self.dynamics_network = DynamicsNetwork(
            state_dim=config.state_dim,
            control_dim=config.control_dim,
            hidden_layers=config.pinn_hidden_layers,
            activation=config.dynamics_activation
        )
        
        # Initialize PINN for measurement model (if needed)
        self.measurement_network = MeasurementNetwork(
            state_dim=config.state_dim,
            measurement_dim=config.measurement_dim,
            hidden_layers=config.pinn_hidden_layers[:-1] if len(config.pinn_hidden_layers) > 1 else [32],
            activation=config.measurement_activation
        )
        
        # Initialize differentiable EKF
        self.ekf = DifferentiableEKF(
            state_dim=config.state_dim,
            measurement_dim=config.measurement_dim,
            initial_Q=config.initial_process_noise,
            initial_R=config.initial_measurement_noise,
            learn_noise_matrices=config.learn_noise_matrices
        )
        
        # Initialize hybrid loss function
        self.loss_fn = HybridLoss(
            config=config,
            state_space=self.state_space
        )
        
        # Move to device
        self.to(config.device)
    
    def forward(
        self, 
        states: torch.Tensor,
        controls: torch.Tensor,
        measurements: torch.Tensor,
        initial_state: torch.Tensor,
        initial_covariance: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through PIDSE.
        
        Args:
            states: Ground truth states [batch, seq_len, state_dim]
            controls: Control inputs [batch, seq_len, control_dim]
            measurements: Sensor measurements [batch, seq_len, measurement_dim]
            initial_state: Initial state estimate [batch, state_dim]
            initial_covariance: Initial covariance [batch, state_dim, state_dim]
            
        Returns:
            Dictionary containing:
                - estimated_states: EKF state estimates
                - predicted_measurements: Predicted measurements
                - covariances: State covariances
                - loss_components: Individual loss terms
        """
        batch_size, seq_len = states.shape[:2]
        device = states.device
        
        # Store outputs
        estimated_states = []
        predicted_measurements = []
        covariances = []
        
        # Initialize EKF state
        current_state = initial_state.clone()
        current_cov = initial_covariance.clone()
        
        # Process sequence
        for t in range(seq_len):
            # Get current inputs
            current_control = controls[:, t] if t > 0 else torch.zeros_like(controls[:, 0])
            current_measurement = measurements[:, t]
            
            if t > 0:
                # Prediction step with learned dynamics
                predicted_state, predicted_cov = self.ekf.predict(
                    estimated_states[t-1], 
                    covariances[t-1],
                    lambda x: self._predict_state(x, current_control)
                )
            else:
                predicted_state = current_state
                predicted_cov = current_cov
            
            # Update step with EKF
            estimated_state, updated_cov, pred_measurement = self.ekf.update(
                predicted_state,
                predicted_cov,
                current_measurement,
                lambda x: self._predict_measurement(x)
            )
            
            # Store results
            estimated_states.append(estimated_state)
            predicted_measurements.append(pred_measurement)
            covariances.append(updated_cov)
            
            # Update for next iteration
            current_state = estimated_state
            current_cov = updated_cov
        
        # Stack results
        estimated_states = torch.stack(estimated_states, dim=1)
        predicted_measurements = torch.stack(predicted_measurements, dim=1)
        covariances = torch.stack(covariances, dim=1)
        
        # Compute loss components
        loss_components = self.loss_fn(
            true_states=states,
            estimated_states=estimated_states,
            predicted_measurements=predicted_measurements,
            measurements=measurements,
            controls=controls
        )
        
        return {
            "estimated_states": estimated_states,
            "predicted_measurements": predicted_measurements,
            "covariances": covariances,
            "loss_components": loss_components
        }
    
    def _predict_state(
        self, 
        state: torch.Tensor, 
        control: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict next state using known physics + learned residuals.
        
        f(x, u) = f_known(x, u) + f_PINN(x, u; θ)
        """
        # Known physics prediction
        known_prediction = self.state_space.dynamics_step(state, control)
        
        # Learned residual dynamics
        residual = self.dynamics_network(state, control)
        
        # Combined prediction
        return known_prediction + residual
    
    def _predict_measurement(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict measurements from state.
        
        h(x) = h_known(x) + h_PINN(x; θ)
        """
        # Known measurement model (e.g., IMU model)
        known_measurement = self.state_space.measurement_model(state)
        
        # Learned residual measurement model
        residual = self.measurement_network(state)
        
        # Combined measurement prediction
        return known_measurement + residual
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Single training step."""
        optimizer.zero_grad()
        
        # Move batch to device
        device = next(self.parameters()).device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        outputs = self.forward(
            states=batch["states"],
            controls=batch["controls"],
            measurements=batch["measurements"],
            initial_state=batch["initial_state"],
            initial_covariance=batch["initial_covariance"]
        )
        
        # Compute total loss
        loss_components = outputs["loss_components"]
        total_loss = loss_components["total_loss"]
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Return loss components as scalars
        return {k: v.item() for k, v in loss_components.items()}
    
    def evaluate(
        self,
        data_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate model on validation/test data."""
        self.eval()
        total_losses = {}
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                device = next(self.parameters()).device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.forward(
                    states=batch["states"],
                    controls=batch["controls"],
                    measurements=batch["measurements"],
                    initial_state=batch["initial_state"],
                    initial_covariance=batch["initial_covariance"]
                )
                
                # Accumulate losses
                loss_components = outputs["loss_components"]
                for k, v in loss_components.items():
                    total_losses[k] = total_losses.get(k, 0) + v.item()
        
        # Average losses
        num_batches = len(data_loader)
        return {k: v / num_batches for k, v in total_losses.items()}
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'q_matrix': self.ekf.Q_matrix.data,
            'r_matrix': self.ekf.R_matrix.data,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint