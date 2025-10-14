"""
Hybrid loss function for PIDSE training.

This module implements the composite loss function that combines:
1. Estimation loss - accuracy of state estimation
2. Physics loss - adherence to physical laws  
3. Regularization loss - model stability and generalization

L_total = L_estimation + λ_P * L_physics + λ_D * L_regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math

from ..core.state_space import StateSpaceModel


class HybridLoss(nn.Module):
    """
    Hybrid loss function for PIDSE training.
    
    Combines multiple loss components to ensure both performance
    and physical consistency of the learned model.
    """
    
    def __init__(
        self,
        config,  # PIDSEConfig
        state_space: StateSpaceModel,
        estimation_loss_type: str = "mse",
        physics_loss_type: str = "conservation",
        adaptive_weights: bool = False
    ):
        super().__init__()
        self.config = config
        self.state_space = state_space
        self.estimation_loss_type = estimation_loss_type
        self.physics_loss_type = physics_loss_type
        self.adaptive_weights = adaptive_weights
        
        # Loss weights
        if adaptive_weights:
            # Learnable loss weights with proper initialization
            self.log_physics_weight = nn.Parameter(torch.log(torch.tensor(config.physics_weight)))
            self.log_reg_weight = nn.Parameter(torch.log(torch.tensor(config.regularization_weight)))
        else:
            # Fixed weights
            self.physics_weight = config.physics_weight
            self.regularization_weight = config.regularization_weight
        
        # Physics constraint implementations
        self.physics_constraints = PhysicsConstraints(
            state_space=state_space,
            constraint_type=physics_loss_type
        )
        
        # Regularization implementations
        self.regularizers = RegularizationTerms(config)
        
        # Loss history for adaptive weighting
        self.register_buffer('loss_history', torch.zeros(3, 100))  # [estimation, physics, reg] x history
        self.register_buffer('history_idx', torch.tensor(0))
    
    @property
    def current_physics_weight(self) -> float:
        """Get current physics loss weight."""
        if self.adaptive_weights:
            return torch.exp(self.log_physics_weight).item()
        else:
            return self.physics_weight
    
    @property
    def current_regularization_weight(self) -> float:
        """Get current regularization loss weight."""
        if self.adaptive_weights:
            return torch.exp(self.log_reg_weight).item()
        else:
            return self.regularization_weight
    
    def forward(
        self,
        true_states: torch.Tensor,
        estimated_states: torch.Tensor,
        predicted_measurements: torch.Tensor,
        measurements: torch.Tensor,
        controls: torch.Tensor,
        model_parameters: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hybrid loss function.
        
        Args:
            true_states: Ground truth states [batch, seq_len, state_dim]
            estimated_states: EKF estimated states [batch, seq_len, state_dim]
            predicted_measurements: Predicted measurements [batch, seq_len, measurement_dim]
            measurements: Actual measurements [batch, seq_len, measurement_dim]
            controls: Control inputs [batch, seq_len, control_dim]
            model_parameters: Optional model parameters for regularization
            
        Returns:
            Dictionary containing individual loss components and total loss
        """
        # 1. Estimation Loss
        estimation_loss = self._compute_estimation_loss(
            true_states, estimated_states, predicted_measurements, measurements
        )
        
        # 2. Physics Loss
        physics_loss = self._compute_physics_loss(
            true_states, estimated_states, controls
        )
        
        # 3. Regularization Loss
        regularization_loss = self._compute_regularization_loss(model_parameters)
        
        # Update loss history for adaptive weighting
        self._update_loss_history(estimation_loss, physics_loss, regularization_loss)
        
        # Adaptive weight adjustment
        if self.adaptive_weights:
            physics_weight, reg_weight = self._compute_adaptive_weights()
        else:
            physics_weight = self.physics_weight
            reg_weight = self.regularization_weight
        
        # Total loss
        total_loss = (
            estimation_loss + 
            physics_weight * physics_loss + 
            reg_weight * regularization_loss
        )
        
        return {
            'total_loss': total_loss,
            'estimation_loss': estimation_loss,
            'physics_loss': physics_loss,
            'regularization_loss': regularization_loss,
            'physics_weight': torch.tensor(physics_weight),
            'regularization_weight': torch.tensor(reg_weight)
        }
    
    def _compute_estimation_loss(
        self,
        true_states: torch.Tensor,
        estimated_states: torch.Tensor,
        predicted_measurements: torch.Tensor,
        measurements: torch.Tensor
    ) -> torch.Tensor:
        """Compute estimation accuracy loss."""
        if self.estimation_loss_type == "mse":
            # Mean Squared Error on states
            state_loss = F.mse_loss(estimated_states, true_states)
            
            # MSE on measurements (observation likelihood)
            measurement_loss = F.mse_loss(predicted_measurements, measurements)
            
            # Weighted combination
            estimation_loss = state_loss + 0.1 * measurement_loss
            
        elif self.estimation_loss_type == "huber":
            # Huber loss for robustness to outliers
            state_loss = F.huber_loss(estimated_states, true_states, delta=1.0)
            measurement_loss = F.huber_loss(predicted_measurements, measurements, delta=0.5)
            estimation_loss = state_loss + 0.1 * measurement_loss
            
        elif self.estimation_loss_type == "weighted_mse":
            # Weighted MSE based on state component importance
            weights = torch.tensor([1.0, 1.0, 1.0,  # position (high importance)
                                  0.5, 0.5, 0.5,  # velocity
                                  0.3, 0.3, 0.3,  # orientation
                                  0.2, 0.2, 0.2], # angular velocity
                                 device=true_states.device)
            
            if true_states.shape[-1] < len(weights):
                weights = weights[:true_states.shape[-1]]
            
            state_errors = (estimated_states - true_states) ** 2
            weighted_errors = state_errors * weights.unsqueeze(0).unsqueeze(0)
            state_loss = weighted_errors.mean()
            
            measurement_loss = F.mse_loss(predicted_measurements, measurements)
            estimation_loss = state_loss + 0.1 * measurement_loss
            
        else:
            raise ValueError(f"Unknown estimation loss type: {self.estimation_loss_type}")
        
        return estimation_loss
    
    def _compute_physics_loss(
        self,
        true_states: torch.Tensor,
        estimated_states: torch.Tensor,
        controls: torch.Tensor
    ) -> torch.Tensor:
        """Compute physics consistency loss."""
        return self.physics_constraints(true_states, estimated_states, controls)
    
    def _compute_regularization_loss(
        self,
        model_parameters: Optional[Dict] = None
    ) -> torch.Tensor:
        """Compute regularization loss."""
        return self.regularizers(model_parameters)
    
    def _update_loss_history(
        self,
        estimation_loss: torch.Tensor,
        physics_loss: torch.Tensor,
        regularization_loss: torch.Tensor
    ):
        """Update loss history for adaptive weighting."""
        idx = self.history_idx % self.loss_history.shape[1]
        self.loss_history[0, idx] = estimation_loss.detach()
        self.loss_history[1, idx] = physics_loss.detach()
        self.loss_history[2, idx] = regularization_loss.detach()
        self.history_idx += 1
    
    def _compute_adaptive_weights(self) -> Tuple[float, float]:
        """Compute adaptive loss weights based on loss history."""
        if self.history_idx < 10:  # Not enough history
            return self.current_physics_weight, self.current_regularization_weight
        
        # Get recent loss statistics
        recent_window = min(50, self.history_idx.item())
        recent_losses = self.loss_history[:, :recent_window]
        
        # Compute relative loss magnitudes
        loss_means = torch.mean(recent_losses, dim=1)
        loss_stds = torch.std(recent_losses, dim=1)
        
        # Adaptive weighting: balance loss magnitudes
        estimation_magnitude = loss_means[0] + 1e-8
        physics_magnitude = loss_means[1] + 1e-8
        reg_magnitude = loss_means[2] + 1e-8
        
        # Scale weights to balance contributions
        physics_weight = self.current_physics_weight * (estimation_magnitude / physics_magnitude).clamp(0.1, 10.0)
        reg_weight = self.current_regularization_weight * (estimation_magnitude / reg_magnitude).clamp(0.1, 10.0)
        
        return physics_weight.item(), reg_weight.item()


class PhysicsConstraints(nn.Module):
    """
    Physics constraint implementations for the physics loss term.
    """
    
    def __init__(
        self,
        state_space: StateSpaceModel,
        constraint_type: str = "conservation"
    ):
        super().__init__()
        self.state_space = state_space
        self.constraint_type = constraint_type
    
    def forward(
        self,
        true_states: torch.Tensor,
        estimated_states: torch.Tensor,
        controls: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute physics constraint violations.
        
        Args:
            true_states: Ground truth states [batch, seq_len, state_dim]
            estimated_states: Estimated states [batch, seq_len, state_dim]
            controls: Control inputs [batch, seq_len, control_dim]
            
        Returns:
            physics_loss: Physics constraint violation penalty
        """
        if self.constraint_type == "conservation":
            return self._energy_momentum_conservation(estimated_states, controls)
        elif self.constraint_type == "dynamics_consistency":
            return self._dynamics_consistency(estimated_states, controls)
        elif self.constraint_type == "smoothness":
            return self._trajectory_smoothness(estimated_states)
        else:
            return torch.tensor(0.0, device=estimated_states.device)
    
    def _energy_momentum_conservation(
        self,
        states: torch.Tensor,
        controls: torch.Tensor
    ) -> torch.Tensor:
        """Enforce energy and momentum conservation laws."""
        batch_size, seq_len = states.shape[:2]
        device = states.device
        
        if seq_len < 2:
            return torch.tensor(0.0, device=device)
        
        # Extract state components
        positions = states[:, :, 0:3]
        velocities = states[:, :, 3:6]
        
        violations = []
        
        # Energy conservation check
        if self.state_space.mass is not None:
            mass = self.state_space.mass
            
            # Kinetic energy
            kinetic_energy = 0.5 * mass * torch.sum(velocities**2, dim=-1)
            
            # Potential energy (gravitational)
            gravity_mag = torch.norm(self.state_space.gravity)
            potential_energy = mass * gravity_mag * positions[:, :, 2]  # height component
            
            # Total energy
            total_energy = kinetic_energy + potential_energy
            
            # Energy change between timesteps
            energy_change = total_energy[:, 1:] - total_energy[:, :-1]
            
            # Work done by controls (simplified)
            dt = self.state_space.dt
            forces = controls[:, :-1, 0:3]  # assume first 3 controls are forces
            displacements = positions[:, 1:] - positions[:, :-1]
            work_done = torch.sum(forces * displacements, dim=-1) * dt
            
            # Energy balance violation
            energy_violation = torch.abs(energy_change - work_done)
            violations.append(energy_violation.mean())
        
        # Momentum conservation (in absence of external forces)
        if len(controls.shape) > 2 and controls.shape[-1] >= 3:
            momentum = velocities  # assuming unit mass
            momentum_change = momentum[:, 1:] - momentum[:, :-1]
            
            # External forces should equal momentum change
            dt = self.state_space.dt
            forces = controls[:, :-1, 0:3]
            expected_momentum_change = forces * dt
            
            momentum_violation = F.mse_loss(momentum_change, expected_momentum_change)
            violations.append(momentum_violation)
        
        if violations:
            return torch.stack(violations).mean()
        else:
            return torch.tensor(0.0, device=device)
    
    def _dynamics_consistency(
        self,
        states: torch.Tensor,
        controls: torch.Tensor
    ) -> torch.Tensor:
        """Enforce consistency with known dynamics equations."""
        batch_size, seq_len = states.shape[:2]
        device = states.device
        
        if seq_len < 2:
            return torch.tensor(0.0, device=device)
        
        violations = []
        
        # Check kinematic consistency
        positions = states[:, :, 0:3]
        velocities = states[:, :, 3:6]
        
        # Position should be integral of velocity
        dt = self.state_space.dt
        predicted_positions = positions[:, :-1] + velocities[:, :-1] * dt
        actual_positions = positions[:, 1:]
        
        position_violation = F.mse_loss(predicted_positions, actual_positions)
        violations.append(position_violation)
        
        # Check orientation consistency (if available)
        if states.shape[-1] > 9:
            orientations = states[:, :, 6:9]
            angular_velocities = states[:, :, 9:12]
            
            # Orientation should be integral of angular velocity (simplified)
            predicted_orientations = orientations[:, :-1] + angular_velocities[:, :-1] * dt
            actual_orientations = orientations[:, 1:]
            
            orientation_violation = F.mse_loss(predicted_orientations, actual_orientations)
            violations.append(orientation_violation)
        
        return torch.stack(violations).mean()
    
    def _trajectory_smoothness(self, states: torch.Tensor) -> torch.Tensor:
        """Enforce smooth trajectories (penalize high accelerations)."""
        if states.shape[1] < 3:
            return torch.tensor(0.0, device=states.device)
        
        # Compute second derivatives (acceleration)
        velocities = states[:, :, 3:6]  # assume velocity is in states
        accelerations = velocities[:, 2:] - 2 * velocities[:, 1:-1] + velocities[:, :-2]
        
        # Penalize large accelerations
        smoothness_penalty = torch.mean(accelerations**2)
        
        return smoothness_penalty


class RegularizationTerms(nn.Module):
    """
    Regularization terms for model stability and generalization.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.l1_weight = getattr(config, 'l1_regularization', 0.0)
        self.l2_weight = getattr(config, 'l2_regularization', 1e-4)
        self.noise_regularization = getattr(config, 'noise_regularization', True)
    
    def forward(self, model_parameters: Optional[Dict] = None) -> torch.Tensor:
        """Compute regularization penalty."""
        regularization_terms = []
        
        # L1/L2 weight regularization
        if model_parameters is not None:
            if 'pinn_params' in model_parameters and self.l1_weight > 0:
                l1_penalty = sum(torch.sum(torch.abs(p)) for p in model_parameters['pinn_params'])
                regularization_terms.append(self.l1_weight * l1_penalty)
            
            if 'pinn_params' in model_parameters and self.l2_weight > 0:
                l2_penalty = sum(torch.sum(p**2) for p in model_parameters['pinn_params'])
                regularization_terms.append(self.l2_weight * l2_penalty)
        
        # Noise matrix regularization (ensure reasonable values)
        if model_parameters is not None and self.noise_regularization:
            if 'Q_matrix' in model_parameters:
                Q = model_parameters['Q_matrix']
                # Penalize extreme eigenvalues
                eigenvals = torch.linalg.eigvals(Q).real
                eigenval_penalty = torch.sum(torch.clamp(eigenvals - 1.0, min=0)**2)
                eigenval_penalty += torch.sum(torch.clamp(1e-6 - eigenvals, min=0)**2)
                regularization_terms.append(0.1 * eigenval_penalty)
            
            if 'R_matrix' in model_parameters:
                R = model_parameters['R_matrix']
                eigenvals = torch.linalg.eigvals(R).real
                eigenval_penalty = torch.sum(torch.clamp(eigenvals - 1.0, min=0)**2)
                eigenval_penalty += torch.sum(torch.clamp(1e-6 - eigenvals, min=0)**2)
                regularization_terms.append(0.1 * eigenval_penalty)
        
        # Residual magnitude regularization (encourage small residuals)
        if model_parameters is not None and 'residual_magnitudes' in model_parameters:
            residual_magnitudes = model_parameters['residual_magnitudes']
            residual_penalty = torch.mean(residual_magnitudes**2)
            regularization_terms.append(0.01 * residual_penalty)
        
        if regularization_terms:
            return torch.stack(regularization_terms).sum()
        else:
            return torch.tensor(0.0)