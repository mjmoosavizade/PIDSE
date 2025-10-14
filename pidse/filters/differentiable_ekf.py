"""
Differentiable Extended Kalman Filter implementation.

This module implements an Extended Kalman Filter that is fully differentiable,
allowing for end-to-end training of the entire PIDSE system including the
learnable noise covariance matrices Q and R.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple, Optional
import math


class LearnableCovariance(nn.Module):
    """
    Learnable covariance matrix that maintains positive definiteness.
    
    Uses Cholesky decomposition parameterization to ensure the matrix
    remains positive definite during training.
    """
    
    def __init__(
        self,
        size: int,
        initial_value: float = 1e-3,
        min_eigenvalue: float = 1e-6,
        diagonal_init: bool = True
    ):
        super().__init__()
        self.size = size
        self.min_eigenvalue = min_eigenvalue
        
        if diagonal_init:
            # Initialize as diagonal matrix
            self.log_diagonal = nn.Parameter(
                torch.log(torch.full((size,), initial_value))
            )
            self.off_diagonal = nn.Parameter(
                torch.zeros(size * (size - 1) // 2)
            )
        else:
            # Full Cholesky factor initialization
            L_init = torch.eye(size) * math.sqrt(initial_value)
            self.cholesky_factor = nn.Parameter(L_init)
    
    def forward(self) -> torch.Tensor:
        """
        Compute positive definite covariance matrix.
        
        Returns:
            Covariance matrix [size, size]
        """
        if hasattr(self, 'log_diagonal'):
            # Diagonal + off-diagonal parameterization
            diagonal = torch.exp(self.log_diagonal).clamp(min=self.min_eigenvalue)
            
            # Construct lower triangular matrix
            L = torch.diag(diagonal)
            if self.off_diagonal.numel() > 0:
                tril_indices = torch.tril_indices(self.size, self.size, offset=-1)
                L[tril_indices[0], tril_indices[1]] = self.off_diagonal
            
            # Covariance = L @ L^T
            covariance = L @ L.T
        else:
            # Full Cholesky parameterization
            L = torch.tril(self.cholesky_factor)
            # Ensure positive diagonal
            diag_indices = torch.arange(self.size)
            L[diag_indices, diag_indices] = F.softplus(L[diag_indices, diag_indices]) + self.min_eigenvalue
            covariance = L @ L.T
        
        return covariance
    
    def get_eigenvalues(self) -> torch.Tensor:
        """Get eigenvalues of the covariance matrix."""
        cov = self.forward()
        eigenvalues = torch.linalg.eigvals(cov).real
        return eigenvalues


class DifferentiableEKF(nn.Module):
    """
    Differentiable Extended Kalman Filter.
    
    Implements the standard EKF equations with automatic differentiation
    support and learnable noise covariance matrices.
    
    Key features:
    - Fully differentiable through all operations
    - Learnable process noise Q and measurement noise R matrices
    - Numerical stability through regularization
    - Support for batch processing
    """
    
    def __init__(
        self,
        state_dim: int,
        measurement_dim: int,
        initial_Q: float = 1e-3,
        initial_R: float = 1e-2,
        learn_noise_matrices: bool = True,
        numerical_stability_eps: float = 1e-6
    ):
        super().__init__()
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.learn_noise_matrices = learn_noise_matrices
        self.eps = numerical_stability_eps
        
        # Initialize learnable covariance matrices
        if learn_noise_matrices:
            self.Q_learnable = LearnableCovariance(
                size=state_dim,
                initial_value=initial_Q,
                diagonal_init=True
            )
            self.R_learnable = LearnableCovariance(
                size=measurement_dim,
                initial_value=initial_R,
                diagonal_init=True
            )
        else:
            # Fixed covariance matrices
            self.register_buffer('Q_matrix', torch.eye(state_dim) * initial_Q)
            self.register_buffer('R_matrix', torch.eye(measurement_dim) * initial_R)
    
    @property
    def Q_matrix(self) -> torch.Tensor:
        """Get process noise covariance matrix."""
        if self.learn_noise_matrices:
            return self.Q_learnable()
        else:
            return self.Q_matrix
    
    @property
    def R_matrix(self) -> torch.Tensor:
        """Get measurement noise covariance matrix."""
        if self.learn_noise_matrices:
            return self.R_learnable()
        else:
            return self.R_matrix
    
    def predict(
        self,
        state: torch.Tensor,
        covariance: torch.Tensor,
        dynamics_fn: Callable[[torch.Tensor], torch.Tensor],
        jacobian_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        EKF prediction step.
        
        Args:
            state: Current state estimate [batch, state_dim]
            covariance: Current covariance [batch, state_dim, state_dim]
            dynamics_fn: Function that predicts next state
            jacobian_fn: Function that computes dynamics Jacobian (optional)
            
        Returns:
            predicted_state: Predicted state [batch, state_dim]
            predicted_covariance: Predicted covariance [batch, state_dim, state_dim]
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Predict state using dynamics function
        predicted_state = dynamics_fn(state)
        
        # Compute Jacobian matrix F = ∂f/∂x
        if jacobian_fn is not None:
            F = jacobian_fn(state)
        else:
            # Use automatic differentiation to compute Jacobian
            F = self._compute_jacobian(dynamics_fn, state)
        
        # Get process noise covariance
        Q = self.Q_matrix
        if Q.dim() == 2:
            Q = Q.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Predict covariance: P = F * P * F^T + Q
        predicted_covariance = torch.bmm(torch.bmm(F, covariance), F.transpose(-2, -1)) + Q
        
        # Ensure numerical stability
        predicted_covariance = self._regularize_covariance(predicted_covariance)
        
        return predicted_state, predicted_covariance
    
    def update(
        self,
        predicted_state: torch.Tensor,
        predicted_covariance: torch.Tensor,
        measurement: torch.Tensor,
        measurement_fn: Callable[[torch.Tensor], torch.Tensor],
        measurement_jacobian_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        EKF update step.
        
        Args:
            predicted_state: Predicted state [batch, state_dim]
            predicted_covariance: Predicted covariance [batch, state_dim, state_dim]
            measurement: Actual measurement [batch, measurement_dim]
            measurement_fn: Function that predicts measurement from state
            measurement_jacobian_fn: Function that computes measurement Jacobian
            
        Returns:
            updated_state: Updated state estimate [batch, state_dim]
            updated_covariance: Updated covariance [batch, state_dim, state_dim]
            predicted_measurement: Predicted measurement [batch, measurement_dim]
        """
        batch_size = predicted_state.shape[0]
        device = predicted_state.device
        
        # Predict measurement
        predicted_measurement = measurement_fn(predicted_state)
        
        # Compute measurement Jacobian H = ∂h/∂x
        if measurement_jacobian_fn is not None:
            H = measurement_jacobian_fn(predicted_state)
        else:
            H = self._compute_jacobian(measurement_fn, predicted_state)
        
        # Get measurement noise covariance
        R = self.R_matrix
        if R.dim() == 2:
            R = R.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Innovation covariance: S = H * P * H^T + R
        S = torch.bmm(torch.bmm(H, predicted_covariance), H.transpose(-2, -1)) + R
        S = self._regularize_covariance(S)
        
        # Kalman gain: K = P * H^T * S^{-1}
        try:
            S_inv = torch.linalg.inv(S)
        except RuntimeError:
            # Fallback to pseudo-inverse for numerical stability
            S_inv = torch.linalg.pinv(S)
        
        K = torch.bmm(torch.bmm(predicted_covariance, H.transpose(-2, -1)), S_inv)
        
        # Innovation
        innovation = measurement - predicted_measurement
        
        # Update state: x = x + K * innovation
        updated_state = predicted_state + torch.bmm(K, innovation.unsqueeze(-1)).squeeze(-1)
        
        # Update covariance: P = (I - K * H) * P
        I = torch.eye(self.state_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        KH = torch.bmm(K, H)
        updated_covariance = torch.bmm(I - KH, predicted_covariance)
        
        # Joseph form for numerical stability (optional)
        # P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        if True:  # Use Joseph form
            IKH = I - KH
            updated_covariance = (
                torch.bmm(torch.bmm(IKH, predicted_covariance), IKH.transpose(-2, -1)) +
                torch.bmm(torch.bmm(K, R), K.transpose(-2, -1))
            )
        
        updated_covariance = self._regularize_covariance(updated_covariance)
        
        return updated_state, updated_covariance, predicted_measurement
    
    def forward(
        self,
        initial_state: torch.Tensor,
        initial_covariance: torch.Tensor,
        measurements: torch.Tensor,
        dynamics_fn: Callable[[torch.Tensor], torch.Tensor],
        measurement_fn: Callable[[torch.Tensor], torch.Tensor],
        dynamics_jacobian_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        measurement_jacobian_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run EKF over a sequence of measurements.
        
        Args:
            initial_state: Initial state [batch, state_dim]
            initial_covariance: Initial covariance [batch, state_dim, state_dim]
            measurements: Measurement sequence [batch, seq_len, measurement_dim]
            dynamics_fn: Dynamics function
            measurement_fn: Measurement function
            dynamics_jacobian_fn: Dynamics Jacobian function (optional)
            measurement_jacobian_fn: Measurement Jacobian function (optional)
            
        Returns:
            states: State estimates [batch, seq_len, state_dim]
            covariances: Covariances [batch, seq_len, state_dim, state_dim]
            predicted_measurements: Predicted measurements [batch, seq_len, measurement_dim]
        """
        batch_size, seq_len = measurements.shape[:2]
        device = measurements.device
        
        # Initialize storage
        states = []
        covariances = []
        predicted_measurements = []
        
        # Current estimates
        current_state = initial_state
        current_covariance = initial_covariance
        
        for t in range(seq_len):
            # Prediction step
            predicted_state, predicted_covariance = self.predict(
                current_state, current_covariance, dynamics_fn, dynamics_jacobian_fn
            )
            
            # Update step
            updated_state, updated_covariance, pred_measurement = self.update(
                predicted_state, predicted_covariance, measurements[:, t],
                measurement_fn, measurement_jacobian_fn
            )
            
            # Store results
            states.append(updated_state)
            covariances.append(updated_covariance)
            predicted_measurements.append(pred_measurement)
            
            # Update for next iteration
            current_state = updated_state
            current_covariance = updated_covariance
        
        # Stack results
        states = torch.stack(states, dim=1)
        covariances = torch.stack(covariances, dim=1)
        predicted_measurements = torch.stack(predicted_measurements, dim=1)
        
        return states, covariances, predicted_measurements
    
    def _compute_jacobian(
        self,
        func: Callable[[torch.Tensor], torch.Tensor],
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Jacobian matrix using automatic differentiation.
        
        Args:
            func: Function to differentiate
            input_tensor: Input tensor [batch, input_dim]
            
        Returns:
            jacobian: Jacobian matrix [batch, output_dim, input_dim]
        """
        batch_size, input_dim = input_tensor.shape
        input_tensor = input_tensor.requires_grad_(True)
        
        # Compute function output
        output = func(input_tensor)
        output_dim = output.shape[-1]
        
        # Initialize Jacobian
        jacobian = torch.zeros(batch_size, output_dim, input_dim, 
                             device=input_tensor.device, dtype=input_tensor.dtype)
        
        # Compute Jacobian row by row
        for i in range(output_dim):
            # Create gradient vector for i-th output
            grad_outputs = torch.zeros_like(output)
            grad_outputs[:, i] = 1.0
            
            # Compute gradients
            grads = torch.autograd.grad(
                outputs=output,
                inputs=input_tensor,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=True,  # Enable higher-order derivatives
                only_inputs=True
            )[0]
            
            jacobian[:, i, :] = grads
        
        return jacobian
    
    def _regularize_covariance(
        self,
        covariance: torch.Tensor
    ) -> torch.Tensor:
        """
        Regularize covariance matrix for numerical stability.
        
        Args:
            covariance: Covariance matrix [batch, dim, dim]
            
        Returns:
            regularized_covariance: Regularized covariance matrix
        """
        batch_size, dim = covariance.shape[:2]
        device = covariance.device
        
        # Ensure symmetry
        covariance = 0.5 * (covariance + covariance.transpose(-2, -1))
        
        # Add small regularization to diagonal
        eye = torch.eye(dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        covariance = covariance + self.eps * eye
        
        return covariance
    
    def get_innovation_covariance(
        self,
        predicted_covariance: torch.Tensor,
        measurement_jacobian: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute innovation covariance for outlier detection.
        
        Args:
            predicted_covariance: Predicted state covariance [batch, state_dim, state_dim]
            measurement_jacobian: Measurement Jacobian [batch, measurement_dim, state_dim]
            
        Returns:
            innovation_covariance: Innovation covariance [batch, measurement_dim, measurement_dim]
        """
        H = measurement_jacobian
        R = self.R_matrix
        if R.dim() == 2:
            R = R.unsqueeze(0).expand(predicted_covariance.shape[0], -1, -1)
        
        S = torch.bmm(torch.bmm(H, predicted_covariance), H.transpose(-2, -1)) + R
        return self._regularize_covariance(S)
    
    def compute_log_likelihood(
        self,
        innovation: torch.Tensor,
        innovation_covariance: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log-likelihood of innovation for model selection.
        
        Args:
            innovation: Innovation vector [batch, measurement_dim]
            innovation_covariance: Innovation covariance [batch, measurement_dim, measurement_dim]
            
        Returns:
            log_likelihood: Log-likelihood [batch]
        """
        measurement_dim = innovation.shape[-1]
        
        # Compute log determinant
        sign, logdet = torch.linalg.slogdet(innovation_covariance)
        
        # Mahalanobis distance
        try:
            S_inv = torch.linalg.inv(innovation_covariance)
        except RuntimeError:
            S_inv = torch.linalg.pinv(innovation_covariance)
        
        mahalanobis = torch.bmm(
            torch.bmm(innovation.unsqueeze(1), S_inv), 
            innovation.unsqueeze(-1)
        ).squeeze()
        
        # Log-likelihood
        log_likelihood = -0.5 * (
            measurement_dim * math.log(2 * math.pi) + 
            logdet + 
            mahalanobis
        )
        
        return log_likelihood