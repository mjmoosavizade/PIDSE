"""
Physics-Informed Neural Networks for dynamics and measurement modeling.

This module implements the neural network components that learn residual dynamics
and measurement models while respecting physical constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


class PhysicsInformedMLP(nn.Module):
    """
    Base class for physics-informed multi-layer perceptrons.
    
    Features:
    - Skip connections for gradient flow
    - Physics-aware initialization
    - Constraint enforcement layers
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int],
        activation: str = "tanh",
        use_skip_connections: bool = True,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.use_skip_connections = use_skip_connections
        
        # Activation function
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        self.layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList() if use_skip_connections else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_layers[0]))
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            
            # Skip connection layers (project to matching dimensions)
            if use_skip_connections and i % 2 == 1:  # Every other layer
                if hidden_layers[i + 1] != input_dim:
                    self.skip_layers.append(nn.Linear(input_dim, hidden_layers[i + 1]))
                else:
                    self.skip_layers.append(nn.Identity())
        
        # Output layer
        self.layers.append(nn.Linear(hidden_layers[-1], output_dim))
        
        # Initialize weights with physics-aware scheme
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Physics-aware weight initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Xavier/Glorot initialization for tanh, He for ReLU
                if isinstance(self.activation, nn.Tanh):
                    nn.init.xavier_normal_(layer.weight)
                else:
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                
                # Small bias initialization
                nn.init.constant_(layer.bias, 0.0)
        
        # Initialize output layer with smaller weights (residual learning)
        nn.init.normal_(self.layers[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.layers[-1].bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional skip connections."""
        identity = x
        
        # First layer
        x = self.activation(self.layers[0](x))
        if self.dropout:
            x = self.dropout(x)
        
        # Hidden layers with skip connections
        skip_idx = 0
        for i in range(1, len(self.layers) - 1):
            x = self.layers[i](x)
            
            # Add skip connection every other layer
            if self.use_skip_connections and i % 2 == 0 and skip_idx < len(self.skip_layers):
                skip_connection = self.skip_layers[skip_idx](identity)
                x = x + skip_connection
                skip_idx += 1
            
            x = self.activation(x)
            if self.dropout:
                x = self.dropout(x)
        
        # Output layer (no activation)
        x = self.layers[-1](x)
        
        return x


class DynamicsNetwork(nn.Module):
    """
    Physics-Informed Neural Network for learning residual dynamics.
    
    Learns f_PINN(x, u; θ) such that:
    f_total(x, u) = f_known(x, u) + f_PINN(x, u; θ)
    
    The network is designed to output small residuals that correct
    the known physics model.
    """
    
    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        hidden_layers: List[int] = [64, 64, 32],
        activation: str = "tanh",
        use_physics_embedding: bool = True,
        residual_scale: float = 0.1
    ):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.residual_scale = residual_scale
        self.use_physics_embedding = use_physics_embedding
        
        # Physics embedding layer (optional)
        if use_physics_embedding:
            self.physics_embedding = self._create_physics_embedding()
            embedding_dim = state_dim + control_dim + 6  # add physics features
        else:
            self.physics_embedding = None
            embedding_dim = state_dim + control_dim
        
        # Main dynamics network
        self.dynamics_net = PhysicsInformedMLP(
            input_dim=embedding_dim,
            output_dim=state_dim,  # predict residual for each state
            hidden_layers=hidden_layers,
            activation=activation,
            use_skip_connections=True,
            dropout_rate=0.1
        )
        
        # Residual scaling layer
        self.residual_scale_layer = nn.Parameter(
            torch.ones(state_dim) * residual_scale, 
            requires_grad=True
        )
    
    def _create_physics_embedding(self) -> nn.Module:
        """Create physics-aware feature embedding."""
        return nn.Sequential(
            nn.Linear(self.state_dim + self.control_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 6)  # Extract 6 physics features
        )
    
    def _extract_physics_features(
        self, 
        state: torch.Tensor, 
        control: torch.Tensor
    ) -> torch.Tensor:
        """Extract physics-relevant features from state and control."""
        # Assume state = [position(3), velocity(3), orientation(3), angular_vel(3)]
        position = state[..., 0:3]
        velocity = state[..., 3:6]
        orientation = state[..., 6:9] if state.shape[-1] > 6 else torch.zeros_like(position)
        angular_vel = state[..., 9:12] if state.shape[-1] > 9 else torch.zeros_like(position)
        
        # Compute physics-relevant quantities
        speed = torch.norm(velocity, dim=-1, keepdim=True)
        kinetic_energy = 0.5 * torch.sum(velocity**2, dim=-1, keepdim=True)
        angular_speed = torch.norm(angular_vel, dim=-1, keepdim=True) if state.shape[-1] > 9 else torch.zeros_like(speed)
        
        # Control magnitude
        control_magnitude = torch.norm(control, dim=-1, keepdim=True)
        
        # Height (assuming z is up)
        height = position[..., 2:3]
        
        # Orientation magnitude (angle from upright)
        orientation_magnitude = torch.norm(orientation, dim=-1, keepdim=True) if state.shape[-1] > 6 else torch.zeros_like(speed)
        
        physics_features = torch.cat([
            speed, kinetic_energy, angular_speed, 
            control_magnitude, height, orientation_magnitude
        ], dim=-1)
        
        return physics_features
    
    def forward(
        self, 
        state: torch.Tensor, 
        control: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass to predict residual dynamics.
        
        Args:
            state: Current state [batch, state_dim]
            control: Control input [batch, control_dim]
            
        Returns:
            residual: Dynamics residual [batch, state_dim]
        """
        # Concatenate state and control
        input_features = torch.cat([state, control], dim=-1)
        
        # Add physics features if enabled
        if self.use_physics_embedding:
            physics_features = self._extract_physics_features(state, control)
            if self.physics_embedding is not None:
                embedded_physics = self.physics_embedding(input_features)
                input_features = torch.cat([input_features, embedded_physics], dim=-1)
            else:
                input_features = torch.cat([input_features, physics_features], dim=-1)
        
        # Predict residual
        residual = self.dynamics_net(input_features)
        
        # Apply learned scaling
        residual = residual * self.residual_scale_layer
        
        return residual
    
    def get_residual_magnitude(self) -> torch.Tensor:
        """Get current residual scaling factors."""
        return self.residual_scale_layer.abs()


class MeasurementNetwork(nn.Module):
    """
    Physics-Informed Neural Network for learning residual measurement model.
    
    Learns h_PINN(x; θ) such that:
    h_total(x) = h_known(x) + h_PINN(x; θ)
    
    Accounts for sensor biases, nonlinearities, and complex measurement dynamics.
    """
    
    def __init__(
        self,
        state_dim: int,
        measurement_dim: int,
        hidden_layers: List[int] = [32, 32],
        activation: str = "relu",
        sensor_noise_learning: bool = True
    ):
        super().__init__()
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.sensor_noise_learning = sensor_noise_learning
        
        # Main measurement network
        self.measurement_net = PhysicsInformedMLP(
            input_dim=state_dim,
            output_dim=measurement_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            use_skip_connections=False,  # Simpler for measurement model
            dropout_rate=0.05
        )
        
        # Learnable sensor bias (common in real sensors)
        self.sensor_bias = nn.Parameter(
            torch.zeros(measurement_dim), 
            requires_grad=True
        )
        
        # Learnable nonlinear scaling (sensor calibration)
        if sensor_noise_learning:
            self.sensor_scale = nn.Parameter(
                torch.ones(measurement_dim) * 0.1, 
                requires_grad=True
            )
        else:
            self.register_parameter('sensor_scale', None)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict measurement residual.
        
        Args:
            state: Current state [batch, state_dim]
            
        Returns:
            residual: Measurement residual [batch, measurement_dim]
        """
        # Predict residual measurement
        residual = self.measurement_net(state)
        
        # Add sensor bias
        residual = residual + self.sensor_bias
        
        # Apply sensor scaling if enabled
        if self.sensor_scale is not None:
            residual = residual * self.sensor_scale
        
        return residual
    
    def get_sensor_bias(self) -> torch.Tensor:
        """Get current sensor bias estimate."""
        return self.sensor_bias.data
    
    def get_sensor_scale(self) -> Optional[torch.Tensor]:
        """Get current sensor scaling factors."""
        if self.sensor_scale is not None:
            return self.sensor_scale.data.abs()
        return None


class EnsembleDynamicsNetwork(nn.Module):
    """
    Ensemble of dynamics networks for uncertainty quantification.
    
    Uses multiple PINN networks to capture epistemic uncertainty
    in the learned dynamics model.
    """
    
    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        num_ensemble: int = 5,
        hidden_layers: List[int] = [64, 64, 32],
        activation: str = "tanh"
    ):
        super().__init__()
        self.num_ensemble = num_ensemble
        
        # Create ensemble of networks
        self.networks = nn.ModuleList([
            DynamicsNetwork(
                state_dim=state_dim,
                control_dim=control_dim,
                hidden_layers=hidden_layers,
                activation=activation,
                residual_scale=0.1 / math.sqrt(num_ensemble)  # Scale down for ensemble
            )
            for _ in range(num_ensemble)
        ])
    
    def forward(
        self, 
        state: torch.Tensor, 
        control: torch.Tensor,
        return_uncertainty: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through ensemble.
        
        Args:
            state: Current state [batch, state_dim]
            control: Control input [batch, control_dim]
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            mean_residual: Mean prediction across ensemble
            uncertainty: Standard deviation across ensemble (if requested)
        """
        # Get predictions from all networks
        predictions = torch.stack([
            network(state, control) for network in self.networks
        ], dim=0)  # [num_ensemble, batch, state_dim]
        
        # Compute ensemble statistics
        mean_residual = torch.mean(predictions, dim=0)
        
        if return_uncertainty:
            uncertainty = torch.std(predictions, dim=0)
            return mean_residual, uncertainty
        else:
            return mean_residual, None
    
    def get_ensemble_size(self) -> int:
        """Get number of networks in ensemble."""
        return self.num_ensemble