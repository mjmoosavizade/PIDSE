"""
State space model with known physics.

This module implements the known physical laws that form the basis
for the PIDSE framework, such as rigid body dynamics.
"""

import torch
import torch.nn as nn
from typing import Optional


class StateSpaceModel(nn.Module):
    """
    Known physics component of the state space model.
    
    Implements f_known(x, u) - the known physical laws that govern
    system dynamics, such as rigid body kinematics and basic dynamics.
    """
    
    def __init__(
        self,
        state_dim: int = 12,
        control_dim: int = 4,
        measurement_dim: int = 9,
        mass: Optional[float] = None,
        dt: float = 0.01
    ):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.measurement_dim = measurement_dim
        self.dt = dt
        self.mass = mass
        
        # Register gravity as a parameter (can be learned if needed)
        self.register_parameter(
            "gravity", 
            nn.Parameter(torch.tensor([0.0, 0.0, -9.81]), requires_grad=False)
        )
    
    def dynamics_step(
        self, 
        state: torch.Tensor, 
        control: torch.Tensor
    ) -> torch.Tensor:
        """
        Known physics dynamics: f_known(x, u)
        
        State vector: [px, py, pz, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        Control vector: [fx, fy, fz, torque_z] or [motor_commands]
        
        Args:
            state: Current state [batch, state_dim]
            control: Control input [batch, control_dim]
            
        Returns:
            Next state according to known physics [batch, state_dim]
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Extract state components
        position = state[:, 0:3]  # [px, py, pz]
        velocity = state[:, 3:6]  # [vx, vy, vz]
        orientation = state[:, 6:9]  # [roll, pitch, yaw]
        angular_velocity = state[:, 9:12]  # [wx, wy, wz]
        
        # Position integration (kinematic)
        next_position = position + velocity * self.dt
        
        # Velocity integration (dynamic) - adapt to control dimensions
        control_dim = control.shape[-1]
        
        if control_dim >= 3:
            control_forces = control[:, 0:3]
        else:
            # Pad control to 3D if needed
            control_forces = torch.cat([
                control, 
                torch.zeros(batch_size, 3 - control_dim, device=device)
            ], dim=1)
            
        if self.mass is not None:
            # If mass is known, use F = ma
            acceleration = control_forces / self.mass + self.gravity.to(device).unsqueeze(0)
        else:
            # If mass unknown, assume control directly gives acceleration
            acceleration = control_forces + self.gravity.to(device).unsqueeze(0)
        
        next_velocity = velocity + acceleration * self.dt
        
        # Orientation integration (kinematic)
        # Simple Euler integration (can be improved with quaternions)
        next_orientation = orientation + angular_velocity * self.dt
        
        # Angular velocity integration (simplified)
        if self.control_dim > 3:
            angular_acceleration = control[:, 3:6] if self.control_dim >= 6 else control[:, 3:4].repeat(1, 3)
        else:
            angular_acceleration = torch.zeros_like(angular_velocity)
        
        next_angular_velocity = angular_velocity + angular_acceleration * self.dt
        
        # Combine next state - match input state dimensions
        state_components = []
        current_dim = 0
        
        if self.state_dim > current_dim:
            needed = min(3, self.state_dim - current_dim)
            state_components.append(next_position[:, :needed])
            current_dim += needed
            
        if self.state_dim > current_dim:
            needed = min(3, self.state_dim - current_dim)  
            state_components.append(next_velocity[:, :needed])
            current_dim += needed
            
        if self.state_dim > current_dim:
            needed = min(3, self.state_dim - current_dim)
            state_components.append(next_orientation[:, :needed])
            current_dim += needed
            
        if self.state_dim > current_dim:
            needed = min(3, self.state_dim - current_dim)
            state_components.append(next_angular_velocity[:, :needed])
            current_dim += needed
        
        next_state = torch.cat(state_components, dim=1)
        
        return next_state
    
    def measurement_model(self, state: torch.Tensor) -> torch.Tensor:
        """
        Known measurement model: h_known(x)
        
        Simulates ideal IMU + encoder measurements:
        - Accelerometer: measures specific force (acceleration - gravity)
        - Gyroscope: measures angular velocity
        - Encoder: measures position/velocity (if available)
        
        Args:
            state: Current state [batch, state_dim]
            
        Returns:
            Expected measurements [batch, measurement_dim]
        """
        batch_size = state.shape[0]
        device = state.device
        state_dim = state.shape[-1]
        
        # Extract state components adaptively based on state dimension
        if state_dim >= 3:
            position = state[:, 0:3]
        else:
            position = torch.cat([state[:, 0:2], torch.zeros(batch_size, 1, device=device)], dim=1)
            
        if state_dim >= 6:
            velocity = state[:, 3:6]
        elif state_dim >= 4:
            velocity = torch.cat([state[:, 2:4], torch.zeros(batch_size, 1, device=device)], dim=1)
        else:
            velocity = torch.zeros(batch_size, 3, device=device)
            
        if state_dim >= 9:
            orientation = state[:, 6:9]
        else:
            orientation = torch.zeros(batch_size, 3, device=device)
            
        if state_dim >= 12:
            angular_velocity = state[:, 9:12]
        else:
            angular_velocity = torch.zeros(batch_size, 3, device=device)
        
        # Ideal accelerometer measurement (specific force)
        # This would need proper rotation from body to world frame
        roll, pitch, yaw = orientation[:, 0], orientation[:, 1], orientation[:, 2]
        
        # Simplified rotation matrices (small angle approximation)
        cos_r, sin_r = torch.cos(roll), torch.sin(roll)
        cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        
        # Gravity in body frame (simplified)
        gravity_magnitude = torch.norm(self.gravity.to(device))
        gravity_body = torch.stack([
            sin_p,
            -sin_r * cos_p,
            -cos_r * cos_p
        ], dim=1) * gravity_magnitude
        
        # Accelerometer measurement (would include linear acceleration in real case)
        accel_measurement = gravity_body  # simplified
        
        # Gyroscope measurement
        gyro_measurement = angular_velocity
        
        # Position measurement (if GPS/mocap available)
        position_measurement = position
        
        # Combine measurements adaptively based on measurement dimension needed
        measurement_components = []
        
        if self.measurement_dim >= 3:
            measurement_components.append(accel_measurement)  # 3D
        if self.measurement_dim >= 6:
            measurement_components.append(gyro_measurement)   # 3D
        if self.measurement_dim >= 9:
            measurement_components.append(position_measurement)  # 3D
        
        # For simple case, just use available state components
        if not measurement_components or self.measurement_dim == state_dim:
            # Direct measurement of available state
            if state_dim <= self.measurement_dim:
                measurements = torch.cat([state, torch.zeros(batch_size, self.measurement_dim - state_dim, device=device)], dim=1)
            else:
                measurements = state[:, :self.measurement_dim]
        else:
            measurements = torch.cat(measurement_components, dim=1)
            # Trim or pad to exact measurement dimension
            if measurements.shape[1] > self.measurement_dim:
                measurements = measurements[:, :self.measurement_dim]
            elif measurements.shape[1] < self.measurement_dim:
                padding = torch.zeros(batch_size, self.measurement_dim - measurements.shape[1], device=device)
                measurements = torch.cat([measurements, padding], dim=1)
        
        return measurements
    
    def get_dynamics_jacobian(
        self, 
        state: torch.Tensor, 
        control: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Jacobian of dynamics w.r.t. state for EKF linearization.
        
        Returns:
            F_jacobian: ∂f/∂x [batch, state_dim, state_dim]
        """
        # For linear dynamics, Jacobian is constant
        batch_size = state.shape[0]
        device = state.device
        
        # Identity for position terms (dx/dt = v)
        F = torch.eye(self.state_dim, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Position derivatives w.r.t. velocity
        F[:, 0:3, 3:6] = torch.eye(3, device=device) * self.dt
        
        # Orientation derivatives w.r.t. angular velocity
        F[:, 6:9, 9:12] = torch.eye(3, device=device) * self.dt
        
        return F
    
    def get_measurement_jacobian(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian of measurement model w.r.t. state for EKF.
        
        Returns:
            H_jacobian: ∂h/∂x [batch, measurement_dim, state_dim]
        """
        batch_size = state.shape[0]
        device = state.device
        
        H = torch.zeros(batch_size, self.measurement_dim, self.state_dim, device=device)
        
        # Accelerometer depends on orientation (gravity rotation)
        # Simplified: assume small angles
        H[:, 0:3, 6:9] = torch.eye(3, device=device) * torch.norm(self.gravity)
        
        # Gyroscope directly measures angular velocity
        H[:, 3:6, 9:12] = torch.eye(3, device=device)
        
        # Position measurement directly measures position
        H[:, 6:9, 0:3] = torch.eye(3, device=device)
        
        return H
    
    def enforce_physics_constraints(
        self, 
        state: torch.Tensor,
        next_state: torch.Tensor,
        control: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute physics violation for the loss function.
        
        Returns:
            physics_violation: Scalar tensor representing constraint violations
        """
        violations = []
        
        # Energy conservation check (simplified)
        if self.mass is not None:
            current_velocity = state[:, 3:6]
            next_velocity = next_state[:, 3:6]
            
            current_ke = 0.5 * self.mass * torch.sum(current_velocity**2, dim=1)
            next_ke = 0.5 * self.mass * torch.sum(next_velocity**2, dim=1)
            
            # Work done by forces
            forces = control[:, 0:3]
            displacement = (next_state[:, 0:3] - state[:, 0:3])
            work_done = torch.sum(forces * displacement, dim=1)
            
            # Energy balance violation
            energy_violation = torch.abs(next_ke - current_ke - work_done)
            violations.append(energy_violation.mean())
        
        # Momentum conservation (if no external forces)
        # This would be more complex in practice
        
        if violations:
            return torch.stack(violations).sum()
        else:
            return torch.tensor(0.0, device=state.device)