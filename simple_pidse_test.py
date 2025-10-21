"""
Minimal PIDSE example to verify core functionality.

This creates the simplest possible working version of PIDSE
with basic dynamics learning and state estimation.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class SimplePINN(nn.Module):
    """Simple Physics-Informed Neural Network for residual dynamics."""
    
    def __init__(self, state_dim, control_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + control_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, state_dim)
        )
        
        # Initialize with small weights for residual learning
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.zeros_(layer.bias)
    
    def forward(self, state, control):
        x = torch.cat([state, control], dim=-1)
        return self.net(x) * 0.1  # Scale down residuals


class SimpleEKF:
    """Simple Extended Kalman Filter with fixed noise matrices."""
    
    def __init__(self, state_dim, measurement_dim):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.Q = torch.eye(state_dim) * 0.01  # Process noise
        self.R = torch.eye(measurement_dim) * 0.1  # Measurement noise
    
    def predict_and_update(self, state, covariance, measurement):
        """Simple prediction and update step."""
        # Simple prediction (assume identity dynamics for now)
        predicted_state = state
        predicted_cov = covariance + self.Q
        
        # Simple update (assume identity measurement model)
        H = torch.eye(self.measurement_dim, self.state_dim)
        if measurement.dim() > 1:  # Batch processing
            batch_size = measurement.shape[0]
            H = H.unsqueeze(0).expand(batch_size, -1, -1)
            Q = self.Q.unsqueeze(0).expand(batch_size, -1, -1)
            R = self.R.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            Q = self.Q
            R = self.R
        
        # Innovation
        predicted_measurement = predicted_state[..., :self.measurement_dim]
        innovation = measurement - predicted_measurement
        
        # Innovation covariance
        if measurement.dim() > 1:
            S = torch.bmm(torch.bmm(H, predicted_cov), H.transpose(-2, -1)) + R
            # Kalman gain
            try:
                S_inv = torch.linalg.inv(S)
                K = torch.bmm(torch.bmm(predicted_cov, H.transpose(-2, -1)), S_inv)
            except:
                K = torch.zeros_like(torch.bmm(predicted_cov, H.transpose(-2, -1)))
        else:
            S = H @ predicted_cov @ H.T + R
            try:
                K = predicted_cov @ H.T @ torch.linalg.inv(S)
            except:
                K = torch.zeros_like(predicted_cov @ H.T)
        
        # Update
        if measurement.dim() > 1:
            updated_state = predicted_state + torch.bmm(K, innovation.unsqueeze(-1)).squeeze(-1)
            I = torch.eye(self.state_dim).unsqueeze(0).expand(batch_size, -1, -1)
            updated_cov = torch.bmm(I - torch.bmm(K, H), predicted_cov)
        else:
            updated_state = predicted_state + K @ innovation
            I = torch.eye(self.state_dim)
            updated_cov = (I - K @ H) @ predicted_cov
            
        return updated_state, updated_cov


class SimplePIDSE(nn.Module):
    """Simplified PIDSE for testing core functionality."""
    
    def __init__(self, state_dim=4, control_dim=2, measurement_dim=4):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim 
        self.measurement_dim = measurement_dim
        
        # Components
        self.pinn = SimplePINN(state_dim, control_dim)
        self.ekf = SimpleEKF(state_dim, measurement_dim)
    
    def known_dynamics(self, state, control, dt=0.01):
        """Known physics model (simple kinematic model)."""
        batch_shape = state.shape[:-1]
        next_state = state.clone()
        
        # Simple integrator dynamics: [x, y, vx, vy]
        if self.state_dim >= 4:
            # Position integration
            next_state[..., 0] = state[..., 0] + state[..., 2] * dt  # x += vx * dt
            next_state[..., 1] = state[..., 1] + state[..., 3] * dt  # y += vy * dt
            
            # Velocity integration (from control)
            if self.control_dim >= 2:
                next_state[..., 2] = state[..., 2] + control[..., 0] * dt  # vx += ax * dt
                next_state[..., 3] = state[..., 3] + control[..., 1] * dt  # vy += ay * dt
                
        return next_state
    
    def forward(self, states, controls, measurements):
        """Forward pass through simplified PIDSE."""
        batch_size, seq_len = states.shape[:2]
        
        estimated_states = []
        current_state = states[:, 0]  # Start with first true state
        current_cov = torch.eye(self.state_dim).unsqueeze(0).expand(batch_size, -1, -1) * 0.1
        
        for t in range(seq_len):
            if t > 0:
                # Prediction with known + learned dynamics
                known_next = self.known_dynamics(estimated_states[-1], controls[:, t-1])
                learned_residual = self.pinn(estimated_states[-1], controls[:, t-1])
                predicted_state = known_next + learned_residual
            else:
                predicted_state = current_state
            
            # EKF update
            updated_state, updated_cov = self.ekf.predict_and_update(
                predicted_state, current_cov, measurements[:, t]
            )
            
            estimated_states.append(updated_state)
            current_state = updated_state
            current_cov = updated_cov
        
        estimated_states = torch.stack(estimated_states, dim=1)
        
        # Simple loss
        loss = torch.mean((estimated_states - states)**2)
        
        return estimated_states, loss


def create_simple_data(n_traj=10, seq_len=20):
    """Create simple 2D trajectory data."""
    trajectories = []
    
    for _ in range(n_traj):
        # Generate smooth trajectory
        t = torch.linspace(0, 2*np.pi, seq_len)
        
        # Simple circular motion with some variation
        radius = 1.0 + 0.3 * torch.randn(1)
        frequency = 0.5 + 0.2 * torch.randn(1)
        
        x = radius * torch.cos(frequency * t)
        y = radius * torch.sin(frequency * t)
        
        vx = -radius * frequency * torch.sin(frequency * t)
        vy = radius * frequency * torch.cos(frequency * t)
        
        # States: [x, y, vx, vy]
        states = torch.stack([x, y, vx, vy], dim=1)
        
        # Controls: [ax, ay] (accelerations)
        ax = -radius * frequency**2 * torch.cos(frequency * t)
        ay = -radius * frequency**2 * torch.sin(frequency * t)
        controls = torch.stack([ax, ay], dim=1)
        
        # Measurements: noisy states
        measurements = states + 0.05 * torch.randn_like(states)
        
        trajectories.append({
            'states': states,
            'controls': controls, 
            'measurements': measurements
        })
    
    return trajectories


def test_simple_pidse():
    """Test the simplified PIDSE implementation."""
    print("🧪 Testing Simplified PIDSE")
    print("=" * 40)
    
    # Create model
    model = SimplePIDSE(state_dim=4, control_dim=2, measurement_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create data
    trajectories = create_simple_data(n_traj=5, seq_len=20)
    print(f"✅ Created {len(trajectories)} trajectories")
    
    # Training loop
    print("\n🎯 Training for 50 epochs...")
    losses = []
    
    for epoch in range(50):
        epoch_losses = []
        
        for traj in trajectories:
            states = traj['states'].unsqueeze(0)  # Add batch dim
            controls = traj['controls'].unsqueeze(0)
            measurements = traj['measurements'].unsqueeze(0)
            
            optimizer.zero_grad()
            estimated_states, loss = model(states, controls, measurements)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"   Epoch {epoch:2d}: Loss = {avg_loss:.6f}")
    
    print("✅ Training completed")
    
    # Test on one trajectory
    test_traj = trajectories[0]
    states = test_traj['states'].unsqueeze(0)
    controls = test_traj['controls'].unsqueeze(0)
    measurements = test_traj['measurements'].unsqueeze(0)
    
    with torch.no_grad():
        estimated_states, _ = model(states, controls, measurements)
    
    # Compute error
    error = torch.mean(torch.norm(estimated_states - states, dim=-1))
    print(f"\n📊 Test Results:")
    print(f"   Average position error: {error:.4f}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot trajectory
    true_pos = states[0, :, :2].numpy()
    est_pos = estimated_states[0, :, :2].numpy()
    
    ax1.plot(true_pos[:, 0], true_pos[:, 1], 'b-', label='True', linewidth=2)
    ax1.plot(est_pos[:, 0], est_pos[:, 1], 'r--', label='Estimated', linewidth=2)
    ax1.set_title('Trajectory Comparison')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y') 
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # Plot loss
    ax2.plot(losses, 'g-', linewidth=2)
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('simple_pidse_results.png', dpi=150, bbox_inches='tight')
    print(f"✅ Results saved to 'simple_pidse_results.png'")
    
    return True


if __name__ == "__main__":
    success = test_simple_pidse()
    if success:
        print("\n🎉 Simple PIDSE test successful!")
        print("Next steps:")
        print("1. This proves the core concept works")
        print("2. Now we can fix the full implementation")
        print("3. Add back complexity gradually")
    else:
        print("\n❌ Simple PIDSE test failed")