"""
Basic example of using PIDSE for quadrotor dynamics learning.

This example demonstrates:
1. Creating synthetic quadrotor data
2. Configuring PIDSE for the task
3. Training the model
4. Evaluating performance
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pidse import PIDSE, PIDSEConfig
from pidse.data import create_data_loaders
from pidse.utils import compute_ate, plot_trajectory

def generate_synthetic_quadrotor_data(n_trajectories=100, seq_length=200):
    """Generate synthetic quadrotor trajectory data."""
    trajectories = []
    
    for _ in range(n_trajectories):
        # Initial state [px, py, pz, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        state = torch.zeros(12)
        state[2] = 1.0  # start at 1m height
        
        states = [state]
        controls = []
        measurements = []
        
        for t in range(seq_length):
            # Simple control policy: hover with small perturbations
            control = torch.tensor([0.0, 0.0, 9.81, 0.0])  # hover thrust + small perturbations
            control += 0.5 * torch.randn(4)  # add noise
            
            # Simple dynamics (this would be f_known in practice)
            dt = 0.01
            next_state = state.clone()
            
            # Position integration
            next_state[0:3] += state[3:6] * dt
            
            # Velocity integration
            next_state[3:6] += (control[0:3] + torch.tensor([0.0, 0.0, -9.81])) * dt
            
            # Orientation integration (simplified)
            next_state[6:9] += state[9:12] * dt
            
            # Angular velocity (simple damping)
            next_state[9:12] = 0.9 * state[9:12] + 0.1 * torch.randn(3)
            
            # Add some unknown dynamics (this is what PINN should learn)
            drag_effect = -0.1 * state[3:6] * torch.norm(state[3:6])
            next_state[3:6] += drag_effect * dt
            
            # Generate measurements (IMU + GPS)
            measurement = torch.cat([
                control[0:3] + torch.tensor([0.0, 0.0, -9.81]) + 0.1 * torch.randn(3),  # accel
                state[9:12] + 0.05 * torch.randn(3),  # gyro
                state[0:3] + 0.2 * torch.randn(3)  # GPS position
            ])
            
            states.append(next_state)
            controls.append(control)
            measurements.append(measurement)
            state = next_state
        
        # Convert to tensors
        states_tensor = torch.stack(states[1:])  # exclude initial state
        controls_tensor = torch.stack(controls)
        measurements_tensor = torch.stack(measurements)
        
        trajectories.append({
            'states': states_tensor,
            'controls': controls_tensor,
            'measurements': measurements_tensor,
            'initial_state': states[0],
            'initial_covariance': torch.eye(12) * 0.1
        })
    
    return trajectories

def main():
    """Main example function."""
    print("🚁 PIDSE Quadrotor Example")
    print("=" * 50)
    
    # Generate synthetic data
    print("📊 Generating synthetic quadrotor data...")
    train_data = generate_synthetic_quadrotor_data(n_trajectories=80, seq_length=100)
    test_data = generate_synthetic_quadrotor_data(n_trajectories=20, seq_length=100)
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        train_data, test_data, batch_size=8, sequence_length=100
    )
    
    # Configure PIDSE
    config = PIDSEConfig(
        state_dim=12,
        control_dim=4,
        measurement_dim=9,
        pinn_hidden_layers=[32, 32, 16],
        learning_rate=1e-3,
        physics_weight=0.1,
        regularization_weight=0.01,
        mass=1.0  # 1kg quadrotor
    )
    
    print(f"📋 Configuration:")
    print(f"   State dim: {config.state_dim}")
    print(f"   Control dim: {config.control_dim}")
    print(f"   Measurement dim: {config.measurement_dim}")
    print(f"   PINN layers: {config.pinn_hidden_layers}")
    print(f"   Physics weight: {config.physics_weight}")
    
    # Initialize PIDSE
    print("\n🏗️  Initializing PIDSE...")
    pidse = PIDSE(config)
    optimizer = torch.optim.Adam(pidse.parameters(), lr=config.learning_rate)
    
    print(f"   Model parameters: {sum(p.numel() for p in pidse.parameters()):,}")
    print(f"   Device: {config.device}")
    
    # Training loop
    print("\n🎯 Training PIDSE...")
    num_epochs = 50
    train_losses = []
    
    for epoch in range(num_epochs):
        pidse.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(config.device) for k, v in batch.items()}
            
            # Training step
            loss_dict = pidse.train_step(batch, optimizer)
            epoch_losses.append(loss_dict['total_loss'])
            
            if batch_idx == 0:  # Print first batch losses
                print(f"   Epoch {epoch+1:2d}: Loss = {loss_dict['total_loss']:.4f} "
                      f"(Est: {loss_dict['estimation_loss']:.4f}, "
                      f"Phys: {loss_dict['physics_loss']:.4f}, "
                      f"Reg: {loss_dict['regularization_loss']:.4f})")
        
        train_losses.append(np.mean(epoch_losses))
        
        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            val_metrics = pidse.evaluate(test_loader)
            print(f"   Validation - Loss: {val_metrics['total_loss']:.4f}")
    
    print("\n✅ Training completed!")
    
    # Evaluation
    print("\n📈 Evaluating on test data...")
    pidse.eval()
    
    # Get a test batch for detailed analysis
    test_batch = next(iter(test_loader))
    test_batch = {k: v.to(config.device) for k, v in test_batch.items()}
    
    with torch.no_grad():
        outputs = pidse.forward(
            states=test_batch["states"],
            controls=test_batch["controls"],
            measurements=test_batch["measurements"],
            initial_state=test_batch["initial_state"],
            initial_covariance=test_batch["initial_covariance"]
        )
    
    # Compute trajectory metrics
    true_states = test_batch["states"].cpu().numpy()
    estimated_states = outputs["estimated_states"].cpu().numpy()
    
    # Compute ATE for first trajectory in batch
    ate = compute_ate(true_states[0, :, 0:3], estimated_states[0, :, 0:3])
    print(f"   Absolute Trajectory Error (ATE): {ate:.3f} m")
    
    # Plot results
    print("\n📊 Plotting results...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 3D trajectory
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    traj_true = true_states[0, :, 0:3]
    traj_est = estimated_states[0, :, 0:3]
    
    ax.plot(traj_true[:, 0], traj_true[:, 1], traj_true[:, 2], 'b-', label='True', linewidth=2)
    ax.plot(traj_est[:, 0], traj_est[:, 1], traj_est[:, 2], 'r--', label='Estimated', linewidth=2)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory')
    ax.legend()
    
    # Plot position errors
    axes[0, 1].plot(np.linalg.norm(traj_true - traj_est, axis=1))
    axes[0, 1].set_title('Position Error')
    axes[0, 1].set_ylabel('Error (m)')
    axes[0, 1].set_xlabel('Time step')
    axes[0, 1].grid(True)
    
    # Plot training loss
    axes[1, 0].plot(train_losses)
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].grid(True)
    
    # Plot learned noise matrices
    Q_learned = pidse.ekf.Q_matrix.detach().cpu().numpy()
    R_learned = pidse.ekf.R_matrix.detach().cpu().numpy()
    
    axes[1, 1].bar(range(len(np.diag(Q_learned))), np.diag(Q_learned), alpha=0.7, label='Q (process)')
    axes[1, 1].bar(range(len(np.diag(R_learned))), np.diag(R_learned), alpha=0.7, label='R (measurement)')
    axes[1, 1].set_title('Learned Noise Covariances')
    axes[1, 1].set_ylabel('Variance')
    axes[1, 1].set_xlabel('State/Measurement Index')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/mj/PIDSE/examples/quadrotor_results.png', dpi=150, bbox_inches='tight')
    print(f"   Results saved to: examples/quadrotor_results.png")
    
    # Save model
    model_path = '/home/mj/PIDSE/examples/pidse_quadrotor.pth'
    pidse.save_checkpoint(model_path)
    print(f"   Model saved to: {model_path}")
    
    print("\n🎉 Example completed successfully!")
    print(f"   Final ATE: {ate:.3f} m")
    print(f"   Learned Q matrix diagonal: {np.diag(Q_learned)}")
    print(f"   Learned R matrix diagonal: {np.diag(R_learned)}")

if __name__ == "__main__":
    main()