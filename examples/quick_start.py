"""
Quick start example for PIDSE.

This script demonstrates how to quickly get started with PIDSE
for learning dynamics and state estimation.
"""

import torch
import matplotlib.pyplot as plt
from pidse import PIDSE, PIDSEConfig
from pidse.data import create_synthetic_dataset, create_data_loaders
from pidse.utils.metrics import compute_trajectory_metrics


def quick_start_example():
    """Quick start example for PIDSE."""
    print("🚀 PIDSE Quick Start Example")
    print("=" * 40)
    
    # 1. Create configuration
    print("📋 Step 1: Configure PIDSE")
    config = PIDSEConfig(
        state_dim=12,          # [px, py, pz, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        control_dim=4,         # [fx, fy, fz, torque_z]
        measurement_dim=9,     # [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, pos_x, pos_y, pos_z]
        pinn_hidden_layers=[32, 32, 16],
        learning_rate=1e-3,
        physics_weight=0.1,
        regularization_weight=0.01,
        device='cpu'  # Use 'cuda' if you have GPU
    )
    print(f"   ✓ Configuration created for {config.device}")
    
    # 2. Generate synthetic data
    print("\n📊 Step 2: Generate synthetic quadrotor data")
    train_data = create_synthetic_dataset(
        n_trajectories=20,
        trajectory_length=100,
        system_type="quadrotor",
        noise_level=0.1
    )
    
    test_data = create_synthetic_dataset(
        n_trajectories=5,
        trajectory_length=100,
        system_type="quadrotor",
        noise_level=0.1
    )
    print(f"   ✓ Generated {len(train_data)} training and {len(test_data)} test trajectories")
    
    # 3. Create data loaders
    print("\n🔄 Step 3: Create data loaders")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, test_data,
        batch_size=4,
        sequence_length=50,
        val_split=0.2
    )
    print(f"   ✓ Created loaders: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test batches")
    
    # 4. Initialize PIDSE model
    print("\n🏗️  Step 4: Initialize PIDSE model")
    model = PIDSE(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    print(f"   ✓ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 5. Quick training loop
    print("\n🎯 Step 5: Quick training (10 epochs)")
    model.train()
    
    for epoch in range(10):
        epoch_losses = []
        
        for batch in train_loader:
            # Training step
            loss_dict = model.train_step(batch, optimizer)
            epoch_losses.append(loss_dict['total_loss'])
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        if epoch % 2 == 0:
            print(f"   Epoch {epoch:2d}: Loss = {avg_loss:.4f}")
    
    print("   ✓ Training completed")
    
    # 6. Evaluation
    print("\n📈 Step 6: Evaluate on test data")
    model.eval()
    
    # Get a test batch
    test_batch = next(iter(test_loader))
    
    with torch.no_grad():
        outputs = model(
            states=test_batch["states"],
            controls=test_batch["controls"],
            measurements=test_batch["measurements"],
            initial_state=test_batch["initial_state"],
            initial_covariance=test_batch["initial_covariance"]
        )
    
    # Compute metrics for first trajectory in batch
    true_states = test_batch["states"][0]
    estimated_states = outputs["estimated_states"][0]
    
    metrics = compute_trajectory_metrics(estimated_states, true_states)
    
    print(f"   ✓ Evaluation completed:")
    print(f"     - ATE: {metrics['ate']:.3f} m")
    print(f"     - RPE: {metrics['rpe_trans_mean']:.3f} m")
    print(f"     - Final drift: {metrics['final_drift']:.3f} m")
    
    # 7. Visualize results
    print("\n📊 Step 7: Visualize results")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Extract positions
    true_pos = true_states[:, 0:3].numpy()
    est_pos = estimated_states[:, 0:3].numpy()
    
    # 3D trajectory
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.plot(true_pos[:, 0], true_pos[:, 1], true_pos[:, 2], 'b-', label='True', linewidth=2)
    ax.plot(est_pos[:, 0], est_pos[:, 1], est_pos[:, 2], 'r--', label='Estimated', linewidth=2)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory')
    ax.legend()
    
    # Position errors
    pos_errors = torch.norm(estimated_states[:, 0:3] - true_states[:, 0:3], dim=1).numpy()
    axes[0, 1].plot(pos_errors)
    axes[0, 1].set_title('Position Error Over Time')
    axes[0, 1].set_ylabel('Error (m)')
    axes[0, 1].set_xlabel('Time step')
    axes[0, 1].grid(True)
    
    # Velocity comparison
    true_vel = true_states[:, 3:6]
    est_vel = estimated_states[:, 3:6]
    
    for i, label in enumerate(['vx', 'vy', 'vz']):
        axes[1, 0].plot(true_vel[:, i].numpy(), label=f'True {label}', linestyle='-')
        axes[1, 0].plot(est_vel[:, i].numpy(), label=f'Est {label}', linestyle='--')
    
    axes[1, 0].set_title('Velocity Components')
    axes[1, 0].set_ylabel('Velocity (m/s)')
    axes[1, 0].set_xlabel('Time step')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learned noise matrices
    Q_learned = model.ekf.Q_matrix.detach().numpy()
    R_learned = model.ekf.R_matrix.detach().numpy()
    
    x_pos = range(len(Q_learned))
    axes[1, 1].bar([x - 0.2 for x in x_pos], torch.diag(torch.tensor(Q_learned)).numpy(), 
                   width=0.4, alpha=0.7, label='Q (process)')
    
    x_pos_r = range(len(R_learned))
    axes[1, 1].bar([x + 0.2 for x in x_pos_r], torch.diag(torch.tensor(R_learned)).numpy(), 
                   width=0.4, alpha=0.7, label='R (measurement)')
    
    axes[1, 1].set_title('Learned Noise Covariances')
    axes[1, 1].set_ylabel('Variance')
    axes[1, 1].set_xlabel('Component Index')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('pidse_quick_start_results.png', dpi=150, bbox_inches='tight')
    print("   ✓ Results saved to 'pidse_quick_start_results.png'")
    
    # 8. Save model
    print("\n💾 Step 8: Save trained model")
    model.save_checkpoint('pidse_quick_start_model.pth')
    print("   ✓ Model saved to 'pidse_quick_start_model.pth'")
    
    # Summary
    print("\n🎉 Quick Start Example Completed!")
    print("=" * 40)
    print("📊 Results Summary:")
    print(f"   • Trained on {len(train_data)} trajectories")
    print(f"   • Final ATE: {metrics['ate']:.3f} m")
    print(f"   • Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   • Learned Q diagonal: {torch.diag(model.ekf.Q_matrix).detach().numpy()}")
    print(f"   • Learned R diagonal: {torch.diag(model.ekf.R_matrix).detach().numpy()}")
    
    print("\n🚀 Next Steps:")
    print("   1. Try with real motion capture data")
    print("   2. Experiment with different network architectures")
    print("   3. Adjust physics and regularization weights")
    print("   4. Train for more epochs with larger datasets")
    print("   5. Use GPU for faster training")


if __name__ == "__main__":
    quick_start_example()