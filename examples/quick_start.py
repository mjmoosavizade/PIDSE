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
    """
    Quick start example showing PIDSE basics.
    
    This example demonstrates:
    1. Creating a PIDSE model
    2. Generating synthetic data
    3. Training the model
    4. Evaluating performance
    """
    
    print("🚀 PIDSE Quick Start Example")
    print("=" * 40)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Configure PIDSE for simplified dynamics (more stable)
    print("\n� Step 1: Configure PIDSE")
    config = PIDSEConfig(
        state_dim=4,           # [px, py, vx, vy] - simplified 2D dynamics  
        control_dim=2,         # [ax, ay] - acceleration commands
        measurement_dim=4,     # Direct state measurements with noise
        pinn_hidden_layers=[32, 32],  # Smaller networks for stability
        learning_rate=1e-4,    # Lower learning rate
        batch_size=4,          # Smaller batches
        sequence_length=20,    # Shorter sequences  
        physics_weight=0.0,    # Start without physics constraints
        regularization_weight=0.0,  # No regularization initially
        initial_process_noise=0.01,
        initial_measurement_noise=0.1,
        learn_noise_matrices=False,  # Fixed noise for stability
        device=device
    )
    print(f"   ✓ Configuration created for {config.device}")
    
    # 2. Generate synthetic data
    print("\n📊 Step 2: Generate synthetic quadrotor data")
    train_data = create_synthetic_dataset(
        n_trajectories=15,     # Fewer trajectories
        trajectory_length=50,  # Shorter sequences
        state_dim=4,           # Match configuration
        control_dim=2,
        measurement_dim=4,
        system_type="generic", # Use stable generic dynamics
        noise_level=0.05       # Lower noise
    )
    
    test_data = create_synthetic_dataset(
        n_trajectories=5,
        trajectory_length=50,
        state_dim=4,
        control_dim=2,
        measurement_dim=4,
        system_type="generic",
        noise_level=0.05
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
    
    # Move batch to device
    device = next(model.parameters()).device
    test_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                  for k, v in test_batch.items()}
    
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
    
    # Extract positions (use first 3 dimensions, or less if not available)
    n_pos_dims = min(3, true_states.shape[1])
    true_pos = true_states[:, 0:n_pos_dims].cpu().numpy()
    est_pos = estimated_states[:, 0:n_pos_dims].cpu().numpy()
    
    # 3D trajectory (if we have 3+ dimensions)
    if n_pos_dims >= 3:
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        ax.plot(true_pos[:, 0], true_pos[:, 1], true_pos[:, 2], 'b-', label='True', linewidth=2)
        ax.plot(est_pos[:, 0], est_pos[:, 1], est_pos[:, 2], 'r--', label='Estimated', linewidth=2)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Trajectory')
        ax.legend()
    else:
        # 2D trajectory if only 2 dimensions
        axes[0, 0].plot(true_pos[:, 0], true_pos[:, 1] if n_pos_dims > 1 else true_pos[:, 0], 
                       'b-', label='True', linewidth=2)
        axes[0, 0].plot(est_pos[:, 0], est_pos[:, 1] if n_pos_dims > 1 else est_pos[:, 0], 
                       'r--', label='Estimated', linewidth=2)
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y' if n_pos_dims > 1 else 'X')
        axes[0, 0].set_title('2D Trajectory')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # Position errors (use available dimensions)
    pos_errors = torch.norm(estimated_states[:, 0:n_pos_dims] - true_states[:, 0:n_pos_dims], dim=1).cpu().numpy()
    axes[0, 1].plot(pos_errors)
    axes[0, 1].set_title('Position Error Over Time')
    axes[0, 1].set_ylabel('Error (m)')
    axes[0, 1].set_xlabel('Time step')
    axes[0, 1].grid(True)
    
    # Velocity comparison (if state has velocity)
    if true_states.shape[1] >= 6:
        true_vel = true_states[:, 3:6]
        est_vel = estimated_states[:, 3:6]
        
        for i, label in enumerate(['vx', 'vy', 'vz']):
            axes[1, 0].plot(true_vel[:, i].cpu().numpy(), label=f'True {label}', linestyle='-')
            axes[1, 0].plot(est_vel[:, i].cpu().numpy(), label=f'Est {label}', linestyle='--')
        
        axes[1, 0].set_title('Velocity Components')
        axes[1, 0].set_ylabel('Velocity (m/s)')
    else:
        # Plot all state components if velocity not available
        for i in range(min(true_states.shape[1], 4)):
            axes[1, 0].plot(true_states[:, i].cpu().numpy(), label=f'True s{i}', linestyle='-')
            axes[1, 0].plot(estimated_states[:, i].cpu().numpy(), label=f'Est s{i}', linestyle='--')
        
        axes[1, 0].set_title('State Components')
        axes[1, 0].set_ylabel('State value')
    
    axes[1, 0].set_xlabel('Time step')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learned noise matrices
    Q_learned = model.ekf.Q_matrix.detach().cpu().numpy()
    R_learned = model.ekf.R_matrix.detach().cpu().numpy()
    
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
    print(f"   • Learned Q diagonal: {torch.diag(model.ekf.Q_matrix).detach().cpu().numpy()}")
    print(f"   • Learned R diagonal: {torch.diag(model.ekf.R_matrix).detach().cpu().numpy()}")
    
    print("\n🚀 Next Steps:")
    print("   1. Try with real motion capture data")
    print("   2. Experiment with different network architectures")
    print("   3. Adjust physics and regularization weights")
    print("   4. Train for more epochs with larger datasets")
    print("   5. Use GPU for faster training")


if __name__ == "__main__":
    quick_start_example()