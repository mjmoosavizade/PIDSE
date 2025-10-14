"""
Test script to verify PIDSE implementation.

This script performs basic functionality tests for all major components
of the PIDSE framework.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pidse import PIDSE, PIDSEConfig
from pidse.data import create_synthetic_dataset, create_data_loaders
from pidse.utils.metrics import compute_trajectory_metrics


def test_basic_functionality():
    """Test basic PIDSE functionality."""
    print("🧪 Testing basic PIDSE functionality...")
    
    # Create configuration
    config = PIDSEConfig(
        state_dim=12,
        control_dim=4,
        measurement_dim=9,
        pinn_hidden_layers=[32, 32],
        learning_rate=1e-3,
        batch_size=4,
        sequence_length=20,
        physics_weight=0.1,
        regularization_weight=0.01,
        device='cpu'  # Use CPU for testing
    )
    
    print(f"✓ Configuration created: {config}")
    
    # Initialize PIDSE model
    model = PIDSE(config)
    print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    batch_size = config.batch_size
    seq_len = config.sequence_length
    
    # Create dummy data
    states = torch.randn(batch_size, seq_len, config.state_dim)
    controls = torch.randn(batch_size, seq_len, config.control_dim)
    measurements = torch.randn(batch_size, seq_len, config.measurement_dim)
    initial_state = torch.randn(batch_size, config.state_dim)
    initial_covariance = torch.eye(config.state_dim).unsqueeze(0).repeat(batch_size, 1, 1)
    
    print("✓ Test data created")
    
    # Forward pass
    try:
        outputs = model(
            states=states,
            controls=controls,
            measurements=measurements,
            initial_state=initial_state,
            initial_covariance=initial_covariance
        )
        print("✓ Forward pass successful")
        print(f"  - Estimated states shape: {outputs['estimated_states'].shape}")
        print(f"  - Predicted measurements shape: {outputs['predicted_measurements'].shape}")
        print(f"  - Covariances shape: {outputs['covariances'].shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    # Test loss computation
    try:
        loss_components = outputs['loss_components']
        total_loss = loss_components['total_loss']
        print(f"✓ Loss computation successful: {total_loss.item():.4f}")
        print(f"  - Estimation loss: {loss_components['estimation_loss'].item():.4f}")
        print(f"  - Physics loss: {loss_components['physics_loss'].item():.4f}")
        print(f"  - Regularization loss: {loss_components['regularization_loss'].item():.4f}")
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        return False
    
    return True


def test_data_handling():
    """Test data handling functionality."""
    print("\n📊 Testing data handling...")
    
    # Create synthetic dataset
    try:
        trajectories = create_synthetic_dataset(
            n_trajectories=10,
            trajectory_length=50,
            system_type="quadrotor",
            noise_level=0.1
        )
        print(f"✓ Synthetic dataset created: {len(trajectories)} trajectories")
        
        # Check trajectory structure
        traj = trajectories[0]
        print(f"  - States shape: {traj['states'].shape}")
        print(f"  - Controls shape: {traj['controls'].shape}")
        print(f"  - Measurements shape: {traj['measurements'].shape}")
        
    except Exception as e:
        print(f"✗ Synthetic dataset creation failed: {e}")
        return False
    
    # Create data loaders
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            trajectories[:8], trajectories[8:],
            batch_size=4, sequence_length=20
        )
        print(f"✓ Data loaders created")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Val batches: {len(val_loader)}")
        print(f"  - Test batches: {len(test_loader)}")
        
        # Test batch loading
        batch = next(iter(train_loader))
        print(f"  - Batch keys: {list(batch.keys())}")
        print(f"  - Batch states shape: {batch['states'].shape}")
        
    except Exception as e:
        print(f"✗ Data loader creation failed: {e}")
        return False
    
    return True


def test_training_step():
    """Test training step functionality."""
    print("\n🎯 Testing training step...")
    
    # Create model and data
    config = PIDSEConfig(
        state_dim=12, control_dim=4, measurement_dim=9,
        pinn_hidden_layers=[16, 16], batch_size=2, sequence_length=10,
        device='cpu'
    )
    
    model = PIDSE(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create synthetic data
    trajectories = create_synthetic_dataset(
        n_trajectories=4, trajectory_length=20, system_type="quadrotor"
    )
    
    train_loader, _, _ = create_data_loaders(
        trajectories, batch_size=2, sequence_length=10
    )
    
    # Test training step
    try:
        batch = next(iter(train_loader))
        loss_dict = model.train_step(batch, optimizer)
        
        print("✓ Training step successful")
        print(f"  - Total loss: {loss_dict['total_loss']:.4f}")
        print(f"  - Loss components: {list(loss_dict.keys())}")
        
        # Check that parameters are updated
        param_sum_before = sum(p.sum().item() for p in model.parameters())
        loss_dict = model.train_step(batch, optimizer)
        param_sum_after = sum(p.sum().item() for p in model.parameters())
        
        if abs(param_sum_before - param_sum_after) > 1e-6:
            print("✓ Parameters updated during training")
        else:
            print("⚠ Parameters may not be updating")
        
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        return False
    
    return True


def test_evaluation_metrics():
    """Test evaluation metrics."""
    print("\n📈 Testing evaluation metrics...")
    
    try:
        # Create dummy trajectory data
        seq_len = 50
        predicted_states = torch.randn(seq_len, 12)
        true_states = predicted_states + 0.1 * torch.randn(seq_len, 12)  # Add small error
        
        # Compute metrics
        metrics = compute_trajectory_metrics(predicted_states, true_states)
        
        print("✓ Trajectory metrics computed")
        print(f"  - ATE: {metrics['ate']:.4f}")
        print(f"  - RPE translation mean: {metrics['rpe_trans_mean']:.4f}")
        print(f"  - Final drift: {metrics['final_drift']:.4f}")
        print(f"  - Available metrics: {list(metrics.keys())}")
        
    except Exception as e:
        print(f"✗ Evaluation metrics failed: {e}")
        return False
    
    return True


def test_model_serialization():
    """Test model saving and loading."""
    print("\n💾 Testing model serialization...")
    
    try:
        # Create and train model briefly
        config = PIDSEConfig(
            state_dim=6, control_dim=2, measurement_dim=4,
            pinn_hidden_layers=[8], device='cpu'
        )
        
        model = PIDSE(config)
        
        # Save checkpoint
        checkpoint_path = "/tmp/test_pidse_checkpoint.pth"
        model.save_checkpoint(checkpoint_path)
        print("✓ Model checkpoint saved")
        
        # Create new model and load checkpoint
        new_model = PIDSE(config)
        checkpoint = new_model.load_checkpoint(checkpoint_path)
        print("✓ Model checkpoint loaded")
        print(f"  - Checkpoint keys: {list(checkpoint.keys())}")
        
        # Clean up
        Path(checkpoint_path).unlink(missing_ok=True)
        
    except Exception as e:
        print(f"✗ Model serialization failed: {e}")
        return False
    
    return True


def test_component_integration():
    """Test integration between different components."""
    print("\n🔗 Testing component integration...")
    
    try:
        config = PIDSEConfig(device='cpu')
        model = PIDSE(config)
        
        # Test individual components
        state = torch.randn(1, config.state_dim)
        control = torch.randn(1, config.control_dim)
        
        # Test dynamics network
        dynamics_residual = model.dynamics_network(state, control)
        print(f"✓ Dynamics network output shape: {dynamics_residual.shape}")
        
        # Test measurement network
        measurement_residual = model.measurement_network(state)
        print(f"✓ Measurement network output shape: {measurement_residual.shape}")
        
        # Test state space model
        known_dynamics = model.state_space.dynamics_step(state, control)
        print(f"✓ State space dynamics output shape: {known_dynamics.shape}")
        
        # Test EKF components
        Q = model.ekf.Q_matrix
        R = model.ekf.R_matrix
        print(f"✓ EKF noise matrices - Q: {Q.shape}, R: {R.shape}")
        
        # Check that noise matrices are positive definite
        Q_eigenvals = torch.linalg.eigvals(Q).real
        R_eigenvals = torch.linalg.eigvals(R).real
        
        if torch.all(Q_eigenvals > 0) and torch.all(R_eigenvals > 0):
            print("✓ Noise matrices are positive definite")
        else:
            print("⚠ Noise matrices may not be positive definite")
        
    except Exception as e:
        print(f"✗ Component integration failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("🚀 PIDSE Implementation Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Data Handling", test_data_handling),
        ("Training Step", test_training_step),
        ("Evaluation Metrics", test_evaluation_metrics),
        ("Model Serialization", test_model_serialization),
        ("Component Integration", test_component_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * 30)
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! PIDSE implementation is working correctly.")
        return 0
    else:
        print("⚠ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit(main())