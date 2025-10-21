"""
Robust PIDSE example - gradual complexity increase.

This takes the working simple version and adds back complexity gradually.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add project to path  
sys.path.append('.')
from pidse.data.loaders import create_synthetic_dataset


def create_robust_pidse_config():
    """Create a robust configuration that should work."""
    from pidse.core.pidse import PIDSEConfig
    
    return PIDSEConfig(
        state_dim=4,  # Keep simple: [x, y, vx, vy]
        control_dim=2,  # [ax, ay]
        measurement_dim=4,  # Direct state measurements
        pinn_hidden_layers=[16, 16],  # Small networks
        learning_rate=1e-4,  # Conservative learning rate
        batch_size=2,  # Small batches
        sequence_length=10,  # Short sequences
        physics_weight=0.0,  # Start without physics loss
        regularization_weight=0.0,  # No regularization initially
        initial_process_noise=0.01,  # Reasonable noise
        initial_measurement_noise=0.1,
        learn_noise_matrices=False,  # Fixed noise matrices first
        device='cpu'
    )


def test_robust_pidse():
    """Test PIDSE with robust settings."""
    print("🔧 Testing Robust PIDSE Configuration")
    print("=" * 50)
    
    # Create config
    config = create_robust_pidse_config()
    print(f"✅ Configuration: {config.state_dim}D state, {config.control_dim}D control")
    
    # Create model
    from pidse.core.pidse import PIDSE
    model = PIDSE(config)
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create simple data (similar to working version)
    trajectories = create_synthetic_dataset(
        n_trajectories=3,
        trajectory_length=15,
        state_dim=4,
        control_dim=2,
        measurement_dim=4,
        system_type="generic",  # Use generic instead of complex dynamics
        noise_level=0.01  # Low noise
    )
    print(f"✅ Created {len(trajectories)} simple trajectories")
    
    # Create data loader
    from pidse.data.loaders import create_data_loaders
    train_loader, _, _ = create_data_loaders(
        trajectories, 
        batch_size=config.batch_size,
        sequence_length=config.sequence_length
    )
    print(f"✅ Data loader created: {len(train_loader)} batches")
    
    # Test forward pass
    batch = next(iter(train_loader))
    print(f"✅ Batch loaded: {batch['states'].shape}")
    
    try:
        outputs = model(
            states=batch["states"],
            controls=batch["controls"], 
            measurements=batch["measurements"],
            initial_state=batch["initial_state"],
            initial_covariance=batch["initial_covariance"]
        )
        
        loss = outputs["loss_components"]["total_loss"]
        print(f"✅ Forward pass successful: Loss = {loss.item():.6f}")
        
        # Check for NaN
        if torch.isnan(loss):
            print("❌ Loss is NaN!")
            return False
        else:
            print(f"✅ Loss is finite: {loss.item()}")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test training step
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    try:
        loss_dict = model.train_step(batch, optimizer)
        print(f"✅ Training step successful: {loss_dict}")
        
        # Check training stability
        for key, value in loss_dict.items():
            if np.isnan(value):
                print(f"❌ {key} is NaN!")
                return False
                
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        return False
    
    # Quick training test
    print(f"\n🎯 Quick training test (5 epochs)...")
    losses = []
    
    for epoch in range(5):
        epoch_losses = []
        for batch in train_loader:
            loss_dict = model.train_step(batch, optimizer) 
            epoch_losses.append(loss_dict['total_loss'])
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"   Epoch {epoch}: Loss = {avg_loss:.6f}")
        
        if np.isnan(avg_loss):
            print(f"❌ Training became unstable at epoch {epoch}")
            return False
    
    print("✅ Training completed successfully")
    print(f"✅ Final loss: {losses[-1]:.6f}")
    
    return True


if __name__ == "__main__":
    success = test_robust_pidse()
    
    if success:
        print("\n🎉 Robust PIDSE test successful!")
        print("\nNext steps:")
        print("1. ✅ Core PIDSE functionality verified") 
        print("2. ✅ Training stability confirmed")
        print("3. 🎯 Ready to fix dimension issues in KITTI example")
        print("4. 🎯 Ready to add back physics constraints gradually")
    else:
        print("\n❌ Robust PIDSE test failed")
        print("Need to debug the full implementation further")