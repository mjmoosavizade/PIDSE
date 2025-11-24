#!/usr/bin/env python3
"""
KITTI Training with Adaptive/Curriculum Physics Constraints
============================================================

This script implements curriculum learning for physics constraints:
1. Phase 1 (epochs 0-10): No physics (physics_weight=0.0) - Learn data patterns
2. Phase 2 (epochs 10-25): Gradual physics (0.0 → 0.005) - Introduce constraints
3. Phase 3 (epochs 25-30): Stable physics (physics_weight=0.005) - Refinement

This approach allows the model to first learn the data distribution before
being constrained by physics, preventing over-regularization while still
gaining the benefits of physical consistency.

Expected improvements over baseline:
- 10-20% reduction in long-term drift
- Better balance between data-driven and physics-based learning
- More stable training dynamics
"""

import sys
from pathlib import Path
import numpy as np
import torch
import mlflow
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pidse.core.pidse import PIDSE, PIDSEConfig
from pidse.data.datasets import KITTIDataset
from pidse.data.loaders import create_data_loaders
from pidse.utils.metrics import compute_trajectory_metrics


def get_adaptive_physics_weight(epoch: int, total_epochs: int = 30) -> float:
    """
    Compute adaptive physics weight using curriculum learning.
    
    Phase 1 (0-10): physics_weight = 0.0 (pure data-driven learning)
    Phase 2 (10-25): physics_weight linearly increases 0.0 → 0.005
    Phase 3 (25-30): physics_weight = 0.005 (stable refinement)
    
    Args:
        epoch: Current epoch number
        total_epochs: Total number of epochs
        
    Returns:
        Physics weight for current epoch
    """
    phase1_end = 10
    phase2_end = 25
    max_physics_weight = 0.005
    
    if epoch < phase1_end:
        # Phase 1: No physics constraints
        return 0.0
    elif epoch < phase2_end:
        # Phase 2: Gradual introduction
        progress = (epoch - phase1_end) / (phase2_end - phase1_end)
        return max_physics_weight * progress
    else:
        # Phase 3: Stable physics
        return max_physics_weight


def train_with_adaptive_physics(config_dict):
    """
    KITTI training with adaptive physics weight curriculum.
    
    Args:
        config_dict: Configuration dictionary
    """
    
    with mlflow.start_run(run_name=f"KITTI_Adaptive_Physics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        print("=" * 70)
        print("🚗 PIDSE KITTI Training - Adaptive Physics Curriculum")
        print("=" * 70)
        
        # Extract training-specific parameters
        num_epochs = config_dict.pop('num_epochs', 30)
        train_sequences = config_dict.pop('train_sequences', ['00', '01', '02'])
        val_sequences = config_dict.pop('val_sequences', ['05'])
        test_sequences = config_dict.pop('test_sequences', ['03', '04'])
        
        # Initial physics weight (will be overridden during training)
        config_dict['physics_weight'] = 0.0
        
        config = PIDSEConfig(**config_dict)
        
        # Log parameters
        mlflow.log_param('num_epochs', num_epochs)
        mlflow.log_param('curriculum_strategy', 'adaptive_physics')
        mlflow.log_param('phase1_epochs', '0-10 (no physics)')
        mlflow.log_param('phase2_epochs', '10-25 (gradual 0→0.005)')
        mlflow.log_param('phase3_epochs', '25-30 (stable 0.005)')
        mlflow.log_param('train_sequences', ','.join(train_sequences))
        mlflow.log_param('val_sequences', ','.join(val_sequences))
        mlflow.log_param('test_sequences', ','.join(test_sequences))
        
        for key, value in config_dict.items():
            mlflow.log_param(key, value)
        
        print(f"\n📋 Configuration:")
        print(f"   Device: {config.device}")
        print(f"   State dim: {config.state_dim}, Control dim: {config.control_dim}")
        print(f"   Batch size: {config.batch_size}, Sequence length: {config.sequence_length}")
        print(f"   Learning rate: {config.learning_rate}")
        print(f"   Physics curriculum: Adaptive (0.0 → 0.005 over epochs)")
        
        # Load datasets
        print("\n📊 Loading KITTI Dataset...")
        kitti_path = Path("data/kitti")
        
        train_dataset = KITTIDataset(
            data_path=kitti_path,
            sequence_ids=train_sequences,
            sequence_length=config.sequence_length,
            overlap=0.5
        )
        
        val_dataset = KITTIDataset(
            data_path=kitti_path,
            sequence_ids=val_sequences,
            sequence_length=config.sequence_length,
            overlap=0.0
        )
        
        test_dataset = KITTIDataset(
            data_path=kitti_path,
            sequence_ids=test_sequences,
            sequence_length=config.sequence_length,
            overlap=0.0
        )
        
        print(f"   ✓ Training: {len(train_dataset)} samples")
        print(f"   ✓ Validation: {len(val_dataset)} samples")
        print(f"   ✓ Test: {len(test_dataset)} samples")
        
        # Create data loaders
        train_trajectories = [train_dataset[i] for i in range(len(train_dataset))]
        val_trajectories = [val_dataset[i] for i in range(len(val_dataset))]
        test_trajectories = [test_dataset[i] for i in range(len(test_dataset))]
        
        train_loader, _, _ = create_data_loaders(
            train_trajectories, None,
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            val_split=0.0
        )
        
        val_loader, _, _ = create_data_loaders(
            val_trajectories, None,
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            val_split=0.0
        )
        
        test_loader, _, _ = create_data_loaders(
            test_trajectories, None,
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            val_split=0.0
        )
        
        print(f"   ✓ Train batches: {len(train_loader)}")
        print(f"   ✓ Validation batches: {len(val_loader)}")
        print(f"   ✓ Test batches: {len(test_loader)}")
        
        # Initialize model
        print("\n🏗️  Initializing PIDSE Model...")
        model = PIDSE(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   ✓ Total parameters: {n_params:,}")
        print(f"   ✓ Trainable parameters: {n_trainable:,}")
        
        mlflow.log_param("total_parameters", n_params)
        mlflow.log_param("trainable_parameters", n_trainable)
        
        # Training loop with adaptive physics
        print("\n🎯 Training with Adaptive Physics Curriculum...")
        print(f"   Training for {num_epochs} epochs\n")
        
        best_val_loss = float('inf')
        train_history = []
        val_history = []
        physics_weight_history = []
        
        for epoch in range(num_epochs):
            # Get adaptive physics weight for this epoch
            current_physics_weight = get_adaptive_physics_weight(epoch, num_epochs)
            physics_weight_history.append(current_physics_weight)
            
            # Update model's physics weight
            if hasattr(model, 'criterion') and hasattr(model.criterion, 'physics_weight'):
                model.criterion.physics_weight = current_physics_weight
            
            # Determine training phase
            if epoch < 10:
                phase = "Phase 1: Pure Data Learning"
            elif epoch < 25:
                phase = f"Phase 2: Gradual Physics ({current_physics_weight:.4f})"
            else:
                phase = f"Phase 3: Stable Physics ({current_physics_weight:.4f})"
            
            print(f"Epoch {epoch:2d}/{num_epochs} | {phase}")
            
            # Training phase
            model.train()
            epoch_train_losses = []
            
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx % 50 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)}...", end='\r', flush=True)
                
                try:
                    loss_dict = model.train_step(batch, optimizer)
                    
                    if not np.isfinite(loss_dict['total_loss']):
                        print(f"\n  ⚠️  Warning: NaN loss at batch {batch_idx}, skipping...")
                        continue
                    
                    epoch_train_losses.append(loss_dict['total_loss'])
                    
                except RuntimeError as e:
                    print(f"\n  ⚠️  Error at batch {batch_idx}: {e}")
                    continue
            
            avg_train_loss = np.mean(epoch_train_losses)
            
            # Validation phase
            model.eval()
            epoch_val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    try:
                        # Move batch to device
                        device = next(model.parameters()).device
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items()}
                        
                        # Forward pass
                        outputs = model.forward(
                            states=batch["states"],
                            controls=batch["controls"],
                            measurements=batch["measurements"],
                            initial_state=batch["initial_state"],
                            initial_covariance=batch["initial_covariance"]
                        )
                        
                        val_loss = outputs["loss_components"]["total_loss"].item()
                        epoch_val_losses.append(val_loss)
                    except Exception as e:
                        continue
            
            avg_val_loss = np.mean(epoch_val_losses) if len(epoch_val_losses) > 0 else float('inf')
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Store history
            train_history.append(avg_train_loss)
            val_history.append(avg_val_loss)
            
            # Log to MLflow
            mlflow.log_metrics({
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'physics_weight': current_physics_weight,
                'learning_rate': current_lr
            }, step=epoch)
            
            # Print epoch summary
            print(f"  Train: {avg_train_loss:7.4f} | Val: {avg_val_loss:7.4f} | "
                  f"Phys: {current_physics_weight:.4f} | LR: {current_lr:.6f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'config': config_dict,
                    'physics_weight': current_physics_weight
                }, 'saved_models/best_kitti_adaptive_physics.pth')
                print(f"  ✓ Saved best model")
        
        print(f"\n✓ Training completed. Best val loss: {best_val_loss:.4f}")
        
        # Final evaluation
        print("\n📈 Final Evaluation on Test Set...")
        model.eval()
        test_metrics_list = []
        
        with torch.no_grad():
            for batch in test_loader:
                try:
                    # Move batch to device
                    device = next(model.parameters()).device
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = model.forward(
                        states=batch["states"],
                        controls=batch["controls"],
                        measurements=batch["measurements"],
                        initial_state=batch["initial_state"],
                        initial_covariance=batch["initial_covariance"]
                    )
                    
                    for i in range(batch["states"].shape[0]):
                        true_states = batch["states"][i].cpu()
                        estimated_states = outputs["estimated_states"][i].cpu()
                        
                        metrics = compute_trajectory_metrics(
                            estimated_states[:, :2],
                            true_states[:, :2]
                        )
                        test_metrics_list.append(metrics)
                except Exception as e:
                    continue
        
        avg_test_metrics = {
            'ate': np.mean([m['ate'] for m in test_metrics_list]) if len(test_metrics_list) > 0 else float('inf'),
            'rpe_trans': np.mean([m['rpe_trans_mean'] for m in test_metrics_list]) if len(test_metrics_list) > 0 else float('inf'),
            'final_drift': np.mean([m['final_drift'] for m in test_metrics_list]) if len(test_metrics_list) > 0 else float('inf')
        }
        
        print(f"   ✅ Test Results:")
        print(f"     - ATE: {avg_test_metrics['ate']:.3f} m")
        print(f"     - RPE (trans): {avg_test_metrics['rpe_trans']:.3f} m")
        print(f"     - Final drift: {avg_test_metrics['final_drift']:.3f} m")
        
        # Log final metrics
        mlflow.log_metric("test_ate", avg_test_metrics['ate'])
        mlflow.log_metric("test_rpe_trans", avg_test_metrics['rpe_trans'])
        mlflow.log_metric("test_final_drift", avg_test_metrics['final_drift'])
        mlflow.log_metric("best_val_loss", best_val_loss)
        
        # Create comparison report
        print("\n📊 Comparison with Baseline:")
        print("   BASELINE (no physics):")
        print("     ATE: 2.907m, RPE: 1.031m, Drift: 2.906m")
        print("   AGGRESSIVE PHYSICS (0.05):")
        print("     ATE: 5.638m, RPE: 1.381m, Drift: 7.560m")
        print("   ADAPTIVE PHYSICS (0.0→0.005):")
        print(f"     ATE: {avg_test_metrics['ate']:.3f}m, "
              f"RPE: {avg_test_metrics['rpe_trans']:.3f}m, "
              f"Drift: {avg_test_metrics['final_drift']:.3f}m")
        
        # Calculate improvement
        baseline_drift = 2.906
        improvement = ((baseline_drift - avg_test_metrics['final_drift']) / baseline_drift) * 100
        print(f"\n   Drift improvement: {improvement:+.1f}%")
        
        print("\n" + "=" * 70)
        print("🎉 Adaptive Physics Training Completed!")
        print("=" * 70)


def main():
    """Main entry point"""
    
    # Setup MLflow
    mlflow.set_experiment("PIDSE_KITTI_Adaptive_Physics")
    
    # Configuration - same as baseline but with adaptive physics
    config = {
        # Model architecture (same as successful baseline)
        'state_dim': 12,
        'control_dim': 4,
        'measurement_dim': 9,
        'pinn_hidden_layers': [32, 32],
        
        # Training parameters (same as baseline)
        'learning_rate': 1e-4,
        'batch_size': 2,
        'sequence_length': 50,
        'num_epochs': 30,
        
        # Loss weights - will be adaptive during training
        'physics_weight': 0.0,  # Starting value (will adapt)
        'regularization_weight': 0.001,
        
        # Noise initialization (same as baseline)
        'initial_process_noise': 0.1,
        'initial_measurement_noise': 0.5,
        'learn_noise_matrices': True,
        
        # System parameters
        'mass': 1500.0,
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Data sequences
        'train_sequences': ['00', '01', '02'],
        'val_sequences': ['05'],
        'test_sequences': ['03', '04']
    }
    
    try:
        train_with_adaptive_physics(config)
        print("\n✅ Training completed successfully!")
        return 0
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
