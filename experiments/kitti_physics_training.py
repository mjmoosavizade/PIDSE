#!/usr/bin/env python3
"""
KITTI Training with Physics Constraints Enabled
================================================
This script enables physics constraints (physics_weight=0.05) to reduce drift
and improve long-term trajectory accuracy.

Expected improvements over baseline:
- 15-25% reduction in long-term drift
- More physically consistent predictions
- Better velocity and orientation alignment

Baseline metrics (physics_weight=0.0):
- Training Loss: 8.545
- Validation Loss: 9.492
- ATE (Seq 03): 2.907 m
- ATE (Seq 04): 4.706 m
- RPE (Seq 03): 1.031 m
- RPE (Seq 04): 0.929 m
- Final Drift (Seq 03): 2.906 m
- Final Drift (Seq 04): 7.024 m
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import mlflow
import mlflow.pytorch
from datetime import datetime
import matplotlib.pyplot as plt
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pidse.core.pidse import PIDSE
from pidse.data.datasets import KITTIDataset
from pidse.data.loaders import create_data_loaders
from pidse.losses.hybrid_loss import HybridLoss
from pidse.utils.metrics import compute_trajectory_metrics

def load_kitti_data(data_dir, sequences, sequence_length=50):
    """Load KITTI dataset sequences"""
    dataset = KITTIDataset(
        data_path=data_dir,
        sequence_ids=sequences,
        sequence_length=sequence_length
    )
    return dataset

def compute_metrics_on_sequence(model, dataset, sequence_id, device, config):
    """Compute metrics on a specific test sequence"""
    # Get full trajectory data for the sequence
    model.eval()
    
    with torch.no_grad():
        # Get the sequence data
        seq_data = dataset.get_full_sequence(sequence_id)
        
        if seq_data is None:
            print(f"Warning: Could not load sequence {sequence_id}")
            return None
        
        states = seq_data['states'].to(device)
        controls = seq_data['controls'].to(device)
        measurements = seq_data['measurements'].to(device)
        gt_poses = seq_data['poses'].cpu().numpy()
        
        # Run prediction through the full sequence
        batch_size = states.size(0)
        seq_len = states.size(1)
        
        # Initialize hidden state
        hidden = None
        predicted_states = []
        
        # Process sequence in chunks if needed
        chunk_size = 100
        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)
            
            state_chunk = states[:, i:end_idx, :]
            control_chunk = controls[:, i:end_idx, :]
            meas_chunk = measurements[:, i:end_idx, :]
            
            outputs, hidden = model(
                state_chunk,
                control_chunk,
                meas_chunk,
                hidden=hidden
            )
            
            predicted_states.append(outputs['predicted_state'].cpu())
        
        # Concatenate predictions
        predicted_states = torch.cat(predicted_states, dim=1)
        predicted_states = predicted_states[0].numpy()  # Remove batch dimension
        
        # Extract positions for trajectory comparison
        pred_positions = predicted_states[:, :3]  # [x, y, z]
        gt_positions = gt_poses[:len(pred_positions), :3, 3]
        
        # Compute metrics
        metrics = compute_trajectory_metrics(pred_positions, gt_positions)
        
        # Compute final drift (distance between final positions)
        final_drift = np.linalg.norm(pred_positions[-1] - gt_positions[-1])
        metrics['final_drift'] = final_drift
        
        return {
            'metrics': metrics,
            'predicted_positions': pred_positions,
            'gt_positions': gt_positions,
            'predicted_states': predicted_states,
            'gt_poses': gt_poses
        }

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_meas_loss = 0
    total_physics_loss = 0
    total_reg_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        states = batch['states'].to(device)
        controls = batch['controls'].to(device)
        measurements = batch['measurements'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(states, controls, measurements)
        
        # Compute loss
        loss_dict = criterion(outputs, measurements, controls)
        loss = loss_dict['total_loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_meas_loss += loss_dict['measurement_loss'].item()
        total_physics_loss += loss_dict['physics_loss'].item()
        total_reg_loss += loss_dict['regularization_loss'].item()
        num_batches += 1
        
        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"(Meas: {loss_dict['measurement_loss'].item():.4f}, "
                  f"Phys: {loss_dict['physics_loss'].item():.4f}, "
                  f"Reg: {loss_dict['regularization_loss'].item():.4f})")
    
    return {
        'total_loss': total_loss / num_batches,
        'measurement_loss': total_meas_loss / num_batches,
        'physics_loss': total_physics_loss / num_batches,
        'regularization_loss': total_reg_loss / num_batches
    }

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    total_meas_loss = 0
    total_physics_loss = 0
    total_reg_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            states = batch['states'].to(device)
            controls = batch['controls'].to(device)
            measurements = batch['measurements'].to(device)
            
            # Forward pass
            outputs = model(states, controls, measurements)
            
            # Compute loss
            loss_dict = criterion(outputs, measurements, controls)
            
            # Accumulate losses
            total_loss += loss_dict['total_loss'].item()
            total_meas_loss += loss_dict['measurement_loss'].item()
            total_physics_loss += loss_dict['physics_loss'].item()
            total_reg_loss += loss_dict['regularization_loss'].item()
            num_batches += 1
    
    return {
        'total_loss': total_loss / num_batches,
        'measurement_loss': total_meas_loss / num_batches,
        'physics_loss': total_physics_loss / num_batches,
        'regularization_loss': total_reg_loss / num_batches
    }

def main():
    # Configuration
    config = {
        # Data paths
        'data_dir': '/home/mj/PIDSE/data/kitti/dataset',
        'train_sequences': ['00', '01', '02'],
        'val_sequences': ['05'],
        'test_sequences': ['03', '04'],
        
        # Model architecture - FROM SUCCESSFUL RUN
        'state_dim': 12,  # [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        'control_dim': 4,  # [ax, ay, az, alpha_z]
        'measurement_dim': 9,  # [x, y, z, vx, vy, vz, roll, pitch, yaw]
        'hidden_dim': 64,
        'num_layers': 3,
        
        # Training parameters - SAME AS BASELINE
        'learning_rate': 1e-4,
        'batch_size': 2,
        'sequence_length': 50,
        'num_epochs': 30,
        
        # Loss weights - ENABLE PHYSICS CONSTRAINTS
        'physics_weight': 0.05,  # Enable physics constraints (was 0.0)
        'regularization_weight': 0.001,
        
        # Noise initialization - SAME AS BASELINE
        'initial_process_noise': 0.1,
        'initial_measurement_noise': 0.5,
        'learn_noise_matrices': True,
        
        # System parameters
        'mass': 1500.0,
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Experiment name
        'experiment_name': f'kitti_physics_enabled_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }
    
    print("="*80)
    print("KITTI TRAINING WITH PHYSICS CONSTRAINTS ENABLED")
    print("="*80)
    print(f"Device: {config['device']}")
    print(f"Physics Weight: {config['physics_weight']} (ENABLED)")
    print(f"Training sequences: {config['train_sequences']}")
    print(f"Validation sequences: {config['val_sequences']}")
    print(f"Test sequences: {config['test_sequences']}")
    print("="*80)
    
    # Setup MLflow
    mlflow.set_experiment("kitti_physics_training")
    
    with mlflow.start_run(run_name=config['experiment_name']) as run:
        # Log configuration
        mlflow.log_params(config)
        
        print("\n[1/6] Loading datasets...")
        # Load training data
        train_dataset = load_kitti_data(
            config['data_dir'],
            config['train_sequences'],
            config['sequence_length']
        )
        
        val_dataset = load_kitti_data(
            config['data_dir'],
            config['val_sequences'],
            config['sequence_length']
        )
        
        # Create dataloaders
        train_loader, val_loader, _ = create_data_loaders(
            train_dataset,
            val_dataset,
            batch_size=config['batch_size'],
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Batches per epoch: {len(train_loader)}")
        
        # Load test datasets (for evaluation)
        test_datasets = {}
        for seq in config['test_sequences']:
            test_datasets[seq] = load_kitti_data(
                config['data_dir'],
                [seq],
                config['sequence_length']
            )
        
        print("\n[2/6] Initializing model...")
        # Initialize model
        model = PIDSE(
            state_dim=config['state_dim'],
            control_dim=config['control_dim'],
            measurement_dim=config['measurement_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            initial_process_noise=config['initial_process_noise'],
            initial_measurement_noise=config['initial_measurement_noise'],
            learn_noise_matrices=config['learn_noise_matrices']
        ).to(config['device'])
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {num_params:,}")
        mlflow.log_param("num_parameters", num_params)
        
        print("\n[3/6] Setting up training...")
        # Initialize loss criterion with physics enabled
        criterion = HybridLoss(
            physics_weight=config['physics_weight'],
            regularization_weight=config['regularization_weight']
        )
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        print(f"  Optimizer: Adam (lr={config['learning_rate']})")
        print(f"  Loss: Hybrid Loss (physics_weight={config['physics_weight']})")
        print(f"  Scheduler: ReduceLROnPlateau (patience=5)")
        
        print("\n[4/6] Training model...")
        print("="*80)
        
        # Training loop
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        train_physics_losses = []
        val_physics_losses = []
        
        for epoch in range(config['num_epochs']):
            print(f"\nEpoch [{epoch+1}/{config['num_epochs']}]")
            
            # Train
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer,
                config['device'], epoch, config['num_epochs']
            )
            
            # Validate
            val_metrics = validate_epoch(
                model, val_loader, criterion, config['device']
            )
            
            # Update learning rate
            scheduler.step(val_metrics['total_loss'])
            
            # Store losses
            train_losses.append(train_metrics['total_loss'])
            val_losses.append(val_metrics['total_loss'])
            train_physics_losses.append(train_metrics['physics_loss'])
            val_physics_losses.append(val_metrics['physics_loss'])
            
            # Log to MLflow
            mlflow.log_metrics({
                'train_loss': train_metrics['total_loss'],
                'train_measurement_loss': train_metrics['measurement_loss'],
                'train_physics_loss': train_metrics['physics_loss'],
                'train_regularization_loss': train_metrics['regularization_loss'],
                'val_loss': val_metrics['total_loss'],
                'val_measurement_loss': val_metrics['measurement_loss'],
                'val_physics_loss': val_metrics['physics_loss'],
                'val_regularization_loss': val_metrics['regularization_loss'],
            }, step=epoch)
            
            # Print epoch summary
            print(f"  Training   - Total: {train_metrics['total_loss']:.4f}, "
                  f"Meas: {train_metrics['measurement_loss']:.4f}, "
                  f"Phys: {train_metrics['physics_loss']:.4f}, "
                  f"Reg: {train_metrics['regularization_loss']:.4f}")
            print(f"  Validation - Total: {val_metrics['total_loss']:.4f}, "
                  f"Meas: {val_metrics['measurement_loss']:.4f}, "
                  f"Phys: {val_metrics['physics_loss']:.4f}, "
                  f"Reg: {val_metrics['regularization_loss']:.4f}")
            
            # Save best model
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                save_path = '/home/mj/PIDSE/saved_models/best_kitti_physics_model.pth'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'config': config
                }, save_path)
                print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
                mlflow.log_artifact(save_path)
        
        print("\n" + "="*80)
        print("Training completed!")
        print("="*80)
        
        print("\n[5/6] Evaluating on test sequences...")
        # Load best model
        checkpoint = torch.load('/home/mj/PIDSE/saved_models/best_kitti_physics_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_results = {}
        for seq_id in config['test_sequences']:
            print(f"\nSequence {seq_id}:")
            result = compute_metrics_on_sequence(
                model, test_datasets[seq_id], seq_id,
                config['device'], config
            )
            
            if result is not None:
                metrics = result['metrics']
                test_results[seq_id] = result
                
                print(f"  ATE: {metrics['ate']:.3f} m")
                print(f"  RPE: {metrics['rpe']:.3f} m")
                print(f"  Final Drift: {metrics['final_drift']:.3f} m")
                
                # Log to MLflow
                mlflow.log_metrics({
                    f'test_ate_seq{seq_id}': metrics['ate'],
                    f'test_rpe_seq{seq_id}': metrics['rpe'],
                    f'test_final_drift_seq{seq_id}': metrics['final_drift']
                })
        
        print("\n[6/6] Saving results and visualizations...")
        
        # Plot training curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Total loss
        axes[0, 0].plot(train_losses, label='Training', linewidth=2)
        axes[0, 0].plot(val_losses, label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Total Loss over Training')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Physics loss
        axes[0, 1].plot(train_physics_losses, label='Training Physics', linewidth=2)
        axes[0, 1].plot(val_physics_losses, label='Validation Physics', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Physics Loss')
        axes[0, 1].set_title('Physics Loss over Training (ENABLED)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Test sequence results - Sequence 03
        if '03' in test_results:
            seq_03 = test_results['03']
            axes[1, 0].plot(seq_03['gt_positions'][:, 0], seq_03['gt_positions'][:, 1],
                          'b-', label='Ground Truth', linewidth=2, alpha=0.7)
            axes[1, 0].plot(seq_03['predicted_positions'][:, 0], seq_03['predicted_positions'][:, 1],
                          'r--', label='Predicted', linewidth=2, alpha=0.7)
            axes[1, 0].set_xlabel('X (m)')
            axes[1, 0].set_ylabel('Y (m)')
            axes[1, 0].set_title(f"Sequence 03 (ATE: {seq_03['metrics']['ate']:.3f}m)")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axis('equal')
        
        # Test sequence results - Sequence 04
        if '04' in test_results:
            seq_04 = test_results['04']
            axes[1, 1].plot(seq_04['gt_positions'][:, 0], seq_04['gt_positions'][:, 1],
                          'b-', label='Ground Truth', linewidth=2, alpha=0.7)
            axes[1, 1].plot(seq_04['predicted_positions'][:, 0], seq_04['predicted_positions'][:, 1],
                          'r--', label='Predicted', linewidth=2, alpha=0.7)
            axes[1, 1].set_xlabel('X (m)')
            axes[1, 1].set_ylabel('Y (m)')
            axes[1, 1].set_title(f"Sequence 04 (ATE: {seq_04['metrics']['ate']:.3f}m)")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axis('equal')
        
        plt.tight_layout()
        results_path = '/home/mj/PIDSE/experiments/kitti_physics_results.png'
        plt.savefig(results_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(results_path)
        print(f"  Saved results plot: {results_path}")
        plt.close()
        
        # Save comparison with baseline
        comparison_path = '/home/mj/PIDSE/experiments/physics_comparison.txt'
        with open(comparison_path, 'w') as f:
            f.write("Physics Constraints Comparison\n")
            f.write("="*80 + "\n\n")
            f.write("BASELINE (physics_weight=0.0):\n")
            f.write("  Sequence 03: ATE=2.907m, RPE=1.031m, Drift=2.906m\n")
            f.write("  Sequence 04: ATE=4.706m, RPE=0.929m, Drift=7.024m\n\n")
            f.write("PHYSICS ENABLED (physics_weight=0.05):\n")
            if '03' in test_results:
                m = test_results['03']['metrics']
                f.write(f"  Sequence 03: ATE={m['ate']:.3f}m, RPE={m['rpe']:.3f}m, Drift={m['final_drift']:.3f}m\n")
            if '04' in test_results:
                m = test_results['04']['metrics']
                f.write(f"  Sequence 04: ATE={m['ate']:.3f}m, RPE={m['rpe']:.3f}m, Drift={m['final_drift']:.3f}m\n")
            f.write("\nIMPROVEMENT:\n")
            if '03' in test_results:
                m = test_results['03']['metrics']
                drift_improve = ((2.906 - m['final_drift']) / 2.906) * 100
                f.write(f"  Sequence 03 Drift: {drift_improve:+.1f}%\n")
            if '04' in test_results:
                m = test_results['04']['metrics']
                drift_improve = ((7.024 - m['final_drift']) / 7.024) * 100
                f.write(f"  Sequence 04 Drift: {drift_improve:+.1f}%\n")
        
        mlflow.log_artifact(comparison_path)
        print(f"  Saved comparison: {comparison_path}")
        
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETE")
        print("="*80)
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Model saved: /home/mj/PIDSE/saved_models/best_kitti_physics_model.pth")
        print("="*80)

if __name__ == "__main__":
    main()
