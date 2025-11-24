"""
Full KITTI Training with MLflow Experiment Tracking

This script trains PIDSE on KITTI dataset with comprehensive
experiment tracking, metrics logging, and visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import mlflow
import mlflow.pytorch
from datetime import datetime
import json
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pidse import PIDSE, PIDSEConfig
from pidse.data import KITTIDataset, create_data_loaders
from pidse.utils.metrics import compute_trajectory_metrics


def setup_mlflow(experiment_name="PIDSE_KITTI_Training"):
    """Setup MLflow experiment tracking"""
    # Set tracking directory
    mlflow_dir = Path("mlruns")
    mlflow_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_dir.absolute()}")
    
    # Create or get experiment
    experiment = mlflow.set_experiment(experiment_name)
    print(f"📊 MLflow Experiment: {experiment_name}")
    print(f"   Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"   Experiment ID: {experiment.experiment_id}")
    
    return experiment


def log_system_info():
    """Log system and environment information"""
    mlflow.log_param("python_version", sys.version.split()[0])
    mlflow.log_param("pytorch_version", torch.__version__)
    mlflow.log_param("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        mlflow.log_param("cuda_version", torch.version.cuda)
        mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))
        mlflow.log_param("gpu_memory_gb", torch.cuda.get_device_properties(0).total_memory / 1e9)


def train_kitti_full(config_dict):
    """
    Full KITTI training run with MLflow tracking
    
    Args:
        config_dict: Dictionary of configuration parameters
    """
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"KITTI_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        print("=" * 70)
        print("🚗 PIDSE KITTI Full Training with MLflow Tracking")
        print("=" * 70)
        
        # Log system info
        log_system_info()
        
        # 1. Configuration
        print("\n🔧 Step 1: Setup Configuration")
        
        # Extract training-specific parameters
        num_epochs = config_dict.pop('num_epochs', 50)
        train_sequences = config_dict.pop('train_sequences', ['00', '01', '02'])
        val_sequences = config_dict.pop('val_sequences', ['05'])
        test_sequences = config_dict.pop('test_sequences', ['03', '04'])
        
        config = PIDSEConfig(**config_dict)
        
        # Log all configuration parameters
        mlflow.log_param('num_epochs', num_epochs)
        mlflow.log_param('train_sequences', ','.join(train_sequences))
        mlflow.log_param('val_sequences', ','.join(val_sequences))
        mlflow.log_param('test_sequences', ','.join(test_sequences))
        
        for key, value in config_dict.items():
            mlflow.log_param(key, value)
        
        print(f"   Device: {config.device}")
        print(f"   State dim: {config.state_dim}, Control dim: {config.control_dim}")
        print(f"   Batch size: {config.batch_size}, Sequence length: {config.sequence_length}")
        print(f"   Learning rate: {config.learning_rate}")
        
        # 2. Load KITTI dataset
        print("\n📊 Step 2: Load KITTI Dataset")
        kitti_path = Path("data/kitti")
        
        if not kitti_path.exists():
            raise FileNotFoundError("KITTI data not found! Run: bash scripts/download_kitti.sh")
        
        # Use sequences extracted earlier
        
        try:
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
            
            print(f"   ✓ Training sequences: {len(train_dataset)} samples")
            print(f"   ✓ Validation sequences: {len(val_dataset)} samples")
            print(f"   ✓ Test sequences: {len(test_dataset)} samples")
            
            mlflow.log_metric("train_dataset_size", len(train_dataset))
            mlflow.log_metric("val_dataset_size", len(val_dataset))
            mlflow.log_metric("test_dataset_size", len(test_dataset))
            
        except Exception as e:
            print(f"   ❌ Failed to load KITTI data: {e}")
            raise
        
        # 3. Create data loaders
        print("\n🔄 Step 3: Create Data Loaders")
        
        # Convert datasets to trajectory list format
        train_trajectories = [train_dataset[i] for i in range(len(train_dataset))]
        val_trajectories = [val_dataset[i] for i in range(len(val_dataset))]
        test_trajectories = [test_dataset[i] for i in range(len(test_dataset))]
        
        train_loader, _, _ = create_data_loaders(
            train_trajectories, None,
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            val_split=0.0  # We have separate validation set
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
        
        # 4. Initialize model
        print("\n🏗️  Step 4: Initialize PIDSE Model")
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
        
        # 5. Training loop
        print("\n🎯 Step 5: Training Loop")
        print(f"   Training for {num_epochs} epochs...")
        best_val_loss = float('inf')
        train_history = []
        val_history = []
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            epoch_train_losses = {
                'total': [], 'estimation': [], 'physics': [], 'regularization': []
            }
            
            for batch_idx, batch in enumerate(train_loader):
                # Progress indicator every 20 batches
                if batch_idx % 20 == 0:
                    print(f"     Batch {batch_idx}/{len(train_loader)}...", flush=True)
                
                # Training step
                try:
                    loss_dict = model.train_step(batch, optimizer)
                    
                    # Check for NaN
                    if not np.isfinite(loss_dict['total_loss']):
                        print(f"   ⚠️  Warning: NaN/Inf loss at batch {batch_idx}, skipping...")
                        continue
                    
                    # Accumulate losses
                    epoch_train_losses['total'].append(loss_dict['total_loss'])
                    epoch_train_losses['estimation'].append(loss_dict.get('estimation_loss', 0))
                    epoch_train_losses['physics'].append(loss_dict.get('physics_loss', 0))
                    epoch_train_losses['regularization'].append(loss_dict.get('regularization_loss', 0))
                    
                except RuntimeError as e:
                    print(f"   ⚠️  Error at batch {batch_idx}: {e}")
                    continue
            
            # Average training losses
            avg_train_loss = np.mean(epoch_train_losses['total'])
            avg_est_loss = np.mean(epoch_train_losses['estimation'])
            avg_phys_loss = np.mean(epoch_train_losses['physics'])
            avg_reg_loss = np.mean(epoch_train_losses['regularization'])
            
            train_history.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            epoch_val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
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
            
            avg_val_loss = np.mean(epoch_val_losses)
            val_history.append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log metrics to MLflow
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("train_estimation_loss", avg_est_loss, step=epoch)
            mlflow.log_metric("train_physics_loss", avg_phys_loss, step=epoch)
            mlflow.log_metric("train_regularization_loss", avg_reg_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("learning_rate", current_lr, step=epoch)
            
            # Print progress
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                print(f"   Epoch {epoch:3d}/{num_epochs}: "
                      f"Train Loss = {avg_train_loss:.4f}, "
                      f"Val Loss = {avg_val_loss:.4f}, "
                      f"LR = {current_lr:.6f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = "best_kitti_model.pth"
                model.save_checkpoint(best_model_path)
                mlflow.log_metric("best_val_loss", best_val_loss)
                mlflow.log_metric("best_epoch", epoch)
        
        print(f"   ✓ Training completed. Best val loss: {best_val_loss:.4f}")
        
        # 6. Final evaluation on test set
        print("\n📈 Step 6: Final Evaluation on Test Set")
        
        model.eval()
        test_metrics_list = []
        
        with torch.no_grad():
            for batch in test_loader:
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
                
                # Compute metrics for each trajectory in batch
                for i in range(batch["states"].shape[0]):
                    true_states = batch["states"][i].cpu()
                    estimated_states = outputs["estimated_states"][i].cpu()
                    
                    # Compute trajectory metrics (position only for vehicle)
                    metrics = compute_trajectory_metrics(
                        estimated_states[:, :2],  # px, py
                        true_states[:, :2]
                    )
                    test_metrics_list.append(metrics)
        
        # Average test metrics
        avg_test_metrics = {
            'ate': np.mean([m['ate'] for m in test_metrics_list]),
            'rpe_trans': np.mean([m['rpe_trans_mean'] for m in test_metrics_list]),
            'final_drift': np.mean([m['final_drift'] for m in test_metrics_list])
        }
        
        print(f"   ✅ Test Results:")
        print(f"     - ATE: {avg_test_metrics['ate']:.3f} m")
        print(f"     - RPE (trans): {avg_test_metrics['rpe_trans']:.3f} m")
        print(f"     - Final drift: {avg_test_metrics['final_drift']:.3f} m")
        
        # Log test metrics
        mlflow.log_metric("test_ate", avg_test_metrics['ate'])
        mlflow.log_metric("test_rpe_trans", avg_test_metrics['rpe_trans'])
        mlflow.log_metric("test_final_drift", avg_test_metrics['final_drift'])
        
        # 7. Visualizations
        print("\n📊 Step 7: Create Visualizations")
        
        # Training curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(train_history, label='Train Loss', linewidth=2)
        axes[0].plot(val_history, label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Get one test batch for trajectory visualization
        test_batch = next(iter(test_loader))
        device = next(model.parameters()).device
        test_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                      for k, v in test_batch.items()}
        
        with torch.no_grad():
            outputs = model.forward(
                states=test_batch["states"],
                controls=test_batch["controls"],
                measurements=test_batch["measurements"],
                initial_state=test_batch["initial_state"],
                initial_covariance=test_batch["initial_covariance"]
            )
        
        true_pos = test_batch["states"][0, :, :2].cpu().numpy()
        est_pos = outputs["estimated_states"][0, :, :2].cpu().numpy()
        
        axes[1].plot(true_pos[:, 0], true_pos[:, 1], 'b-', label='Ground Truth', linewidth=2)
        axes[1].plot(est_pos[:, 0], est_pos[:, 1], 'r--', label='PIDSE Estimate', linewidth=2)
        axes[1].set_xlabel('X Position (m)')
        axes[1].set_ylabel('Y Position (m)')
        axes[1].set_title('KITTI Test Trajectory')
        axes[1].legend()
        axes[1].grid(True)
        axes[1].axis('equal')
        
        plt.tight_layout()
        plot_path = "kitti_training_results.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(plot_path)
        print(f"   ✓ Saved training visualization")
        
        # 8. Log learned parameters
        print("\n🔍 Step 8: Log Learned Parameters")
        
        Q_learned = model.ekf.Q_matrix.detach().cpu().numpy()
        R_learned = model.ekf.R_matrix.detach().cpu().numpy()
        
        print(f"   Process noise Q diagonal: {np.diag(Q_learned)}")
        print(f"   Measurement noise R diagonal: {np.diag(R_learned)}")
        
        # Save noise matrices
        np.save("learned_Q_matrix.npy", Q_learned)
        np.save("learned_R_matrix.npy", R_learned)
        mlflow.log_artifact("learned_Q_matrix.npy")
        mlflow.log_artifact("learned_R_matrix.npy")
        
        # Log final model
        mlflow.pytorch.log_model(model, "model")
        print(f"   ✓ Logged model and artifacts to MLflow")
        
        # 9. Save summary
        print("\n💾 Step 9: Save Experiment Summary")
        
        summary = {
            "experiment": "PIDSE KITTI Full Training",
            "timestamp": datetime.now().isoformat(),
            "config": config_dict,
            "results": {
                "best_val_loss": float(best_val_loss),
                "test_ate": float(avg_test_metrics['ate']),
                "test_rpe_trans": float(avg_test_metrics['rpe_trans']),
                "test_final_drift": float(avg_test_metrics['final_drift']),
                "total_params": n_params,
                "trainable_params": n_trainable
            }
        }
        
        summary_path = "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(summary_path)
        
        print("\n" + "=" * 70)
        print("🎉 KITTI Full Training Completed!")
        print("=" * 70)
        print(f"📊 Final Results:")
        print(f"   • Best validation loss: {best_val_loss:.4f}")
        print(f"   • Test ATE: {avg_test_metrics['ate']:.3f} m")
        print(f"   • Model parameters: {n_params:,}")
        print(f"\n📁 MLflow UI: Run 'mlflow ui' in the project directory")
        print(f"   Then visit: http://127.0.0.1:5000")
        
        return summary


def main():
    """Main entry point"""
    
    # Setup MLflow
    setup_mlflow()
    
    # Configuration for KITTI training
    config = {
        # Model architecture
        'state_dim': 12,          # [px, py, pz, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        'control_dim': 4,         # [ax, ay, az, torque] (derived from accelerations)
        'measurement_dim': 9,     # [px, py, pz, vx, vy, vz, roll, pitch, yaw] (GPS + IMU)
        'pinn_hidden_layers': [32, 32],  # Smaller network to start
        
        # Training parameters - CONSERVATIVE SETTINGS
        'learning_rate': 1e-4,    # Lower learning rate
        'batch_size': 2,          # Smaller batches for stability
        'sequence_length': 50,    # Shorter sequences
        'num_epochs': 30,         # Fewer epochs for first run
        
        # Loss weights - OPTIMAL FROM SYSTEMATIC TESTING
        'physics_weight': 0.001,   # Optimal: Energy/momentum conservation
        'regularization_weight': 0.001,
        
        # Noise initialization - MORE CONSERVATIVE
        'initial_process_noise': 0.1,     # Higher initial noise
        'initial_measurement_noise': 0.5,  # Higher measurement noise
        'learn_noise_matrices': True,
        
        # System parameters
        'mass': 1500.0,           # Vehicle mass (kg)
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Data sequences
        'train_sequences': ['00', '01', '02'],
        'val_sequences': ['05'],
        'test_sequences': ['03', '04']
    }
    
    # Run training
    try:
        summary = train_kitti_full(config)
        print("\n✅ Training completed successfully!")
        return 0
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
