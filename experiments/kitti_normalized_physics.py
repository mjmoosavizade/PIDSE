"""
KITTI Training with Normalized Physics Loss

Tests the hypothesis that loss scale normalization can make
physics constraints effective by matching magnitude scales.
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


def train_with_normalized_physics(config_dict):
    """Train PIDSE with normalized physics loss"""
    
    # Setup MLflow
    mlflow.set_tracking_uri("file://mlruns")
    mlflow.set_experiment("KITTI_Normalized_Physics")
    
    with mlflow.start_run(run_name=f"normalized_physics_{config_dict['physics_weight']}"):
        # Log configuration
        mlflow.log_params({
            'physics_weight': config_dict['physics_weight'],
            'normalize_physics_loss': config_dict.get('normalize_physics_loss', True),
            'physics_loss_type': config_dict.get('physics_loss_type', 'conservation'),
            'learning_rate': config_dict['learning_rate'],
            'batch_size': config_dict['batch_size'],
            'num_epochs': config_dict['num_epochs']
        })
        
        # Create config
        config = PIDSEConfig(**config_dict)
        
        # Load data
        print("📦 Loading KITTI dataset...")
        train_loader, val_loader, test_loaders = create_data_loaders(
            dataset_name="kitti",
            data_dir="data/kitti",
            train_sequences=config_dict['train_sequences'],
            val_sequences=config_dict['val_sequences'],
            test_sequences=config_dict['test_sequences'],
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            num_workers=0
        )
        
        # Initialize model
        print("🔧 Initializing PIDSE model...")
        model = PIDSE(config)
        
        # IMPORTANT: Enable normalized physics loss
        if hasattr(model, 'loss_fn'):
            model.loss_fn.normalize_physics_loss = config_dict.get('normalize_physics_loss', True)
            print(f"✅ Physics loss normalization: {model.loss_fn.normalize_physics_loss}")
        
        device = torch.device(config_dict['device'])
        model = model.to(device)
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Training loop
        print(f"\n🚀 Starting training for {config.num_epochs} epochs...")
        best_val_loss = float('inf')
        
        for epoch in range(config.num_epochs):
            # Training
            model.train()
            train_losses = []
            train_est_losses = []
            train_phys_losses_raw = []
            train_phys_losses_norm = []
            train_scale_ratios = []
            
            for batch_idx, batch in enumerate(train_loader):
                measurements = batch['measurements'].to(device)
                controls = batch['controls'].to(device)
                true_states = batch['states'].to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                estimated_states, predicted_measurements, _, _ = model(measurements, controls)
                
                # Compute loss
                loss_dict = model.loss_fn(
                    true_states=true_states,
                    estimated_states=estimated_states,
                    predicted_measurements=predicted_measurements,
                    measurements=measurements,
                    controls=controls
                )
                
                total_loss = loss_dict['total_loss']
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Track metrics
                train_losses.append(total_loss.item())
                train_est_losses.append(loss_dict['estimation_loss'].item())
                train_phys_losses_norm.append(loss_dict['physics_loss'].item())
                if 'physics_loss_raw' in loss_dict:
                    train_phys_losses_raw.append(loss_dict['physics_loss_raw'].item())
                if 'physics_scale_ratio' in loss_dict:
                    train_scale_ratios.append(loss_dict['physics_scale_ratio'].item())
            
            # Validation
            model.eval()
            val_losses = []
            val_est_losses = []
            val_phys_losses_raw = []
            val_phys_losses_norm = []
            val_predictions = []
            val_ground_truth = []
            
            with torch.no_grad():
                for batch in val_loader:
                    measurements = batch['measurements'].to(device)
                    controls = batch['controls'].to(device)
                    true_states = batch['states'].to(device)
                    
                    estimated_states, predicted_measurements, _, _ = model(measurements, controls)
                    
                    loss_dict = model.loss_fn(
                        true_states=true_states,
                        estimated_states=estimated_states,
                        predicted_measurements=predicted_measurements,
                        measurements=measurements,
                        controls=controls
                    )
                    
                    val_losses.append(loss_dict['total_loss'].item())
                    val_est_losses.append(loss_dict['estimation_loss'].item())
                    val_phys_losses_norm.append(loss_dict['physics_loss'].item())
                    if 'physics_loss_raw' in loss_dict:
                        val_phys_losses_raw.append(loss_dict['physics_loss_raw'].item())
                    
                    val_predictions.append(estimated_states.cpu().numpy())
                    val_ground_truth.append(true_states.cpu().numpy())
            
            # Compute trajectory metrics
            val_pred = np.concatenate(val_predictions, axis=0)
            val_true = np.concatenate(val_ground_truth, axis=0)
            
            metrics = compute_trajectory_metrics(val_true, val_pred)
            
            # Log metrics
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            avg_train_est = np.mean(train_est_losses)
            avg_val_est = np.mean(val_est_losses)
            avg_train_phys_norm = np.mean(train_phys_losses_norm)
            avg_val_phys_norm = np.mean(val_phys_losses_norm)
            
            mlflow.log_metrics({
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_estimation_loss': avg_train_est,
                'val_estimation_loss': avg_val_est,
                'train_physics_loss_normalized': avg_train_phys_norm,
                'val_physics_loss_normalized': avg_val_phys_norm,
                'val_ate': metrics['ate'],
                'val_rpe': metrics['rpe'],
                'val_drift': metrics['drift_percentage']
            }, step=epoch)
            
            # Log raw physics loss if available
            if train_phys_losses_raw:
                avg_train_phys_raw = np.mean(train_phys_losses_raw)
                avg_val_phys_raw = np.mean(val_phys_losses_raw) if val_phys_losses_raw else 0
                mlflow.log_metrics({
                    'train_physics_loss_raw': avg_train_phys_raw,
                    'val_physics_loss_raw': avg_val_phys_raw,
                }, step=epoch)
            
            # Log scale ratio if available
            if train_scale_ratios:
                avg_scale_ratio = np.mean(train_scale_ratios)
                mlflow.log_metric('physics_to_estimation_ratio', avg_scale_ratio, step=epoch)
            
            # Print progress
            print(f"Epoch {epoch+1}/{config.num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"  Est: {avg_train_est:.4f} | Phys(norm): {avg_train_phys_norm:.4f}")
            if train_phys_losses_raw:
                print(f"  Phys(raw): {avg_train_phys_raw:.2f} | Scale ratio: {avg_scale_ratio:.1f}x")
            print(f"  ATE: {metrics['ate']:.3f}m | RPE: {metrics['rpe']:.3f}m | Drift: {metrics['drift_percentage']:.3f}%")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path = Path("saved_models") / f"best_kitti_normalized_{config_dict['physics_weight']}.pth"
                model_path.parent.mkdir(exist_ok=True)
                torch.save(model.state_dict(), model_path)
                mlflow.log_artifact(str(model_path))
                print(f"  ✅ Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Final test evaluation
        print("\n📊 Evaluating on test sequences...")
        test_results = {}
        
        for seq_name, test_loader in test_loaders.items():
            model.eval()
            test_predictions = []
            test_ground_truth = []
            
            with torch.no_grad():
                for batch in test_loader:
                    measurements = batch['measurements'].to(device)
                    controls = batch['controls'].to(device)
                    true_states = batch['states'].to(device)
                    
                    estimated_states, _, _, _ = model(measurements, controls)
                    
                    test_predictions.append(estimated_states.cpu().numpy())
                    test_ground_truth.append(true_states.cpu().numpy())
            
            test_pred = np.concatenate(test_predictions, axis=0)
            test_true = np.concatenate(test_ground_truth, axis=0)
            
            metrics = compute_trajectory_metrics(test_true, test_pred)
            test_results[seq_name] = metrics
            
            mlflow.log_metrics({
                f'test_{seq_name}_ate': metrics['ate'],
                f'test_{seq_name}_rpe': metrics['rpe'],
                f'test_{seq_name}_drift': metrics['drift_percentage']
            })
            
            print(f"  Sequence {seq_name}: ATE={metrics['ate']:.3f}m, RPE={metrics['rpe']:.3f}m, Drift={metrics['drift_percentage']:.3f}%")
        
        # Compute average test metrics
        avg_test_ate = np.mean([m['ate'] for m in test_results.values()])
        avg_test_rpe = np.mean([m['rpe'] for m in test_results.values()])
        avg_test_drift = np.mean([m['drift_percentage'] for m in test_results.values()])
        
        mlflow.log_metrics({
            'test_avg_ate': avg_test_ate,
            'test_avg_rpe': avg_test_rpe,
            'test_avg_drift': avg_test_drift
        })
        
        print(f"\n📈 Average Test Performance:")
        print(f"   ATE: {avg_test_ate:.3f}m")
        print(f"   RPE: {avg_test_rpe:.3f}m")
        print(f"   Drift: {avg_test_drift:.3f}%")
        
        return {
            'val_loss': best_val_loss,
            'test_ate': avg_test_ate,
            'test_rpe': avg_test_rpe,
            'test_drift': avg_test_drift
        }


def main():
    """Run normalized physics loss experiments"""
    
    print("=" * 80)
    print("KITTI Training with Normalized Physics Loss")
    print("=" * 80)
    
    # Test multiple physics weights with normalization
    physics_weights = [0.0, 0.01, 0.05, 0.1, 0.5]  # Can use higher weights now!
    
    results = {}
    
    for weight in physics_weights:
        print(f"\n{'='*80}")
        print(f"Testing physics_weight = {weight}")
        print(f"{'='*80}\n")
        
        config = {
            # Model architecture
            'state_dim': 12,
            'control_dim': 4,
            'measurement_dim': 9,
            'pinn_hidden_layers': [32, 32],
            
            # Training parameters
            'learning_rate': 1e-4,
            'batch_size': 2,
            'sequence_length': 50,
            'num_epochs': 30,
            
            # Loss weights - NORMALIZED PHYSICS
            'physics_weight': weight,
            'regularization_weight': 0.001,
            'normalize_physics_loss': True,  # Enable normalization
            'physics_loss_type': 'conservation',
            
            # Noise initialization
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
            result = train_with_normalized_physics(config)
            results[weight] = result
            print(f"\n✅ Completed training with weight={weight}")
        except Exception as e:
            print(f"\n❌ Failed with weight={weight}: {e}")
            import traceback
            traceback.print_exc()
            results[weight] = {'error': str(e)}
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Normalized Physics Loss Results")
    print("=" * 80)
    print(f"{'Weight':<10} {'Val Loss':<12} {'ATE (m)':<10} {'RPE (m)':<10} {'Drift (%)':<10}")
    print("-" * 80)
    
    for weight, result in results.items():
        if 'error' not in result:
            print(f"{weight:<10.3f} {result['val_loss']:<12.4f} {result['test_ate']:<10.3f} "
                  f"{result['test_rpe']:<10.3f} {result['test_drift']:<10.3f}")
        else:
            print(f"{weight:<10.3f} ERROR: {result['error']}")
    
    # Save results
    results_file = Path("experiments") / "normalized_physics_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to {results_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
