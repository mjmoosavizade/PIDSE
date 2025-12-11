#!/usr/bin/env python3
"""
Multi-Hypothesis Physics Constraints Testing with Normalization
================================================================

Test different physics constraint formulations WITH LOSS NORMALIZATION:

Hypothesis 1: Energy/Momentum Conservation + Normalization
Hypothesis 2: Dynamics Consistency + Normalization
Hypothesis 3: Smoothness Constraints + Normalization

Each hypothesis tests multiple weights [0.0, 0.01, 0.05, 0.1, 0.5] with normalized physics loss.
Runs 15 epochs per configuration for quick assessment.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import mlflow
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from pidse.core.pidse import PIDSE, PIDSEConfig
from pidse.data.datasets import KITTIDataset
from pidse.data.loaders import create_data_loaders
from pidse.utils.metrics import compute_trajectory_metrics


def train_hypothesis(
    physics_type: str,
    physics_weight: float,
    normalize: bool = True,
    num_epochs: int = 15
):
    """
    Train model with specific physics constraint type and normalization.
    
    Args:
        physics_type: 'conservation', 'dynamics_consistency', 'smoothness'
        physics_weight: Weight for physics loss
        normalize: Enable loss normalization
        num_epochs: Number of training epochs
        
    Returns:
        dict: Training results and metrics
    """
    
    run_name = f"{physics_type}_w{physics_weight}_norm{normalize}"
    print(f"\n{'='*80}")
    print(f"Testing: {physics_type.upper()} | Weight={physics_weight} | Normalize={normalize}")
    print(f"{'='*80}")
    
    # Configuration
    config_dict = {
        'state_dim': 12,
        'control_dim': 4,
        'measurement_dim': 9,
        'pinn_hidden_layers': [32, 32],
        'learning_rate': 1e-4,
        'batch_size': 2,
        'sequence_length': 50,
        'physics_weight': physics_weight,
        'regularization_weight': 0.001,
        'initial_process_noise': 0.1,
        'initial_measurement_noise': 0.5,
        'learn_noise_matrices': True,
        'mass': 1500.0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    config = PIDSEConfig(**config_dict)
    
    # Load datasets
    kitti_path = Path("data/kitti")
    train_sequences = ['00', '01', '02']
    val_sequences = ['05']
    test_sequences = ['03', '04']
    
    print("Loading train dataset...")
    train_dataset = KITTIDataset(
        data_path=kitti_path,
        sequence_ids=train_sequences,
        sequence_length=config.sequence_length,
        overlap=0.5
    )
    print(f"✓ Train dataset loaded: {len(train_dataset)} samples")
    
    print("Loading val dataset...")
    val_dataset = KITTIDataset(
        data_path=kitti_path,
        sequence_ids=val_sequences,
        sequence_length=config.sequence_length,
        overlap=0.0
    )
    print(f"✓ Val dataset loaded: {len(val_dataset)} samples")
    
    print("Loading test dataset...")
    test_dataset = KITTIDataset(
        data_path=kitti_path,
        sequence_ids=test_sequences,
        sequence_length=config.sequence_length,
        overlap=0.0
    )
    print(f"✓ Test dataset loaded: {len(test_dataset)} samples")
    
    # Create data loaders
    print("Creating data loaders...")
    train_trajectories = [train_dataset[i] for i in range(len(train_dataset))]
    print(f"✓ Loaded {len(train_trajectories)} train trajectories")
    val_trajectories = [val_dataset[i] for i in range(len(val_dataset))]
    test_trajectories = [test_dataset[i] for i in range(len(test_dataset))]
    
    train_loader, _, _ = create_data_loaders(
        train_trajectories, None,
        batch_size=config.batch_size,
        sequence_length=config.sequence_length,
        val_split=0.0,
        num_workers=0
    )
    
    print("Creating val loader...")
    val_loader, _, _ = create_data_loaders(
        val_trajectories, None,
        batch_size=config.batch_size,
        sequence_length=config.sequence_length,
        val_split=0.0,
        num_workers=0
    )
    print(f"✓ Val loader created")
    
    print("Creating test loader...")
    test_loader, _, _ = create_data_loaders(
        test_trajectories, None,
        batch_size=config.batch_size,
        sequence_length=config.sequence_length,
        val_split=0.0,
        num_workers=0
    )
    print(f"✓ Test loader created")
    
    # Initialize model
    print("Initializing model...")
    device = torch.device(config_dict['device'])
    model = PIDSE(config).to(device)
    print(f"✓ Model initialized on {device}")
    
    # Configure physics constraints and normalization
    print("Configuring physics constraints...")
    if hasattr(model, 'loss_fn'):
        # Set physics constraint type
        if hasattr(model.loss_fn, 'physics_constraints'):
            model.loss_fn.physics_constraints.constraint_type = physics_type
            print(f"✓ Physics type: {physics_type}")
        
        # Enable/disable normalization
        model.loss_fn.normalize_physics_loss = normalize
        print(f"✓ Loss normalization: {normalize}")
    
    print("Creating optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    print("✓ Optimizer created")
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    best_val_loss = float('inf')
    best_val_ate = float('inf')
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_estimation_loss': [],
        'train_physics_loss_raw': [],
        'train_physics_loss_normalized': [],
        'val_estimation_loss': [],
        'val_ate': [],
        'val_rpe': [],
        'val_drift': [],
        'physics_scale_ratio': []
    }
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_losses = {
            'total': [],
            'estimation': [],
            'physics_raw': [],
            'physics_norm': [],
            'scale_ratio': []
        }
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}")
            
            try:
                measurements = batch['measurements'].to(device)
                controls = batch['controls'].to(device)
                true_states = batch['states'].to(device)
                initial_state = batch['initial_state'].to(device)
                initial_covariance = batch['initial_covariance'].to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(
                    states=true_states,
                    controls=controls,
                    measurements=measurements,
                    initial_state=initial_state,
                    initial_covariance=initial_covariance
                )
                estimated_states = outputs['estimated_states']
                predicted_measurements = outputs['predicted_measurements']
                
                # Compute loss
                loss_dict = model.loss_fn(
                    true_states=true_states,
                    estimated_states=estimated_states,
                    predicted_measurements=predicted_measurements,
                    measurements=measurements,
                    controls=controls
                )
                
                total_loss = loss_dict['total_loss']
                
                # Check for NaN
                if not torch.isfinite(total_loss):
                    continue
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Track metrics
                epoch_losses['total'].append(total_loss.item())
                epoch_losses['estimation'].append(loss_dict['estimation_loss'].item())
                epoch_losses['physics_norm'].append(loss_dict['physics_loss'].item())
                if 'physics_loss_raw' in loss_dict:
                    epoch_losses['physics_raw'].append(loss_dict['physics_loss_raw'].item())
                if 'physics_scale_ratio' in loss_dict:
                    epoch_losses['scale_ratio'].append(loss_dict['physics_scale_ratio'].item())
                
            except Exception as e:
                print(f"\nWarning: Batch {batch_idx} failed: {e}")
                continue
        
        # Compute training averages
        avg_train_loss = np.mean(epoch_losses['total']) if epoch_losses['total'] else float('inf')
        avg_train_est = np.mean(epoch_losses['estimation']) if epoch_losses['estimation'] else 0
        avg_train_phys_norm = np.mean(epoch_losses['physics_norm']) if epoch_losses['physics_norm'] else 0
        avg_train_phys_raw = np.mean(epoch_losses['physics_raw']) if epoch_losses['physics_raw'] else 0
        avg_scale_ratio = np.mean(epoch_losses['scale_ratio']) if epoch_losses['scale_ratio'] else 1.0
        
        # Validation
        model.eval()
        val_losses = []
        val_est_losses = []
        val_predictions = []
        val_ground_truth = []
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    measurements = batch['measurements'].to(device)
                    controls = batch['controls'].to(device)
                    true_states = batch['states'].to(device)
                    initial_state = batch['initial_state'].to(device)
                    initial_covariance = batch['initial_covariance'].to(device)
                    
                    outputs = model(
                        states=true_states,
                        controls=controls,
                        measurements=measurements,
                        initial_state=initial_state,
                        initial_covariance=initial_covariance
                    )
                    estimated_states = outputs['estimated_states']
                    predicted_measurements = outputs['predicted_measurements']
                    
                    loss_dict = model.loss_fn(
                        true_states=true_states,
                        estimated_states=estimated_states,
                        predicted_measurements=predicted_measurements,
                        measurements=measurements,
                        controls=controls
                    )
                    
                    val_losses.append(loss_dict['total_loss'].item())
                    val_est_losses.append(loss_dict['estimation_loss'].item())
                    
                    val_predictions.append(estimated_states.cpu().numpy())
                    val_ground_truth.append(true_states.cpu().numpy())
                    
                except Exception as e:
                    continue
        
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        avg_val_est = np.mean(val_est_losses) if val_est_losses else 0
        
        # Compute trajectory metrics
        if val_predictions:
            val_pred = np.concatenate(val_predictions, axis=0)
            val_true = np.concatenate(val_ground_truth, axis=0)
            # Reshape from [batch, seq, state_dim] to [N, state_dim]
            val_pred = val_pred.reshape(-1, val_pred.shape[-1])
            val_true = val_true.reshape(-1, val_true.shape[-1])
            # Extract positions (first 3 dimensions) and convert to torch tensors
            val_pred_pos = torch.from_numpy(val_pred[:, :3]).float()
            val_true_pos = torch.from_numpy(val_true[:, :3]).float()
            metrics = compute_trajectory_metrics(val_true_pos, val_pred_pos)
            val_ate = metrics.get('ate', float('inf'))
            val_rpe = metrics.get('rpe', float('inf'))
            val_drift = metrics.get('drift_percentage', float('inf'))
        else:
            val_ate = val_rpe = val_drift = float('inf')
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_estimation_loss'].append(avg_train_est)
        history['train_physics_loss_raw'].append(avg_train_phys_raw)
        history['train_physics_loss_normalized'].append(avg_train_phys_norm)
        history['val_estimation_loss'].append(avg_val_est)
        history['val_ate'].append(val_ate)
        history['val_rpe'].append(val_rpe)
        history['val_drift'].append(val_drift)
        history['physics_scale_ratio'].append(avg_scale_ratio)
        
        # Print progress
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  Est: {avg_train_est:.4f} | Phys(raw): {avg_train_phys_raw:.2f} | Phys(norm): {avg_train_phys_norm:.4f}")
        if avg_scale_ratio != 1.0:
            print(f"  Scale Ratio: {avg_scale_ratio:.1f}x (before normalization)")
        print(f"  ATE: {val_ate:.3f}m | RPE: {val_rpe:.3f}m | Drift: {val_drift:.3f}%")
        
        # Track best model
        if val_ate < best_val_ate:
            best_val_ate = val_ate
            best_val_loss = avg_val_loss
            print(f"  ✓ New best ATE: {best_val_ate:.3f}m")
    
    # Final test evaluation
    print("\n📊 Final test evaluation...")
    model.eval()
    test_predictions = []
    test_ground_truth = []
    
    with torch.no_grad():
        for batch in test_loader:
            try:
                measurements = batch['measurements'].to(device)
                controls = batch['controls'].to(device)
                true_states = batch['states'].to(device)
                initial_state = batch['initial_state'].to(device)
                initial_covariance = batch['initial_covariance'].to(device)
                
                outputs = model(
                    states=true_states,
                    controls=controls,
                    measurements=measurements,
                    initial_state=initial_state,
                    initial_covariance=initial_covariance
                )
                estimated_states = outputs['estimated_states']
                
                test_predictions.append(estimated_states.cpu().numpy())
                test_ground_truth.append(true_states.cpu().numpy())
                
            except Exception as e:
                continue
    
    if test_predictions:
        test_pred = np.concatenate(test_predictions, axis=0)
        test_true = np.concatenate(test_ground_truth, axis=0)
        # Reshape from [batch, seq, state_dim] to [N, state_dim]
        test_pred = test_pred.reshape(-1, test_pred.shape[-1])
        test_true = test_true.reshape(-1, test_true.shape[-1])
        # Extract positions (first 3 dimensions) and convert to torch tensors
        test_pred_pos = torch.from_numpy(test_pred[:, :3]).float()
        test_true_pos = torch.from_numpy(test_true[:, :3]).float()
        test_metrics = compute_trajectory_metrics(test_true_pos, test_pred_pos)
    else:
        test_metrics = {'ate': float('inf'), 'rpe': float('inf'), 'drift_percentage': float('inf')}
    
    print(f"  Test ATE: {test_metrics.get('ate', float('inf')):.3f}m")
    print(f"  Test RPE: {test_metrics.get('rpe', float('inf')):.3f}m")
    print(f"  Test Drift: {test_metrics.get('drift_percentage', float('inf')):.3f}%")
    
    # Return results
    return {
        'config': {
            'physics_type': physics_type,
            'physics_weight': physics_weight,
            'normalize': normalize,
            'num_epochs': num_epochs
        },
        'best': {
            'val_loss': best_val_loss,
            'val_ate': best_val_ate,
            'test_ate': test_metrics['ate'],
            'test_rpe': test_metrics['rpe'],
            'test_drift': test_metrics['drift_percentage']
        },
        'history': history
    }


def main():
    """Run multi-hypothesis testing with normalization"""
    
    print("=" * 80)
    print("Multi-Hypothesis Physics Testing with Loss Normalization")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup MLflow
    mlflow_dir = Path("mlruns").absolute()
    mlflow_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"file:{mlflow_dir}")
    mlflow.set_experiment("KITTI_Normalized_Physics_Hypotheses")
    
    # Test configurations
    hypotheses = [
        ('conservation', [0.0, 0.01, 0.05, 0.1, 0.5]),
        ('dynamics_consistency', [0.0, 0.01, 0.05, 0.1, 0.5]),
        ('smoothness', [0.0, 0.01, 0.05, 0.1, 0.5])
    ]
    
    all_results = []
    
    for physics_type, weights in hypotheses:
        print(f"\n{'#'*80}")
        print(f"# HYPOTHESIS: {physics_type.upper()}")
        print(f"{'#'*80}")
        
        for weight in weights:
            try:
                with mlflow.start_run(run_name=f"{physics_type}_w{weight}_normalized"):
                    # Log parameters
                    mlflow.log_params({
                        'physics_type': physics_type,
                        'physics_weight': weight,
                        'normalize_physics_loss': True,
                        'num_epochs': 15
                    })
                    
                    # Train
                    result = train_hypothesis(
                        physics_type=physics_type,
                        physics_weight=weight,
                        normalize=True,
                        num_epochs=15
                    )
                    
                    # Log metrics
                    mlflow.log_metrics({
                        'best_val_loss': result['best']['val_loss'],
                        'best_val_ate': result['best']['val_ate'],
                        'test_ate': result['best']['test_ate'],
                        'test_rpe': result['best']['test_rpe'],
                        'test_drift': result['best']['test_drift']
                    })
                    
                    # Log history as artifacts
                    for key, values in result['history'].items():
                        for epoch, value in enumerate(values):
                            mlflow.log_metric(key, value, step=epoch)
                    
                    all_results.append(result)
                    
                    print(f"\n✅ Completed {physics_type} with weight={weight}")
                    
            except Exception as e:
                print(f"\n❌ Failed {physics_type} with weight={weight}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save comprehensive results
    results_file = Path("experiments/normalized_hypothesis_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print("SUMMARY: Normalized Physics Hypothesis Testing")
    print(f"{'='*80}")
    print(f"{'Hypothesis':<25} {'Weight':<10} {'Val ATE':<12} {'Test ATE':<12} {'Test RPE':<12}")
    print("-" * 80)
    
    for result in all_results:
        cfg = result['config']
        best = result['best']
        print(f"{cfg['physics_type']:<25} {cfg['physics_weight']:<10.3f} "
              f"{best['val_ate']:<12.3f} {best['test_ate']:<12.3f} {best['test_rpe']:<12.3f}")
    
    # Find best configuration
    if all_results:
        best_result = min(all_results, key=lambda x: x['best']['test_ate'])
        print(f"\n{'='*80}")
        print("🏆 BEST CONFIGURATION:")
        print(f"  Physics Type: {best_result['config']['physics_type']}")
        print(f"  Physics Weight: {best_result['config']['physics_weight']}")
        print(f"  Test ATE: {best_result['best']['test_ate']:.3f}m")
        print(f"  Test RPE: {best_result['best']['test_rpe']:.3f}m")
        print(f"  Test Drift: {best_result['best']['test_drift']:.3f}%")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print("❌ NO SUCCESSFUL RESULTS - All configurations failed")
        print(f"{'='*80}")
    
    print(f"\n📁 Results saved to: {results_file}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0


if __name__ == "__main__":
    exit(main())
