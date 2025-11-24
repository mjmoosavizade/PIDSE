#!/usr/bin/env python3
"""
Multi-Hypothesis Physics Constraints Testing
============================================

Test different physics constraint formulations to find which works best for KITTI:

Hypothesis 1: Energy/Momentum Conservation (current default)
Hypothesis 2: Kinematic Consistency (vehicle-specific)
Hypothesis 3: Smoothness Constraints (trajectory-based)
Hypothesis 4: Velocity-Position Consistency (numerical)
Hypothesis 5: No Physics (baseline control)

Each hypothesis runs a short training (10 epochs) and we compare results.
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


def train_with_physics_type(physics_type: str, physics_weight: float, num_epochs: int = 10):
    """
    Train model with specific physics constraint type.
    
    Args:
        physics_type: 'conservation', 'dynamics_consistency', 'smoothness', 'velocity_consistency', 'none'
        physics_weight: Weight for physics loss
        num_epochs: Number of training epochs
        
    Returns:
        dict: Training results and metrics
    """
    
    print(f"\n{'='*70}")
    print(f"Testing: {physics_type.upper()} (weight={physics_weight})")
    print(f"{'='*70}")
    
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
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'train_sequences': ['00', '01', '02'],
        'val_sequences': ['05'],
        'test_sequences': ['03', '04']
    }
    
    train_sequences = config_dict.pop('train_sequences')
    val_sequences = config_dict.pop('val_sequences')
    test_sequences = config_dict.pop('test_sequences')
    
    config = PIDSEConfig(**config_dict)
    
    # Load datasets
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
    
    # Initialize model
    model = PIDSE(config)
    
    # Modify physics constraint type if model supports it
    if hasattr(model, 'criterion') and hasattr(model.criterion, 'physics_constraints'):
        model.criterion.physics_constraints.constraint_type = physics_type
        print(f"✓ Set physics constraint type to: {physics_type}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    results = {
        'physics_type': physics_type,
        'physics_weight': physics_weight,
        'train_losses': [],
        'val_losses': [],
        'physics_losses': []
    }
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_losses = []
        epoch_physics_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}", end='\r', flush=True)
            
            try:
                loss_dict = model.train_step(batch, optimizer)
                
                if not np.isfinite(loss_dict['total_loss']):
                    continue
                
                epoch_train_losses.append(loss_dict['total_loss'])
                if 'physics_loss' in loss_dict:
                    epoch_physics_losses.append(loss_dict['physics_loss'])
                
            except RuntimeError:
                continue
        
        avg_train_loss = np.mean(epoch_train_losses)
        avg_physics_loss = np.mean(epoch_physics_losses) if epoch_physics_losses else 0.0
        
        # Validation
        model.eval()
        epoch_val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    device = next(model.parameters()).device
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    outputs = model.forward(
                        states=batch["states"],
                        controls=batch["controls"],
                        measurements=batch["measurements"],
                        initial_state=batch["initial_state"],
                        initial_covariance=batch["initial_covariance"]
                    )
                    
                    val_loss = outputs["loss_components"]["total_loss"].item()
                    epoch_val_losses.append(val_loss)
                except:
                    continue
        
        avg_val_loss = np.mean(epoch_val_losses) if epoch_val_losses else float('inf')
        
        results['train_losses'].append(avg_train_loss)
        results['val_losses'].append(avg_val_loss)
        results['physics_losses'].append(avg_physics_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        print(f"  Epoch {epoch+1}/{num_epochs}: Train={avg_train_loss:.2f}, Val={avg_val_loss:.2f}, Physics={avg_physics_loss:.4f}")
    
    # Test evaluation
    model.eval()
    test_metrics_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            try:
                device = next(model.parameters()).device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
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
            except:
                continue
    
    if test_metrics_list:
        results['test_ate'] = np.mean([m['ate'] for m in test_metrics_list])
        results['test_rpe'] = np.mean([m['rpe_trans_mean'] for m in test_metrics_list])
        results['test_drift'] = np.mean([m['final_drift'] for m in test_metrics_list])
    else:
        results['test_ate'] = float('inf')
        results['test_rpe'] = float('inf')
        results['test_drift'] = float('inf')
    
    results['best_val_loss'] = best_val_loss
    
    print(f"\nResults:")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Test ATE: {results['test_ate']:.3f}m")
    print(f"  Test RPE: {results['test_rpe']:.3f}m")
    print(f"  Test Drift: {results['test_drift']:.3f}m")
    
    return results


def main():
    """Run multi-hypothesis testing"""
    
    mlflow.set_experiment("PIDSE_Physics_Hypothesis_Testing")
    
    print("="*70)
    print("MULTI-HYPOTHESIS PHYSICS CONSTRAINTS TESTING")
    print("="*70)
    print("\nTesting 5 different physics formulations with 10 epochs each")
    print("Target: Find physics constraints that help (not hurt) KITTI performance")
    print("Baseline to beat: ATE=2.907m, RPE=1.031m, Drift=2.906m")
    print("="*70)
    
    # Define hypotheses to test
    hypotheses = [
        {
            'name': 'H0: No Physics (Baseline)',
            'type': 'none',
            'weight': 0.0,
            'description': 'Pure data-driven learning, no physics constraints'
        },
        {
            'name': 'H1: Energy/Momentum Conservation',
            'type': 'conservation',
            'weight': 0.001,
            'description': 'Conservation laws (current default, may not fit vehicles)'
        },
        {
            'name': 'H2: Dynamics Consistency',
            'type': 'dynamics_consistency',
            'weight': 0.001,
            'description': 'State transition consistency with dynamics model'
        },
        {
            'name': 'H3: Trajectory Smoothness',
            'type': 'smoothness',
            'weight': 0.002,
            'description': 'Penalize jerky/discontinuous trajectories'
        },
        {
            'name': 'H4: Energy Conservation (Strong)',
            'type': 'conservation',
            'weight': 0.005,
            'description': 'Same as H1 but stronger weight'
        },
        {
            'name': 'H5: Smoothness (Strong)',
            'type': 'smoothness',
            'weight': 0.01,
            'description': 'Stronger smoothness constraint'
        }
    ]
    
    all_results = []
    
    for i, hypothesis in enumerate(hypotheses, 1):
        print(f"\n\n{'#'*70}")
        print(f"HYPOTHESIS {i}/{len(hypotheses)}: {hypothesis['name']}")
        print(f"Description: {hypothesis['description']}")
        print(f"{'#'*70}")
        
        with mlflow.start_run(run_name=f"H{i}_{hypothesis['type']}_{hypothesis['weight']}"):
            # Log hypothesis details
            mlflow.log_param('hypothesis_name', hypothesis['name'])
            mlflow.log_param('physics_type', hypothesis['type'])
            mlflow.log_param('physics_weight', hypothesis['weight'])
            
            # Run training
            results = train_with_physics_type(
                physics_type=hypothesis['type'],
                physics_weight=hypothesis['weight'],
                num_epochs=10
            )
            
            # Log results
            mlflow.log_metric('final_val_loss', results['best_val_loss'])
            mlflow.log_metric('test_ate', results['test_ate'])
            mlflow.log_metric('test_rpe', results['test_rpe'])
            mlflow.log_metric('test_drift', results['test_drift'])
            
            # Store for comparison
            results['hypothesis'] = hypothesis['name']
            all_results.append(results)
    
    # Comparative analysis
    print("\n\n" + "="*70)
    print("COMPARATIVE RESULTS")
    print("="*70)
    print(f"\n{'Hypothesis':<40} {'Val Loss':>10} {'ATE':>8} {'RPE':>8} {'Drift':>8}")
    print("-"*70)
    
    for result in all_results:
        print(f"{result['hypothesis']:<40} {result['best_val_loss']:>10.2f} "
              f"{result['test_ate']:>8.3f} {result['test_rpe']:>8.3f} {result['test_drift']:>8.3f}")
    
    # Find best hypothesis
    best_by_ate = min(all_results, key=lambda x: x['test_ate'])
    best_by_drift = min(all_results, key=lambda x: x['test_drift'])
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print(f"\nBest by ATE: {best_by_ate['hypothesis']}")
    print(f"  ATE: {best_by_ate['test_ate']:.3f}m (Baseline: 2.907m)")
    print(f"\nBest by Drift: {best_by_drift['hypothesis']}")
    print(f"  Drift: {best_by_drift['test_drift']:.3f}m (Baseline: 2.906m)")
    
    # Save results
    results_file = Path('experiments/physics_hypothesis_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    mlflow.log_artifact(str(results_file))
    print(f"\n✓ Results saved to: {results_file}")
    print("="*70)


if __name__ == "__main__":
    main()
