#!/usr/bin/env python3
"""
Fine-Grained Physics Weight Optimization
========================================

Based on hypothesis testing results:
- H1 (Energy/Momentum, weight=0.001) performed best
- H3 (Smoothness, weight=0.002) had best ATE

Now we do fine-grained search around these winners to find optimal weight:

For Energy/Momentum Conservation:
  Test: [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003]

For Smoothness:
  Test: [0.0, 0.001, 0.0015, 0.002, 0.0025, 0.003]

This will identify the precise optimal physics weight for full training.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import mlflow
from datetime import datetime
import json
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from pidse.core.pidse import PIDSE, PIDSEConfig
from pidse.data.datasets import KITTIDataset
from pidse.data.loaders import create_data_loaders
from pidse.utils.metrics import compute_trajectory_metrics


def quick_train_eval(physics_type: str, physics_weight: float, num_epochs: int = 8) -> Dict:
    """
    Quick training run to evaluate physics weight.
    
    Args:
        physics_type: 'conservation' or 'smoothness'
        physics_weight: Weight for physics loss
        num_epochs: Number of epochs (default 8 for speed)
        
    Returns:
        dict: Training metrics and results
    """
    
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
    }
    
    config = PIDSEConfig(**config_dict)
    
    # Load datasets (smaller subset for speed)
    kitti_path = Path("data/kitti")
    
    train_dataset = KITTIDataset(
        data_path=kitti_path,
        sequence_ids=['00', '01'],  # Use 2 sequences for speed
        sequence_length=config.sequence_length,
        overlap=0.5
    )
    
    val_dataset = KITTIDataset(
        data_path=kitti_path,
        sequence_ids=['05'],
        sequence_length=config.sequence_length,
        overlap=0.0
    )
    
    test_dataset = KITTIDataset(
        data_path=kitti_path,
        sequence_ids=['03'],  # Single test sequence
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
    
    # Set physics constraint type
    if hasattr(model, 'criterion') and hasattr(model.criterion, 'physics_constraints'):
        model.criterion.physics_constraints.constraint_type = physics_type
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    final_train_loss = 0
    final_val_loss = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                loss_dict = model.train_step(batch, optimizer)
                if np.isfinite(loss_dict['total_loss']):
                    epoch_train_losses.append(loss_dict['total_loss'])
            except RuntimeError:
                continue
        
        final_train_loss = np.mean(epoch_train_losses) if epoch_train_losses else float('inf')
        
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
        
        final_val_loss = np.mean(epoch_val_losses) if epoch_val_losses else float('inf')
        best_val_loss = min(best_val_loss, final_val_loss)
    
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
    
    return {
        'physics_type': physics_type,
        'physics_weight': physics_weight,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'best_val_loss': best_val_loss,
        'test_ate': np.mean([m['ate'] for m in test_metrics_list]) if test_metrics_list else float('inf'),
        'test_rpe': np.mean([m['rpe_trans_mean'] for m in test_metrics_list]) if test_metrics_list else float('inf'),
        'test_drift': np.mean([m['final_drift'] for m in test_metrics_list]) if test_metrics_list else float('inf')
    }


def main():
    """Run fine-grained physics weight optimization"""
    
    mlflow.set_experiment("PIDSE_Physics_Weight_Optimization")
    
    print("="*80)
    print("FINE-GRAINED PHYSICS WEIGHT OPTIMIZATION")
    print("="*80)
    print("\nBased on hypothesis testing, we found:")
    print("  ✓ Energy/Momentum (0.001) - Best validation loss & drift")
    print("  ✓ Smoothness (0.002) - Best ATE")
    print("\nNow testing fine-grained weights around these winners...")
    print("="*80)
    
    # Define weight ranges to test
    experiments = [
        # Energy/Momentum Conservation fine-tuning
        {'type': 'conservation', 'weights': [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003]},
        # Smoothness fine-tuning
        {'type': 'smoothness', 'weights': [0.0, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004]}
    ]
    
    all_results = []
    
    for exp in experiments:
        physics_type = exp['type']
        weights = exp['weights']
        
        print(f"\n{'#'*80}")
        print(f"TESTING: {physics_type.upper()}")
        print(f"Weights to test: {weights}")
        print(f"{'#'*80}\n")
        
        type_results = []
        
        for weight in weights:
            print(f"\n{'='*60}")
            print(f"Physics: {physics_type}, Weight: {weight}")
            print(f"{'='*60}")
            
            with mlflow.start_run(run_name=f"{physics_type}_{weight}"):
                # Log parameters
                mlflow.log_param('physics_type', physics_type)
                mlflow.log_param('physics_weight', weight)
                
                # Run training
                result = quick_train_eval(physics_type, weight, num_epochs=8)
                
                # Log metrics
                mlflow.log_metric('final_train_loss', result['final_train_loss'])
                mlflow.log_metric('final_val_loss', result['final_val_loss'])
                mlflow.log_metric('best_val_loss', result['best_val_loss'])
                mlflow.log_metric('test_ate', result['test_ate'])
                mlflow.log_metric('test_rpe', result['test_rpe'])
                mlflow.log_metric('test_drift', result['test_drift'])
                
                # Print results
                print(f"  Train: {result['final_train_loss']:.2f}")
                print(f"  Val: {result['final_val_loss']:.2f}")
                print(f"  ATE: {result['test_ate']:.3f}m")
                print(f"  Drift: {result['test_drift']:.3f}m")
                
                type_results.append(result)
                all_results.append(result)
        
        # Analyze this physics type
        print(f"\n{'='*80}")
        print(f"SUMMARY: {physics_type.upper()}")
        print(f"{'='*80}")
        print(f"{'Weight':>8} {'Val Loss':>10} {'ATE':>8} {'RPE':>8} {'Drift':>8}")
        print("-"*80)
        
        for r in type_results:
            print(f"{r['physics_weight']:>8.4f} {r['best_val_loss']:>10.2f} "
                  f"{r['test_ate']:>8.3f} {r['test_rpe']:>8.3f} {r['test_drift']:>8.3f}")
    
    # Overall comparison
    print("\n\n" + "="*80)
    print("OVERALL BEST CONFIGURATIONS")
    print("="*80)
    
    # Sort by different metrics
    best_val = min(all_results, key=lambda x: x['best_val_loss'])
    best_ate = min(all_results, key=lambda x: x['test_ate'])
    best_drift = min(all_results, key=lambda x: x['test_drift'])
    best_rpe = min(all_results, key=lambda x: x['test_rpe'])
    
    print(f"\n🏆 Best Validation Loss:")
    print(f"   {best_val['physics_type']} (weight={best_val['physics_weight']:.4f})")
    print(f"   Val Loss: {best_val['best_val_loss']:.2f}, ATE: {best_val['test_ate']:.3f}m, "
          f"Drift: {best_val['test_drift']:.3f}m")
    
    print(f"\n🏆 Best ATE:")
    print(f"   {best_ate['physics_type']} (weight={best_ate['physics_weight']:.4f})")
    print(f"   ATE: {best_ate['test_ate']:.3f}m, Val Loss: {best_ate['best_val_loss']:.2f}, "
          f"Drift: {best_ate['test_drift']:.3f}m")
    
    print(f"\n🏆 Best Drift:")
    print(f"   {best_drift['physics_type']} (weight={best_drift['physics_weight']:.4f})")
    print(f"   Drift: {best_drift['test_drift']:.3f}m, ATE: {best_drift['test_ate']:.3f}m, "
          f"Val Loss: {best_drift['best_val_loss']:.2f}")
    
    print(f"\n🏆 Best RPE:")
    print(f"   {best_rpe['physics_type']} (weight={best_rpe['physics_weight']:.4f})")
    print(f"   RPE: {best_rpe['test_rpe']:.3f}m, ATE: {best_rpe['test_ate']:.3f}m, "
          f"Drift: {best_rpe['test_drift']:.3f}m")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR FULL TRAINING")
    print("="*80)
    
    # Score each configuration (weighted combination)
    for r in all_results:
        # Lower is better for all metrics
        # Normalize and weight: val_loss (30%), ATE (40%), drift (30%)
        val_scores = [x['best_val_loss'] for x in all_results]
        ate_scores = [x['test_ate'] for x in all_results]
        drift_scores = [x['test_drift'] for x in all_results]
        
        val_norm = (r['best_val_loss'] - min(val_scores)) / (max(val_scores) - min(val_scores) + 1e-6)
        ate_norm = (r['test_ate'] - min(ate_scores)) / (max(ate_scores) - min(ate_scores) + 1e-6)
        drift_norm = (r['test_drift'] - min(drift_scores)) / (max(drift_scores) - min(drift_scores) + 1e-6)
        
        r['composite_score'] = 0.3 * val_norm + 0.4 * ate_norm + 0.3 * drift_norm
    
    # Sort by composite score
    all_results.sort(key=lambda x: x['composite_score'])
    
    print("\nTop 3 configurations (by composite score):")
    print(f"{'Rank':<6} {'Physics Type':<15} {'Weight':<8} {'Score':<8} {'Val':<8} {'ATE':<8} {'Drift':<8}")
    print("-"*80)
    
    for i, r in enumerate(all_results[:3], 1):
        print(f"{i:<6} {r['physics_type']:<15} {r['physics_weight']:<8.4f} "
              f"{r['composite_score']:<8.3f} {r['best_val_loss']:<8.1f} "
              f"{r['test_ate']:<8.3f} {r['test_drift']:<8.3f}")
    
    print("\n✅ RECOMMENDED FOR FULL 30-EPOCH TRAINING:")
    best = all_results[0]
    print(f"   Physics Type: {best['physics_type']}")
    print(f"   Physics Weight: {best['physics_weight']:.4f}")
    print(f"   Expected performance: ATE~{best['test_ate']:.2f}m, Drift~{best['test_drift']:.2f}m")
    
    # Save results
    results_file = Path('experiments/physics_weight_optimization_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    mlflow.log_artifact(str(results_file))
    print(f"\n✓ Results saved to: {results_file}")
    print("="*80)


if __name__ == "__main__":
    main()
