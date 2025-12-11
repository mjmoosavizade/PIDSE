#!/usr/bin/env python3
"""
Test 4 Physics Loss Normalization Strategies
=============================================

This script tests the 4 normalization strategies now built into HybridLoss:
1. Fixed: Calibrate scale once at start (first 10 batches), keep constant
2. Manual: Simple divide by 1000 (rule of thumb)
3. Learnable: Gradient-optimized scale parameter
4. Adaptive: EMA-based (current approach - for comparison)

Goal: Find which strategy enables physics constraints to improve performance.
Baseline (no physics): 2.907m ATE

Each trains for 30 epochs on KITTI with conservation physics, weight=0.1
"""

import sys
from pathlib import Path
import numpy as np
import torch
import mlflow
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from pidse.models.pidse import PIDSE
from pidse.data.kitti import KITTIDataModule
from pidse.losses.hybrid_loss import HybridLoss
from pidse.core.vehicle_dynamics import VehicleDynamics
from pidse.core.physics_constraints import PhysicsInformedConstraints


def setup_mlflow():
    """Setup MLflow tracking."""
    project_root = Path(__file__).parent.parent
    mlflow_dir = project_root / "mlruns"
    mlflow_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    mlflow.set_experiment("normalization_strategies")


def train_with_strategy(
    strategy: str,
    physics_type: str = "conservation",
    physics_weight: float = 0.1,
    num_epochs: int = 30,
    device: str = "cuda"
):
    """
    Train PIDSE with specific normalization strategy using built-in HybridLoss.
    
    Args:
        strategy: "fixed", "manual", "learnable", or "adaptive"
        physics_type: Type of physics constraint
        physics_weight: Weight for physics loss
        num_epochs: Number of training epochs
        device: Device to train on
    """
    print(f"\n{'='*80}")
    print(f"Strategy: {strategy.upper()} | Physics: {physics_type} | Weight: {physics_weight}")
    print(f"{'='*80}\n")
    
    project_root = Path(__file__).parent.parent
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"{strategy}_{physics_type}_{physics_weight}"):
        # Log params
        mlflow.log_params({
            "normalization_strategy": strategy,
            "physics_type": physics_type,
            "physics_weight": physics_weight,
            "epochs": num_epochs,
        })
        
        # Setup data
        data_module = KITTIDataModule(
            data_root=str(project_root / "data" / "kitti"),
            batch_size=32,
            train_sequences=['00', '01', '02'],
            val_sequences=['05'],
            test_sequences=['03', '04']
        )
        data_module.setup()
        
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()
        
        # Initialize model
        model = PIDSE(
            state_dim=12,
            control_dim=4,
            measurement_dim=9,
            hidden_dims=[32, 32],
            dynamics=VehicleDynamics()
        ).to(device)
        
        print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
        
        # Setup physics constraints
        physics_constraints = PhysicsInformedConstraints(
            dynamics=model.dynamics,
            constraint_types=[physics_type]
        )
        
        # Setup loss with specific strategy
        loss_fn = HybridLoss(
            estimation_loss_type='mse',
            physics_constraints=physics_constraints,
            physics_weight=physics_weight,
            regularization_weight=1e-4,
            normalize_physics_loss=True,
            normalization_strategy=strategy
        ).to(device)
        
        # Optimizer (include learnable scale if applicable)
        if strategy == "learnable":
            params = list(model.parameters()) + [loss_fn.log_physics_scale]
            optimizer = torch.optim.Adam(params, lr=1e-3)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Training
        best_val_ate = float('inf')
        best_epoch = 0
        
        for epoch in range(num_epochs):
            model.train()
            train_losses = []
            
            for batch_idx, batch in enumerate(train_loader):
                states = batch['states'].to(device)
                controls = batch['controls'].to(device)
                measurements = batch['measurements'].to(device)
                
                optimizer.zero_grad()
                outputs = model(measurements, controls)
                loss_dict = loss_fn(
                    true_states=states,
                    estimated_states=outputs['filtered_states'],
                    predicted_measurements=outputs['predicted_measurements'],
                    measurements=measurements,
                    controls=controls
                )
                
                total_loss = loss_dict['total']
                total_loss.backward()
                optimizer.step()
                
                train_losses.append(total_loss.item())
                
                if batch_idx % 20 == 0:
                    msg = (f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | "
                           f"Loss: {total_loss.item():.4f} | Est: {loss_dict['estimation'].item():.4f} | "
                           f"Phys: {loss_dict['physics'].item():.4f}")
                    if strategy == "learnable":
                        scale = torch.exp(loss_fn.log_physics_scale).item()
                        msg += f" | Scale: {scale:.6f}"
                    print(msg)
            
            avg_train_loss = np.mean(train_losses)
            
            # Validation
            model.eval()
            val_losses = []
            val_ates = []
            
            with torch.no_grad():
                for batch in val_loader:
                    states = batch['states'].to(device)
                    controls = batch['controls'].to(device)
                    measurements = batch['measurements'].to(device)
                    
                    outputs = model(measurements, controls)
                    loss_dict = loss_fn(
                        true_states=states,
                        estimated_states=outputs['filtered_states'],
                        predicted_measurements=outputs['predicted_measurements'],
                        measurements=measurements,
                        controls=controls
                    )
                    
                    val_losses.append(loss_dict['total'].item())
                    
                    positions_true = states[:, :, :3]
                    positions_est = outputs['filtered_states'][:, :, :3]
                    ate = torch.sqrt(((positions_true - positions_est) ** 2).sum(dim=-1)).mean()
                    val_ates.append(ate.item())
            
            avg_val_loss = np.mean(val_losses)
            avg_val_ate = np.mean(val_ates)
            
            print(f"\nEpoch {epoch+1}/{num_epochs}: Train={avg_train_loss:.4f}, "
                  f"Val={avg_val_loss:.4f}, ATE={avg_val_ate:.4f}m")
            
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_ate": avg_val_ate,
            }, step=epoch)
            
            if avg_val_ate < best_val_ate:
                best_val_ate = avg_val_ate
                best_epoch = epoch + 1
                torch.save(
                    model.state_dict(),
                    project_root / f"saved_models/best_{strategy}_{physics_type}.pth"
                )
        
        print(f"\nBest val ATE: {best_val_ate:.4f}m at epoch {best_epoch}")
        
        # Test
        model.load_state_dict(
            torch.load(project_root / f"saved_models/best_{strategy}_{physics_type}.pth")
        )
        model.eval()
        
        test_ates = []
        with torch.no_grad():
            for batch in test_loader:
                states = batch['states'].to(device)
                controls = batch['controls'].to(device)
                measurements = batch['measurements'].to(device)
                
                outputs = model(measurements, controls)
                positions_true = states[:, :, :3]
                positions_est = outputs['filtered_states'][:, :, :3]
                ate = torch.sqrt(((positions_true - positions_est) ** 2).sum(dim=-1)).mean()
                test_ates.append(ate.item())
        
        test_ate = np.mean(test_ates)
        print(f"Test ATE: {test_ate:.4f}m")
        
        mlflow.log_metrics({
            "best_val_ate": best_val_ate,
            "test_ate": test_ate,
            "best_epoch": best_epoch
        })
        
        return {
            "strategy": strategy,
            "physics_type": physics_type,
            "physics_weight": physics_weight,
            "val_ate": best_val_ate,
            "test_ate": test_ate,
            "best_epoch": best_epoch
        }


def main():
    """Run normalization strategy comparison."""
    setup_mlflow()
    
    project_root = Path(__file__).parent.parent
    
    # Configuration
    strategies = ["fixed", "manual", "learnable", "adaptive"]
    physics_type = "conservation"
    physics_weight = 0.1
    num_epochs = 30
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*80)
    print("NORMALIZATION STRATEGY COMPARISON")
    print("="*80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Strategies: {', '.join(strategies)}")
    print(f"Physics: {physics_type}, Weight: {physics_weight}")
    print(f"Epochs: {num_epochs}")
    print(f"\nBaseline (no physics, 30 epochs): 2.907m ATE\n")
    
    # Run experiments
    results = []
    for strategy in strategies:
        try:
            result = train_with_strategy(
                strategy=strategy,
                physics_type=physics_type,
                physics_weight=physics_weight,
                num_epochs=num_epochs,
                device=device
            )
            results.append(result)
        except Exception as e:
            print(f"\n❌ {strategy} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "strategy": strategy,
                "physics_type": physics_type,
                "physics_weight": physics_weight,
                "error": str(e)
            })
    
    # Save results
    results_file = project_root / "experiments" / "normalization_strategy_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\nBaseline (no physics): 2.907m ATE")
    print(f"Physics: {physics_type}, Weight: {physics_weight}\n")
    print("-"*80)
    
    valid_results = [r for r in results if 'test_ate' in r]
    valid_results.sort(key=lambda x: x['test_ate'])
    
    for rank, r in enumerate(valid_results, 1):
        improvement = ((2.907 - r['test_ate']) / 2.907) * 100
        print(f"{rank}. {r['strategy']:12s} | Test: {r['test_ate']:6.3f}m | "
              f"Val: {r['val_ate']:6.3f}m | Change: {improvement:+6.1f}%")
    
    failed = [r for r in results if 'error' in r]
    if failed:
        print("\nFailed:")
        for r in failed:
            print(f"  {r['strategy']}: {r['error']}")
    
    print("\n" + "="*80)
    print(f"Saved: {results_file}")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
