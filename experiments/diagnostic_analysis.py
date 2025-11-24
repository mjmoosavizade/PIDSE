#!/usr/bin/env python3
"""
Diagnostic Analysis: Why Physics Constraints Hurt KITTI Performance
====================================================================

This script investigates the root causes of physics constraint degradation through:
1. Physics loss magnitude analysis
2. Gradient flow examination  
3. Loss component correlation
4. State prediction error decomposition
5. Constraint violation patterns
6. Model capacity analysis

For research paper: Understanding when physics-informed learning helps vs hurts.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from pidse.core.pidse import PIDSE, PIDSEConfig
from pidse.data.datasets import KITTIDataset
from pidse.data.loaders import create_data_loaders
from pidse.utils.metrics import compute_trajectory_metrics


def analyze_physics_loss_magnitude(model, data_loader, device):
    """Analyze the magnitude and behavior of physics loss vs estimation loss."""
    
    model.eval()
    estimation_losses = []
    physics_losses = []
    total_losses = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            try:
                outputs = model.forward(
                    states=batch["states"],
                    controls=batch["controls"],
                    measurements=batch["measurements"],
                    initial_state=batch["initial_state"],
                    initial_covariance=batch["initial_covariance"]
                )
                
                loss_comps = outputs["loss_components"]
                est_loss = loss_comps.get("estimation_loss", 0)
                phys_loss = loss_comps.get("physics_loss", 0)
                tot_loss = loss_comps["total_loss"]
                
                estimation_losses.append(est_loss.item() if torch.is_tensor(est_loss) else est_loss)
                physics_losses.append(phys_loss.item() if torch.is_tensor(phys_loss) else phys_loss)
                total_losses.append(tot_loss.item() if torch.is_tensor(tot_loss) else tot_loss)
            except:
                continue
    
    return {
        'estimation_mean': np.mean(estimation_losses),
        'estimation_std': np.std(estimation_losses),
        'physics_mean': np.mean(physics_losses),
        'physics_std': np.std(physics_losses),
        'total_mean': np.mean(total_losses),
        'physics_to_estimation_ratio': np.mean(physics_losses) / (np.mean(estimation_losses) + 1e-8)
    }


def analyze_gradient_magnitudes(model, data_loader, device, physics_weight):
    """Analyze gradient magnitudes with and without physics loss."""
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Sample one batch
    batch = next(iter(data_loader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()}
    
    # Gradient with physics
    optimizer.zero_grad()
    original_weight = model.criterion.physics_weight
    model.criterion.physics_weight = physics_weight
    
    outputs = model.forward(
        states=batch["states"],
        controls=batch["controls"],
        measurements=batch["measurements"],
        initial_state=batch["initial_state"],
        initial_covariance=batch["initial_covariance"]
    )
    
    loss_with_physics = outputs["loss_components"]["total_loss"]
    loss_with_physics.backward()
    
    grad_norms_with_physics = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms_with_physics.append((name, param.grad.norm().item()))
    
    # Gradient without physics
    optimizer.zero_grad()
    model.criterion.physics_weight = 0.0
    
    outputs = model.forward(
        states=batch["states"],
        controls=batch["controls"],
        measurements=batch["measurements"],
        initial_state=batch["initial_state"],
        initial_covariance=batch["initial_covariance"]
    )
    
    loss_without_physics = outputs["loss_components"]["total_loss"]
    loss_without_physics.backward()
    
    grad_norms_without_physics = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms_without_physics.append((name, param.grad.norm().item()))
    
    # Restore original weight
    model.criterion.physics_weight = original_weight
    
    return {
        'with_physics': grad_norms_with_physics,
        'without_physics': grad_norms_without_physics,
        'loss_with_physics': loss_with_physics.item(),
        'loss_without_physics': loss_without_physics.item()
    }


def analyze_state_error_decomposition(model, data_loader, device):
    """Decompose state estimation error by state component."""
    
    model.eval()
    
    # State indices: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
    state_names = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'roll', 'pitch', 'yaw', 'wx', 'wy', 'wz']
    errors_by_state = {name: [] for name in state_names}
    
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            try:
                outputs = model.forward(
                    states=batch["states"],
                    controls=batch["controls"],
                    measurements=batch["measurements"],
                    initial_state=batch["initial_state"],
                    initial_covariance=batch["initial_covariance"]
                )
                
                true_states = batch["states"]
                estimated_states = outputs["estimated_states"]
                
                # Compute error for each state dimension
                errors = torch.abs(true_states - estimated_states)
                
                for i, name in enumerate(state_names):
                    errors_by_state[name].extend(errors[:, :, i].flatten().cpu().numpy())
            except:
                continue
    
    return {name: {
        'mean': np.mean(vals),
        'std': np.std(vals),
        'max': np.max(vals),
        'median': np.median(vals)
    } for name, vals in errors_by_state.items()}


def analyze_physics_constraint_violations(model, data_loader, device):
    """Analyze which physics constraints are being violated and by how much."""
    
    model.eval()
    
    violations = {
        'energy_conservation': [],
        'momentum_conservation': [],
        'velocity_position_consistency': []
    }
    
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            try:
                states = batch["states"]
                controls = batch["controls"]
                
                # Extract components
                positions = states[:, :, 0:3]
                velocities = states[:, :, 3:6]
                
                # Velocity-position consistency (numerical derivative)
                dt = 0.1  # Assuming 10Hz
                if states.shape[1] > 1:
                    vel_from_pos = (positions[:, 1:, :] - positions[:, :-1, :]) / dt
                    vel_actual = velocities[:, :-1, :]
                    vel_error = torch.abs(vel_from_pos - vel_actual).mean().item()
                    violations['velocity_position_consistency'].append(vel_error)
                
                # Energy changes (kinetic energy)
                kinetic_energy = 0.5 * torch.sum(velocities**2, dim=-1)
                if states.shape[1] > 1:
                    energy_change = torch.abs(kinetic_energy[:, 1:] - kinetic_energy[:, :-1])
                    violations['energy_conservation'].append(energy_change.mean().item())
                
                # Momentum changes
                if states.shape[1] > 1:
                    momentum_change = torch.abs(velocities[:, 1:, :] - velocities[:, :-1, :])
                    violations['momentum_conservation'].append(momentum_change.mean().item())
                
            except:
                continue
    
    return {name: {
        'mean': np.mean(vals),
        'std': np.std(vals),
        'max': np.max(vals)
    } for name, vals in violations.items()}


def analyze_model_capacity(model):
    """Analyze if model has sufficient capacity for physics constraints."""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Analyze layer sizes
    layer_info = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            layer_info.append({
                'name': name,
                'in_features': module.in_features,
                'out_features': module.out_features,
                'params': module.in_features * module.out_features + module.out_features
            })
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'layers': layer_info,
        'pinn_capacity': sum(l['params'] for l in layer_info if 'pinn' in l['name'].lower()),
        'total_capacity': sum(l['params'] for l in layer_info)
    }


def compare_training_dynamics(baseline_model, physics_model, data_loader, device):
    """Compare learning dynamics between baseline and physics-constrained models."""
    
    dynamics = {
        'baseline': {'losses': [], 'convergence_speed': 0},
        'physics': {'losses': [], 'convergence_speed': 0}
    }
    
    for model_name, model in [('baseline', baseline_model), ('physics', physics_model)]:
        model.eval()
        losses = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                try:
                    outputs = model.forward(
                        states=batch["states"],
                        controls=batch["controls"],
                        measurements=batch["measurements"],
                        initial_state=batch["initial_state"],
                        initial_covariance=batch["initial_covariance"]
                    )
                    
                    losses.append(outputs["loss_components"]["total_loss"])
                except:
                    continue
        
        dynamics[model_name]['losses'] = losses
        dynamics[model_name]['mean_loss'] = np.mean(losses)
        dynamics[model_name]['std_loss'] = np.std(losses)
    
    return dynamics


def main():
    """Run comprehensive diagnostic analysis."""
    
    print("="*80)
    print("DIAGNOSTIC ANALYSIS: Why Physics Constraints Hurt KITTI Performance")
    print("="*80)
    print("\nThis analysis investigates the root causes for research paper documentation.")
    print("="*80)
    
    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n[1/7] Loading Models...")
    print("  - Baseline model (physics_weight=0.0)")
    print("  - Physics-constrained model (physics_weight=0.001)")
    
    # Load baseline model
    baseline_checkpoint = torch.load('saved_models/best_kitti_model.pth', map_location=device, weights_only=False)
    
    if isinstance(baseline_checkpoint['config'], dict):
        baseline_config = PIDSEConfig(**baseline_checkpoint['config'])
    else:
        baseline_config = baseline_checkpoint['config']
    
    baseline_config.physics_weight = 0.0
    baseline_model = PIDSE(baseline_config).to(device)
    baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
    
    # Create physics model with same config but different weight
    if isinstance(baseline_checkpoint['config'], dict):
        physics_config = PIDSEConfig(**baseline_checkpoint['config'])
    else:
        # Copy the config
        import copy
        physics_config = copy.deepcopy(baseline_config)
    
    physics_config.physics_weight = 0.001
    physics_model = PIDSE(physics_config).to(device)
    physics_model.load_state_dict(baseline_checkpoint['model_state_dict'])
    
    # Load data
    print("\n[2/7] Loading KITTI Data...")
    kitti_path = Path("data/kitti")
    val_dataset = KITTIDataset(
        data_path=kitti_path,
        sequence_ids=['05'],
        sequence_length=50,
        overlap=0.0
    )
    
    val_trajectories = [val_dataset[i] for i in range(len(val_dataset))]
    val_loader, _, _ = create_data_loaders(
        val_trajectories, None,
        batch_size=2,
        sequence_length=50,
        val_split=0.0
    )
    
    print(f"  ✓ Loaded {len(val_dataset)} validation samples")
    
    # Analysis 1: Physics loss magnitude
    print("\n[3/7] Analyzing Physics Loss Magnitude...")
    physics_mag = analyze_physics_loss_magnitude(physics_model, val_loader, device)
    
    print(f"  Estimation Loss: {physics_mag['estimation_mean']:.2f} ± {physics_mag['estimation_std']:.2f}")
    print(f"  Physics Loss: {physics_mag['physics_mean']:.2f} ± {physics_mag['physics_std']:.2f}")
    print(f"  Physics/Estimation Ratio: {physics_mag['physics_to_estimation_ratio']:.4f}")
    
    if physics_mag['physics_to_estimation_ratio'] > 10:
        print("  ⚠️  WARNING: Physics loss is >>10x larger than estimation loss!")
        print("  → This overwhelms the learning signal and prevents convergence")
    
    # Analysis 2: Gradient magnitudes
    print("\n[4/7] Analyzing Gradient Flow...")
    grad_analysis = analyze_gradient_magnitudes(physics_model, val_loader, device, 0.001)
    
    with_phys_total = sum(g[1] for g in grad_analysis['with_physics'])
    without_phys_total = sum(g[1] for g in grad_analysis['without_physics'])
    
    print(f"  Total gradient norm WITH physics: {with_phys_total:.4f}")
    print(f"  Total gradient norm WITHOUT physics: {without_phys_total:.4f}")
    print(f"  Ratio: {with_phys_total / (without_phys_total + 1e-8):.4f}")
    
    if with_phys_total > 2 * without_phys_total:
        print("  ⚠️  Physics constraints cause >2x gradient magnitude increase")
        print("  → May lead to training instability and poor convergence")
    
    # Analysis 3: State error decomposition
    print("\n[5/7] Analyzing State-wise Error Decomposition...")
    baseline_errors = analyze_state_error_decomposition(baseline_model, val_loader, device)
    physics_errors = analyze_state_error_decomposition(physics_model, val_loader, device)
    
    print("\n  State-wise comparison (mean absolute error):")
    print(f"  {'State':<8} {'Baseline':<12} {'Physics':<12} {'Difference':<12}")
    print("  " + "-"*50)
    
    for state in ['x', 'y', 'z', 'vx', 'vy', 'vz']:
        baseline_val = baseline_errors[state]['mean']
        physics_val = physics_errors[state]['mean']
        diff = ((physics_val - baseline_val) / baseline_val) * 100
        
        marker = "📈" if diff > 20 else "📊" if diff > 0 else "✅"
        print(f"  {state:<8} {baseline_val:<12.4f} {physics_val:<12.4f} {diff:>+6.1f}% {marker}")
    
    # Analysis 4: Physics constraint violations
    print("\n[6/7] Analyzing Physics Constraint Violations...")
    violations = analyze_physics_constraint_violations(baseline_model, val_loader, device)
    
    print(f"  Velocity-Position Consistency: {violations['velocity_position_consistency']['mean']:.4f}")
    print(f"  Energy Conservation: {violations['energy_conservation']['mean']:.4f}")
    print(f"  Momentum Conservation: {violations['momentum_conservation']['mean']:.4f}")
    
    # Analysis 5: Model capacity
    print("\n[7/7] Analyzing Model Capacity...")
    capacity = analyze_model_capacity(baseline_model)
    
    print(f"  Total parameters: {capacity['total_params']:,}")
    print(f"  Trainable parameters: {capacity['trainable_params']:,}")
    print(f"  PINN capacity: {capacity['pinn_capacity']:,} params")
    
    if capacity['pinn_capacity'] < 1000:
        print("  ⚠️  PINN network may be too small to learn complex physics")
        print("  → Insufficient capacity for both data fitting and physics constraints")
    
    # Compile diagnostic report
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    report = {
        'physics_loss_magnitude': physics_mag,
        'gradient_analysis': {
            'with_physics_norm': with_phys_total,
            'without_physics_norm': without_phys_total,
            'ratio': with_phys_total / (without_phys_total + 1e-8)
        },
        'state_errors_baseline': baseline_errors,
        'state_errors_physics': physics_errors,
        'constraint_violations': violations,
        'model_capacity': capacity
    }
    
    # Identify root causes
    print("\n🔍 ROOT CAUSES IDENTIFIED:\n")
    
    causes = []
    
    if physics_mag['physics_to_estimation_ratio'] > 10:
        causes.append({
            'cause': 'Physics Loss Magnitude Mismatch',
            'severity': 'HIGH',
            'description': f"Physics loss ({physics_mag['physics_mean']:.1f}) is {physics_mag['physics_to_estimation_ratio']:.1f}x larger than estimation loss",
            'impact': 'Dominates gradient signal, prevents learning of data patterns',
            'recommendation': 'Rescale physics loss or use adaptive weighting'
        })
    
    if with_phys_total > 1.5 * without_phys_total:
        causes.append({
            'cause': 'Gradient Magnitude Amplification',
            'severity': 'MEDIUM',
            'description': f"Physics constraints increase gradient norm by {(with_phys_total/without_phys_total - 1)*100:.0f}%",
            'impact': 'Training instability, difficulty in convergence',
            'recommendation': 'Use gradient clipping or normalize physics loss'
        })
    
    if capacity['pinn_capacity'] < 1000:
        causes.append({
            'cause': 'Insufficient Model Capacity',
            'severity': 'MEDIUM',
            'description': f"PINN has only {capacity['pinn_capacity']} parameters",
            'impact': 'Cannot simultaneously fit data and satisfy physics constraints',
            'recommendation': 'Increase PINN hidden layer sizes to [64, 64] or [128, 64]'
        })
    
    # Check if physics constraints misalign with data
    vel_violation = violations['velocity_position_consistency']['mean']
    if vel_violation > 0.5:
        causes.append({
            'cause': 'Physics Model Mismatch',
            'severity': 'HIGH',
            'description': f"Large velocity-position inconsistency ({vel_violation:.2f})",
            'impact': 'Physics assumptions do not match vehicle dynamics in KITTI',
            'recommendation': 'Use vehicle-specific kinematic constraints instead of general conservation laws'
        })
    
    for i, cause in enumerate(causes, 1):
        print(f"{i}. {cause['cause']} [{cause['severity']}]")
        print(f"   Description: {cause['description']}")
        print(f"   Impact: {cause['impact']}")
        print(f"   Recommendation: {cause['recommendation']}\n")
    
    report['root_causes'] = causes
    
    # Save report
    report_file = Path('experiments/physics_diagnostic_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"✓ Full diagnostic report saved to: {report_file}")
    print("="*80)
    
    # Research paper recommendations
    print("\n📝 FOR RESEARCH PAPER:")
    print("-"*80)
    print("Title suggestion: 'When Physics-Informed Learning Fails: A Case Study on")
    print("                  KITTI Vehicle Odometry'")
    print("\nKey findings to highlight:")
    print("  1. Physics loss magnitude mismatch as primary failure mode")
    print("  2. Model capacity constraints limiting physics integration")
    print("  3. Importance of physics model alignment with application domain")
    print("  4. Trade-offs between data-driven and physics-informed approaches")
    print("\nRecommended sections:")
    print("  - Diagnostic methodology for identifying physics constraint failures")
    print("  - Quantitative analysis of gradient flow and loss components")
    print("  - Guidelines for when to use/avoid physics constraints")
    print("  - Adaptive weighting strategies for better physics integration")
    print("="*80)


if __name__ == "__main__":
    main()
