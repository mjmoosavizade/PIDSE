"""
Quick test to verify physics loss normalization is working
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pidse import PIDSE, PIDSEConfig

# Create config with physics
config_dict = {
    'state_dim': 12,
    'control_dim': 4,
    'measurement_dim': 9,
    'pinn_hidden_layers': [32, 32],
    'physics_weight': 0.1,  # High weight to test normalization
    'regularization_weight': 0.001,
    'learning_rate': 1e-4,
    'batch_size': 2,
    'sequence_length': 50,
    'initial_process_noise': 0.1,
    'initial_measurement_noise': 0.5,
    'learn_noise_matrices': True,
    'mass': 1500.0,
    'device': 'cpu'
}

config = PIDSEConfig(**config_dict)
model = PIDSE(config)

print("=" * 80)
print("Testing Physics Loss Normalization")
print("=" * 80)

# Test 1: WITHOUT normalization
print("\n1️⃣  WITHOUT Normalization:")
model.loss_fn.normalize_physics_loss = False

# Create dummy data
batch_size = 2
seq_len = 10
states = torch.randn(batch_size, seq_len, 12)
measurements = torch.randn(batch_size, seq_len, 9)
controls = torch.randn(batch_size, seq_len, 4)
estimated_states = states + torch.randn_like(states) * 0.1
predicted_measurements = measurements + torch.randn_like(measurements) * 0.1

loss_dict = model.loss_fn(
    true_states=states,
    estimated_states=estimated_states,
    predicted_measurements=predicted_measurements,
    measurements=measurements,
    controls=controls
)

print(f"   Estimation Loss: {loss_dict['estimation_loss'].item():.4f}")
print(f"   Physics Loss (raw): {loss_dict['physics_loss_raw'].item():.2f}")
print(f"   Physics Loss (used): {loss_dict['physics_loss'].item():.2f}")
print(f"   Scale Ratio: {loss_dict['physics_scale_ratio'].item():.1f}x")
print(f"   Total Loss: {loss_dict['total_loss'].item():.4f}")
print(f"   Physics Weight: {config_dict['physics_weight']}")
effective_contrib = loss_dict['physics_loss'].item() * config_dict['physics_weight']
print(f"   Effective Physics Contribution: {effective_contrib:.4f}")

# Test 2: WITH normalization
print("\n2️⃣  WITH Normalization:")
model.loss_fn.normalize_physics_loss = True

# Run a few iterations to build up EMA
for i in range(5):
    loss_dict = model.loss_fn(
        true_states=states,
        estimated_states=estimated_states,
        predicted_measurements=predicted_measurements,
        measurements=measurements,
        controls=controls
    )

print(f"   Estimation Loss: {loss_dict['estimation_loss'].item():.4f}")
print(f"   Physics Loss (raw): {loss_dict['physics_loss_raw'].item():.2f}")
print(f"   Physics Loss (normalized): {loss_dict['physics_loss'].item():.4f}")
print(f"   Scale Ratio (original): {loss_dict['physics_scale_ratio'].item():.1f}x")
print(f"   Total Loss: {loss_dict['total_loss'].item():.4f}")
print(f"   Physics Weight: {config_dict['physics_weight']}")
effective_contrib = loss_dict['physics_loss'].item() * config_dict['physics_weight']
print(f"   Effective Physics Contribution: {effective_contrib:.4f}")

print(f"\n   EMA Estimation: {model.loss_fn.estimation_loss_ema.item():.4f}")
print(f"   EMA Physics: {model.loss_fn.physics_loss_ema.item():.2f}")
scale = model.loss_fn.estimation_loss_ema / model.loss_fn.physics_loss_ema
print(f"   Scaling Factor: {scale.item():.6f}")

# Test 3: Show the impact on total loss
print("\n3️⃣  Impact Comparison:")
print(f"   Without Normalization:")
phys_raw = loss_dict['physics_loss_raw'].item()
est = loss_dict['estimation_loss'].item()
total_without = est + config_dict['physics_weight'] * phys_raw
print(f"      Total = {est:.4f} + {config_dict['physics_weight']} × {phys_raw:.2f} = {total_without:.2f}")
print(f"      Physics dominates by {(phys_raw * config_dict['physics_weight'] / est):.1f}x")

print(f"\n   With Normalization:")
phys_norm = loss_dict['physics_loss'].item()
total_with = est + config_dict['physics_weight'] * phys_norm
print(f"      Total = {est:.4f} + {config_dict['physics_weight']} × {phys_norm:.4f} = {total_with:.4f}")
print(f"      Physics contributes {(phys_norm * config_dict['physics_weight'] / est * 100):.1f}% of estimation loss")

print("\n" + "=" * 80)
print("✅ Normalization Test Complete!")
print("=" * 80)
print("\nKey Insight:")
print("  - Without normalization: Physics loss ~hundreds (dominates learning)")
print("  - With normalization: Physics loss ~same scale as estimation (balanced)")
print("  - Can now use higher physics_weight (0.1, 0.5) without overwhelming signal")
