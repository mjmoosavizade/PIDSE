"""Quick KITTI test to debug training issues"""

import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from pidse import PIDSE, PIDSEConfig
from pidse.data import KITTIDataset, create_data_loaders

print("🧪 Quick KITTI Training Test")
print("=" * 50)

# Minimal config
config = PIDSEConfig(
    state_dim=12,
    control_dim=4,
    measurement_dim=9,
    pinn_hidden_layers=[16, 16],
    learning_rate=1e-4,
    batch_size=1,  # Single batch
    sequence_length=20,  # Very short
    physics_weight=0.0,
    regularization_weight=0.001,
    initial_process_noise=0.1,
    initial_measurement_noise=0.5,
    learn_noise_matrices=False,  # Fixed for stability
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"✅ Config: {config.device}, {config.state_dim}D state")

# Load minimal data
dataset = KITTIDataset(
    data_path=Path('data/kitti'),
    sequence_ids=['00'],  # Just one sequence
    sequence_length=config.sequence_length
)

print(f"✅ Dataset: {len(dataset)} samples")

# Get one batch
train_loader, _, _ = create_data_loaders(
    [dataset[0], dataset[1]],  # Just 2 samples
    None,
    batch_size=1,
    sequence_length=config.sequence_length
)

print(f"✅ Data loader: {len(train_loader)} batches")

# Create model
model = PIDSE(config)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

print(f"✅ Model: {sum(p.numel() for p in model.parameters())} parameters")

# Try one training step
print("\n🔄 Testing one forward pass...")
model.train()

batch = next(iter(train_loader))
print(f"  Batch shapes: states={batch['states'].shape}, controls={batch['controls'].shape}")

try:
    loss_dict = model.train_step(batch, optimizer)
    print(f"✅ Forward pass SUCCESS!")
    print(f"   Loss: {loss_dict['total_loss']:.4f}")
    print(f"   Components: est={loss_dict.get('estimation_loss', 0):.4f}, "
          f"phys={loss_dict.get('physics_loss', 0):.4f}")
    
    # Try a few more steps
    print("\n🔄 Testing 5 training steps...")
    for i in range(5):
        loss_dict = model.train_step(batch, optimizer)
        print(f"   Step {i+1}: Loss = {loss_dict['total_loss']:.4f}")
    
    print("\n✅ KITTI training is functional!")
    print("   Issue might be with:")
    print("   - Large number of batches (204) taking too long")
    print("   - Need to reduce batch count or add progress bars")
    
except Exception as e:
    print(f"❌ Forward pass FAILED: {e}")
    import traceback
    traceback.print_exc()
