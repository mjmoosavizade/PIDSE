# Physics-Informed Differentiable State Estimator (PIDSE)

A novel framework that integrates machine learning with classical estimation theory for robust dynamics learning and optimal state estimation.

## Overview

PIDSE combines Physics-Informed Neural Networks (PINNs) with differentiable Extended Kalman Filters to:
- Learn complex system dynamics while respecting physical laws
- Automatically tune noise covariance matrices (Q and R)
- Provide robust state estimation for autonomous and robotic systems

## Key Features

- **Physics-Informed Dynamics Network (PINN)**: Learns residual dynamics while preserving known physics
- **Differentiable Extended Kalman Filter (D-EKF)**: End-to-end trainable state estimation
- **Hybrid Loss Function**: Balances estimation performance, physical consistency, and stability
- **Multi-Dataset Support**: Works with motion capture data, KITTI, EuroC, and custom datasets

## Architecture

```
PIDSE Framework:
├── Layer 1: Physics-Informed Dynamics Network (PINN)
│   ├── f_known: Known physical laws (rigid body dynamics)
│   └── f_PINN: Learned residual dynamics (θ parameters)
├── Layer 2: Differentiable Extended Kalman Filter (D-EKF)
│   ├── Learnable Q matrix (process noise covariance)
│   └── Learnable R matrix (measurement noise covariance)
└── Layer 3: Hybrid Loss Function
    ├── L_estimation: State estimation error (MSE)
    ├── L_physics: Physical consistency penalty
    └── L_regularization: Stability constraints
```

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from pidse import PIDSE, PIDSEConfig
from pidse.data import load_motion_capture_data

# Load data
data = load_motion_capture_data('path/to/dataset')

# Configure PIDSE
config = PIDSEConfig(
    state_dim=12,  # [position, velocity, orientation, angular_velocity]
    control_dim=4,  # control inputs
    measurement_dim=9,  # IMU + encoder measurements
    pinn_hidden_layers=[64, 64, 32],
    physics_weight=0.1,
    regularization_weight=0.01
)

# Initialize and train
pidse = PIDSE(config)
pidse.train(data, epochs=1000)

# Evaluate
metrics = pidse.evaluate(test_data)
print(f"ATE: {metrics['ate']:.3f}m, RPE: {metrics['rpe']:.3f}m")
```

## Project Structure

```
├── pidse/                  # Main package
│   ├── core/              # Core PIDSE components
│   ├── models/            # Neural network architectures
│   ├── filters/           # Kalman filter implementations
│   ├── losses/            # Loss functions
│   ├── data/              # Data handling utilities
│   └── utils/             # Helper functions
├── experiments/           # Experiment configurations and scripts
├── data/                 # Dataset storage
├── tests/                # Unit tests
├── docs/                 # Documentation
└── examples/             # Usage examples
```

## Research Background

This implementation is based on the research proposal:
**"Physics-Informed Differentiable State Estimator (PIDSE) for Robust Dynamics Learning"**

### Key Innovation
The PIDSE framework addresses critical limitations in autonomous systems by:
1. **Closing the Modeling Gap**: Learning unknown dynamics (friction, aerodynamic effects, component degradation) that are difficult to model from first principles
2. **Eliminating Manual Tuning**: Automatically optimizing noise covariance matrices Q and R through end-to-end training
3. **Ensuring Physical Consistency**: Enforcing known physical laws while learning residual dynamics

## Development

### Running Tests
```bash
pytest tests/
```

### Development Setup
```bash
pip install -e ".[dev]"
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for discussion.