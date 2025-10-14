"""
Data loading utilities and helper functions.

This module provides utilities for creating data loaders, preprocessing,
and data manipulation for PIDSE training.
"""

import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

from .datasets import PIDSEDataset, MotionCaptureDataset, KITTIDataset, EuroCDataset


def create_data_loaders(
    train_data: Union[List[Dict], str, Path],
    test_data: Optional[Union[List[Dict], str, Path]] = None,
    val_split: float = 0.2,
    batch_size: int = 32,
    sequence_length: int = 50,
    overlap: float = 0.5,
    num_workers: int = 4,
    dataset_type: str = "pidse",
    **kwargs
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        train_data: Training data (trajectories or path to data)
        test_data: Test data (optional)
        val_split: Validation split ratio
        batch_size: Batch size
        sequence_length: Sequence length for training
        overlap: Overlap between sequences
        num_workers: Number of data loading workers
        dataset_type: Type of dataset ('pidse', 'mocap', 'kitti', 'euroc')
        **kwargs: Additional arguments for dataset
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    if isinstance(train_data, (str, Path)):
        # Load from file
        train_dataset = _create_dataset_from_path(
            train_data, dataset_type, sequence_length, overlap, **kwargs
        )
    else:
        # Use provided trajectories
        train_dataset = PIDSEDataset(
            train_data, sequence_length, overlap, **kwargs
        )
    
    # Split training data into train/val
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    # Create train and validation loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_sequences
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_sequences
    )
    
    # Create test loader if test data provided
    test_loader = None
    if test_data is not None:
        if isinstance(test_data, (str, Path)):
            test_dataset = _create_dataset_from_path(
                test_data, dataset_type, sequence_length, overlap, normalize=False, **kwargs
            )
        else:
            test_dataset = PIDSEDataset(
                test_data, sequence_length, overlap, normalize=False, **kwargs
            )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_sequences
        )
    
    return train_loader, val_loader, test_loader


def _create_dataset_from_path(
    data_path: Union[str, Path],
    dataset_type: str,
    sequence_length: int,
    overlap: float,
    **kwargs
) -> PIDSEDataset:
    """Create dataset from file path."""
    if dataset_type == "mocap":
        return MotionCaptureDataset(data_path, sequence_length, overlap, **kwargs)
    elif dataset_type == "kitti":
        return KITTIDataset(data_path, sequence_length=sequence_length, overlap=overlap, **kwargs)
    elif dataset_type == "euroc":
        return EuroCDataset(data_path, sequence_length=sequence_length, overlap=overlap, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def collate_sequences(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching sequences.
    
    Args:
        batch: List of sequence dictionaries
        
    Returns:
        Batched sequences
    """
    # Stack sequences
    batched = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            batched[key] = torch.stack([item[key] for item in batch])
        else:
            batched[key] = [item[key] for item in batch]
    
    return batched


def load_motion_capture_data(
    data_path: Union[str, Path],
    format: str = "auto"
) -> List[Dict]:
    """
    Load motion capture data from various formats.
    
    Args:
        data_path: Path to data file
        format: Data format ('h5', 'npz', 'json', 'auto')
        
    Returns:
        List of trajectory dictionaries
    """
    data_path = Path(data_path)
    
    if format == "auto":
        format = data_path.suffix[1:]  # Remove dot
    
    if format == "h5":
        return _load_h5_mocap(data_path)
    elif format == "npz":
        return _load_npz_mocap(data_path)
    elif format == "json":
        return _load_json_mocap(data_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _load_h5_mocap(data_path: Path) -> List[Dict]:
    """Load motion capture data from HDF5 file."""
    import h5py
    
    trajectories = []
    with h5py.File(data_path, 'r') as f:
        for traj_name in f.keys():
            traj_group = f[traj_name]
            
            trajectory = {
                'states': torch.from_numpy(traj_group['states'][:]).float(),
                'controls': torch.from_numpy(traj_group['controls'][:]).float(),
                'measurements': torch.from_numpy(traj_group['measurements'][:]).float(),
                'metadata': dict(traj_group.attrs) if hasattr(traj_group, 'attrs') else {}
            }
            trajectories.append(trajectory)
    
    return trajectories


def _load_npz_mocap(data_path: Path) -> List[Dict]:
    """Load motion capture data from NPZ file."""
    data = np.load(data_path)
    
    trajectory = {
        'states': torch.from_numpy(data['states']).float(),
        'controls': torch.from_numpy(data['controls']).float(),
        'measurements': torch.from_numpy(data['measurements']).float(),
        'metadata': {}
    }
    
    return [trajectory]


def _load_json_mocap(data_path: Path) -> List[Dict]:
    """Load motion capture data from JSON file."""
    import json
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    trajectories = []
    for traj_data in data['trajectories']:
        trajectory = {
            'states': torch.tensor(traj_data['states']).float(),
            'controls': torch.tensor(traj_data['controls']).float(),
            'measurements': torch.tensor(traj_data['measurements']).float(),
            'metadata': traj_data.get('metadata', {})
        }
        trajectories.append(trajectory)
    
    return trajectories


def save_trajectories(
    trajectories: List[Dict],
    save_path: Union[str, Path],
    format: str = "h5"
) -> None:
    """
    Save trajectories to file.
    
    Args:
        trajectories: List of trajectory dictionaries
        save_path: Path to save file
        format: Save format ('h5', 'npz', 'json')
    """
    save_path = Path(save_path)
    
    if format == "h5":
        _save_h5_trajectories(trajectories, save_path)
    elif format == "npz":
        _save_npz_trajectories(trajectories, save_path)
    elif format == "json":
        _save_json_trajectories(trajectories, save_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _save_h5_trajectories(trajectories: List[Dict], save_path: Path) -> None:
    """Save trajectories to HDF5 file."""
    import h5py
    
    with h5py.File(save_path, 'w') as f:
        for i, traj in enumerate(trajectories):
            traj_group = f.create_group(f'trajectory_{i:04d}')
            
            # Save arrays
            traj_group.create_dataset('states', data=traj['states'].numpy())
            traj_group.create_dataset('controls', data=traj['controls'].numpy())
            traj_group.create_dataset('measurements', data=traj['measurements'].numpy())
            
            # Save metadata as attributes
            if 'metadata' in traj:
                for key, value in traj['metadata'].items():
                    traj_group.attrs[key] = value


def _save_npz_trajectories(trajectories: List[Dict], save_path: Path) -> None:
    """Save trajectories to NPZ file (single trajectory only)."""
    if len(trajectories) != 1:
        raise ValueError("NPZ format only supports single trajectory")
    
    traj = trajectories[0]
    np.savez(
        save_path,
        states=traj['states'].numpy(),
        controls=traj['controls'].numpy(),
        measurements=traj['measurements'].numpy()
    )


def _save_json_trajectories(trajectories: List[Dict], save_path: Path) -> None:
    """Save trajectories to JSON file."""
    import json
    
    data = {
        'trajectories': [
            {
                'states': traj['states'].tolist(),
                'controls': traj['controls'].tolist(),
                'measurements': traj['measurements'].tolist(),
                'metadata': traj.get('metadata', {})
            }
            for traj in trajectories
        ]
    }
    
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)


def create_synthetic_dataset(
    n_trajectories: int = 100,
    trajectory_length: int = 200,
    state_dim: int = 12,
    control_dim: int = 4,
    measurement_dim: int = 9,
    system_type: str = "quadrotor",
    noise_level: float = 0.1
) -> List[Dict]:
    """
    Create synthetic dataset for testing and development.
    
    Args:
        n_trajectories: Number of trajectories to generate
        trajectory_length: Length of each trajectory
        state_dim: State dimension
        control_dim: Control dimension
        measurement_dim: Measurement dimension
        system_type: Type of system ('quadrotor', 'ground_vehicle', 'generic')
        noise_level: Noise level for measurements
        
    Returns:
        List of synthetic trajectories
    """
    trajectories = []
    
    for _ in range(n_trajectories):
        if system_type == "quadrotor":
            trajectory = _generate_quadrotor_trajectory(
                trajectory_length, noise_level
            )
        elif system_type == "ground_vehicle":
            trajectory = _generate_vehicle_trajectory(
                trajectory_length, noise_level
            )
        else:
            trajectory = _generate_generic_trajectory(
                trajectory_length, state_dim, control_dim, measurement_dim, noise_level
            )
        
        trajectories.append(trajectory)
    
    return trajectories


def _generate_quadrotor_trajectory(
    length: int,
    noise_level: float
) -> Dict:
    """Generate synthetic quadrotor trajectory."""
    dt = 0.01
    
    # Initialize state [px, py, pz, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
    state = torch.zeros(12)
    state[2] = 1.0  # Start at 1m height
    
    states = []
    controls = []
    measurements = []
    
    for t in range(length):
        # Simple control policy: hover with perturbations
        control = torch.tensor([0.0, 0.0, 9.81, 0.0])  # Hover thrust
        control += noise_level * torch.randn(4)
        
        # Simple dynamics
        next_state = state.clone()
        
        # Position integration
        next_state[0:3] += state[3:6] * dt
        
        # Velocity integration with gravity
        acceleration = control[0:3] + torch.tensor([0.0, 0.0, -9.81])
        next_state[3:6] += acceleration * dt
        
        # Orientation integration
        next_state[6:9] += state[9:12] * dt
        
        # Angular velocity with damping
        next_state[9:12] = 0.9 * state[9:12] + 0.1 * control[3] * torch.randn(3)
        
        # Add nonlinear effects (what PINN should learn)
        drag = -0.1 * state[3:6] * torch.norm(state[3:6])
        next_state[3:6] += drag * dt
        
        # Generate measurements
        measurement = torch.cat([
            acceleration + noise_level * torch.randn(3),  # Accelerometer
            state[9:12] + noise_level * 0.5 * torch.randn(3),  # Gyroscope
            state[0:3] + noise_level * 2.0 * torch.randn(3)  # GPS/position
        ])
        
        states.append(next_state)
        controls.append(control)
        measurements.append(measurement)
        state = next_state
    
    return {
        'states': torch.stack(states),
        'controls': torch.stack(controls),
        'measurements': torch.stack(measurements),
        'metadata': {'system_type': 'quadrotor', 'dt': dt}
    }


def _generate_vehicle_trajectory(
    length: int,
    noise_level: float
) -> Dict:
    """Generate synthetic ground vehicle trajectory."""
    dt = 0.1
    
    # Initialize state [px, py, vx, vy, heading, heading_rate]
    state = torch.zeros(6)
    
    states = []
    controls = []
    measurements = []
    
    for t in range(length):
        # Simple control: [throttle, steering]
        control = torch.randn(2) * 0.5
        
        # Bicycle model dynamics
        next_state = state.clone()
        
        # Position update
        next_state[0] += state[2] * torch.cos(state[4]) * dt
        next_state[1] += state[2] * torch.sin(state[4]) * dt
        
        # Velocity update
        next_state[2] += control[0] * dt - 0.1 * state[2]  # Throttle with drag
        next_state[3] = 0.0  # No lateral velocity for simplicity
        
        # Heading update
        next_state[4] += state[5] * dt
        next_state[5] = control[1] * state[2] * 0.1  # Steering
        
        # Generate measurements
        measurement = torch.cat([
            state[0:2] + noise_level * torch.randn(2),  # GPS
            state[2:4] + noise_level * 0.5 * torch.randn(2),  # Velocity
            state[4:6] + noise_level * 0.1 * torch.randn(2)  # Heading
        ])
        
        states.append(next_state)
        controls.append(control)
        measurements.append(measurement)
        state = next_state
    
    return {
        'states': torch.stack(states),
        'controls': torch.stack(controls),
        'measurements': torch.stack(measurements),
        'metadata': {'system_type': 'ground_vehicle', 'dt': dt}
    }


def _generate_generic_trajectory(
    length: int,
    state_dim: int,
    control_dim: int,
    measurement_dim: int,
    noise_level: float
) -> Dict:
    """Generate generic synthetic trajectory."""
    # Simple linear system with noise
    A = torch.randn(state_dim, state_dim) * 0.1
    A += torch.eye(state_dim) * 0.9  # Stable system
    
    B = torch.randn(state_dim, control_dim) * 0.1
    C = torch.randn(measurement_dim, state_dim) * 0.1
    C += torch.eye(min(measurement_dim, state_dim), state_dim) # Partial observability
    
    state = torch.randn(state_dim) * 0.1
    states = []
    controls = []
    measurements = []
    
    for t in range(length):
        control = torch.randn(control_dim) * 0.5
        
        # Linear dynamics with noise
        next_state = A @ state + B @ control + noise_level * torch.randn(state_dim)
        
        # Measurements with noise
        measurement = C @ state + noise_level * torch.randn(measurement_dim)
        
        states.append(next_state)
        controls.append(control)
        measurements.append(measurement)
        state = next_state
    
    return {
        'states': torch.stack(states),
        'controls': torch.stack(controls),
        'measurements': torch.stack(measurements),
        'metadata': {'system_type': 'generic'}
    }