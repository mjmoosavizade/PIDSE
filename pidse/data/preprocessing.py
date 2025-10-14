"""
Data preprocessing utilities for PIDSE.

This module provides functions for normalizing, augmenting, and preprocessing
trajectory data for training and evaluation.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy.spatial.transform import Rotation as R


def normalize_trajectory(
    trajectory: Dict[str, torch.Tensor],
    stats: Optional[Dict[str, torch.Tensor]] = None
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Normalize trajectory data.
    
    Args:
        trajectory: Dictionary containing states, controls, measurements
        stats: Normalization statistics (if None, computed from trajectory)
        
    Returns:
        Tuple of (normalized_trajectory, normalization_stats)
    """
    if stats is None:
        # Compute normalization statistics
        stats = {}
        for key in ['states', 'controls', 'measurements']:
            if key in trajectory:
                data = trajectory[key]
                stats[f'{key}_mean'] = torch.mean(data, dim=0, keepdim=True)
                stats[f'{key}_std'] = torch.std(data, dim=0, keepdim=True) + 1e-8
    
    # Apply normalization
    normalized = {}
    for key in trajectory:
        if key in ['states', 'controls', 'measurements']:
            mean = stats.get(f'{key}_mean', 0.0)
            std = stats.get(f'{key}_std', 1.0)
            normalized[key] = (trajectory[key] - mean) / std
        else:
            normalized[key] = trajectory[key]
    
    return normalized, stats


def denormalize_trajectory(
    normalized_trajectory: Dict[str, torch.Tensor],
    stats: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Denormalize trajectory data.
    
    Args:
        normalized_trajectory: Normalized trajectory data
        stats: Normalization statistics
        
    Returns:
        Denormalized trajectory
    """
    denormalized = {}
    for key in normalized_trajectory:
        if key in ['states', 'controls', 'measurements']:
            mean = stats.get(f'{key}_mean', 0.0)
            std = stats.get(f'{key}_std', 1.0)
            denormalized[key] = normalized_trajectory[key] * std + mean
        else:
            denormalized[key] = normalized_trajectory[key]
    
    return denormalized


def add_noise(
    trajectory: Dict[str, torch.Tensor],
    noise_config: Dict[str, float]
) -> Dict[str, torch.Tensor]:
    """
    Add noise to trajectory data for augmentation.
    
    Args:
        trajectory: Input trajectory
        noise_config: Noise configuration with scales for different components
        
    Returns:
        Noisy trajectory
    """
    noisy = trajectory.copy()
    
    # Add measurement noise
    if 'measurements' in trajectory and 'measurement_noise' in noise_config:
        noise_scale = noise_config['measurement_noise']
        noise = torch.randn_like(trajectory['measurements']) * noise_scale
        noisy['measurements'] = trajectory['measurements'] + noise
    
    # Add state noise (for ground truth corruption)
    if 'states' in trajectory and 'state_noise' in noise_config:
        noise_scale = noise_config['state_noise']
        noise = torch.randn_like(trajectory['states']) * noise_scale
        noisy['states'] = trajectory['states'] + noise
    
    # Add control noise
    if 'controls' in trajectory and 'control_noise' in noise_config:
        noise_scale = noise_config['control_noise']
        noise = torch.randn_like(trajectory['controls']) * noise_scale
        noisy['controls'] = trajectory['controls'] + noise
    
    return noisy


def resample_trajectory(
    trajectory: Dict[str, torch.Tensor],
    target_length: int,
    method: str = 'linear'
) -> Dict[str, torch.Tensor]:
    """
    Resample trajectory to target length.
    
    Args:
        trajectory: Input trajectory
        target_length: Target sequence length
        method: Interpolation method ('linear', 'nearest')
        
    Returns:
        Resampled trajectory
    """
    current_length = trajectory['states'].shape[0]
    
    if current_length == target_length:
        return trajectory
    
    # Create interpolation indices
    old_indices = torch.linspace(0, current_length - 1, current_length)
    new_indices = torch.linspace(0, current_length - 1, target_length)
    
    resampled = {}
    for key, data in trajectory.items():
        if isinstance(data, torch.Tensor) and data.dim() >= 2:
            # Interpolate each dimension
            resampled_data = []
            for dim in range(data.shape[1]):
                if method == 'linear':
                    resampled_dim = torch.interp(new_indices, old_indices, data[:, dim])
                else:  # nearest
                    indices = torch.round(new_indices).long().clamp(0, current_length - 1)
                    resampled_dim = data[indices, dim]
                resampled_data.append(resampled_dim)
            
            resampled[key] = torch.stack(resampled_data, dim=1)
        else:
            resampled[key] = data
    
    return resampled


def split_trajectory(
    trajectory: Dict[str, torch.Tensor],
    sequence_length: int,
    overlap: float = 0.0
) -> List[Dict[str, torch.Tensor]]:
    """
    Split long trajectory into shorter sequences.
    
    Args:
        trajectory: Input trajectory
        sequence_length: Length of each sequence
        overlap: Overlap ratio between sequences (0-1)
        
    Returns:
        List of trajectory sequences
    """
    total_length = trajectory['states'].shape[0]
    step_size = int(sequence_length * (1 - overlap))
    
    if step_size <= 0:
        step_size = 1
    
    sequences = []
    for start_idx in range(0, total_length - sequence_length + 1, step_size):
        end_idx = start_idx + sequence_length
        
        sequence = {}
        for key, data in trajectory.items():
            if isinstance(data, torch.Tensor) and data.dim() >= 1:
                sequence[key] = data[start_idx:end_idx]
            else:
                sequence[key] = data
        
        sequences.append(sequence)
    
    return sequences


def filter_trajectory(
    trajectory: Dict[str, torch.Tensor],
    filter_type: str = 'lowpass',
    cutoff_freq: float = 10.0,
    sampling_freq: float = 100.0
) -> Dict[str, torch.Tensor]:
    """
    Apply temporal filtering to trajectory data.
    
    Args:
        trajectory: Input trajectory
        filter_type: Type of filter ('lowpass', 'highpass', 'bandpass')
        cutoff_freq: Cutoff frequency in Hz
        sampling_freq: Sampling frequency in Hz
        
    Returns:
        Filtered trajectory
    """
    try:
        from scipy import signal
    except ImportError:
        print("Warning: scipy not available, skipping filtering")
        return trajectory
    
    # Design filter
    nyquist = sampling_freq / 2
    normalized_cutoff = cutoff_freq / nyquist
    
    if filter_type == 'lowpass':
        b, a = signal.butter(4, normalized_cutoff, btype='low')
    elif filter_type == 'highpass':
        b, a = signal.butter(4, normalized_cutoff, btype='high')
    else:
        print(f"Warning: unsupported filter type {filter_type}")
        return trajectory
    
    # Apply filter
    filtered = {}
    for key, data in trajectory.items():
        if isinstance(data, torch.Tensor) and data.dim() >= 2:
            # Filter each dimension
            filtered_data = []
            for dim in range(data.shape[1]):
                filtered_dim = signal.filtfilt(b, a, data[:, dim].numpy())
                filtered_data.append(torch.from_numpy(filtered_dim).float())
            
            filtered[key] = torch.stack(filtered_data, dim=1)
        else:
            filtered[key] = data
    
    return filtered


def align_timestamps(
    trajectories: List[Dict[str, torch.Tensor]],
    reference_timestamps: Optional[torch.Tensor] = None
) -> List[Dict[str, torch.Tensor]]:
    """
    Align multiple trajectories to common timestamps.
    
    Args:
        trajectories: List of trajectories with timestamps
        reference_timestamps: Reference timestamps (if None, use first trajectory)
        
    Returns:
        List of aligned trajectories
    """
    if not trajectories or 'timestamps' not in trajectories[0]:
        return trajectories
    
    if reference_timestamps is None:
        reference_timestamps = trajectories[0]['timestamps']
    
    aligned = []
    for traj in trajectories:
        aligned_traj = {}
        traj_timestamps = traj['timestamps']
        
        for key, data in traj.items():
            if key == 'timestamps':
                aligned_traj[key] = reference_timestamps
            elif isinstance(data, torch.Tensor) and data.dim() >= 2:
                # Interpolate to reference timestamps
                aligned_data = []
                for dim in range(data.shape[1]):
                    aligned_dim = torch.interp(reference_timestamps, traj_timestamps, data[:, dim])
                    aligned_data.append(aligned_dim)
                aligned_traj[key] = torch.stack(aligned_data, dim=1)
            else:
                aligned_traj[key] = data
        
        aligned.append(aligned_traj)
    
    return aligned


def convert_coordinates(
    trajectory: Dict[str, torch.Tensor],
    source_frame: str,
    target_frame: str,
    transformation: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """
    Convert trajectory between coordinate frames.
    
    Args:
        trajectory: Input trajectory
        source_frame: Source coordinate frame
        target_frame: Target coordinate frame
        transformation: 4x4 transformation matrix (if custom)
        
    Returns:
        Transformed trajectory
    """
    converted = trajectory.copy()
    
    # Common coordinate frame conversions
    if source_frame == 'body' and target_frame == 'world':
        # Convert from body frame to world frame
        if 'states' in trajectory:
            states = trajectory['states']
            if states.shape[1] >= 9:  # Has orientation
                positions = states[:, 0:3]
                orientations = states[:, 6:9]  # Assume Euler angles
                
                # Convert positions using orientation
                transformed_positions = []
                for i in range(len(positions)):
                    # Create rotation matrix from Euler angles
                    euler = orientations[i].numpy()
                    rotation_matrix = R.from_euler('xyz', euler).as_matrix()
                    rotation_tensor = torch.from_numpy(rotation_matrix).float()
                    
                    # Transform position
                    transformed_pos = rotation_tensor @ positions[i]
                    transformed_positions.append(transformed_pos)
                
                converted_states = states.clone()
                converted_states[:, 0:3] = torch.stack(transformed_positions)
                converted['states'] = converted_states
    
    elif transformation is not None:
        # Apply custom transformation
        if 'states' in trajectory:
            states = trajectory['states']
            positions = states[:, 0:3]
            
            # Apply transformation to positions
            ones = torch.ones(positions.shape[0], 1)
            homogeneous_pos = torch.cat([positions, ones], dim=1)  # [N, 4]
            transformed_pos = (transformation @ homogeneous_pos.T).T[:, 0:3]
            
            converted_states = states.clone()
            converted_states[:, 0:3] = transformed_pos
            converted['states'] = converted_states
    
    return converted


def validate_trajectory(
    trajectory: Dict[str, torch.Tensor],
    checks: List[str] = None
) -> Dict[str, bool]:
    """
    Validate trajectory data for common issues.
    
    Args:
        trajectory: Trajectory to validate
        checks: List of checks to perform
        
    Returns:
        Dictionary of validation results
    """
    if checks is None:
        checks = ['shape_consistency', 'finite_values', 'reasonable_ranges']
    
    results = {}
    
    # Check shape consistency
    if 'shape_consistency' in checks:
        shapes = {key: data.shape[0] for key, data in trajectory.items() 
                 if isinstance(data, torch.Tensor)}
        consistent = len(set(shapes.values())) <= 1
        results['shape_consistency'] = consistent
    
    # Check for finite values
    if 'finite_values' in checks:
        finite = True
        for key, data in trajectory.items():
            if isinstance(data, torch.Tensor):
                finite = finite and torch.all(torch.isfinite(data))
        results['finite_values'] = finite
    
    # Check reasonable value ranges
    if 'reasonable_ranges' in checks:
        reasonable = True
        
        if 'states' in trajectory:
            states = trajectory['states']
            # Check position ranges (assuming reasonable for most systems)
            if states.shape[1] >= 3:
                positions = states[:, 0:3]
                reasonable = reasonable and torch.all(torch.abs(positions) < 1000)  # < 1km
            
            # Check velocity ranges
            if states.shape[1] >= 6:
                velocities = states[:, 3:6]
                reasonable = reasonable and torch.all(torch.abs(velocities) < 100)  # < 100 m/s
        
        results['reasonable_ranges'] = reasonable
    
    return results


def compute_trajectory_statistics(
    trajectory: Dict[str, torch.Tensor]
) -> Dict[str, Dict[str, float]]:
    """
    Compute basic statistics for trajectory data.
    
    Args:
        trajectory: Input trajectory
        
    Returns:
        Dictionary of statistics for each data type
    """
    stats = {}
    
    for key, data in trajectory.items():
        if isinstance(data, torch.Tensor) and data.dim() >= 2:
            stats[key] = {
                'mean': torch.mean(data).item(),
                'std': torch.std(data).item(),
                'min': torch.min(data).item(),
                'max': torch.max(data).item(),
                'shape': list(data.shape)
            }
            
            # Per-dimension statistics
            if data.shape[1] <= 12:  # Only for reasonable number of dimensions
                for dim in range(data.shape[1]):
                    stats[key][f'dim_{dim}_mean'] = torch.mean(data[:, dim]).item()
                    stats[key][f'dim_{dim}_std'] = torch.std(data[:, dim]).item()
    
    return stats