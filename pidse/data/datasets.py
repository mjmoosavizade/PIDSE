"""
Dataset classes for PIDSE training and evaluation.

This module provides dataset interfaces for various types of data:
- Motion capture data
- IMU/sensor data  
- Public datasets (KITTI, EuroC)
- Synthetic data
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import h5py
import json
from pathlib import Path


class PIDSEDataset(Dataset):
    """
    Base dataset class for PIDSE.
    
    Handles trajectory data with states, controls, and measurements.
    """
    
    def __init__(
        self,
        trajectories: List[Dict],
        sequence_length: int = 50,
        overlap: float = 0.5,
        normalize: bool = True,
        augment: bool = False
    ):
        """
        Initialize PIDSE dataset.
        
        Args:
            trajectories: List of trajectory dictionaries
            sequence_length: Length of training sequences
            overlap: Overlap ratio between sequences (0-1)
            normalize: Whether to normalize data
            augment: Whether to apply data augmentation
        """
        self.trajectories = trajectories
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.normalize = normalize
        self.augment = augment
        
        # Extract sequences from trajectories
        self.sequences = self._extract_sequences()
        
        # Compute normalization statistics
        if normalize:
            self.norm_stats = self._compute_normalization_stats()
        else:
            self.norm_stats = None
    
    def _extract_sequences(self) -> List[Dict]:
        """Extract fixed-length sequences from trajectories."""
        sequences = []
        step_size = int(self.sequence_length * (1 - self.overlap))
        
        for traj in self.trajectories:
            traj_length = traj['states'].shape[0]
            
            # Extract sequences with overlap
            for start_idx in range(0, traj_length - self.sequence_length + 1, step_size):
                end_idx = start_idx + self.sequence_length
                
                sequence = {
                    'states': traj['states'][start_idx:end_idx],
                    'controls': traj['controls'][start_idx:end_idx],
                    'measurements': traj['measurements'][start_idx:end_idx],
                    'initial_state': traj['states'][start_idx],
                    'initial_covariance': traj.get('initial_covariance', torch.eye(traj['states'].shape[-1]) * 0.1)
                }
                sequences.append(sequence)
        
        return sequences
    
    def _compute_normalization_stats(self) -> Dict:
        """Compute normalization statistics across all sequences."""
        all_states = torch.cat([seq['states'] for seq in self.sequences], dim=0)
        all_controls = torch.cat([seq['controls'] for seq in self.sequences], dim=0)
        all_measurements = torch.cat([seq['measurements'] for seq in self.sequences], dim=0)
        
        stats = {
            'states_mean': torch.mean(all_states, dim=0),
            'states_std': torch.std(all_states, dim=0) + 1e-8,
            'controls_mean': torch.mean(all_controls, dim=0),
            'controls_std': torch.std(all_controls, dim=0) + 1e-8,
            'measurements_mean': torch.mean(all_measurements, dim=0),
            'measurements_std': torch.std(all_measurements, dim=0) + 1e-8
        }
        
        return stats
    
    def normalize_data(self, data: Dict) -> Dict:
        """Apply normalization to data."""
        if self.norm_stats is None:
            return data
        
        normalized = data.copy()
        normalized['states'] = (data['states'] - self.norm_stats['states_mean']) / self.norm_stats['states_std']
        normalized['controls'] = (data['controls'] - self.norm_stats['controls_mean']) / self.norm_stats['controls_std']
        normalized['measurements'] = (data['measurements'] - self.norm_stats['measurements_mean']) / self.norm_stats['measurements_std']
        normalized['initial_state'] = (data['initial_state'] - self.norm_stats['states_mean']) / self.norm_stats['states_std']
        
        return normalized
    
    def denormalize_states(self, normalized_states: torch.Tensor) -> torch.Tensor:
        """Denormalize states back to original scale."""
        if self.norm_stats is None:
            return normalized_states
        
        return normalized_states * self.norm_stats['states_std'] + self.norm_stats['states_mean']
    
    def augment_data(self, data: Dict) -> Dict:
        """Apply data augmentation."""
        if not self.augment:
            return data
        
        augmented = data.copy()
        
        # Add noise to measurements
        noise_scale = 0.01
        measurement_noise = torch.randn_like(data['measurements']) * noise_scale
        augmented['measurements'] = data['measurements'] + measurement_noise
        
        # Slight temporal shifts
        if np.random.random() < 0.3:
            shift = np.random.randint(-2, 3)
            if shift != 0:
                if shift > 0:
                    augmented['states'] = torch.cat([data['states'][shift:], data['states'][-shift:]])
                    augmented['controls'] = torch.cat([data['controls'][shift:], data['controls'][-shift:]])
                    augmented['measurements'] = torch.cat([data['measurements'][shift:], data['measurements'][-shift:]])
                else:
                    augmented['states'] = torch.cat([data['states'][:shift], data['states'][:shift]])
                    augmented['controls'] = torch.cat([data['controls'][:shift], data['controls'][:shift]])
                    augmented['measurements'] = torch.cat([data['measurements'][:shift], data['measurements'][:shift]])
        
        return augmented
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx].copy()
        
        # Apply augmentation
        sequence = self.augment_data(sequence)
        
        # Apply normalization
        sequence = self.normalize_data(sequence)
        
        return sequence


class MotionCaptureDataset(PIDSEDataset):
    """
    Dataset for motion capture data (OptiTrack, Vicon, etc.).
    
    Handles high-frequency, high-precision ground truth data
    with corresponding sensor measurements.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        sequence_length: int = 50,
        overlap: float = 0.5,
        downsample_factor: int = 1,
        **kwargs
    ):
        """
        Initialize motion capture dataset.
        
        Args:
            data_path: Path to motion capture data file
            sequence_length: Length of training sequences
            overlap: Overlap ratio between sequences
            downsample_factor: Factor to downsample high-frequency data
        """
        self.data_path = Path(data_path)
        self.downsample_factor = downsample_factor
        
        # Load trajectories
        trajectories = self._load_mocap_data()
        
        super().__init__(trajectories, sequence_length, overlap, **kwargs)
    
    def _load_mocap_data(self) -> List[Dict]:
        """Load motion capture data from file."""
        if self.data_path.suffix == '.h5':
            return self._load_h5_data()
        elif self.data_path.suffix == '.npz':
            return self._load_npz_data()
        elif self.data_path.suffix == '.json':
            return self._load_json_data()
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
    
    def _load_h5_data(self) -> List[Dict]:
        """Load data from HDF5 file."""
        trajectories = []
        
        with h5py.File(self.data_path, 'r') as f:
            for traj_name in f.keys():
                traj_group = f[traj_name]
                
                # Load data arrays
                states = torch.from_numpy(traj_group['states'][:]).float()
                controls = torch.from_numpy(traj_group['controls'][:]).float()
                measurements = torch.from_numpy(traj_group['measurements'][:]).float()
                
                # Downsample if specified
                if self.downsample_factor > 1:
                    states = states[::self.downsample_factor]
                    controls = controls[::self.downsample_factor]
                    measurements = measurements[::self.downsample_factor]
                
                # Create trajectory dict
                trajectory = {
                    'states': states,
                    'controls': controls,
                    'measurements': measurements,
                    'metadata': dict(traj_group.attrs) if hasattr(traj_group, 'attrs') else {}
                }
                
                trajectories.append(trajectory)
        
        return trajectories
    
    def _load_npz_data(self) -> List[Dict]:
        """Load data from NPZ file."""
        data = np.load(self.data_path)
        trajectories = []
        
        # Assume data contains arrays: states, controls, measurements
        states = torch.from_numpy(data['states']).float()
        controls = torch.from_numpy(data['controls']).float()
        measurements = torch.from_numpy(data['measurements']).float()
        
        # Downsample if specified
        if self.downsample_factor > 1:
            states = states[::self.downsample_factor]
            controls = controls[::self.downsample_factor]
            measurements = measurements[::self.downsample_factor]
        
        trajectory = {
            'states': states,
            'controls': controls,
            'measurements': measurements,
            'metadata': {}
        }
        
        trajectories.append(trajectory)
        return trajectories
    
    def _load_json_data(self) -> List[Dict]:
        """Load data from JSON file (with numpy arrays)."""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        trajectories = []
        for traj_data in data['trajectories']:
            states = torch.tensor(traj_data['states']).float()
            controls = torch.tensor(traj_data['controls']).float()
            measurements = torch.tensor(traj_data['measurements']).float()
            
            # Downsample if specified
            if self.downsample_factor > 1:
                states = states[::self.downsample_factor]
                controls = controls[::self.downsample_factor]
                measurements = measurements[::self.downsample_factor]
            
            trajectory = {
                'states': states,
                'controls': controls,
                'measurements': measurements,
                'metadata': traj_data.get('metadata', {})
            }
            
            trajectories.append(trajectory)
        
        return trajectories


class KITTIDataset(PIDSEDataset):
    """
    Dataset for KITTI odometry data.
    
    Handles KITTI dataset format with GPS/IMU and visual odometry.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        sequence_ids: Optional[List[str]] = None,
        use_stereo: bool = True,
        **kwargs
    ):
        """
        Initialize KITTI dataset.
        
        Args:
            data_path: Path to KITTI dataset root
            sequence_ids: List of sequence IDs to load (e.g., ['00', '01'])
            use_stereo: Whether to use stereo camera data
        """
        self.data_path = Path(data_path)
        self.sequence_ids = sequence_ids or ['00', '01', '02']
        self.use_stereo = use_stereo
        
        # Load trajectories
        trajectories = self._load_kitti_data()
        
        super().__init__(trajectories, **kwargs)
    
    def _load_kitti_data(self) -> List[Dict]:
        """Load KITTI odometry data."""
        trajectories = []
        
        for seq_id in self.sequence_ids:
            seq_path = self.data_path / 'sequences' / seq_id
            
            if not seq_path.exists():
                print(f"Warning: Sequence {seq_id} not found at {seq_path}")
                continue
            
            # Load poses (ground truth)
            poses_file = self.data_path / 'poses' / f'{seq_id}.txt'
            if poses_file.exists():
                poses = self._load_kitti_poses(poses_file)
                
                # Convert poses to state representation
                states = self._poses_to_states(poses)
                
                # Generate mock controls and measurements for KITTI
                controls = self._generate_kitti_controls(states)
                measurements = self._generate_kitti_measurements(states)
                
                trajectory = {
                    'states': states,
                    'controls': controls,
                    'measurements': measurements,
                    'metadata': {'sequence_id': seq_id, 'dataset': 'kitti'}
                }
                
                trajectories.append(trajectory)
        
        return trajectories
    
    def _load_kitti_poses(self, poses_file: Path) -> torch.Tensor:
        """Load KITTI pose file."""
        poses = []
        with open(poses_file, 'r') as f:
            for line in f:
                pose = [float(x) for x in line.strip().split()]
                pose_matrix = np.array(pose).reshape(3, 4)
                poses.append(pose_matrix)
        
        return torch.tensor(np.array(poses)).float()
    
    def _poses_to_states(self, poses: torch.Tensor) -> torch.Tensor:
        """Convert KITTI poses to state representation."""
        num_poses = poses.shape[0]
        
        # Extract positions
        positions = poses[:, :, 3]  # [N, 3]
        
        # Extract rotations and convert to Euler angles (simplified)
        rotation_matrices = poses[:, :3, :3]  # [N, 3, 3]
        orientations = self._rotation_matrix_to_euler(rotation_matrices)
        
        # Compute velocities by finite differences
        dt = 0.1  # Assume 10Hz
        velocities = torch.zeros_like(positions)
        if num_poses > 1:
            velocities[1:] = (positions[1:] - positions[:-1]) / dt
        
        # Compute angular velocities
        angular_velocities = torch.zeros_like(orientations)
        if num_poses > 1:
            angular_velocities[1:] = (orientations[1:] - orientations[:-1]) / dt
        
        # Combine into state vector
        states = torch.cat([positions, velocities, orientations, angular_velocities], dim=1)
        
        return states
    
    def _rotation_matrix_to_euler(self, rotation_matrices: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrices to Euler angles (ZYX convention)."""
        # Simplified conversion - in practice, use proper rotation library
        roll = torch.atan2(rotation_matrices[:, 2, 1], rotation_matrices[:, 2, 2])
        pitch = torch.atan2(-rotation_matrices[:, 2, 0], 
                          torch.sqrt(rotation_matrices[:, 2, 1]**2 + rotation_matrices[:, 2, 2]**2))
        yaw = torch.atan2(rotation_matrices[:, 1, 0], rotation_matrices[:, 0, 0])
        
        return torch.stack([roll, pitch, yaw], dim=1)
    
    def _generate_kitti_controls(self, states: torch.Tensor) -> torch.Tensor:
        """Generate mock control inputs for KITTI (since not available)."""
        # Simple model: assume controls are proportional to acceleration
        velocities = states[:, 3:6]
        
        controls = torch.zeros(states.shape[0], 4)
        if states.shape[0] > 1:
            accelerations = velocities[1:] - velocities[:-1]
            controls[1:, :3] = accelerations
        
        return controls
    
    def _generate_kitti_measurements(self, states: torch.Tensor) -> torch.Tensor:
        """Generate mock IMU measurements for KITTI."""
        # Simple model: noisy position + velocity measurements
        positions = states[:, 0:3]
        velocities = states[:, 3:6]
        orientations = states[:, 6:9]
        
        # Add noise to simulate real measurements
        pos_noise = torch.randn_like(positions) * 0.1
        vel_noise = torch.randn_like(velocities) * 0.05
        ori_noise = torch.randn_like(orientations) * 0.02
        
        measurements = torch.cat([
            positions + pos_noise,
            velocities + vel_noise,
            orientations + ori_noise
        ], dim=1)
        
        return measurements


class EuroCDataset(PIDSEDataset):
    """
    Dataset for EuRoC MAV dataset.
    
    Handles EuRoC format with IMU, camera, and ground truth data.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        sequences: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize EuRoC dataset.
        
        Args:
            data_path: Path to EuRoC dataset root
            sequences: List of sequence names to load
        """
        self.data_path = Path(data_path)
        self.sequences = sequences or ['MH_01_easy', 'MH_02_easy']
        
        # Load trajectories
        trajectories = self._load_euroc_data()
        
        super().__init__(trajectories, **kwargs)
    
    def _load_euroc_data(self) -> List[Dict]:
        """Load EuRoC dataset."""
        trajectories = []
        
        for seq_name in self.sequences:
            seq_path = self.data_path / seq_name
            
            if not seq_path.exists():
                print(f"Warning: Sequence {seq_name} not found")
                continue
            
            # Load ground truth
            gt_file = seq_path / 'mav0' / 'state_groundtruth_estimate0' / 'data.csv'
            if gt_file.exists():
                states = self._load_euroc_groundtruth(gt_file)
                
                # Load IMU data
                imu_file = seq_path / 'mav0' / 'imu0' / 'data.csv'
                if imu_file.exists():
                    measurements = self._load_euroc_imu(imu_file, states.shape[0])
                    
                    # Generate mock controls
                    controls = torch.zeros(states.shape[0], 4)
                    
                    trajectory = {
                        'states': states,
                        'controls': controls,
                        'measurements': measurements,
                        'metadata': {'sequence': seq_name, 'dataset': 'euroc'}
                    }
                    
                    trajectories.append(trajectory)
        
        return trajectories
    
    def _load_euroc_groundtruth(self, gt_file: Path) -> torch.Tensor:
        """Load EuRoC ground truth data."""
        import pandas as pd
        
        df = pd.read_csv(gt_file)
        
        # Extract state components
        positions = df[['p_RS_R_x [m]', 'p_RS_R_y [m]', 'p_RS_R_z [m]']].values
        velocities = df[['v_RS_R_x [m s^-1]', 'v_RS_R_y [m s^-1]', 'v_RS_R_z [m s^-1]']].values
        
        # Convert quaternions to Euler angles
        quaternions = df[['q_RS_w []', 'q_RS_x []', 'q_RS_y []', 'q_RS_z []']].values
        orientations = self._quaternions_to_euler(quaternions)
        
        angular_velocities = df[['w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]']].values
        
        # Combine state
        states = np.concatenate([positions, velocities, orientations, angular_velocities], axis=1)
        
        return torch.tensor(states).float()
    
    def _load_euroc_imu(self, imu_file: Path, target_length: int) -> torch.Tensor:
        """Load EuRoC IMU data."""
        import pandas as pd
        
        df = pd.read_csv(imu_file)
        
        # Extract IMU measurements
        accel = df[['a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]']].values
        gyro = df[['w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]']].values
        
        # Combine measurements
        measurements = np.concatenate([accel, gyro], axis=1)
        
        # Resample to match ground truth length
        if len(measurements) != target_length:
            from scipy.interpolate import interp1d
            old_indices = np.linspace(0, len(measurements) - 1, len(measurements))
            new_indices = np.linspace(0, len(measurements) - 1, target_length)
            
            interpolated = []
            for i in range(measurements.shape[1]):
                f = interp1d(old_indices, measurements[:, i], kind='linear')
                interpolated.append(f(new_indices))
            
            measurements = np.column_stack(interpolated)
        
        # Add mock position measurements (zeros)
        pos_measurements = np.zeros((target_length, 3))
        measurements = np.concatenate([measurements, pos_measurements], axis=1)
        
        return torch.tensor(measurements).float()
    
    def _quaternions_to_euler(self, quaternions: np.ndarray) -> np.ndarray:
        """Convert quaternions to Euler angles."""
        # Simple conversion - in practice use proper rotation library
        w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.where(np.abs(sinp) >= 1, np.copysign(np.pi / 2, sinp), np.arcsin(sinp))
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.column_stack([roll, pitch, yaw])