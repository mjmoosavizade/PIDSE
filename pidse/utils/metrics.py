"""
Evaluation metrics for trajectory estimation.

This module implements standard metrics for evaluating trajectory estimation
performance, including ATE, RPE, and other trajectory-specific metrics.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


def compute_ate(
    predicted_trajectory: torch.Tensor,
    ground_truth_trajectory: torch.Tensor,
    align_trajectories: bool = True
) -> float:
    """
    Compute Absolute Trajectory Error (ATE).
    
    ATE measures the absolute distances between the predicted and ground truth
    trajectory after optimal alignment.
    
    Args:
        predicted_trajectory: Predicted positions [N, 3]
        ground_truth_trajectory: Ground truth positions [N, 3]
        align_trajectories: Whether to align trajectories before computing error
        
    Returns:
        ATE value (RMSE of position errors)
    """
    pred = predicted_trajectory.clone()
    gt = ground_truth_trajectory.clone()
    
    if align_trajectories:
        pred = align_trajectory(pred, gt)
    
    # Compute position errors
    position_errors = torch.norm(pred - gt, dim=1)
    
    # Root Mean Square Error
    ate = torch.sqrt(torch.mean(position_errors**2))
    
    return ate.item()


def compute_rpe(
    predicted_trajectory: torch.Tensor,
    ground_truth_trajectory: torch.Tensor,
    delta: int = 1,
    pose_relation: str = "translation_part"
) -> Dict[str, float]:
    """
    Compute Relative Pose Error (RPE).
    
    RPE measures the local consistency of the trajectory by comparing
    relative transformations over fixed time intervals.
    
    Args:
        predicted_trajectory: Predicted poses [N, 3] or [N, 7] (pos + quat)
        ground_truth_trajectory: Ground truth poses [N, 3] or [N, 7]
        delta: Time interval for relative pose computation
        pose_relation: Type of pose relation ("translation_part", "rotation_part", "full")
        
    Returns:
        Dictionary with RPE statistics
    """
    pred = predicted_trajectory
    gt = ground_truth_trajectory
    
    if len(pred) <= delta:
        return {"mean": 0.0, "std": 0.0, "rmse": 0.0, "median": 0.0, "max": 0.0}
    
    errors = []
    
    for i in range(len(pred) - delta):
        # Compute relative transformations
        if pose_relation == "translation_part":
            pred_rel = pred[i + delta] - pred[i]
            gt_rel = gt[i + delta] - gt[i]
            error = torch.norm(pred_rel - gt_rel)
        elif pose_relation == "rotation_part":
            # For rotation part, need to handle quaternions or angles
            if pred.shape[1] >= 6:  # Assume [pos(3), euler(3)] or similar
                pred_rel = pred[i + delta, 3:6] - pred[i, 3:6]
                gt_rel = gt[i + delta, 3:6] - gt[i, 3:6]
                error = torch.norm(pred_rel - gt_rel)
            else:
                error = torch.tensor(0.0)
        else:  # "full"
            pred_rel = pred[i + delta] - pred[i]
            gt_rel = gt[i + delta] - gt[i]
            error = torch.norm(pred_rel - gt_rel)
        
        errors.append(error.item())
    
    errors = np.array(errors)
    
    return {
        "mean": np.mean(errors),
        "std": np.std(errors),
        "rmse": np.sqrt(np.mean(errors**2)),
        "median": np.median(errors),
        "max": np.max(errors),
        "min": np.min(errors)
    }


def compute_trajectory_metrics(
    predicted_states: torch.Tensor,
    ground_truth_states: torch.Tensor,
    predicted_measurements: Optional[torch.Tensor] = None,
    ground_truth_measurements: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute comprehensive trajectory evaluation metrics.
    
    Args:
        predicted_states: Predicted state trajectory [N, state_dim]
        ground_truth_states: Ground truth state trajectory [N, state_dim]
        predicted_measurements: Predicted measurements [N, measurement_dim]
        ground_truth_measurements: Ground truth measurements [N, measurement_dim]
        
    Returns:
        Dictionary containing various evaluation metrics
    """
    metrics = {}
    
    # Extract positions (assume first 3 dimensions are position)
    pred_pos = predicted_states[:, :3]
    gt_pos = ground_truth_states[:, :3]
    
    # Absolute Trajectory Error
    metrics["ate"] = compute_ate(pred_pos, gt_pos)
    
    # Relative Pose Error
    rpe_trans = compute_rpe(pred_pos, gt_pos, delta=1)
    metrics["rpe_trans_mean"] = rpe_trans["mean"]
    metrics["rpe_trans_rmse"] = rpe_trans["rmse"]
    
    # RPE over longer intervals
    rpe_trans_long = compute_rpe(pred_pos, gt_pos, delta=10)
    metrics["rpe_trans_long_mean"] = rpe_trans_long["mean"]
    
    # Velocity errors (if available)
    if predicted_states.shape[1] >= 6:
        pred_vel = predicted_states[:, 3:6]
        gt_vel = ground_truth_states[:, 3:6]
        
        vel_errors = torch.norm(pred_vel - gt_vel, dim=1)
        metrics["velocity_rmse"] = torch.sqrt(torch.mean(vel_errors**2)).item()
        metrics["velocity_mean_error"] = torch.mean(vel_errors).item()
    
    # Orientation errors (if available)
    if predicted_states.shape[1] >= 9:
        pred_ori = predicted_states[:, 6:9]
        gt_ori = ground_truth_states[:, 6:9]
        
        ori_errors = torch.norm(pred_ori - gt_ori, dim=1)
        metrics["orientation_rmse"] = torch.sqrt(torch.mean(ori_errors**2)).item()
        metrics["orientation_mean_error"] = torch.mean(ori_errors).item()
    
    # Measurement prediction errors (if available)
    if predicted_measurements is not None and ground_truth_measurements is not None:
        meas_errors = torch.norm(predicted_measurements - ground_truth_measurements, dim=1)
        metrics["measurement_rmse"] = torch.sqrt(torch.mean(meas_errors**2)).item()
        metrics["measurement_mean_error"] = torch.mean(meas_errors).item()
    
    # Trajectory smoothness metrics
    metrics.update(compute_smoothness_metrics(predicted_states, ground_truth_states))
    
    # Final drift (end-to-end error)
    final_error = torch.norm(pred_pos[-1] - gt_pos[-1])
    metrics["final_drift"] = final_error.item()
    
    return metrics


def compute_smoothness_metrics(
    predicted_states: torch.Tensor,
    ground_truth_states: torch.Tensor
) -> Dict[str, float]:
    """
    Compute trajectory smoothness metrics.
    
    Args:
        predicted_states: Predicted trajectory [N, state_dim]
        ground_truth_states: Ground truth trajectory [N, state_dim]
        
    Returns:
        Dictionary with smoothness metrics
    """
    metrics = {}
    
    # Extract positions
    pred_pos = predicted_states[:, :3]
    gt_pos = ground_truth_states[:, :3]
    
    if len(pred_pos) < 3:
        return {"jerk_error": 0.0, "acceleration_error": 0.0}
    
    # Compute accelerations (second derivatives)
    pred_accel = pred_pos[2:] - 2 * pred_pos[1:-1] + pred_pos[:-2]
    gt_accel = gt_pos[2:] - 2 * gt_pos[1:-1] + gt_pos[:-2]
    
    accel_errors = torch.norm(pred_accel - gt_accel, dim=1)
    metrics["acceleration_rmse"] = torch.sqrt(torch.mean(accel_errors**2)).item()
    
    # Compute jerk (third derivatives) if enough points
    if len(pred_pos) >= 4:
        pred_jerk = pred_accel[1:] - pred_accel[:-1]
        gt_jerk = gt_accel[1:] - gt_accel[:-1]
        
        jerk_errors = torch.norm(pred_jerk - gt_jerk, dim=1)
        metrics["jerk_rmse"] = torch.sqrt(torch.mean(jerk_errors**2)).item()
    else:
        metrics["jerk_rmse"] = 0.0
    
    return metrics


def align_trajectory(
    trajectory: torch.Tensor,
    reference: torch.Tensor,
    method: str = "horn"
) -> torch.Tensor:
    """
    Align trajectory to reference using optimal transformation.
    
    Args:
        trajectory: Trajectory to align [N, 3]
        reference: Reference trajectory [N, 3]
        method: Alignment method ("horn", "umeyama", "simple")
        
    Returns:
        Aligned trajectory [N, 3]
    """
    if method == "simple":
        # Simple translation alignment (center both trajectories)
        trajectory_centered = trajectory - torch.mean(trajectory, dim=0)
        reference_centered = reference - torch.mean(reference, dim=0)
        return trajectory_centered + torch.mean(reference, dim=0)
    
    elif method == "horn":
        # Horn's method for optimal rotation and translation
        return horn_alignment(trajectory, reference)
    
    elif method == "umeyama":
        # Umeyama's method (includes scaling)
        return umeyama_alignment(trajectory, reference)
    
    else:
        raise ValueError(f"Unknown alignment method: {method}")


def horn_alignment(
    trajectory: torch.Tensor,
    reference: torch.Tensor
) -> torch.Tensor:
    """
    Align trajectory using Horn's method (rotation + translation).
    
    Args:
        trajectory: Source points [N, 3]
        reference: Target points [N, 3]
        
    Returns:
        Aligned trajectory [N, 3]
    """
    # Center both point sets
    traj_centroid = torch.mean(trajectory, dim=0)
    ref_centroid = torch.mean(reference, dim=0)
    
    traj_centered = trajectory - traj_centroid
    ref_centered = reference - ref_centroid
    
    # Compute cross-covariance matrix
    H = traj_centered.T @ ref_centered
    
    # SVD
    U, S, Vt = torch.linalg.svd(H)
    
    # Compute rotation matrix
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply transformation
    aligned = (R @ traj_centered.T).T + ref_centroid
    
    return aligned


def umeyama_alignment(
    trajectory: torch.Tensor,
    reference: torch.Tensor
) -> torch.Tensor:
    """
    Align trajectory using Umeyama's method (rotation + translation + scaling).
    
    Args:
        trajectory: Source points [N, 3]
        reference: Target points [N, 3]
        
    Returns:
        Aligned trajectory [N, 3]
    """
    # Center both point sets
    traj_centroid = torch.mean(trajectory, dim=0)
    ref_centroid = torch.mean(reference, dim=0)
    
    traj_centered = trajectory - traj_centroid
    ref_centered = reference - ref_centroid
    
    # Compute scaling factor
    traj_scale = torch.sqrt(torch.sum(traj_centered**2))
    ref_scale = torch.sqrt(torch.sum(ref_centered**2))
    scale = ref_scale / (traj_scale + 1e-8)
    
    # Scale source points
    traj_scaled = traj_centered * scale
    
    # Compute cross-covariance matrix
    H = traj_scaled.T @ ref_centered
    
    # SVD
    U, S, Vt = torch.linalg.svd(H)
    
    # Compute rotation matrix
    R = Vt.T @ U.T
    
    # Ensure proper rotation
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply transformation
    aligned = (R @ traj_scaled.T).T + ref_centroid
    
    return aligned


def compute_covariance_metrics(
    predicted_covariances: torch.Tensor,
    estimation_errors: torch.Tensor
) -> Dict[str, float]:
    """
    Compute metrics for covariance estimation quality.
    
    Args:
        predicted_covariances: Predicted covariances [N, state_dim, state_dim]
        estimation_errors: Actual estimation errors [N, state_dim]
        
    Returns:
        Dictionary with covariance quality metrics
    """
    metrics = {}
    
    # Extract diagonal covariances (variances)
    predicted_variances = torch.diagonal(predicted_covariances, dim1=-2, dim1=-1)
    predicted_stds = torch.sqrt(predicted_variances + 1e-8)
    
    # Compute actual squared errors
    actual_squared_errors = estimation_errors**2
    
    # Normalized Estimation Error Squared (NEES)
    # Should be close to state_dim if covariances are well-calibrated
    nees = []
    for i in range(len(estimation_errors)):
        error = estimation_errors[i].unsqueeze(0)  # [1, state_dim]
        cov = predicted_covariances[i].unsqueeze(0)  # [1, state_dim, state_dim]
        try:
            cov_inv = torch.linalg.inv(cov)
            nees_val = (error @ cov_inv @ error.T).item()
            nees.append(nees_val)
        except:
            nees.append(0.0)
    
    nees = np.array(nees)
    metrics["nees_mean"] = np.mean(nees)
    metrics["nees_std"] = np.std(nees)
    
    # Coverage probability (what fraction of errors fall within predicted bounds)
    # For 1-sigma, should be ~68%
    coverage_1sigma = torch.mean(
        (torch.abs(estimation_errors) <= predicted_stds).float()
    ).item()
    
    # For 2-sigma, should be ~95%
    coverage_2sigma = torch.mean(
        (torch.abs(estimation_errors) <= 2 * predicted_stds).float()
    ).item()
    
    metrics["coverage_1sigma"] = coverage_1sigma
    metrics["coverage_2sigma"] = coverage_2sigma
    
    # Sharpness (average predicted uncertainty)
    metrics["mean_predicted_std"] = torch.mean(predicted_stds).item()
    
    # Calibration error (difference between predicted and actual uncertainty)
    calibration_error = torch.mean(torch.abs(predicted_stds - torch.sqrt(actual_squared_errors))).item()
    metrics["calibration_error"] = calibration_error
    
    return metrics


def compute_innovation_metrics(
    innovations: torch.Tensor,
    innovation_covariances: torch.Tensor
) -> Dict[str, float]:
    """
    Compute innovation sequence metrics for filter consistency.
    
    Args:
        innovations: Innovation sequence [N, measurement_dim]
        innovation_covariances: Innovation covariances [N, measurement_dim, measurement_dim]
        
    Returns:
        Dictionary with innovation metrics
    """
    metrics = {}
    
    # Normalized Innovation Squared (NIS)
    nis = []
    for i in range(len(innovations)):
        innov = innovations[i].unsqueeze(0)  # [1, measurement_dim]
        cov = innovation_covariances[i].unsqueeze(0)  # [1, measurement_dim, measurement_dim]
        try:
            cov_inv = torch.linalg.inv(cov)
            nis_val = (innov @ cov_inv @ innov.T).item()
            nis.append(nis_val)
        except:
            nis.append(0.0)
    
    nis = np.array(nis)
    metrics["nis_mean"] = np.mean(nis)
    metrics["nis_std"] = np.std(nis)
    
    # Innovation autocorrelation (should be close to zero for consistent filter)
    if len(innovations) > 1:
        autocorr = torch.corrcoef(torch.stack([innovations[:-1].flatten(), innovations[1:].flatten()]))[0, 1]
        metrics["innovation_autocorr"] = autocorr.item() if not torch.isnan(autocorr) else 0.0
    else:
        metrics["innovation_autocorr"] = 0.0
    
    return metrics