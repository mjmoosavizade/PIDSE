"""
Visualization utilities for PIDSE.

Basic plotting functions for trajectory visualization and analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Optional


def plot_trajectory(
    trajectories: Dict[str, torch.Tensor],
    labels: Optional[List[str]] = None,
    title: str = "Trajectory Comparison",
    save_path: Optional[str] = None
):
    """
    Plot 2D trajectory comparison.
    
    Args:
        trajectories: Dictionary of trajectories to plot
        labels: Labels for each trajectory
        title: Plot title
        save_path: Optional path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (name, traj) in enumerate(trajectories.items()):
        if isinstance(traj, torch.Tensor):
            traj = traj.detach().numpy()
        
        # Extract position (assume first 2 or 3 dims are position)
        if traj.shape[-1] >= 2:
            x, y = traj[:, 0], traj[:, 1]
            color = colors[i % len(colors)]
            label = labels[i] if labels and i < len(labels) else name
            ax.plot(x, y, color=color, label=label, linewidth=2)
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    plt.show()


def plot_estimation_error(
    errors: torch.Tensor,
    title: str = "Estimation Error",
    save_path: Optional[str] = None
):
    """
    Plot estimation error over time.
    
    Args:
        errors: Error values over time
        title: Plot title
        save_path: Optional path to save plot
    """
    if isinstance(errors, torch.Tensor):
        errors = errors.detach().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time_steps = np.arange(len(errors))
    ax.plot(time_steps, errors, 'b-', linewidth=2)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Error')
    ax.set_title(title)
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    plt.show()


def plot_loss_curves(
    losses: Dict[str, List[float]],
    title: str = "Training Loss Curves",
    save_path: Optional[str] = None
):
    """
    Plot training loss curves.
    
    Args:
        losses: Dictionary of loss curves
        title: Plot title
        save_path: Optional path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for loss_name, loss_values in losses.items():
        epochs = range(len(loss_values))
        ax.plot(epochs, loss_values, label=loss_name, linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    plt.show()