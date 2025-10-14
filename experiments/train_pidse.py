"""
Training script for PIDSE.

This script provides a complete training pipeline for the PIDSE model
with configuration management, logging, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
import json
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any
import numpy as np

# PIDSE imports
from pidse import PIDSE, PIDSEConfig
from pidse.data import create_data_loaders, create_synthetic_dataset
from pidse.utils import compute_trajectory_metrics, plot_trajectory, plot_loss_curves


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: Path) -> None:
    """Save configuration to file."""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_pidse_config(config: Dict[str, Any]) -> PIDSEConfig:
    """Create PIDSEConfig from dictionary."""
    model_config = config.get('model', {})
    
    return PIDSEConfig(
        state_dim=model_config.get('state_dim', 12),
        control_dim=model_config.get('control_dim', 4),
        measurement_dim=model_config.get('measurement_dim', 9),
        pinn_hidden_layers=model_config.get('pinn_hidden_layers', [64, 64, 32]),
        dynamics_activation=model_config.get('dynamics_activation', 'tanh'),
        measurement_activation=model_config.get('measurement_activation', 'relu'),
        learning_rate=model_config.get('learning_rate', 1e-3),
        batch_size=model_config.get('batch_size', 32),
        sequence_length=model_config.get('sequence_length', 50),
        physics_weight=model_config.get('physics_weight', 0.1),
        regularization_weight=model_config.get('regularization_weight', 0.01),
        initial_process_noise=model_config.get('initial_process_noise', 1e-3),
        initial_measurement_noise=model_config.get('initial_measurement_noise', 1e-2),
        learn_noise_matrices=model_config.get('learn_noise_matrices', True),
        enforce_energy_conservation=model_config.get('enforce_energy_conservation', True),
        enforce_momentum_conservation=model_config.get('enforce_momentum_conservation', True),
        mass=model_config.get('mass', None),
        device=model_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    )


def train_epoch(
    model: PIDSE,
    train_loader,
    optimizer: optim.Optimizer,
    epoch: int,
    logger: logging.Logger,
    writer: SummaryWriter
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    epoch_losses = {
        'total_loss': [],
        'estimation_loss': [],
        'physics_loss': [],
        'regularization_loss': []
    }
    
    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        batch = {k: v.to(model.config.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Training step
        loss_dict = model.train_step(batch, optimizer)
        
        # Accumulate losses
        for key in epoch_losses:
            if key in loss_dict:
                epoch_losses[key].append(loss_dict[key])
        
        # Log batch losses
        if batch_idx % 10 == 0:
            logger.info(
                f'Epoch {epoch}, Batch {batch_idx}: '
                f'Loss = {loss_dict["total_loss"]:.4f} '
                f'(Est: {loss_dict.get("estimation_loss", 0):.4f}, '
                f'Phys: {loss_dict.get("physics_loss", 0):.4f}, '
                f'Reg: {loss_dict.get("regularization_loss", 0):.4f})'
            )
        
        # Log to tensorboard
        global_step = epoch * len(train_loader) + batch_idx
        for key, value in loss_dict.items():
            writer.add_scalar(f'train_batch/{key}', value, global_step)
    
    # Average losses over epoch
    avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
    
    # Log epoch averages
    logger.info(f'Epoch {epoch} - Average losses: {avg_losses}')
    for key, value in avg_losses.items():
        writer.add_scalar(f'train_epoch/{key}', value, epoch)
    
    return avg_losses


def validate_epoch(
    model: PIDSE,
    val_loader,
    epoch: int,
    logger: logging.Logger,
    writer: SummaryWriter
) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    val_losses = {
        'total_loss': [],
        'estimation_loss': [],
        'physics_loss': [],
        'regularization_loss': []
    }
    
    trajectory_metrics = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Move batch to device
            batch = {k: v.to(model.config.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model.forward(
                states=batch["states"],
                controls=batch["controls"],
                measurements=batch["measurements"],
                initial_state=batch["initial_state"],
                initial_covariance=batch["initial_covariance"]
            )
            
            # Accumulate losses
            loss_components = outputs["loss_components"]
            for key in val_losses:
                if key in loss_components:
                    val_losses[key].append(loss_components[key].item())
            
            # Compute trajectory metrics for first sequence in batch
            if batch_idx == 0:
                true_states = batch["states"][0].cpu()
                estimated_states = outputs["estimated_states"][0].cpu()
                
                metrics = compute_trajectory_metrics(estimated_states, true_states)
                trajectory_metrics.append(metrics)
    
    # Average losses
    avg_losses = {key: np.mean(values) for key, values in val_losses.items()}
    
    # Average trajectory metrics
    if trajectory_metrics:
        avg_metrics = {}
        for key in trajectory_metrics[0]:
            avg_metrics[key] = np.mean([m[key] for m in trajectory_metrics])
    else:
        avg_metrics = {}
    
    # Log validation results
    logger.info(f'Validation Epoch {epoch} - Losses: {avg_losses}')
    logger.info(f'Validation Epoch {epoch} - Metrics: {avg_metrics}')
    
    # Log to tensorboard
    for key, value in avg_losses.items():
        writer.add_scalar(f'val/{key}', value, epoch)
    
    for key, value in avg_metrics.items():
        writer.add_scalar(f'metrics/{key}', value, epoch)
    
    return avg_losses, avg_metrics


def save_checkpoint(
    model: PIDSE,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: Path,
    is_best: bool = False
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': model.config
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = checkpoint_dir / 'best_model.pth'
        torch.save(checkpoint, best_path)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train PIDSE model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='./experiments', help='Output directory')
    parser.add_argument('--experiment_name', type=str, help='Experiment name')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = args.experiment_name or f'pidse_experiment_{timestamp}'
    experiment_dir = Path(args.output_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(experiment_dir)
    logger.info(f'Starting experiment: {experiment_name}')
    logger.info(f'Experiment directory: {experiment_dir}')
    
    # Save configuration
    save_config(config, experiment_dir / 'config.yaml')
    
    # Setup tensorboard
    writer = SummaryWriter(experiment_dir / 'tensorboard')
    
    # Create PIDSE configuration
    pidse_config = create_pidse_config(config)
    logger.info(f'PIDSE Configuration: {pidse_config}')
    
    # Initialize model
    model = PIDSE(pidse_config)
    logger.info(f'Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters')
    
    # Setup optimizer
    optimizer_config = config.get('optimizer', {})
    optimizer = optim.Adam(
        model.parameters(),
        lr=optimizer_config.get('learning_rate', pidse_config.learning_rate),
        weight_decay=optimizer_config.get('weight_decay', 1e-5)
    )
    
    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )
    
    # Load data
    data_config = config.get('data', {})
    
    if data_config.get('use_synthetic', True):
        # Generate synthetic data
        logger.info('Generating synthetic data...')
        train_data = create_synthetic_dataset(
            n_trajectories=data_config.get('n_train_trajectories', 80),
            trajectory_length=data_config.get('trajectory_length', 200),
            state_dim=pidse_config.state_dim,
            control_dim=pidse_config.control_dim,
            measurement_dim=pidse_config.measurement_dim,
            system_type=data_config.get('system_type', 'quadrotor'),
            noise_level=data_config.get('noise_level', 0.1)
        )
        
        test_data = create_synthetic_dataset(
            n_trajectories=data_config.get('n_test_trajectories', 20),
            trajectory_length=data_config.get('trajectory_length', 200),
            state_dim=pidse_config.state_dim,
            control_dim=pidse_config.control_dim,
            measurement_dim=pidse_config.measurement_dim,
            system_type=data_config.get('system_type', 'quadrotor'),
            noise_level=data_config.get('noise_level', 0.1)
        )
        
        train_loader, val_loader, test_loader = create_data_loaders(
            train_data, test_data,
            batch_size=pidse_config.batch_size,
            sequence_length=pidse_config.sequence_length,
            val_split=data_config.get('val_split', 0.2)
        )
    else:
        # Load real data
        data_path = data_config.get('data_path')
        dataset_type = data_config.get('dataset_type', 'mocap')
        
        train_loader, val_loader, test_loader = create_data_loaders(
            data_path,
            batch_size=pidse_config.batch_size,
            sequence_length=pidse_config.sequence_length,
            dataset_type=dataset_type,
            val_split=data_config.get('val_split', 0.2)
        )
    
    logger.info(f'Data loaded - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches')
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=pidse_config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
    
    # Training loop
    training_config = config.get('training', {})
    num_epochs = training_config.get('num_epochs', 100)
    save_interval = training_config.get('save_interval', 10)
    
    logger.info(f'Starting training for {num_epochs} epochs')
    
    for epoch in range(start_epoch, num_epochs):
        # Train epoch
        train_losses = train_epoch(model, train_loader, optimizer, epoch, logger, writer)
        
        # Validate epoch
        val_losses, val_metrics = validate_epoch(model, val_loader, epoch, logger, writer)
        
        # Learning rate scheduling
        scheduler.step(val_losses['total_loss'])
        
        # Save checkpoint
        is_best = val_losses['total_loss'] < best_loss
        if is_best:
            best_loss = val_losses['total_loss']
            logger.info(f'New best model at epoch {epoch} with loss {best_loss:.4f}')
        
        if epoch % save_interval == 0 or is_best:
            checkpoint_dir = experiment_dir / 'checkpoints'
            checkpoint_dir.mkdir(exist_ok=True)
            save_checkpoint(model, optimizer, epoch, val_losses['total_loss'], 
                          checkpoint_dir, is_best)
    
    # Final evaluation on test set
    if test_loader is not None:
        logger.info('Running final evaluation on test set...')
        test_losses, test_metrics = validate_epoch(model, test_loader, num_epochs, logger, writer)
        
        # Save test results
        results = {
            'test_losses': test_losses,
            'test_metrics': test_metrics,
            'config': config
        }
        
        with open(experiment_dir / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f'Final test results: {test_metrics}')
    
    # Close tensorboard writer
    writer.close()
    logger.info('Training completed!')


if __name__ == '__main__':
    main()