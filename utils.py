"""Utility functions for Tunix training pipeline."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch_xla.core.xla_model as xm


def save_config(config: Dict[str, Any], output_dir: str):
    """Save configuration to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info(f"Configuration saved to {config_file}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logging.info(f"Configuration loaded from {config_path}")
    return config


def get_device_info():
    """Get TPU device information."""
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    ordinal = xm.get_ordinal()
    
    info = {
        'device': str(device),
        'world_size': world_size,
        'ordinal': ordinal,
        'is_master': xm.is_master_ordinal(),
    }
    
    return info


def print_model_summary(model: torch.nn.Module):
    """Print model parameter summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model Summary:")
    print(f"  Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e9:.2f}B)")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def estimate_training_time(
    num_samples: int,
    batch_size: int,
    num_epochs: int,
    seconds_per_batch: float
) -> float:
    """Estimate total training time."""
    batches_per_epoch = num_samples // batch_size
    total_batches = batches_per_epoch * num_epochs
    total_seconds = total_batches * seconds_per_batch
    
    return total_seconds


def create_output_structure(output_dir: str):
    """Create standardized output directory structure."""
    output_path = Path(output_dir)
    
    directories = [
        output_path,
        output_path / "checkpoints",
        output_path / "logs",
        output_path / "metrics",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Output directory structure created at {output_dir}")


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Get the latest checkpoint from directory."""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        return None
    
    checkpoints = [
        d for d in checkpoint_path.iterdir() 
        if d.is_dir() and d.name.startswith("checkpoint-")
    ]
    
    if not checkpoints:
        return None
    
    def get_step(checkpoint_dir):
        try:
            return int(checkpoint_dir.name.split("-")[1])
        except (IndexError, ValueError):
            return -1
    
    latest = max(checkpoints, key=get_step)
    return str(latest)


def load_training_state(checkpoint_dir: str) -> Optional[Dict[str, Any]]:
    """Load training state from checkpoint."""
    state_file = Path(checkpoint_dir) / "training_state.json"
    
    if not state_file.exists():
        return None
    
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    return state


def calculate_metrics_summary(metrics_file: str) -> Dict[str, Any]:
    """Calculate summary statistics from metrics log."""
    if not Path(metrics_file).exists():
        return {}
    
    metrics_list = []
    with open(metrics_file, 'r') as f:
        for line in f:
            metrics_list.append(json.loads(line))
    
    if not metrics_list:
        return {}
    
    summary = {}
    
    metric_keys = set()
    for m in metrics_list:
        metric_keys.update(k for k in m.keys() if k not in ['step', 'timestamp'])
    
    for key in metric_keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            summary[key] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'final': values[-1],
            }
    
    return summary


def cleanup_old_checkpoints(
    checkpoint_dir: str, 
    keep_last_n: int = 3,
    keep_best: bool = True,
    keep_final: bool = True
):
    """Clean up old checkpoints, keeping only recent ones."""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        return
    
    protected = set()
    if keep_best:
        protected.add("checkpoint-best")
    if keep_final:
        protected.add("checkpoint-final")
    
    checkpoints = [
        d for d in checkpoint_path.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint-") and d.name not in protected
    ]
    
    def get_step(checkpoint_dir):
        try:
            return int(checkpoint_dir.name.split("-")[1])
        except (IndexError, ValueError):
            return -1
    
    checkpoints.sort(key=get_step, reverse=True)
    
    to_remove = checkpoints[keep_last_n:]
    
    for checkpoint in to_remove:
        import shutil
        shutil.rmtree(checkpoint)
        logging.info(f"Removed old checkpoint: {checkpoint}")


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_parameter_count_by_layer(model: torch.nn.Module) -> Dict[str, int]:
    """Get parameter count breakdown by layer."""
    layer_params = {}
    
    for name, param in model.named_parameters():
        layer_name = name.split('.')[0]
        if layer_name not in layer_params:
            layer_params[layer_name] = 0
        layer_params[layer_name] += param.numel()
    
    return layer_params


def memory_summary():
    """Print memory usage summary for TPU."""
    device = xm.xla_device()
    
    try:
        mem_info = xm.get_memory_info(device)

        print("TPU Memory Summary:")
        if mem_info:
            for key, value in mem_info.items():
                print(f"  {key}: {value / 1e9:.2f} GB")
    except Exception as e:
        logging.warning(f"Could not get memory info: {e}")


def verify_tpu_setup():
    """Verify TPU setup and configuration."""
    try:
        device = xm.xla_device()
        world_size = xm.xrt_world_size()
        ordinal = xm.get_ordinal()

        print("TPU Setup Verification:")
        print(f"  Device: {device}")
        print(f"  World size: {world_size}")
        print(f"  Current ordinal: {ordinal}")
        print(f"  Is master: {xm.is_master_ordinal()}")

        test_tensor = torch.ones(2, 2).to(device)
        _ = test_tensor + test_tensor
        xm.mark_step()

        print("  Test computation: PASSED")
        return True

    except Exception as e:
        print(f"  TPU verification failed: {e}")
        return False
