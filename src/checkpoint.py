# src/checkpoint.py
"""Checkpoint management utilities."""

import torch
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict
from datetime import datetime

from src.config import Config, default_config
from src.model import create_model

def list_checkpoints(checkpoint_dir: str= None) -> list:
    """List all available checkpoints."""
    if checkpoint_dir is None:
        checkpoint_dir = default_config.checkpoint_dir

    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"No checkpoint directory found at {checkpoint_path}")
        return []

    checkpoints = list(checkpoint_path.glob("*.pth"))
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    print(f"\nAvailable checkpoints in {checkpoint_dir}:")
    print("-" * 60)
    for cp in checkpoints:
        size = cp.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(cp.stat().st_mtime)
        print(f"{cp.name:30} {size:6.2f} MB  {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

    return checkpoints


def load_best_model(
    checkpoint_dir: str= None,
    device: str = None
) -> Tuple[torch.nn.Module, Dict, Config]:
    """
    Load the best model from checkpoint directory.

    Args:
        checkpoint_dir: Directory containing checkpoints
        device: Device to load model on

    Returns:
        Tuple of (model, vocab, config)
    """
    if checkpoint_dir is None:
        checkpoint_dir = default_config.checkpoint_dir

    if device is None:
        device = default_config.device

    checkpoint_path = Path(checkpoint_dir) / "best_model.pth"
    vocab_path = Path(checkpoint_dir) / "vocab.pth"
    config_path = Path(checkpoint_dir) / "config.pth"

    # Load config
    config = default_config
    if config_path.exists():
        with open(config_path, "rb") as f:
            config = pickle.load(f)

    # Load vocabulary
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found at {vocab_path}")

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Load Model
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best model not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    model = create_model(len(vocab), config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded best model from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint.get('loss', 'N/A')}")

    return model, vocab, config

def load_model(
        checkpoint_path: str,
        vocab_path: Optional[str] = None,
        config: Optional[Config] = None,
        device: Optional[str] = None
) -> Tuple[torch.nn.Module, Dict, Config]:
    """
    Load a specific model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        vocab_path: Path to vocabulary file
        config: Configuration object
        device: Device to load model on

    Returns:
        Tuple of (model, vocab, config)
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Determine paths
    checkpoint_dir = checkpoint_path.parent

    if vocab_path is None:
        vocab_path = checkpoint_dir / "vocab.pth"
    else:
        vocab_path = Path(vocab_path)

    if config is None:
        config_path = checkpoint_dir / "config.pth"
        if config_path.exists():
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
        else:
            config = default_config

    if device is None:
        device = config.device

    # Load vocabulary
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found at {vocab_path}")

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    model = create_model(len(vocab), config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint.get('loss', 'N/A')}")

    return model, vocab, config

def load_latest_model(
        checkpoint_dir: str = None,
        device: str = None
) -> Tuple[torch.nn.Module, Dict, Config]:
    """
    Load the latest model from checkpoint directory.

    Args:
        checkpoint_dir: Directory containing checkpoints
        device: Device to load model on

    Returns:
        Tuple of (model, vocab, config)
    """
    if checkpoint_dir is None:
        checkpoint_dir = default_config.checkpoint_dir

    if device is None:
        device = default_config.device

    checkpoint_path = Path(checkpoint_dir) / "latest_model.pth"

    if not checkpoint_path.exists():
        # If latest doesn't exist, try best
        return load_best_model(checkpoint_dir, device)

    return load_model(str(checkpoint_path), device=device)