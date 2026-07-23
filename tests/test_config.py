# tests/test_config.py
"""Unit tests for configuration module."""


import torch
from pathlib import Path

from src.config import Config, default_config

class TestConfig:
    """Test configuration class."""

    def test_default_config_creation(self):
        config = default_config
        assert isinstance(config, Config)
        assert config.embedding_dim == 50
        assert config.hidden_dim == 128
        assert config.num_layers == 2
        assert config.dropout == 0.3
        assert config.bidirectional == False
        assert config.bias == True
        assert config.learning_rate == 0.001
        assert config.epochs == 50
        assert config.batch_size == 1
        assert config.threshold == 0.4

    def test_device_property(self):
        """Test that device property returns correct device."""
        config = Config()
        device = config.device
        assert isinstance(device, torch.device)
        assert device.type == "cuda"

    def test_checkpoint_path(self):
        """Test checkpoint path generation."""
        config = Config()
        path = config.checkpoint_path
        assert isinstance(path, Path)
        assert path.parent == Path("checkpoints")
        assert path.name == "best_model.pth"

    def test_vocab_path(self):
        """Test checkpoint path generation."""
        config = Config()
        path = config.vocab_path
        assert isinstance(path, Path)
        assert path.parent == Path("checkpoints")
        assert path.name == "vocab.pth"

    def test_generate_model_name(self):
        """Test checkpoint path generation."""
        config = Config()
        name = config.generate_model_name()
        assert name.startswith("model_epoch50_hidden128_")
        assert name.endswith(".pth")
        assert len(name) > 30

