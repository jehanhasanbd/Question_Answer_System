# tests/test_train.py
"""Unit tests for training module."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import pickle
from unittest.mock import Mock, patch

from src.train import (
    train_one_epoch,
    train_model,
    save_checkpoint
)
from src.config import Config
from src.model import create_model

class TestTraining:
    """Test training functions."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config(
            embedding_dim=16,
            hidden_dim=32,
            num_layers=1,
            dropout=0.1,
            batch_size=2,
            epochs=3,
            learning_rate=0.001,
            checkpoint_dir="test_checkpoints"
        )

    @pytest.fixture
    def moc_dataloader(self):
        """Create a mock dataloader."""
        class MockDataloader:
            def __init__(self):
                self.batch_size = 2
                self.shuffle = True

            def __iter__(self):
                # Yield some sample batches
                for _ in range(3):
                    question = torch.randint(0, 50, (2,10))
                    answer = torch.randint(0, 50, (2,5))
                    yield question, answer

        return MockDataloader()

    @pytest.fixture
    def model(self, config):
        """Create a test model."""
        return create_model(50, config)

    def test_train_one_epoch(self, model, mock_dataloader, config):
        """Test training for one epoch."""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        avg_loss = train_one_epoch(
            model, mock_dataloader, criterion, optimizer, config.device
        )

        assert isinstance(avg_loss, float)
        assert avg_loss > 0

    def test_train_one_epoch_updates_weights(self, model, mock_dataloader, config):
        """Test that training updates model weights."""
        # Get initial weights
        initial_wights = next(model.parameters()).data.clone()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        train_one_epoch(
            model, mock_dataloader, criterion, optimizer, config.device
        )

        # Get weights after training
        final_weight = next(model.parameters()).data

        assert not torch.equal(initial_wights, final_weight)

    def test_save_checkpoint(self, model, config, tmp_path):
        """Test checkpoint saving."""
        # Override checkpoint directory to temp path
        config.checkpoint_dir = str(tmp_path / "checkpoints")

        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        vocab = {
            "UNK": 0,
            "test": 1
        }

        save_checkpoint(
            model, optimizer, vocab, config, epoch=1, loss=0.5, is_best=True
        )

        # Check that files were created
        checkpoint_dir = Path(config.checkpoint_dir)
        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "best_model.pth").exists()
        assert (checkpoint_dir / "vocab.pth").exists()
        assert (checkpoint_dir / "config.pth").exists()

        # Verify vocabulary was saved correctly
        with open(checkpoint_dir / "vocab.pth", "rb") as f:
            saved_vocab = pickle.load(f)
            assert saved_vocab == vocab

    def test_train_model(self, tmp_path, config):
        """Test full training pipeline."""
        # Create minimal test data
        import pandas as pd
        data = pd.DataFrame({
            'question': ['What is this?', 'Who are you?'],
            'answer': ['test', 'unknown'],
            'question_type': ['test', 'test'],
            'image': ['', '']
        })

        csv_path = tmp_path / "test_data.csv"
        data.to_csv(csv_path, index=False)

        config.data_path = str(csv_path)
        config.checkpoint_dir = str(tmp_path / "checkpoints")
        config.epochs = 2

        model, losses, vocab = train_model(config,verbose=False, save_checkpoints=True)

        assert isinstance(model, nn.Module)
        assert len(losses) == config.epochs
        assert isinstance(vocab, dict)
        assert "UNK" in vocab

        # Check that checkpoints were saved
        checkpoint_dir = Path(config.checkpoint_dir)
        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "best_model.pth").exists()
        assert (checkpoint_dir / "vocab.pth").exists()
        assert (checkpoint_dir / "training_loss.pth").exists()