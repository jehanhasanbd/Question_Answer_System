# tests/test_model.py
"""Unit tests for model module."""

import pytest
import torch
import torch.nn as nn

from src.model import QuesAnsLSTM, create_model
from src.config import Config

class TestModel:
    """Test LSTM model."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config(
            embedding_dim=16,
            hidden_dim=32,
            num_layers=2,
            dropout=0.2,
            bidirectional=False,
            bias=True
        )

    @pytest.fixture
    def vocab_size(self):
        """Create a test vocabulary size."""
        return 100

    @pytest.fixture
    def model(self, config, vocab_size):
        """Create a test model."""
        return QuesAnsLSTM(vocab_size, config)

    @pytest.fixture
    def sample_input(self):
        return torch.randint(0, 100, (1, 10))  # batch_size=1, seq_len=10

    def test_model_initialization(self, model, config, vocab_size):
        """Test model initialization."""
        assert isinstance(model, QuesAnsLSTM)

        assert isinstance(model.embedding, nn.Embedding)
        assert model.embedding.num_embeddings == vocab_size
        assert model.embedding.embedding_dim == config.embedding_dim

        assert isinstance(model.lstm, nn.LSTM)
        assert model.lstm.input_size == config.embedding_dim
        assert model.lstm.hidden_size == config.hidden_dim
        assert model.lstm.num_layers == config.num_layers
        assert model.lstm.bidirectional == config.bidirectional

        assert isinstance(model.fc, nn.Linear)
        assert model.fc.in_features == config.hidden_dim
        assert model.fc.out_features == vocab_size

    def test_forward_pass(self, model, sample_input):
        """Test forward pass of the model."""
        output = model(sample_input)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (sample_input.size(0), 100)  # batch_size x vocab_size
        assert torch.dtype == torch.float32

    def test_forward_pass_with_gradients(self, model, sample_input):
        """Test that gradients flow through the model."""

        output = model(sample_input)
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed for all parameters
        for param in model.parameters():
            assert param.grad is not None

    def test_model_device(self, model):
        """Test model device placement."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Check that a parameter is on the right device
        param_device = next(model.parameters()).device
        assert param_device.type == device.type

    def test_create_model(self, config, vocab_size):
        """Test model factory function."""
        model = create_model(vocab_size, config)
        assert isinstance(model, QuesAnsLSTM)
        assert model.embedding.num_embeddings == vocab_size

    def test_model_parameters_count(self, model):
        """Test that model has reasonable number of parameters."""

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params

    def test_dropout_layers(self, model):
        """Test that dropout layers are present and functional."""
        dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout)]
        assert len(dropout_layers) >= 1
        assert all(isinstance(layer, nn.Dropout) for layer in dropout_layers)

    def test_bidirectional_model(self, config, vocab_size):
        """Test bidirectional LSTM configuration."""
        config.bidirectional = True
        model = QuesAnsLSTM(vocab_size, config)

        assert model.lstm.bidirectional == True
        # Bidirectional should double hidden dimension for FC layer
        assert model.fc.in_features == config.hidden_dim * 2
