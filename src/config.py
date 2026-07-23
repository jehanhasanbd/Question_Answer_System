# src/config.py
"""Configuration settings for the Q&A LSTM model."""
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    """Configuration class for model training and inference."""

    # Data path
    data_path: str = "Data/general_knowledge_qa.csv"

    # Model Parameters
    embedding_dim: int = 50
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = False
    bias: bool = True

    # Training parameters
    learning_rate: float = 0.001
    epochs: int = 50
    batch_size: int = 1
    shuffle_data: bool = True
    epoch_jump: int = 10

    # Inference parameters
    threshold: float = 0.4
    max_words: int = 1

    # Special tokens
    unk_token: str = "UNK"
    unk_index: int = 0

    # Checkpoint paths
    checkpoint_dir: str = 'checkpoints'
    model_filename: str = 'best_model.pth'
    vocab_filename: str = 'vocab.pth'

    @property
    def device(self):
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def checkpoint_path(self) -> Path:
        """Full path to model checkpoint file."""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        return Path(self.checkpoint_dir) / self.model_filename

    @property
    def vocab_path(self) -> Path:
        """Full path to vocabulary file."""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        return Path(self.checkpoint_dir) / self.vocab_filename

    def generate_model_name(self) -> str:
        """Generate a unique model name with timestamp."""
        timestampt = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"model_epoch{self.epochs}_hidden{self.hidden_dim}_{timestampt}.pth"


# Create default configuration
default_config = Config()