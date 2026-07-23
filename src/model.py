# src/model.py
"""Model definition for Question-Answering LSTM."""

import torch
import torch.nn as nn

from src.config import Config


class QuesAnsLSTM(nn.Module):
    """LSTM-based model for question answering."""

    def __init__(self, vocab_size: int, config: Config):
        """
            Initialize the model.

            Args:
                vocab_size: Size of vocabulary
                config: Configuration object
        """
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=0
        )
        self.embedding_dropout = nn.Dropout(config.dropout)

        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            bias=config.bias
        )

        # Regularization after LSTM
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.fc = nn.Linear(config.hidden_dim, vocab_size)

    def forward(self, question: torch.Tensor) -> torch.Tensor:
        """
            Forward pass of the model.

            Args:
                question: Input question tensor

            Returns:
                Output logits
        """
        embedding_ques = self.embedding_dropout(self.embedding(question))

        # LSTM returns (output, (hidden_state, cell_states))
        lstm_output, (hidden_state, cell_states) = self.lstm(embedding_ques)

        # Take last layer's hidden state
        last_hidden = hidden_state[-1]

        # LayerNorm + Dropout
        last_hidden = self.layer_norm(last_hidden)
        last_hidden = self.dropout(last_hidden)

        return self.fc(last_hidden)

def create_model(vocab_size: int, config: Config) -> QuesAnsLSTM:
    """Factory function to create the model."""
    model = QuesAnsLSTM(vocab_size, config)
    model.to(config.device)
    return model