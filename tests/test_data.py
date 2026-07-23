# tests/test_data.py
"""Unit tests for data processing module."""

import pytest
import pandas as pd
import torch
from pathlib import Path

from src.data import QADataset, create_dataloader, load_and_process_data
from src.config import default_config
from src.utils import tokenize, text_to_numeric_rep

class TestDataProcessing:
    """Test data processing functions."""

    @pytest.fixture
    def sample_vocab(self):
        """Create a sample vocabulary for testing."""
        return {
            "UNK": 0,
            "what": 1,
            "is": 2,
            "the": 3,
            "capital": 4,
            "of": 5,
            "france": 6,
            "paris": 7,
            "how": 8,
            "many": 9,
            "days": 10,
            "in": 11,
            "a": 12,
            "year": 13,
            "365": 14
        }

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        data = {
            'question': ['What is the capital of France?', 'How many days in a year?'],
            'answer': ['Paris', '365'],
            'question_type': ['fact', 'fact'],
            'image': ['', '']
        }
        return pd.DataFrame(data)


    def test_tokenize(self):
        """Test text tokenization."""
        text = "What is the capital of France?"
        tokens = tokenize(text)
        assert tokens == ["what", "is", "the", "capital", "of", "france"]

        # Test multiple punctuation
        text = "Hello, world! How are you?"
        tokens = tokenize(text)
        assert tokens == ["hello", "world", "how", "are", "you"]

        # Test empty string
        assert tokenize("") == []

    def test_text_to_numeric_rep(self, sample_vocab):
        """Test converting text to numeric representation."""
        text = "what is the capital of france"
        numeric = text_to_numeric_rep(text, sample_vocab)
        expected = [1, 2, 3, 4, 5, 6]
        assert numeric == expected

        # Test with unknown word
        text = "unknown jehan here"
        numeric = text_to_numeric_rep(text, sample_vocab)
        expected = [0, 0, 0]
        assert expected == numeric

    def test_qadataset_init(self, sample_df, sample_vocab):
        """Test QADataset initialization."""
        dataset = QADataset(sample_df, sample_vocab)
        assert len(dataset) == 2
        assert dataset.vocab == sample_vocab
        assert dataset.df is not None

    def test_qadataset_getitem(self, sample_df, sample_vocab):
        """Test QADataset item retrieval."""
        dataset = QADataset(sample_df, sample_vocab)
        question_tensor, answer_tensor = dataset[0]

        assert isinstance(question_tensor, torch.Tensor)
        assert isinstance(answer_tensor, torch.Tensor)
        assert question_tensor.dtype == torch.int64
        assert answer_tensor.dtype == torch.int64
        assert len(question_tensor) > 0
        assert len(answer_tensor) > 0

    def test_create_dataloader(self, sample_df, sample_vocab):
        """Test DataLoader creation."""
        dataloader = create_dataloader(sample_df, sample_vocab, default_config)
        assert dataloader.batch_size == default_config.batch_size
        assert dataloader.shuffle == default_config.shuffle_data

        # Test iterating through dataloader
        batch = next(iter(dataloader))
        assert len(batch) == 2
        assert isinstance(batch[0], torch.Tensor)
        assert isinstance(batch[1], torch.Tensor)

    def test_load_and_process_data(self, temp_path, sample_df):
        """Test loading and processing data."""
        # Save sample data to temporary file
        csv_path = temp_path / "test_data.csv"
        sample_df.to_csv(csv_path, index=False)

        # Create config with temporary data path
        config = default_config
        config.data_path = str(csv_path)

        df, vocab = load_and_process_data(config)
        # Check that unnecessary columns are dropped
        assert "question_type" not in df.columns
        assert "image" not in df.columns
        assert "question" in df.columns
        assert "answer" in df.columns

        # Check vocabulary building
        assert "UNK" in vocab
        assert len(vocab) > 0

        # Check that words from questions and answers are in vocabulary
        assert "what" in vocab or "how" in vocab
        assert "paris" in vocab or "365" in vocab



