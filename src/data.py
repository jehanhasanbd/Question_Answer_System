"""Data processing and loading utilities."""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple

from src.config import Config
from src.utils import tokenize, text_to_numeric_rep


class QADataset(Dataset):
    """Custom Dataset for Question-Answer pairs."""

    def __init__(self, df: pd.DataFrame, vocab: Dict[str,int]):
        self.df = df
        self.vocab = vocab

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]

        numerical_question = text_to_numeric_rep(row['question'], self.vocab)
        numerical_answer = text_to_numeric_rep(row['answer'], self.vocab)

        return torch.tensor(numerical_question), torch.tensor(numerical_answer)


def create_dataloader(df: pd.DataFrame, vocab: Dict[str, int], config: Config) -> DataLoader:
    dataset = QADataset(df, vocab)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_data
    )
    return dataloader


def load_and_process_data(config: Config) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Load and preprocess the data, build vocabulary."""

    # Load Data
    df = pd.read_csv(config.data_path)

    # Drop unnecessary columns
    df.drop(columns=['question_type', 'image'], inplace=True)

    # Build vocabulary
    vocab = {config.unk_token: config.unk_index}

    def build_vocab(row):
        question_tokens = tokenize(row['question'])
        answer_tokens = tokenize(row['answer'])
        merge_tokens = question_tokens + answer_tokens

        for token in merge_tokens:
            if token not in vocab:
                vocab[token] = len(token)

    df.apply(build_vocab, axis=1)  # For every row

    return df, vocab