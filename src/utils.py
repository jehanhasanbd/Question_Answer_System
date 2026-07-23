# src/utils.py
import torch
from typing import List, Dict, Tuple

from .config import default_config
def tokenize(text: str) -> List[str]:
    """
        Tokenize text by lowercasing and removing punctuation.

        Args:
            text: Input text string

        Returns:
            List of tokens
    """
    return (text.lower()
            .replace("?", "")
            .replace("!", "")
            .replace("'", "")
            .replace(",", "")
            .replace(".", "")
            .replace(":", "")
            .replace("(", "")
            .replace(")", "")
            .split())


def text_to_numeric_rep(text: str, vocab: Dict[str, int]) -> List[int]:
    """
        Convert text to numeric representation using vocabulary.

        Args:
            text: Input text string
            vocab: Vocabulary dictionary

        Returns:
            List of token indices
    """
    numerical_present = []
    for word in tokenize(text):
        if word in vocab:
            numerical_present.append(vocab[word])
        else:
            numerical_present.append(default_config.unk_token)

    return numerical_present


def inverse_vocab(vocab: Dict[str,int]) -> Dict[int,str]:
    """Create inverse vocabulary mapping."""
    return {value:key for key, value in vocab.items()}


def predict(
        model: torch.nn.Module,
        question: str,
        vocab: Dict[str, int],
        config,
        threshold: float = None
) -> Tuple[str, float]:
    """
        Make prediction for a question.

        Args:
            model: Trained model
            question: Input question string
            vocab: Vocabulary dictionary
            config: Configuration object
            threshold: Probability threshold (optional)

        Returns:
            Tuple of (predicted_word, probability)
    """
    if threshold is None:
        threshold = config.threshold

    device = config.device

    # Convert text to numbers
    numerical_question = text_to_numeric_rep(question, vocab)

    # Make it tensor
    tensor_question = torch.tensor(numerical_question).unsqueeze(0).to(device)

    # Get model output
    output_probability = model(tensor_question)

    # Apply softmax
    probs = torch.nn.functional.softmax(output_probability, dim=1)

    # Get max probabilities
    max_prob, max_index = torch.max(probs, dim=1)
    max_prob, max_index = max_prob.item(), max_index.item()

    inverse_vocab_dict = inverse_vocab(vocab)
    predicted_word = inverse_vocab_dict.get(max_index, config.unk_token)

    if max_prob >= threshold:
        return predicted_word, max_prob
    else:
        return "I don't know", max_prob