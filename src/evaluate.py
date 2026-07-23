# src/evaluate.py
"""Evaluation and inference module for the Q&A system."""

import torch
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from src.config import Config, default_config
from src.utils import predict
from src.model import create_model
from src.train import train_model


def load_model(
    checkpoint_path: Optional[str] = None,
    vocab_path: Optional[str] = None,
    config: Optional[Config] = None
) -> Tuple[torch.nn.Module, Dict, Config]:
    """
        Load a trained model and vocabulary.

        Args:
            checkpoint_path: Path to model checkpoint (if None, load best model)
            vocab_path: Path to vocabulary file (if None, load from checkpoint dir)
            config: Configuration object (if None, load from checkpoint)

        Returns:
            Tuple of (model, vocab, config)
    """
    if config is None:
        config = default_config

    # Set default paths
    if checkpoint_path is None:
        checkpoint_path = Path(config.checkpoint_dir) / "best_model.pth"
    else:
        checkpoint_path = Path(checkpoint_path)

    if vocab_path is None:
        vocab_path = Path(config.checkpoint_dir) / "vocab.pth"
    else:
        vocab_path = Path(vocab_path)

    # Check if files exist
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found at {vocab_path}")

    # Load vocabulary
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.device)

    # Get config from checkpoint if available
    if 'config' in checkpoint:
        config = checkpoint['config']

    # Create model with saved vocabulary size
    model = create_model(len(vocab), config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    if "epoch" in checkpoint:
        print(f"Trained for {checkpoint['epoch']} epochs, Loss: {checkpoint.get('loss', 'N/A')}")

    return model, vocab, config

def evaluate_model(
    model: torch.nn.Module,
    vocab: Dict[str, int],
    test_questions: List[str],
    expected_answers: List[str] = None,
    threshold: float = None,
    config: Config = None,
) -> List[Tuple[str, str, float, str]]:
    """
        Evaluate model on test questions.

        Args:
            model: Trained model
            vocab: Vocabulary dictionary
            test_questions: List of questions to test
            expected_answers: List of expected answers (optional)
            threshold: Probability threshold
            config: Configuration object

        Returns:
            List of tuples (question, predicted_answer, probability, expected_answer)
    """
    if config is None:
        config = default_config

    if threshold is None:
        threshold = config.threshold

    model.eval()
    results = []

    with torch.no_grad():
        for i, question in enumerate(test_questions):
            predicted_word, probability = predict(
                model, question, vocab, config, threshold
            )

            expected = expected_answers[i] if expected_answers else None
            results.append((question, predicted_word, probability, expected))

    return results

def print_evaluation_results(results: List[Tuple[str, str, float, str]]):
    """Pretty print evaluation results."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    correct = 0
    total = 0

    for question, prediction, prob, expected in results:
        print(f"\nQ: {question}")
        print(f"A: {prediction} (confidence: {prob:.3f})")
        if expected:
            is_correct = prediction.lower() == expected.lower()
            print(f"Expected: {expected} {'✓' if is_correct else '✗'}")
            if is_correct:
                correct += 1
            total += 1
        print("-" * 40)

    if total > 0:
        accuracy = correct / total * 100
        print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
    print("=" * 80)

def test_sample_questions(
        model: torch.nn.Module = None,
        vocab: Dict[str, int] = None,
        config: Config = None
):
    """Run inference on sample questions."""

    if config is None:
        config = default_config

    if model is None or vocab is None:
        print("Loading saved model...")
        try:
            model, vocab, config = load_model(config=config)
        except FileNotFoundError:
            print("No saved model found. Training new model...")
            model, _, vocab = train_model(config, verbose=False)

    # Test questions
    test_questions = [
        "How many days are there in a normal year?",
        "Which animal is known as the 'Ship of the Desert?'",
        "How many sides are there in a triangle?",
        "What is the capital of France?"
        ]

    expected = ["365", "camel", "3", "paris"]

    results = evaluate_model(
        model, vocab, test_questions, expected,
        threshold=config.threshold, config=config
    )

    print_evaluation_results(results)


if __name__ == "__main__":
    # Test the trained model
    test_sample_questions()