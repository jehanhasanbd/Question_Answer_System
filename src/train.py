# src/train.py
"""Training module for the Q&A LSTM model."""

import torch
import torch.nn as  nn
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

from src.config import Config, default_config
from src.data import load_and_process_data, create_dataloader
from src.model import create_model


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    vocab: dict,
    config: Config,
    epoch: int,
    loss: float,
    is_best: bool = False
):
    """
        Save model checkpoint.

        Args:
            model: Trained model
            optimizer: Optimizer state
            vocab: Vocabulary dictionary
            config: Configuration object
            epoch: Current epoch
            loss: Current loss
            is_best: Whether this is the best model so far
    """
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model state
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": config,
        "vocab_size": len(vocab)
    }

    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        torch.save(checkpoint, best_path)
        print(f"Checkpoint saved to {checkpoint_dir}")

    # Save vocabulary separately
    vocab_path = checkpoint_dir / 'vocab.pth'
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    # Save config
    config_path = checkpoint_dir / "config.pth"
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)




def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """
        Train for one epoch.

        Returns:
            Average loss for the epoch
    """
    model.train()

    epoch_loss = 0
    for question, answer in dataloader:
        question, answer = question.to(device), answer.to(device)

        optimizer.zero_grad()

        y_pred = model(question)
        loss = criterion(y_pred, answer[:,0])

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def train_model(
    config: Config = None,
    verbose: bool = True,
    save_checkpoints: bool = True
) -> Tuple[nn.Module, List[float], Dict]:
    """
        Train the model.

        Args:
            config: Configuration object
            verbose: Whether to print training progress
            save_checkpoints: Whether to save model checkpoints

        Returns:
            Tuple of (trained_model, loss_history, vocabulary)
    """
    if config is None:
        config = default_config

    # Load and preprocess data
    df, vocab = load_and_process_data(config)

    # Create dataloader
    dataloader = create_dataloader(df, vocab, config)

    # Create model
    model = create_model(len(vocab), config)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    losses = []
    best_loss = float("inf")

    for epoch in range(config.epochs):
        avg_loss = train_one_epoch(
            model, dataloader, criterion, optimizer, config.device
        )

        losses.append(avg_loss)

        is_best = avg_loss < best_loss
        if best_loss:
            best_loss = avg_loss

        if verbose:
            print(f"Epoch {epoch + 1}/{config.epochs}: Loss {avg_loss:.4f}" + (" (Best!)" if is_best else ""))

        if save_checkpoints and (is_best or (epoch + 1) % config.epoch_jump == 0 or epoch == config.epochs - 1):
            save_checkpoint(
                model, optimizer, vocab, config, epoch+1, avg_loss, is_best
            )

    # Save final model
    if save_checkpoints:
        final_path = Path(config.checkpoint_dir) / "final_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'vocab_size': len(vocab)
        }, final_path)
        print(f"Final model saved to {final_path}")


    # Plot training loss
    if verbose:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, config.epochs + 1), losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.grid(True)
        plt.savefig(Path(config.checkpoint_dir) / "training_loss.png")
        plt.show()

        return model, losses, vocab


if __name__ == '__main__':
    model, losses, vocab = train_model()