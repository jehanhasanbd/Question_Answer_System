"""Run training script - can be executed from any directory."""

import sys
import os

# Add the parent directory to path so we can import src modules
# This works whether we're running from src/ or from project root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now we can import using absolute imports
from src.train import train_model
from src.checkpoint import load_best_model
from src.evaluate import test_sample_questions

if __name__ == "__main__":
    # Train the model
    print("Starting training...")
    print(f"Current directory: {os.getcwd()}")
    print(f"Script directory: {current_dir}")

    model, losses, vocab = train_model(save_checkpoints=True)
    print("Training completed!")

    # Test the model
    print("\nTesting the model...")
    test_sample_questions(model, vocab)