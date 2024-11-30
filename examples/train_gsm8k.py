import os
import sys
import torch
from pathlib import Path

# Add project root to PYTHONPATH
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from src.micro_o1.transformer import TransformerWithValueHead
from src.micro_o1.training.gsm8k_trainer import GSM8KTrainer
from src.micro_o1.config.logging import setup_logger

logger = setup_logger("training", "logs/training.log")

def main():
    # Initialize model
    model = TransformerWithValueHead(
        vocab_size=50257,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        max_seq_length=1024
    )
    
    # Initialize trainer
    trainer = GSM8KTrainer(
        model=model,
        learning_rate=1e-4,
        batch_size=16,
        max_steps=10000,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train
    logger.info("Starting training")
    history = trainer.train()
    
    # Save model
    torch.save(model.state_dict(), 'gsm8k_model.pt')
    logger.info("Training completed")

if __name__ == '__main__':
    main()