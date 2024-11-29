import torch
import torch.nn as nn
from transformers import AutoTokenizer
from typing import List, Dict, Union, Tuple
import numpy as np
from loguru import logger

class MicroO1Tokenizer:
    def __init__(self, model_name: str = "gpt2", max_length: int = 1024):
        """
        Initialize the tokenizer for MicroO1
        
        Args:
            model_name: Base tokenizer to use (default: gpt2)
            max_length: Maximum sequence length (default: 1024)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        # Add special tokens for reasoning
        special_tokens = {
            'pad_token': '[PAD]',
            'sep_token': '[SEP]',
            'reasoning_token': '[REASON]',
            'solution_token': '[SOLUTION]'
        }
        
        # Add special tokens if they don't exist
        self.tokenizer.add_special_tokens({'additional_special_tokens': list(special_tokens.values())})
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = special_tokens['pad_token']
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(special_tokens['pad_token'])
    
    def encode(self, text: Union[str, List[str]], padding: bool = True) -> Dict[str, torch.Tensor]:
        """
        Encode text into tokens
        
        Args:
            text: Input text or list of texts
            padding: Whether to pad sequences to max_length
            
        Returns:
            Dictionary containing input_ids and attention_mask
        """
        encoded = self.tokenizer(
            text,
            padding=padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def decode(self, token_ids: torch.Tensor) -> Union[str, List[str]]:
        """
        Decode token ids back to text
        
        Args:
            token_ids: Tensor of token ids
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size"""
        return len(self.tokenizer)

class MicroO1Embeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_position_embeddings: int = 1024, dropout: float = 0.1):
        """Initialize the embeddings layer"""
        super().__init__()
        logger.info(f"Initializing embeddings with vocab_size={vocab_size}, hidden_size={hidden_size}")
        
        self.vocab_size = vocab_size
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Register buffer for position ids
        position_ids = torch.arange(max_position_embeddings).unsqueeze(0)
        self.register_buffer('position_ids', position_ids)
        
        logger.debug("Embeddings layer initialized")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of the embeddings layer"""
        seq_length = input_ids.size(1)
        logger.debug(f"Processing sequence of length {seq_length}")
        
        # Ensure input_ids are within vocab size
        if (input_ids >= self.vocab_size).any():
            logger.warning(f"Found input_ids >= vocab_size ({self.vocab_size}), clipping")
            input_ids = torch.clamp(input_ids, max=self.vocab_size-1)
        
        # Get word embeddings
        word_embeds = self.word_embeddings(input_ids)
        logger.debug(f"Word embeddings shape: {word_embeds.shape}")
        
        # Get position embeddings
        position_embeds = self.position_embeddings(self.position_ids[:, :seq_length])
        logger.debug(f"Position embeddings shape: {position_embeds.shape}")
        
        # Combine embeddings
        embeddings = word_embeds + position_embeds
        
        # Apply layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        if attention_mask is not None:
            # Apply attention mask
            embeddings = embeddings * attention_mask.unsqueeze(-1)
            logger.debug("Applied attention mask to embeddings")
        
        logger.debug(f"Final embeddings shape: {embeddings.shape}")
        return embeddings

def test_tokenizer():
    """Test the tokenizer implementation"""
    # Initialize tokenizer
    tokenizer = MicroO1Tokenizer()
    
    # Test text
    test_text = "Solve this math problem: If John has 5 apples and gives 2 to Mary, how many does he have left?"
    
    # Test encoding
    encoded = tokenizer.encode(test_text)
    print("Encoded shape:", encoded['input_ids'].shape)
    print("Attention mask shape:", encoded['attention_mask'].shape)
    
    # Test decoding
    decoded = tokenizer.decode(encoded['input_ids'][0])
    print("\nOriginal text:", test_text)
    print("Decoded text:", decoded)
    
    # Test special tokens
    special_text = "[REASON] Let's solve this step by step [SOLUTION] John has 3 apples left"
    encoded_special = tokenizer.encode(special_text)
    decoded_special = tokenizer.decode(encoded_special['input_ids'][0])
    print("\nSpecial tokens test:")
    print("Original:", special_text)
    print("Decoded:", decoded_special)

if __name__ == "__main__":
    test_tokenizer() 