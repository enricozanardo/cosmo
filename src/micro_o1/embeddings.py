import torch
import torch.nn as nn
from typing import Dict, Optional
from loguru import logger

class MicroEmbeddings(nn.Module):
    """Embeddings with special handling for reasoning tokens"""
    
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 max_position_embeddings: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Reasoning-specific embeddings
        self.reasoning_embeddings = nn.Embedding(4, hidden_size)  # For special tokens
        self.step_position_embeddings = nn.Embedding(10, hidden_size)  # For reasoning steps
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"Initialized embeddings with size {hidden_size}")
    
    def forward(self,
                input_ids: torch.Tensor,
                reasoning_ids: Optional[torch.Tensor] = None,
                step_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional reasoning embeddings"""
        # Get basic embeddings
        token_emb = self.token_embeddings(input_ids)
        
        # Add position embeddings
        position_ids = torch.arange(input_ids.size(1), device=input_ids.device)
        pos_emb = self.position_embeddings(position_ids)
        
        # Combine embeddings
        embeddings = token_emb + pos_emb
        
        # Add reasoning embeddings if present
        if reasoning_ids is not None:
            reasoning_emb = self.reasoning_embeddings(reasoning_ids)
            embeddings = embeddings + reasoning_emb
            
            if step_positions is not None:
                step_pos_emb = self.step_position_embeddings(step_positions)
                embeddings = embeddings + step_pos_emb
        
        # Apply layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings 