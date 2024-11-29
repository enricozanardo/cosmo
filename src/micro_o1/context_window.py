import torch
import torch.nn as nn
from typing import Optional, Tuple
from loguru import logger
import math

class SlidingWindowAttention(nn.Module):
    """Sliding window attention for processing long sequences"""
    
    def __init__(self, hidden_size: int, num_heads: int, window_size: int = 512, stride: int = 256):
        super().__init__()
        logger.info(f"Initializing sliding window attention with window_size={window_size}")
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.stride = stride
        self.head_dim = hidden_size // num_heads
        
        # Attention projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Initialize positional bias for each head
        self.pos_bias = nn.Parameter(torch.zeros(num_heads, window_size, window_size))
        
        logger.debug("Sliding window attention initialized")
    
    def _get_attention_scores(self, q: torch.Tensor, k: torch.Tensor, 
                            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute attention scores for a single chunk"""
        # q, k shape: [batch_size, num_heads, window_size, head_dim]
        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Add positional bias [num_heads, window_size, window_size]
        attention_scores = attention_scores + self.pos_bias.unsqueeze(0)
        
        if mask is not None:
            # Expand mask for attention heads
            expanded_mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            attention_scores = attention_scores.masked_fill(~expanded_mask, float('-inf'))
        
        return attention_scores
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with sliding window attention"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project queries, keys, and values
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Process in chunks
        outputs = []
        for i in range(0, seq_len - self.window_size + 1, self.stride):
            end_idx = i + self.window_size
            
            # Get chunk
            q_chunk = q[:, :, i:end_idx]
            k_chunk = k[:, :, i:end_idx]
            v_chunk = v[:, :, i:end_idx]
            
            # Get chunk mask
            chunk_mask = None
            if attention_mask is not None:
                chunk_mask = attention_mask[:, i:end_idx]
            
            # Compute attention scores for chunk
            attention_scores = self._get_attention_scores(q_chunk, k_chunk, chunk_mask)
            
            # Apply softmax and compute chunk output
            attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
            chunk_output = torch.matmul(attention_probs, v_chunk)
            outputs.append(chunk_output)
        
        # Handle remaining sequence if any
        if seq_len % self.stride != 0:
            last_start = max(0, seq_len - self.window_size)
            
            q_chunk = q[:, :, last_start:seq_len]
            k_chunk = k[:, :, last_start:seq_len]
            v_chunk = v[:, :, last_start:seq_len]
            
            chunk_mask = None
            if attention_mask is not None:
                chunk_mask = attention_mask[:, last_start:seq_len]
            
            attention_scores = self._get_attention_scores(q_chunk, k_chunk, chunk_mask)
            attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
            chunk_output = torch.matmul(attention_probs, v_chunk)
            outputs.append((chunk_output, last_start))
        
        # Merge chunks with overlap
        merged = torch.zeros(batch_size, self.num_heads, seq_len, self.head_dim, 
                           device=hidden_states.device)
        counts = torch.zeros(seq_len, device=hidden_states.device)
        
        for i, output in enumerate(outputs[:-1]):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            merged[:, :, start_idx:end_idx] += output
            counts[start_idx:end_idx] += 1
        
        # Add last chunk if it exists
        if len(outputs) > 0 and isinstance(outputs[-1], tuple):
            last_output, last_start = outputs[-1]
            merged[:, :, last_start:] += last_output
            counts[last_start:] += 1
        
        # Average overlapping regions
        merged = merged / counts.view(1, 1, -1, 1).clamp(min=1)
        context_layer = merged
        
        # Reshape and project output
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.view(batch_size, seq_len, self.hidden_size)
        context_layer = self.out_proj(context_layer)
        
        return context_layer

class ExtendedContextTransformer(nn.Module):
    """Transformer block with extended context window support"""
    
    def __init__(self, hidden_size: int, num_heads: int, ff_dim: int, 
                 window_size: int = 512, stride: int = 256, dropout: float = 0.1):
        super().__init__()
        logger.info(f"Initializing extended context transformer with window_size={window_size}")
        
        self.attention = SlidingWindowAttention(hidden_size, num_heads, window_size, stride)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_size),
            nn.Dropout(dropout)
        )
        
        self.gradient_checkpointing = False
        logger.debug("Extended context transformer initialized")
    
    def _forward_attention(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through attention layer"""
        return self.attention(x, attention_mask)
    
    def _forward_ff(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feed-forward network"""
        return self.ff(x)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with gradient checkpointing support"""
        if self.gradient_checkpointing and self.training:
            # Use gradient checkpointing for attention
            def create_custom_forward(module, has_mask=False):
                def custom_forward(*inputs):
                    if has_mask:
                        return module(inputs[0], inputs[1])
                    return module(inputs[0])
                return custom_forward
            
            # Checkpoint attention
            attention_output = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self._forward_attention, has_mask=attention_mask is not None),
                x, attention_mask if attention_mask is not None else None
            )
            x = self.norm1(x + attention_output)
            
            # Checkpoint feed-forward
            ff_output = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self._forward_ff),
                x
            )
            x = self.norm2(x + ff_output)
        else:
            # Regular forward pass
            attention_output = self.attention(x, attention_mask)
            x = self.norm1(x + attention_output)
            x = self.norm2(x + self.ff(x))
        
        return x