import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from loguru import logger



class TransformerWithValueHead(nn.Module):
    """Transformer model with value head for RL and CoT"""
    
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 max_seq_length: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)
        self.reasoning_embeddings = nn.Embedding(4, hidden_size)  # For special tokens
        self.step_position_embeddings = nn.Embedding(10, hidden_size)  # For reasoning steps
        
        # Layer norm and dropout for embeddings
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            batch_first=True  # Add this for better performance
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            batch_first=True  # Add this for better performance
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output heads
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.value_head = nn.Linear(hidden_size, 1)
        
        logger.info(f"Initialized transformer with {num_layers} layers and value head")
    
    def _get_embeddings(self,
                       input_ids: torch.Tensor,
                       reasoning_ids: Optional[torch.Tensor] = None,
                       step_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Combine all embeddings"""
        # Get token embeddings
        embeddings = self.token_embeddings(input_ids)
        
        # Add position embeddings
        position_ids = torch.arange(input_ids.size(1), device=input_ids.device)
        embeddings = embeddings + self.position_embeddings(position_ids)
        
        # Add reasoning embeddings if present
        if reasoning_ids is not None:
            embeddings = embeddings + self.reasoning_embeddings(reasoning_ids)
            if step_positions is not None:
                embeddings = embeddings + self.step_position_embeddings(step_positions)
        
        # Apply layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                return_value: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass with optional value estimation"""
        # Get embeddings
        embeddings = self._get_embeddings(input_ids)
        
        # Encoder
        if attention_mask is not None:
            encoder_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            encoder_mask = None
        
        encoder_output = self.encoder(embeddings.transpose(0, 1), src_key_padding_mask=encoder_mask)
        
        # Decoder (for autoregressive generation)
        decoder_output = self.decoder(
            encoder_output,
            embeddings.transpose(0, 1),
            memory_key_padding_mask=encoder_mask
        )
        
        # Language modeling head
        lm_logits = self.lm_head(decoder_output.transpose(0, 1))
        
        outputs = {'logits': lm_logits}
        
        # Calculate loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            outputs['loss'] = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        
        # Value estimation for RL if requested
        if return_value:
            # Use last hidden state for value estimation
            value = self.value_head(encoder_output[-1])
            outputs['value'] = value
        
        return outputs
    
    def _has_reasoning_tokens(self, input_ids: torch.Tensor) -> bool:
        """Check if input contains reasoning tokens"""
        # Check for special token IDs
        reasoning_token_ids = {0, 1, 2, 3}  # Example IDs for reasoning tokens
        return any(token_id in reasoning_token_ids for token_id in input_ids.unique())
    
    def _get_reasoning_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for reasoning tokens"""
        # Map token IDs to reasoning embeddings
        reasoning_ids = torch.zeros_like(input_ids)
        step_positions = torch.zeros_like(input_ids)
        
        current_step = 0
        for i, token_id in enumerate(input_ids[0]):
            if token_id in {1}:  # Step token
                current_step += 1
                reasoning_ids[0, i] = 1
                step_positions[0, i] = current_step
            elif token_id in {2}:  # Therefore token
                reasoning_ids[0, i] = 2
            elif token_id in {3}:  # Conclusion token
                reasoning_ids[0, i] = 3
        
        reasoning_emb = self.reasoning_embeddings(reasoning_ids)
        step_pos_emb = self.step_position_embeddings(step_positions)
        
        return reasoning_emb + step_pos_emb
    
    def generate(self, 
                input_ids: torch.Tensor,
                max_length: int = 100,
                temperature: float = 1.0,
                return_value: bool = False) -> Dict[str, torch.Tensor]:
        """Generate tokens with optional value estimation"""
        batch_size = input_ids.size(0)
        current_ids = input_ids
        
        outputs = []
        values = [] if return_value else None
        
        for _ in range(max_length):
            # Forward pass
            model_outputs = self.forward(current_ids, return_value=return_value)
            
            # Get next token probabilities
            next_token_logits = model_outputs['logits'][:, -1, :] / temperature
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            
            # Sample next tokens
            next_tokens = torch.multinomial(next_token_probs, num_samples=1)
            
            # Append to outputs
            outputs.append(next_tokens)
            if return_value:
                values.append(model_outputs['value'])
            
            # Update input ids
            current_ids = torch.cat([current_ids, next_tokens], dim=1)
        
        # Combine outputs
        generated_ids = torch.cat(outputs, dim=1)
        result = {'generated_ids': generated_ids}
        
        if return_value:
            result['values'] = torch.cat(values, dim=0)
        
        return result 