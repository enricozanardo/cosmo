import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
from loguru import logger



class TransformerWithValueHead(nn.Module):
    """Transformer model with value head for RL and CoT"""
    
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 max_seq_length: int = 1024,
                 dropout: float = 0.1,
                 tokenizer = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.tokenizer = tokenizer
        
        # Token IDs for reasoning
        self.step_token_id = 1
        self.therefore_token_id = 2
        self.conclusion_token_id = 3
        self.reasoning_token_ids = [1, 2, 3]
        
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
    
    def _get_reasoning_positions(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract reasoning token positions from input"""
        batch_size, seq_len = input_ids.size()
        reasoning_ids = torch.zeros_like(input_ids)
        step_positions = torch.zeros_like(input_ids)
        
        for i in range(batch_size):
            step_count = 0
            for j in range(seq_len):
                token_id = input_ids[i, j].item()
                if token_id == self.step_token_id:
                    step_count += 1
                    reasoning_ids[i, j] = 1
                    step_positions[i, j] = step_count
                elif token_id == self.therefore_token_id:
                    reasoning_ids[i, j] = 2
                elif token_id == self.conclusion_token_id:
                    reasoning_ids[i, j] = 3
        
        return reasoning_ids, step_positions
    
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
    
    def _estimate_reasoning_value(self, encoder_output: torch.Tensor, reasoning_ids: torch.Tensor) -> torch.Tensor:
        """Estimate value of reasoning steps"""
        # Get batch size from input
        batch_size = encoder_output.size(0)
        
        # Get value prediction
        value = self.value_head(encoder_output.mean(dim=1))
        
        # Reshape to [batch_size, 1] instead of [1, 1]
        return value.view(batch_size, 1)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                return_value: bool = False,
                use_reasoning: bool = True) -> Dict[str, torch.Tensor]:
        """Forward pass with CoT reasoning and RL value estimation"""
        # Get reasoning positions if needed
        reasoning_ids = None
        step_positions = None
        if use_reasoning:
            reasoning_ids, step_positions = self._get_reasoning_positions(input_ids)
        
        # Get embeddings with reasoning
        embeddings = self._get_embeddings(
            input_ids=input_ids,
            reasoning_ids=reasoning_ids,
            step_positions=step_positions
        )
        
        # Encoder-Decoder processing
        encoder_output = self.encoder(embeddings)
        decoder_output = self.decoder(encoder_output, embeddings)
        lm_logits = self.lm_head(decoder_output)
        
        outputs = {'logits': lm_logits}
        
        # Add loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            outputs['loss'] = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)),
                labels.view(-1)
            )
        
        # Add value estimation if requested
        if return_value:
            if use_reasoning and reasoning_ids is not None:
                value = self._estimate_reasoning_value(encoder_output, reasoning_ids)
            else:
                # Use last token's hidden state for value estimation
                value = self.value_head(encoder_output[:, -1]).view(-1, 1)  # Shape: (batch_size, 1)
            outputs['value'] = value
        
        # Add reasoning metrics
        if use_reasoning and reasoning_ids is not None:
            outputs['reasoning_metrics'] = {
                'num_steps': (reasoning_ids == 1).sum(1),
                'has_conclusion': (reasoning_ids == 3).any(1),
                'step_positions': step_positions
            }
        
        return outputs
    
    def generate_with_reasoning(self,
                              input_ids: torch.Tensor,
                              max_length: int = 100,
                              temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """Generate text with CoT reasoning and RL guidance"""
        batch_size = input_ids.size(0)
        outputs = []
        values = []
        current_ids = input_ids
        
        for _ in range(max_length):
            # Forward pass with reasoning and value estimation
            model_outputs = self.forward(
                current_ids,
                use_reasoning=True,
                return_value=True
            )
            
            # Get next token probabilities
            next_token_logits = model_outputs['logits'][:, -1, :] / temperature
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            
            # Use value to guide sampling - now handling batch dimension
            values_batch = model_outputs['value']  # Shape: [batch_size, 1]
            
            # Boost probabilities of reasoning tokens for each example in batch
            for i in range(batch_size):
                value = values_batch[i, 0].item()  # Extract scalar value properly
                if value > 0:
                    # Boost probabilities of reasoning tokens for this example
                    for token_id in self.reasoning_token_ids:
                        next_token_probs[i, token_id] *= (1 + value)
                    next_token_probs[i] = next_token_probs[i] / next_token_probs[i].sum()
            
            # Sample next token
            next_tokens = torch.multinomial(next_token_probs, num_samples=1)
            outputs.append(next_tokens)
            values.append(values_batch)  # Store the whole batch of values
            
            # Update input ids
            current_ids = torch.cat([current_ids, next_tokens], dim=1)
        
        # Stack values along sequence dimension
        stacked_values = torch.stack(values, dim=1)  # Changed from cat to stack
        
        return {
            'generated_ids': torch.cat(outputs, dim=1),
            'values': stacked_values,
            'reasoning_metrics': model_outputs.get('reasoning_metrics', {})
        }
    
    def generate(self, 
                input_ids: torch.Tensor,
                max_length: int = 100,
                temperature: float = 1.0,
                return_value: bool = False) -> Dict[str, torch.Tensor]:
        """Basic generation without reasoning"""
        outputs = []
        values = [] if return_value else None
        current_ids = input_ids
        
        for _ in range(max_length):
            # Forward pass
            model_outputs = self.forward(
                current_ids,
                return_value=return_value,
                use_reasoning=False
            )
            
            # Get next token probabilities
            next_token_logits = model_outputs['logits'][:, -1, :] / temperature
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            
            # Sample next token
            next_tokens = torch.multinomial(next_token_probs, num_samples=1)
            outputs.append(next_tokens)
            
            if return_value:
                # Get value for current step (already has shape batch_size, 1)
                values.append(model_outputs['value'])
            
            # Update input ids
            current_ids = torch.cat([current_ids, next_tokens], dim=1)
        
        # Combine outputs
        result = {
            'generated_ids': torch.cat(outputs, dim=1)
        }
        
        if return_value:
            # Stack values along time dimension
            result['values'] = torch.cat(values, dim=0)  # Shape: (max_length, 1)
        
        return result
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        # Enable gradient checkpointing for transformer layers
        def create_encoder_forward(module):
            def custom_forward(src, src_mask=None, src_key_padding_mask=None, 
                             is_causal=False):
                def closure(*inputs):
                    return module(inputs[0], 
                                src_mask=src_mask,
                                src_key_padding_mask=src_key_padding_mask,
                                is_causal=is_causal)
                return torch.utils.checkpoint.checkpoint(
                    closure,
                    src,
                    use_reentrant=False
                )
            return custom_forward

        def create_decoder_forward(module):
            def custom_forward(tgt, memory, tgt_mask=None, memory_mask=None,
                             tgt_key_padding_mask=None, memory_key_padding_mask=None,
                             tgt_is_causal=False, is_causal=False):
                def closure(*inputs):
                    return module(inputs[0], inputs[1],
                                tgt_mask=tgt_mask,
                                memory_mask=memory_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask,
                                tgt_is_causal=tgt_is_causal,
                                is_causal=is_causal)
                return torch.utils.checkpoint.checkpoint(
                    closure,
                    tgt, memory,
                    use_reentrant=False
                )
            return custom_forward

        # Enable for encoder layers
        for layer in self.encoder.layers:
            original_forward = layer.forward
            layer.forward = create_encoder_forward(original_forward)
        
        # Enable for decoder layers
        for layer in self.decoder.layers:
            original_forward = layer.forward
            layer.forward = create_decoder_forward(original_forward)
        
        logger.info("Enabled gradient checkpointing")