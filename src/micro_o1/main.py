import torch
import torch.nn as nn
from transformers import AutoTokenizer
from typing import List, Dict, Union, Tuple, Optional
import numpy as np
from loguru import logger
import math
import torch.nn.functional as F

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

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        """Initialize multi-head attention"""
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Linear layers for Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        logger.debug(f"Initialized MultiHeadAttention with {num_heads} heads")
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of multi-head attention"""
        batch_size = query.size(0)
        
        # Project and reshape for multi-head attention
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        logger.debug(f"Attention scores shape: {scores.shape}")
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute output
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        output = self.out_proj(output)
        
        return output

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        """Initialize transformer block"""
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
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
        
        logger.debug(f"Initialized TransformerBlock with hidden_size={hidden_size}, ff_dim={ff_dim}")
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of transformer block"""
        # Self-attention with residual connection and layer norm
        attention_output = self.attention(x, x, x, attention_mask)
        x = self.norm1(x + attention_output)
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        
        return x

class MicroO1Transformer(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 hidden_size: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 ff_dim: int = 3072,
                 max_position_embeddings: int = 1024,
                 dropout: float = 0.1):
        """Initialize the transformer model"""
        super().__init__()
        logger.info(f"Initializing MicroO1Transformer with {num_layers} layers")
        
        # Embeddings
        self.embeddings = MicroO1Embeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Output layer
        self.norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)
        
        logger.debug("MicroO1Transformer initialization complete")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the transformer"""
        # Get embeddings
        x = self.embeddings(input_ids, attention_mask)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Apply final norm and output projection
        x = self.norm(x)
        logits = self.output(x)
        
        return logits

class ChainOfThoughtProcessor(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        """Initialize Chain of Thought processor"""
        super().__init__()
        logger.info("Initializing Chain of Thought processor")
        
        self.hidden_size = hidden_size
        
        # Reasoning step attention
        self.reasoning_attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.step_norm = nn.LayerNorm(hidden_size)
        
        # Step aggregation
        self.step_aggregator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Step markers for identifying reasoning boundaries
        self.register_parameter('step_start_embedding', 
                              nn.Parameter(torch.randn(1, 1, hidden_size)))
        self.register_parameter('step_end_embedding', 
                              nn.Parameter(torch.randn(1, 1, hidden_size)))
        
        logger.debug(f"CoT processor initialized with hidden_size={hidden_size}")
    
    def identify_reasoning_steps(self, hidden_states: torch.Tensor, 
                               attention_mask: torch.Tensor,
                               reason_token_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Identify and process reasoning steps in the sequence"""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        assert hidden_dim == self.hidden_size, f"Hidden dimension mismatch: got {hidden_dim}, expected {self.hidden_size}"
        
        # Find reasoning token positions
        reason_positions = torch.nonzero(reason_token_mask, as_tuple=True)
        logger.debug(f"Found {len(reason_positions[0])} reasoning tokens")
        
        # Process each reasoning step
        processed_states = hidden_states.clone()
        step_masks = []
        
        for batch_idx in range(batch_size):
            batch_positions = reason_positions[0] == batch_idx
            batch_reason_pos = reason_positions[1][batch_positions]
            
            if len(batch_reason_pos) > 0:
                # Create step segments
                for i in range(len(batch_reason_pos) - 1):
                    start_pos = batch_reason_pos[i]
                    end_pos = batch_reason_pos[i + 1]
                    segment_length = end_pos - start_pos
                    
                    # Process step segment
                    step_hidden = hidden_states[batch_idx, start_pos:end_pos]  # [seq_len, hidden_dim]
                    step_mask = attention_mask[batch_idx, start_pos:end_pos]
                    
                    # Ensure correct dimensions
                    step_hidden = step_hidden.unsqueeze(0)  # [1, seq_len, hidden_dim]
                    
                    # Create step markers with correct size
                    start_marker = self.step_start_embedding.expand(1, 1, hidden_dim)
                    end_marker = self.step_end_embedding.expand(1, 1, hidden_dim)
                    
                    # Concatenate along sequence dimension
                    step_markers = torch.cat([
                        start_marker,
                        step_hidden,
                        end_marker
                    ], dim=1)
                    
                    step_masks.append(step_mask)
                    
                    # Process step with attention
                    processed_step = self.reasoning_attention(step_markers, step_markers, step_markers)
                    processed_step = self.step_norm(processed_step)
                    
                    # Remove markers and ensure correct size
                    processed_step = processed_step.squeeze(0)  # Remove batch dimension
                    processed_step = processed_step[1:-1]  # Remove markers
                    
                    # Verify shapes before assignment
                    logger.debug(f"Processed step shape: {processed_step.shape}")
                    logger.debug(f"Target shape: [{segment_length}, {hidden_dim}]")
                    assert processed_step.shape == (segment_length, hidden_dim), \
                        f"Shape mismatch: got {processed_step.shape}, expected ({segment_length}, {hidden_dim})"
                    
                    # Update hidden states
                    processed_states[batch_idx, start_pos:end_pos] = processed_step
        
        if not step_masks:
            return processed_states, None
            
        # Stack masks only if we have any
        try:
            stacked_masks = torch.stack(step_masks)
            logger.debug(f"Stacked masks shape: {stacked_masks.shape}")
            return processed_states, stacked_masks
        except:
            logger.warning("Failed to stack masks, returning None for masks")
            return processed_states, None

    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: torch.Tensor,
                reason_token_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of Chain of Thought processor"""
        logger.debug(f"Processing sequence of shape {hidden_states.shape}")
        
        # Process reasoning steps
        processed_states, step_masks = self.identify_reasoning_steps(
            hidden_states, attention_mask, reason_token_mask
        )
        
        # Aggregate step information
        if step_masks is not None:
            aggregated = self.step_aggregator(processed_states)
            # Combine with original hidden states
            output = hidden_states + aggregated
        else:
            output = hidden_states
        
        logger.debug("Chain of Thought processing complete")
        return output

class MicroO1TransformerWithCoT(MicroO1Transformer):
    def __init__(self, *args, **kwargs):
        """Initialize transformer with Chain of Thought reasoning"""
        super().__init__(*args, **kwargs)
        logger.info("Initializing transformer with Chain of Thought reasoning")
        
        # Add CoT processor
        self.cot_processor = ChainOfThoughtProcessor(
            hidden_size=kwargs.get('hidden_size', 768)
        )
        
        # Add reasoning token embedding
        self.register_parameter('reasoning_token_embedding',
                              nn.Parameter(torch.randn(1, 1, kwargs.get('hidden_size', 768))))
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                reason_token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with Chain of Thought reasoning"""
        # Get base transformer embeddings
        x = self.embeddings(input_ids, attention_mask)
        
        # Process first half of transformer layers
        mid_layer = len(self.layers) // 2
        for layer in self.layers[:mid_layer]:
            x = layer(x, attention_mask)
        
        # Apply Chain of Thought reasoning if reason_token_mask is provided
        if reason_token_mask is not None:
            # Add reasoning token embeddings where reason_token_mask is 1
            reasoning_embeds = self.reasoning_token_embedding * reason_token_mask.unsqueeze(-1)
            x = x + reasoning_embeds
            
            # Apply CoT processing
            x = self.cot_processor(x, attention_mask, reason_token_mask)
        
        # Process remaining transformer layers
        for layer in self.layers[mid_layer:]:
            x = layer(x, attention_mask)
        
        # Final processing
        x = self.norm(x)
        logits = self.output(x)
        
        return logits

class PPORewardModel(nn.Module):
    """Reward model for evaluating reasoning quality"""
    def __init__(self, hidden_size: int):
        super().__init__()
        logger.info("Initializing PPO Reward Model")
        
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, 1)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute rewards for each token"""
        return self.reward_head(hidden_states).squeeze(-1)

class PPOCritic(nn.Module):
    """Value network for PPO"""
    def __init__(self, hidden_size: int):
        super().__init__()
        logger.info("Initializing PPO Critic")
        
        self.value_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute value estimates"""
        return self.value_net(hidden_states).squeeze(-1)

class MicroO1TransformerWithRL(MicroO1TransformerWithCoT):
    def __init__(self, *args, **kwargs):
        """Initialize transformer with RL components"""
        super().__init__(*args, **kwargs)
        logger.info("Initializing transformer with RL components")
        
        hidden_size = kwargs.get('hidden_size', 768)
        
        # RL components
        self.reward_model = PPORewardModel(hidden_size)
        self.critic = PPOCritic(hidden_size)
        
        # PPO hyperparameters
        self.clip_ratio = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        
        logger.debug("RL components initialized")
    
    def compute_ppo_loss(self, 
                        old_logits: torch.Tensor,
                        new_logits: torch.Tensor,
                        values: torch.Tensor,
                        rewards: torch.Tensor,
                        attention_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute PPO loss components"""
        # Compute log probabilities
        old_probs = F.softmax(old_logits, dim=-1)
        new_probs = F.softmax(new_logits, dim=-1)
        
        # Compute ratio and clipped ratio
        ratio = (new_probs / (old_probs + 1e-8)).mean(dim=-1)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        
        # Compute advantages
        with torch.no_grad():
            advantages = rewards - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss
        policy_loss_1 = ratio * advantages
        policy_loss_2 = clipped_ratio * advantages
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, rewards)
        
        # Entropy loss for exploration
        entropy = -(new_probs * torch.log(new_probs + 1e-8)).mean()
        
        # Total loss
        total_loss = (
            policy_loss + 
            self.value_loss_coef * value_loss - 
            self.entropy_coef * entropy
        )
        
        # Return losses for logging
        losses = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
        
        return total_loss, losses
    
    def forward_rl(self, 
                  input_ids: torch.Tensor,
                  attention_mask: torch.Tensor,
                  reason_token_mask: torch.Tensor,
                  old_logits: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with RL components"""
        # Get model outputs
        logits = self.forward(input_ids, attention_mask, reason_token_mask)
        
        # Get hidden states for value and reward computation
        with torch.no_grad():
            hidden_states = self.get_hidden_states(input_ids, attention_mask)
        
        # Compute values and rewards
        values = self.critic(hidden_states)
        rewards = self.reward_model(hidden_states)
        
        # Apply attention mask
        if attention_mask is not None:
            values = values * attention_mask
            rewards = rewards * attention_mask
        
        return logits, values, rewards
    
    def get_hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get hidden states for value/reward computation"""
        x = self.embeddings(input_ids, attention_mask)
        
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        return self.norm(x)

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