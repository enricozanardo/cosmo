import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from loguru import logger

class AdaptiveController(nn.Module):
    """Controller for adaptive computation in transformer layers"""
    
    def __init__(self, hidden_size: int, num_layers: int, threshold: float = 0.5):
        super().__init__()
        logger.info(f"Initializing adaptive controller with {num_layers} layers")
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.threshold = threshold
        
        # Halting mechanism
        self.halt_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Layer importance predictor
        self.importance_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_layers),
            nn.Softmax(dim=-1)
        )
        
        # Computation budget
        self.max_steps = num_layers
        self.remaining_budget = 1.0
        
        logger.debug("Adaptive controller initialized")
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Predict halting probability and layer importance"""
        # Compute halting probability
        halt_prob = self.halt_predictor(hidden_states)  # [batch_size, seq_len, 1]
        
        # Compute layer importance scores
        importance_scores = self.importance_predictor(hidden_states)  # [batch_size, seq_len, num_layers]
        
        # Compute remaining computation budget
        batch_size, seq_len, _ = hidden_states.shape
        budget_per_token = torch.ones(batch_size, seq_len, device=hidden_states.device)
        
        # Collect metrics
        metrics = {
            'halt_prob': halt_prob.mean().item(),
            'importance_scores': importance_scores.mean(dim=(0, 1)).tolist(),
            'remaining_budget': budget_per_token.mean().item()
        }
        
        return halt_prob, importance_scores, metrics

class AdaptiveTransformerLayer(nn.Module):
    """Transformer layer with adaptive computation"""
    
    def __init__(self, layer: nn.Module, layer_idx: int):
        super().__init__()
        self.layer = layer
        self.layer_idx = layer_idx
        
        # Layer-specific adaptive threshold
        self.threshold = nn.Parameter(torch.tensor(0.5))
        
        logger.debug(f"Adaptive transformer layer {layer_idx} initialized")
    
    def forward(self, 
                hidden_states: torch.Tensor,
                importance_score: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with adaptive computation"""
        # Skip computation if importance score is below threshold
        if importance_score.mean() < self.threshold:
            return hidden_states
        
        # Scale computation by importance
        output = self.layer(hidden_states, attention_mask)
        output = output * importance_score.unsqueeze(-1)
        
        return hidden_states + output

class AdaptiveInference(nn.Module):
    """Adaptive inference module for dynamic computation paths"""
    
    def __init__(self, 
                 transformer_layers: nn.ModuleList,
                 hidden_size: int,
                 early_exit_threshold: float = 0.9):
        super().__init__()
        logger.info("Initializing adaptive inference module")
        
        num_layers = len(transformer_layers)
        self.controller = AdaptiveController(hidden_size, num_layers)
        
        # Wrap transformer layers with adaptive computation
        self.layers = nn.ModuleList([
            AdaptiveTransformerLayer(layer, idx) 
            for idx, layer in enumerate(transformer_layers)
        ])
        
        self.early_exit_threshold = early_exit_threshold
        self.training_steps = 0
        
        logger.debug(f"Adaptive inference initialized with {num_layers} layers")
    
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """Forward pass with adaptive computation"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Get halting probabilities and layer importance
        halt_prob, importance_scores, metrics = self.controller(hidden_states)
        
        # Process through layers with adaptive computation
        layer_outputs = []
        total_computation = 0
        
        for layer_idx, layer in enumerate(self.layers):
            # Get importance score for current layer
            layer_importance = importance_scores[:, :, layer_idx].unsqueeze(-1)
            
            # Process layer
            hidden_states = layer(hidden_states, layer_importance, attention_mask)
            layer_outputs.append(hidden_states)
            
            # Track computation
            total_computation += (layer_importance > layer.threshold).float().mean()
            
            # Check for early exit
            if (not self.training and 
                halt_prob.mean() > self.early_exit_threshold and 
                layer_idx >= len(self.layers) // 2):
                break
        
        # Update metrics
        metrics.update({
            'num_layers_used': layer_idx + 1,
            'total_computation': total_computation.item(),
            'early_exit': layer_idx < len(self.layers) - 1
        })
        
        return hidden_states, metrics 