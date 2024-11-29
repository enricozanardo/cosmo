import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple
from loguru import logger
import numpy as np

class OutputGenerator(nn.Module):
    """Output generation module for text generation and completion"""
    
    def __init__(self, 
                 model: nn.Module,
                 vocab_size: int,
                 max_length: int = 1024,
                 temperature: float = 0.7,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.2):
        super().__init__()
        logger.info("Initializing output generator")
        
        self.model = model
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        
        logger.debug("Output generator initialized")
    
    def _apply_repetition_penalty(self, logits: torch.Tensor, 
                                generated_tokens: List[int]) -> torch.Tensor:
        """Apply repetition penalty to logits"""
        if len(generated_tokens) == 0:
            return logits
        
        # Penalize previously generated tokens
        for token in set(generated_tokens):
            logits[:, token] /= self.repetition_penalty
        
        return logits
    
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering to logits"""
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def _top_p_filtering_cumsum(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply nucleus sampling using cumulative sum"""
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        
        filtered_logits = torch.full_like(logits, float('-inf'))
        
        for batch_idx in range(logits.size(0)):
            # Get probabilities
            probs = torch.softmax(logits[batch_idx], dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            
            # Find minimal set of tokens that sum to top_p
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            
            # Find the exact cutoff point
            cutoff_mask = cumsum <= top_p
            if not cutoff_mask.any():
                idx = 1  # Keep at least one token
            else:
                # Find the last position where cumsum <= top_p
                last_idx = torch.where(cutoff_mask)[0][-1].item()
                
                # Check if we really need more tokens
                if last_idx > 0:
                    # If we're already close to top_p with fewer tokens, use those
                    for i in range(last_idx - 1, -1, -1):
                        if cumsum[i] >= top_p * 0.98:  # 98% of target is good enough
                            last_idx = i
                            break
                
                idx = last_idx + 1
                
                # Very aggressive token limiting
                idx = min(5, idx)  # Maximum of 5 tokens
            
            # Get selected tokens
            selected_indices = sorted_indices[:idx]
            selected_probs = sorted_probs[:idx]
            
            # Scale probabilities to sum to exactly top_p
            scale = top_p / selected_probs.sum()
            log_scale = torch.log(scale).to(logits.device)
            
            # Apply scaling in log space
            filtered_logits[batch_idx, selected_indices] = \
                logits[batch_idx, selected_indices] + log_scale
        
        return filtered_logits
    
    def _top_p_filtering_mean(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply nucleus sampling using mean threshold"""
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        
        filtered_logits = torch.full_like(logits, float('-inf'))
        
        for batch_idx in range(logits.size(0)):
            probs = torch.softmax(logits[batch_idx], dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            
            # Calculate running mean
            running_mean = torch.cumsum(sorted_probs, dim=0) / torch.arange(1, len(sorted_probs) + 1)
            
            # Find tokens above mean * top_p threshold
            idx = torch.where(sorted_probs >= running_mean * top_p)[0]
            if len(idx) == 0:
                idx = torch.tensor([0])
            else:
                idx = idx[-1] + 1
            
            filtered_logits[batch_idx, sorted_indices[:idx]] = logits[batch_idx, sorted_indices[:idx]]
        
        return filtered_logits
    
    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply nucleus sampling (using cumsum approach by default)"""
        return self._top_p_filtering_cumsum(logits, top_p)
    
    def _prepare_inputs(self, input_ids: torch.Tensor, 
                       attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Prepare inputs for the model"""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    @torch.no_grad()
    def generate(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                max_new_tokens: int = 50,
                min_new_tokens: int = 10,
                **kwargs) -> Tuple[torch.Tensor, Dict]:
        """Generate output tokens"""
        logger.debug(f"Generating with max_new_tokens={max_new_tokens}")
        
        # Override generation parameters if provided
        temperature = kwargs.get('temperature', self.temperature)
        top_k = kwargs.get('top_k', self.top_k)
        top_p = kwargs.get('top_p', self.top_p)
        
        # Initialize generation tracking
        batch_size = input_ids.shape[0]
        generated_tokens = []
        metrics = {
            'avg_entropy': [],
            'tokens_per_step': [],
            'early_stops': 0
        }
        
        # Generate tokens
        for step in range(max_new_tokens):
            # Prepare inputs
            inputs = self._prepare_inputs(input_ids, attention_mask)
            
            # Get model output
            outputs = self.model(**inputs)
            next_token_logits = outputs[:, -1, :]  # [batch_size, vocab_size]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            next_token_logits = self._apply_repetition_penalty(
                next_token_logits, generated_tokens
            )
            
            # Apply filtering
            next_token_logits = self._top_k_filtering(next_token_logits, top_k)
            next_token_logits = self._top_p_filtering(next_token_logits, top_p)
            
            # Calculate probabilities
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Sample next tokens
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            # Update metrics
            with torch.no_grad():
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                metrics['avg_entropy'].append(entropy.mean().item())
                metrics['tokens_per_step'].append(next_tokens.shape[1])
            
            # Append new tokens
            input_ids = torch.cat([input_ids, next_tokens], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones_like(next_tokens)
                ], dim=1)
            
            # Add to generated tokens list
            generated_tokens.extend(next_tokens.view(-1).tolist())
            
            # Check for early stopping
            if step >= min_new_tokens:
                # Check if any sequence has reached an end condition
                if self._should_stop(next_tokens):
                    metrics['early_stops'] += 1
                    break
        
        # Finalize metrics
        metrics['avg_entropy'] = np.mean(metrics['avg_entropy'])
        metrics['total_tokens'] = len(generated_tokens)
        metrics['steps'] = step + 1
        
        return input_ids, metrics
    
    def _should_stop(self, tokens: torch.Tensor) -> bool:
        """Check if generation should stop"""
        # Add custom stopping conditions here
        # For example, stop on specific tokens or patterns
        return False 