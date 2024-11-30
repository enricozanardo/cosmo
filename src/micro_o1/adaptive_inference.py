import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from .reasoning_cot import CoTReasoningGenerator
from .generation import OutputGenerator
from loguru import logger

class AdaptiveInference(nn.Module):
    """Adaptive inference module that combines CoT and output generation"""
    
    def __init__(self,
                 model: nn.Module,
                 tokenizer: 'MicroTokenizer',
                 hidden_size: int = 768,
                 max_reasoning_steps: int = 5,
                 temperature: float = 0.7):
        super().__init__()
        
        # Base components
        self.model = model
        self.tokenizer = tokenizer
        
        # Initialize value head for RL
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Initialize reasoning and generation components
        self.cot_generator = CoTReasoningGenerator(
            model=model,
            tokenizer=tokenizer,
            value_head=self.value_head,
            max_reasoning_steps=max_reasoning_steps
        )
        
        self.output_generator = OutputGenerator(
            model=model,
            vocab_size=tokenizer.vocab_size,
            temperature=temperature
        )
        
        logger.info("Initialized adaptive inference module")
    
    def forward(self,
                input_text: str,
                require_reasoning: bool = True,
                max_tokens: int = 100) -> Dict[str, torch.Tensor]:
        """Forward pass with adaptive reasoning"""
        # First, encode input
        encoded = self.tokenizer.encode(input_text)
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        # Decide if we need reasoning
        if require_reasoning:
            # Generate reasoning steps and conclusion
            steps, conclusion, metrics = self.cot_generator.generate_with_cot(
                input_text,
                max_tokens=max_tokens
            )
            
            # Add reasoning to context
            reasoning_text = self._format_reasoning(steps, conclusion)
            full_context = f"{input_text}\n{reasoning_text}"
            
            # Re-encode with reasoning
            encoded = self.tokenizer.encode(full_context)
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
        
        # Generate final output
        output_ids, output_metrics = self.output_generator.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens
        )
        
        # Decode output
        output_text = self.tokenizer.decode(output_ids)[0]
        
        # Combine metrics
        metrics = {
            'output_metrics': output_metrics,
            'reasoning_used': require_reasoning
        }
        if require_reasoning:
            metrics['reasoning_metrics'] = metrics
        
        return {
            'output_text': output_text,
            'reasoning_steps': steps if require_reasoning else None,
            'metrics': metrics
        }
    
    def _format_reasoning(self, steps: List[str], conclusion: str) -> str:
        """Format reasoning steps and conclusion"""
        reasoning = f"{self.tokenizer.special_tokens['reason_start']}\n"
        
        for i, step in enumerate(steps, 1):
            reasoning += f"{self.tokenizer.special_tokens['step_token']}{i}: {step}\n"
        
        reasoning += f"{self.tokenizer.special_tokens['therefore_token']} {conclusion}\n"
        reasoning += f"{self.tokenizer.special_tokens['reason_end']}"
        
        return reasoning
    
    def adapt_reasoning_depth(self, 
                            input_complexity: float,
                            performance_metrics: Dict) -> int:
        """Adapt maximum reasoning steps based on input and performance"""
        base_steps = self.cot_generator.max_reasoning_steps
        
        # Adjust based on input complexity
        complexity_factor = min(max(input_complexity, 0.5), 2.0)
        
        # Adjust based on performance
        if 'reasoning_metrics' in performance_metrics:
            metrics = performance_metrics['reasoning_metrics']
            performance_factor = metrics.get('total_reward', 1.0)
        else:
            performance_factor = 1.0
        
        # Calculate adapted steps
        adapted_steps = int(base_steps * complexity_factor * performance_factor)
        adapted_steps = max(1, min(adapted_steps, 10))  # Limit range
        
        logger.debug(f"Adapted reasoning steps: {adapted_steps} (base: {base_steps})")
        return adapted_steps 