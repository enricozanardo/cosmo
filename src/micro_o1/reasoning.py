import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple
from loguru import logger

class ReasoningTokenizer:
    """Handles reasoning-specific tokens and prompts"""
    
    def __init__(self):
        # Special tokens for reasoning
        self.reasoning_start = "<reason>"
        self.reasoning_end = ""
        self.step_token = "<step>"
        self.therefore_token = "<therefore>"
        
        # Prompt templates
        self.cot_template = "Let's solve this step by step:\n{reasoning}\nTherefore, {conclusion}"
        self.scratchpad_template = "Reasoning:\n{steps}\nConclusion: {answer}"
    
    def add_reasoning_tokens(self, text: str, steps: List[str]) -> str:
        """Add reasoning tokens to text"""
        reasoning = f"{self.reasoning_start}\n"
        for i, step in enumerate(steps, 1):
            reasoning += f"{self.step_token}{i}: {step}\n"
        reasoning += f"{self.therefore_token}\n"
        reasoning += f"{self.reasoning_end}"
        return reasoning

class ReasoningGenerator:
    """Generates reasoning steps for problem solving"""
    
    def __init__(self, 
                 model: nn.Module,
                 tokenizer: ReasoningTokenizer,
                 max_reasoning_steps: int = 5):
        self.model = model
        self.tokenizer = tokenizer
        self.max_reasoning_steps = max_reasoning_steps
        logger.info(f"Initialized reasoning generator with {max_reasoning_steps} max steps")
    
    def generate_reasoning(self, 
                         prompt: str,
                         temperature: float = 0.7,
                         max_tokens: int = 100) -> Tuple[List[str], str]:
        """Generate reasoning steps and conclusion"""
        # Start with reasoning token
        current_text = f"{prompt}\n{self.tokenizer.reasoning_start}\n"
        steps = []
        
        # Generate reasoning steps
        for step in range(self.max_reasoning_steps):
            # Generate next step
            step_text = self._generate_step(current_text, temperature, max_tokens)
            
            # Check if we've reached a conclusion
            if self.tokenizer.therefore_token in step_text:
                break
                
            # Add step
            steps.append(step_text)
            current_text += f"{self.tokenizer.step_token}{step+1}: {step_text}\n"
        
        # Generate conclusion
        conclusion = self._generate_conclusion(current_text, temperature, max_tokens)
        
        return steps, conclusion
    
    def _generate_step(self, context: str, temperature: float, max_tokens: int) -> str:
        """Generate a single reasoning step"""
        # Add step token to prompt
        prompt = f"{context}{self.tokenizer.step_token}"
        
        # Generate step text
        output = self.model.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_tokens=["\n", self.tokenizer.therefore_token]
        )
        
        return output.strip()
    
    def _generate_conclusion(self, context: str, temperature: float, max_tokens: int) -> str:
        """Generate final conclusion"""
        # Add therefore token to prompt
        prompt = f"{context}{self.tokenizer.therefore_token}"
        
        # Generate conclusion
        output = self.model.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_tokens=[self.tokenizer.reasoning_end]
        )
        
        return output.strip() 