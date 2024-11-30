import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from .reasoning import ReasoningTokenizer, ReasoningGenerator


class CoTReasoningGenerator(ReasoningGenerator):
    """Chain of Thought reasoning generator with RL optimization"""
    
    def __init__(self, 
                 model: nn.Module,
                 tokenizer: ReasoningTokenizer,
                 value_head: nn.Module,  # For RL value estimation
                 max_reasoning_steps: int = 5,
                 reward_weights: Dict[str, float] = None):
        super().__init__(model, tokenizer, max_reasoning_steps)
        self.value_head = value_head
        self.reward_weights = reward_weights or {
            'correctness': 1.0,
            'step_coherence': 0.5,
            'conciseness': 0.3
        }
    
    def generate_with_cot(self, 
                         prompt: str,
                         temperature: float = 0.7,
                         max_tokens: int = 100) -> Tuple[List[str], str, Dict]:
        """Generate reasoning with Chain of Thought and collect RL metrics"""
        # Initialize RL tracking
        step_values = []
        step_logprobs = []
        
        # Start reasoning chain
        current_text = f"{prompt}\n{self.tokenizer.reasoning_start}\n"
        steps = []
        
        # Generate reasoning steps with value estimation
        for step in range(self.max_reasoning_steps):
            # Generate next step and get logprobs
            step_text, step_logprob = self._generate_step_with_logprobs(
                current_text, temperature, max_tokens
            )
            
            # Estimate value of current state
            state_value = self._estimate_value(current_text + step_text)
            step_values.append(state_value)
            step_logprobs.append(step_logprob)
            
            # Check for conclusion
            if self.tokenizer.therefore_token in step_text:
                break
            
            # Add step
            steps.append(step_text)
            current_text += f"{self.tokenizer.step_token}{step+1}: {step_text}\n"
        
        # Generate conclusion
        conclusion = self._generate_conclusion(current_text, temperature, max_tokens)
        
        # Calculate rewards
        metrics = self._calculate_rewards(steps, conclusion)
        
        # Calculate total reward
        total_reward = sum(
            self.reward_weights[k] * metrics[k] 
            for k in ['correctness', 'step_coherence', 'conciseness']
        )
        
        # Add RL metrics
        metrics.update({
            'step_values': step_values,
            'step_logprobs': step_logprobs,
            'total_steps': len(steps),
            'total_reward': total_reward  # Add total reward to metrics
        })
        
        return steps, conclusion, metrics
    
    def _generate_step_with_logprobs(self, 
                                   context: str, 
                                   temperature: float,
                                   max_tokens: int) -> Tuple[str, torch.Tensor]:
        """Generate step and return logprobs for RL"""
        prompt = f"{context}{self.tokenizer.step_token}"
        
        # Generate with logprobs
        output, logprobs = self.model.generate_with_logprobs(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_tokens=["\n", self.tokenizer.therefore_token]
        )
        
        return output.strip(), logprobs
    
    def _estimate_value(self, state: str) -> float:
        """Estimate value of current reasoning state"""
        # Encode state
        state_encoding = self.model.encode(state)
        
        # Get value estimate
        value = self.value_head(state_encoding)
        return value.item()
    
    def _calculate_rewards(self, steps: List[str], conclusion: str) -> Dict[str, float]:
        """Calculate rewards for RL training"""
        rewards = {}
        
        # Correctness reward (based on conclusion quality)
        rewards['correctness'] = self._evaluate_correctness(conclusion)
        
        # Step coherence reward
        rewards['step_coherence'] = self._evaluate_coherence(steps)
        
        # Conciseness reward (penalize unnecessary steps)
        rewards['conciseness'] = max(0, 1 - (len(steps) / self.max_reasoning_steps))
        
        # Calculate weighted total
        total_reward = sum(
            self.reward_weights[k] * v for k, v in rewards.items()
        )
        rewards['total'] = total_reward
        
        return rewards
    
    def _evaluate_correctness(self, conclusion: str) -> float:
        """Evaluate correctness of conclusion"""
        # This would be implemented based on task-specific metrics
        # For now, return a dummy score
        return 0.8
    
    def _evaluate_coherence(self, steps: List[str]) -> float:
        """Evaluate coherence between reasoning steps"""
        if len(steps) <= 1:
            return 1.0
            
        # Calculate coherence between consecutive steps
        coherence_scores = []
        for i in range(len(steps) - 1):
            score = self._step_coherence(steps[i], steps[i+1])
            coherence_scores.append(score)
            
        return sum(coherence_scores) / len(coherence_scores)
    
    def _step_coherence(self, step1: str, step2: str) -> float:
        """Calculate coherence between two steps"""
        # This would use the model to evaluate logical connection
        # For now, return a dummy score
        return 0.9 