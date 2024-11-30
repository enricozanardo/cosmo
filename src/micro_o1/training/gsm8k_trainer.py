import torch
import torch.nn as nn
from datasets import load_dataset
from typing import Dict, List, Tuple
from loguru import logger
from ..transformer import TransformerWithValueHead

class GSM8KTrainer:
    """Trainer for GSM8K math reasoning dataset with RL and CoT"""
    
    def __init__(self,
                 model: TransformerWithValueHead,
                 learning_rate: float = 1e-4,
                 batch_size: int = 16,
                 max_steps: int = 10000,
                 warmup_steps: int = 1000,
                 device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Load dataset
        self.dataset = load_dataset("openai/gsm8k", "main")
        logger.info(f"Loaded GSM8K dataset with {len(self.dataset['train'])} examples")
        
        # Training config
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
    
    def prepare_batch(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Prepare batch with reasoning structure"""
        questions = [ex['question'] for ex in examples]
        answers = [ex['answer'] for ex in examples]
        
        # Format with reasoning template
        inputs = []
        tokenizer = self.model.tokenizer  # Get tokenizer from model
        
        for q, a in zip(questions, answers):
            # Extract numerical answer
            final_answer = a.split('####')[-1].strip()
            
            # Format with reasoning
            formatted = (
                f"Question: {q}\n"
                f"{tokenizer.reasoning_start}\n"
                f"Let's solve this step by step:\n"
                f"{a}\n"
                f"{tokenizer.therefore_token}\n"
                f"Therefore, the answer is {final_answer}\n"
                f"{tokenizer.reasoning_end}"
            )
            inputs.append(formatted)
        
        # Tokenize
        encoded = tokenizer.encode(
            inputs,
            padding=True,
            truncation=True,
            max_length=1024
        )
        
        # Move tensors to device and ensure they're within vocab range
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Safety check - clip token IDs to vocab size
        vocab_size = self.model.token_embeddings.num_embeddings
        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def compute_policy_loss(self,
                          logprobs: torch.Tensor,
                          values: torch.Tensor,
                          rewards: torch.Tensor) -> torch.Tensor:
        """Compute PPO policy loss"""
        # Policy gradient loss
        advantages = rewards - values.detach()
        policy_loss = -(logprobs * advantages).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(values, rewards)
        
        # Combined loss
        loss = policy_loss + 0.5 * value_loss
        
        return loss
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with RL"""
        self.model.train()
        
        # Generate with reasoning - limit max_length to save memory
        outputs = self.model.generate_with_reasoning(
            input_ids=batch['input_ids'],
            max_length=32  # Limit generation length
        )
        
        # Clear unnecessary tensors
        del batch['input_ids']
        torch.cuda.empty_cache()
        
        # Get metrics
        reasoning_metrics = outputs['reasoning_metrics']
        num_steps = reasoning_metrics['num_steps']
        has_conclusion = reasoning_metrics['has_conclusion']
        
        # Calculate rewards on CPU to save GPU memory
        rewards = torch.zeros_like(outputs['values'], device='cpu')
        rewards += 0.1 * num_steps.cpu()
        rewards += 0.5 * has_conclusion.cpu()
        rewards = rewards.to(self.device)
        
        # Compute loss
        loss = self.compute_policy_loss(
            outputs['logprobs'],
            outputs['values'],
            rewards
        )
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Clear memory
        del outputs
        torch.cuda.empty_cache()
        
        return {
            'loss': loss.item(),
            'avg_reward': rewards.mean().item(),
            'num_steps': num_steps.mean().item()
        }
    
    def train(self) -> Dict[str, List[float]]:
        """Full training loop"""
        logger.info("Starting training")
        history = {
            'loss': [],
            'reward': [],
            'steps': []
        }
        
        # Reduce batch size if needed
        batch_size = min(self.batch_size, 8)  # Smaller batch size
        
        for step in range(self.max_steps):
            # Get batch
            batch_indices = torch.randint(
                0, len(self.dataset['train']),
                (batch_size,)
            )
            batch = self.dataset['train'].select(batch_indices.tolist())
            inputs = self.prepare_batch(batch)
            
            # Training step
            metrics = self.train_step(inputs)
            
            # Log metrics
            history['loss'].append(metrics['loss'])
            history['reward'].append(metrics['avg_reward'])
            history['steps'].append(metrics['num_steps'])
            
            if (step + 1) % 100 == 0:
                # Clear some memory
                torch.cuda.empty_cache()
                
                logger.info(
                    f"Step {step+1}/{self.max_steps} - "
                    f"Loss: {metrics['loss']:.4f} - "
                    f"Reward: {metrics['avg_reward']:.4f} - "
                    f"Steps: {metrics['num_steps']:.1f}"
                )
        
        return history