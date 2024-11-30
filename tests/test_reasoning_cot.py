import pytest
import torch
import torch.nn as nn
from src.micro_o1.reasoning_cot import CoTReasoningGenerator
from src.micro_o1.reasoning import ReasoningTokenizer
from src.micro_o1.config.logging import setup_logger

# Setup logger
logger = setup_logger("reasoning_cot", "tests/reasoning_cot.log")

@pytest.fixture
def tokenizer():
    """Fixture for reasoning tokenizer"""
    return ReasoningTokenizer()

@pytest.fixture
def value_head():
    """Fixture for value estimator"""
    return nn.Linear(768, 1)  # Simple value estimator

@pytest.fixture
def cot_model():
    """Fixture for CoT model"""
    class DummyCoTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size = 768

        def generate(self, prompt, max_tokens=100, temperature=0.7, stop_tokens=None):
            """Dummy generate method"""
            return "This is a generated response"

        def generate_with_logprobs(self, prompt, **kwargs):
            """Generate with logprobs"""
            return "This is a step", torch.randn(1)
            
        def encode(self, text):
            """Encode text to hidden states"""
            return torch.randn(1, self.hidden_size)
        
        def forward(self, *args, **kwargs):
            """Forward pass"""
            return self.encode("")
    
    return DummyCoTModel()

def test_cot_generation(tokenizer, cot_model, value_head):
    """Test CoT reasoning generation"""
    generator = CoTReasoningGenerator(cot_model, tokenizer, value_head)
    prompt = "What is 2+2?"
    
    steps, conclusion, metrics = generator.generate_with_cot(prompt)
    
    # Check basic outputs
    assert len(steps) > 0
    assert conclusion != ""
    assert isinstance(conclusion, str)
    
    # Check RL metrics
    assert 'step_values' in metrics
    assert 'step_logprobs' in metrics
    assert 'total_reward' in metrics
    
    # Check reward components
    assert 'correctness' in metrics
    assert 'step_coherence' in metrics
    assert 'conciseness' in metrics
    
    # Check values are reasonable
    assert all(0 <= metrics[k] <= 1 for k in ['correctness', 'step_coherence', 'conciseness'])
    assert len(metrics['step_values']) == len(steps)
    
    # Log some debug info
    logger.debug(f"Generated steps: {steps}")
    logger.debug(f"Conclusion: {conclusion}")
    logger.debug(f"Metrics: {metrics}")
    
    # Check step format
    for step in steps:
        assert isinstance(step, str)
        assert len(step) > 0

def test_reward_calculation(tokenizer, cot_model, value_head):
    """Test reward calculation"""
    generator = CoTReasoningGenerator(cot_model, tokenizer, value_head)
    
    # Test with custom reward weights
    custom_weights = {
        'correctness': 0.8,
        'step_coherence': 0.6,
        'conciseness': 0.4
    }
    generator.reward_weights = custom_weights
    
    steps = ["Step 1", "Step 2"]
    conclusion = "Therefore, 4"
    
    rewards = generator._calculate_rewards(steps, conclusion)
    
    # Check reward components
    assert all(k in rewards for k in custom_weights.keys())
    assert 'total' in rewards
    
    # Check weighted sum
    expected_total = sum(
        custom_weights[k] * rewards[k] for k in custom_weights.keys()
    )
    assert abs(rewards['total'] - expected_total) < 1e-6
    
    # Log reward details
    logger.debug(f"Rewards: {rewards}")
    logger.debug(f"Expected total: {expected_total}")

def test_value_estimation(tokenizer, cot_model, value_head):
    """Test value estimation"""
    generator = CoTReasoningGenerator(cot_model, tokenizer, value_head)
    state = "What is 2+2?\nStep 1: First, let's understand that 2+2 means adding two twos"
    
    value = generator._estimate_value(state)
    assert isinstance(value, float)
    logger.debug(f"Estimated value: {value}")

def test_step_coherence(tokenizer, cot_model, value_head):
    """Test step coherence calculation"""
    generator = CoTReasoningGenerator(cot_model, tokenizer, value_head)
    steps = [
        "First, let's understand that 2+2 means adding two twos",
        "Next, we can count: 2 plus 2 equals 4",
        "Finally, we verify that 4 is our answer"
    ]
    
    coherence = generator._evaluate_coherence(steps)
    assert 0 <= coherence <= 1
    logger.debug(f"Step coherence: {coherence}")