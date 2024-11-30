import pytest
import torch
from src.micro_o1.reasoning import ReasoningTokenizer, ReasoningGenerator
from src.micro_o1.config.logging import setup_logger

logger = setup_logger("reasoning", "tests/reasoning.log")

@pytest.fixture
def tokenizer():
    return ReasoningTokenizer()

@pytest.fixture
def dummy_model():
    class DummyModel:
        def generate(self, prompt, max_tokens=100, temperature=0.7, stop_tokens=None):
            # Simulate simple reasoning generation
            if "<step>" in prompt:
                return "This is a reasoning step"
            elif "<therefore>" in prompt:
                return "This is the conclusion"
            return ""
    return DummyModel()

def test_reasoning_tokens(tokenizer):
    """Test reasoning token handling"""
    steps = ["First step", "Second step"]
    result = tokenizer.add_reasoning_tokens("Test", steps)
    
    assert tokenizer.reasoning_start in result
    assert tokenizer.reasoning_end in result
    assert tokenizer.step_token in result
    assert tokenizer.therefore_token in result
    
    for step in steps:
        assert step in result

def test_reasoning_generation(tokenizer, dummy_model):
    """Test reasoning generation"""
    generator = ReasoningGenerator(dummy_model, tokenizer)
    prompt = "What is 2+2?"
    
    steps, conclusion = generator.generate_reasoning(prompt)
    
    assert len(steps) > 0
    assert all(isinstance(step, str) for step in steps)
    assert isinstance(conclusion, str)
    assert conclusion != ""

def test_max_steps_limit(tokenizer, dummy_model):
    """Test maximum steps limit"""
    max_steps = 3
    generator = ReasoningGenerator(dummy_model, tokenizer, max_reasoning_steps=max_steps)
    prompt = "Complex problem"
    
    steps, _ = generator.generate_reasoning(prompt)
    assert len(steps) <= max_steps 