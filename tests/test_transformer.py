import pytest
import torch
from src.micro_o1.transformer import TransformerWithValueHead

@pytest.fixture
def model_params():
    return {
        'vocab_size': 50257,
        'hidden_size': 768,
        'num_layers': 12,
        'num_heads': 12,
        'max_seq_length': 1024
    }

def test_transformer_with_cot_and_rl(model_params):
    """Test transformer with CoT and RL components"""
    model = TransformerWithValueHead(**model_params)
    
    # Test input with reasoning tokens
    input_ids = torch.randint(0, model_params['vocab_size'], (1, 10))
    input_ids[0, 5] = 1  # Add step token
    
    outputs = model(input_ids, return_value=True)
    
    # Check outputs
    assert 'logits' in outputs
    assert 'value' in outputs
    assert outputs['logits'].shape == (1, 10, model_params['vocab_size'])
    assert outputs['value'].shape == (1, 1)

def test_generation_with_value(model_params):
    """Test generation with value estimation"""
    model = TransformerWithValueHead(**model_params)
    
    input_ids = torch.randint(0, model_params['vocab_size'], (1, 5))
    outputs = model.generate(input_ids, max_length=10, return_value=True)
    
    assert 'generated_ids' in outputs
    assert 'values' in outputs
    assert outputs['generated_ids'].shape[1] == 10
    assert outputs['values'].shape == (10, 1)

def test_reasoning_embeddings(model_params):
    """Test CoT-specific embeddings"""
    model = TransformerWithValueHead(**model_params)
    
    # Create sequence with reasoning tokens
    input_ids = torch.zeros((1, 15), dtype=torch.long)
    input_ids[0, 0] = 1  # Step token
    input_ids[0, 5] = 1  # Another step
    input_ids[0, 10] = 2  # Therefore token
    
    outputs = model(input_ids)
    assert outputs['logits'].shape == (1, 15, model_params['vocab_size']) 