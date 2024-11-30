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

def test_reasoning_positions(model_params):
    """Test extraction of reasoning positions"""
    model = TransformerWithValueHead(**model_params)
    
    # Create input with reasoning tokens
    input_ids = torch.zeros((1, 10), dtype=torch.long)
    input_ids[0, 2] = model.step_token_id      # Step 1
    input_ids[0, 5] = model.step_token_id      # Step 2
    input_ids[0, 8] = model.therefore_token_id # Therefore
    
    reasoning_ids, step_positions = model._get_reasoning_positions(input_ids)
    
    # Check reasoning IDs
    assert (reasoning_ids[0, 2] == 1).item()  # Step token
    assert (reasoning_ids[0, 5] == 1).item()  # Step token
    assert (reasoning_ids[0, 8] == 2).item()  # Therefore token
    
    # Check step positions
    assert (step_positions[0, 2] == 1).item()  # First step
    assert (step_positions[0, 5] == 2).item()  # Second step

def test_value_estimation(model_params):
    """Test value estimation with reasoning"""
    model = TransformerWithValueHead(**model_params)
    
    # Create input with reasoning
    input_ids = torch.zeros((1, 15), dtype=torch.long)
    input_ids[0, [2, 5, 8]] = model.step_token_id  # Add steps
    
    outputs = model.forward(input_ids, return_value=True, use_reasoning=True)
    
    assert 'value' in outputs
    assert 'reasoning_metrics' in outputs
    assert outputs['reasoning_metrics']['num_steps'].item() == 3

def test_reasoning_guided_generation(model_params):
    """Test generation with reasoning guidance"""
    model = TransformerWithValueHead(**model_params)
    
    input_ids = torch.zeros((1, 5), dtype=torch.long)
    outputs = model.generate_with_reasoning(
        input_ids,
        max_length=10,
        temperature=0.7
    )
    
    assert 'generated_ids' in outputs
    assert 'values' in outputs
    assert 'reasoning_metrics' in outputs
    assert len(outputs['values']) == 10  # One value per generation step

def test_reasoning_embeddings_combination(model_params):
    """Test combination of different embedding types"""
    model = TransformerWithValueHead(**model_params)
    
    # Create input with reasoning tokens
    input_ids = torch.zeros((1, 10), dtype=torch.long)
    reasoning_ids = torch.zeros_like(input_ids)
    step_positions = torch.zeros_like(input_ids)
    
    # Add some reasoning positions
    reasoning_ids[0, [2, 5]] = 1
    step_positions[0, [2, 5]] = torch.tensor([1, 2])
    
    embeddings = model._get_embeddings(
        input_ids,
        reasoning_ids=reasoning_ids,
        step_positions=step_positions
    )
    
    assert embeddings.shape == (1, 10, model_params['hidden_size'])
    assert torch.isfinite(embeddings).all() 