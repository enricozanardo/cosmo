import pytest
import torch
from src.micro_o1.config.logging import setup_logger
from src.micro_o1.main import (
    ChainOfThoughtProcessor,
    MicroO1TransformerWithCoT,
    MicroO1Tokenizer
)

# Setup logger
logger = setup_logger("cot", "tests/cot.log")

@pytest.fixture
def cot_processor():
    """Fixture for CoT processor"""
    return ChainOfThoughtProcessor(hidden_size=768)

@pytest.fixture
def model_params():
    """Fixture for model parameters"""
    return {
        'vocab_size': 50257,
        'hidden_size': 768,
        'num_layers': 12,
        'num_heads': 12,
        'ff_dim': 3072,
        'max_position_embeddings': 1024,
        'dropout': 0.1
    }

@pytest.fixture
def transformer_cot(model_params):
    """Fixture for transformer with CoT"""
    return MicroO1TransformerWithCoT(**model_params)

def test_cot_processor_shape(cot_processor):
    """Test CoT processor output shapes"""
    batch_size, seq_len, hidden_size = 2, 20, 768
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    reason_token_mask = torch.zeros(batch_size, seq_len)
    reason_token_mask[:, [5, 10, 15]] = 1  # Add reasoning steps
    
    output = cot_processor(hidden_states, attention_mask, reason_token_mask)
    assert output.shape == hidden_states.shape
    logger.debug(f"CoT processor output shape: {output.shape}")

def test_transformer_cot_forward(transformer_cot, model_params):
    """Test forward pass with CoT"""
    batch_size, seq_len = 2, 20
    input_ids = torch.randint(0, model_params['vocab_size'], (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    reason_token_mask = torch.zeros(batch_size, seq_len)
    reason_token_mask[:, [5, 10, 15]] = 1
    
    output = transformer_cot(input_ids, attention_mask, reason_token_mask)
    assert output.shape == (batch_size, seq_len, model_params['vocab_size'])
    assert torch.isfinite(output).all()
    logger.debug(f"Transformer CoT output shape: {output.shape}")

def test_reasoning_steps(transformer_cot):
    """Test processing of reasoning steps"""
    batch_size, seq_len = 2, 30
    input_ids = torch.randint(0, transformer_cot.embeddings.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Create reasoning steps pattern
    reason_token_mask = torch.zeros(batch_size, seq_len)
    reason_positions = [[5, 10, 15, 20], [8, 16, 24]]
    for batch_idx, positions in enumerate(reason_positions):
        reason_token_mask[batch_idx, positions] = 1
    
    output = transformer_cot(input_ids, attention_mask, reason_token_mask)
    assert torch.isfinite(output).all()
    logger.debug("Reasoning steps test passed")

def test_cot_gradient_flow(transformer_cot):
    """Test gradient flow through CoT"""
    batch_size, seq_len = 2, 20
    input_ids = torch.randint(0, transformer_cot.embeddings.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    reason_token_mask = torch.zeros(batch_size, seq_len)
    reason_token_mask[:, [5, 10, 15]] = 1
    
    output = transformer_cot(input_ids, attention_mask, reason_token_mask)
    loss = output.sum()
    loss.backward()
    
    # Check gradients
    assert transformer_cot.cot_processor.reasoning_attention.q_proj.weight.grad is not None
    assert transformer_cot.reasoning_token_embedding.grad is not None
    logger.debug("CoT gradient flow test passed")

@pytest.mark.parametrize("num_steps", [1, 3, 5])
def test_multiple_reasoning_steps(transformer_cot, num_steps):
    """Test different numbers of reasoning steps"""
    batch_size, seq_len = 2, 50
    input_ids = torch.randint(0, transformer_cot.embeddings.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Create reasoning steps
    reason_token_mask = torch.zeros(batch_size, seq_len)
    step_positions = torch.linspace(5, seq_len-5, num_steps).long()
    reason_token_mask[:, step_positions] = 1
    
    output = transformer_cot(input_ids, attention_mask, reason_token_mask)
    assert output.shape == (batch_size, seq_len, transformer_cot.embeddings.vocab_size)
    logger.debug(f"Processed {num_steps} reasoning steps successfully") 