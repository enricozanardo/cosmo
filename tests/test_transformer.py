import pytest
import torch
from src.micro_o1.config.logging import setup_logger
from src.micro_o1.main import (
    MultiHeadAttention,
    TransformerBlock,
    MicroO1Transformer,
    MicroO1Tokenizer
)

# Setup logger
logger = setup_logger("transformer", "tests/transformer.log")

@pytest.fixture
def model_params():
    """Fixture for model parameters"""
    return {
        'vocab_size': 50257,  # GPT-2 vocab size
        'hidden_size': 768,
        'num_layers': 12,
        'num_heads': 12,
        'ff_dim': 3072,
        'max_position_embeddings': 1024,
        'dropout': 0.1
    }

@pytest.fixture
def attention():
    """Fixture for attention layer"""
    return MultiHeadAttention(hidden_size=768, num_heads=12)

@pytest.fixture
def transformer_block():
    """Fixture for transformer block"""
    return TransformerBlock(hidden_size=768, num_heads=12, ff_dim=3072)

@pytest.fixture
def transformer(model_params):
    """Fixture for full transformer model"""
    return MicroO1Transformer(**model_params)

def test_attention_shape(attention):
    """Test attention output shapes"""
    batch_size, seq_len, hidden_size = 2, 10, 768
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    output = attention(x, x, x)
    assert output.shape == (batch_size, seq_len, hidden_size)
    logger.debug(f"Attention output shape: {output.shape}")

def test_transformer_block(transformer_block):
    """Test transformer block forward pass"""
    batch_size, seq_len, hidden_size = 2, 10, 768
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    output = transformer_block(x)
    assert output.shape == (batch_size, seq_len, hidden_size)
    assert torch.isfinite(output).all()
    logger.debug(f"Transformer block output shape: {output.shape}")

def test_transformer_forward(transformer, model_params):
    """Test full transformer forward pass"""
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, model_params['vocab_size'], (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    output = transformer(input_ids, attention_mask)
    assert output.shape == (batch_size, seq_len, model_params['vocab_size'])
    assert torch.isfinite(output).all()
    logger.debug(f"Transformer output shape: {output.shape}")

def test_attention_mask(transformer):
    """Test attention masking"""
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, transformer.embeddings.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, seq_len//2:] = 0  # Mask second half of sequence
    
    output = transformer(input_ids, attention_mask)
    assert torch.isfinite(output).all()
    logger.debug("Attention mask test passed")

def test_gradient_flow(transformer):
    """Test gradient flow through the model"""
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, transformer.embeddings.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    output = transformer(input_ids, attention_mask)
    loss = output.sum()
    loss.backward()
    
    # Check if gradients exist
    for name, param in transformer.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
    logger.debug("Gradient flow test passed")

@pytest.mark.parametrize("batch_size,seq_length", [
    (1, 10),
    (4, 64),
    (8, 128)
])
def test_different_sizes(transformer, batch_size, seq_length):
    """Test transformer with different batch sizes and sequence lengths"""
    input_ids = torch.randint(0, transformer.embeddings.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    output = transformer(input_ids, attention_mask)
    assert output.shape == (batch_size, seq_length, transformer.embeddings.vocab_size)
    logger.debug(f"Successfully processed batch of shape {output.shape}") 