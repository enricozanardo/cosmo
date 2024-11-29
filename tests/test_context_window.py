import pytest
import torch
from src.micro_o1.context_window import SlidingWindowAttention, ExtendedContextTransformer
from src.micro_o1.config.logging import setup_logger

# Setup logger
logger = setup_logger("context_window", "tests/context_window.log")

@pytest.fixture
def attention_params():
    """Fixture for attention parameters"""
    return {
        'hidden_size': 768,
        'num_heads': 12,
        'window_size': 512,
        'stride': 256
    }

@pytest.fixture
def transformer_params():
    """Fixture for transformer parameters"""
    return {
        'hidden_size': 768,
        'num_heads': 12,
        'ff_dim': 3072,
        'window_size': 512,
        'stride': 256,
        'dropout': 0.1
    }

def test_sliding_window_attention_shape(attention_params):
    """Test sliding window attention output shapes"""
    model = SlidingWindowAttention(**attention_params)
    batch_size, seq_len = 2, 1024
    hidden_states = torch.randn(batch_size, seq_len, attention_params['hidden_size'])
    
    output = model(hidden_states)
    assert output.shape == (batch_size, seq_len, attention_params['hidden_size'])
    logger.debug(f"Output shape: {output.shape}")

def test_sliding_window_attention_mask(attention_params):
    """Test attention masking"""
    model = SlidingWindowAttention(**attention_params)
    batch_size, seq_len = 2, 1024
    hidden_states = torch.randn(batch_size, seq_len, attention_params['hidden_size'])
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    attention_mask[:, seq_len//2:] = 0  # Mask second half
    
    output = model(hidden_states, attention_mask)
    assert output.shape == (batch_size, seq_len, attention_params['hidden_size'])
    assert torch.isfinite(output).all()
    logger.debug("Attention masking verified")

def test_extended_context_transformer(transformer_params):
    """Test extended context transformer"""
    model = ExtendedContextTransformer(**transformer_params)
    batch_size, seq_len = 2, 2048
    hidden_states = torch.randn(batch_size, seq_len, transformer_params['hidden_size'])
    
    output = model(hidden_states)
    assert output.shape == (batch_size, seq_len, transformer_params['hidden_size'])
    assert torch.isfinite(output).all()
    logger.debug(f"Transformer output shape: {output.shape}")

def test_gradient_checkpointing(transformer_params):
    """Test gradient checkpointing"""
    try:
        import torch.utils.checkpoint
    except ImportError:
        pytest.skip("torch.utils.checkpoint not available")
    
    model = ExtendedContextTransformer(**transformer_params)
    model.gradient_checkpointing = True
    model.train()
    
    batch_size, seq_len = 2, 2048
    hidden_states = torch.randn(batch_size, seq_len, transformer_params['hidden_size'], requires_grad=True)
    
    try:
        # Forward pass with gradient checkpointing
        output = model(hidden_states)
        loss = output.mean()  # Use mean instead of sum for better numerical stability
        loss.backward()
        
        # Verify gradients
        assert hidden_states.grad is not None, "No gradients computed"
        assert torch.isfinite(hidden_states.grad).all(), "Found infinite gradients"
        assert (hidden_states.grad != 0).any(), "All gradients are zero"
        
        logger.debug("Gradient checkpointing verified")
    except Exception as e:
        logger.error(f"Gradient checkpointing failed: {e}")
        raise

@pytest.mark.parametrize("seq_length", [512, 1024, 2048, 4096])
def test_different_sequence_lengths(transformer_params, seq_length):
    """Test different sequence lengths"""
    model = ExtendedContextTransformer(**transformer_params)
    batch_size = 2
    hidden_states = torch.randn(batch_size, seq_length, transformer_params['hidden_size'])
    
    output = model(hidden_states)
    assert output.shape == (batch_size, seq_length, transformer_params['hidden_size'])
    assert torch.isfinite(output).all()
    logger.debug(f"Successfully processed sequence length {seq_length}")

def test_memory_efficiency(transformer_params):
    """Test memory efficiency"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    model = ExtendedContextTransformer(**transformer_params)
    batch_size, seq_len = 1, 8192
    hidden_states = torch.randn(batch_size, seq_len, transformer_params['hidden_size'])
    
    # Track memory usage
    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
        torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        output = model(hidden_states)
    
    # Check memory usage if available
    if hasattr(torch.cuda, 'max_memory_allocated'):
        mem_used = torch.cuda.max_memory_allocated()
        logger.debug(f"Memory used for sequence length {seq_len}: {mem_used/1024/1024:.2f}MB")
        
        # Memory usage should scale roughly linearly with sequence length
        expected_mem = seq_len * transformer_params['hidden_size'] * 4 * 2  # Rough estimate
        assert mem_used < expected_mem, "Memory usage higher than expected"
    else:
        logger.warning("CUDA memory tracking not available")