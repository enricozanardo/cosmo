import pytest
import torch
from src.micro_o1.config.logging import setup_logger
from src.micro_o1.main import (
    MixedPrecisionWrapper,
    MicroO1TransformerWithMixedPrecision
)

# Setup logger
logger = setup_logger("mixed_precision", "tests/mixed_precision.log")

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
def mixed_precision_model(model_params):
    """Fixture for mixed precision model"""
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    return MicroO1TransformerWithMixedPrecision(
        **model_params,
        enable_mixed_precision=True,
        device_type=device_type
    )

def test_mixed_precision_conversion(mixed_precision_model):
    """Test conversion of modules to fp16"""
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for name, module in mixed_precision_model.named_modules():
        if hasattr(module, 'weight'):
            if device_type == 'cuda':
                # On CUDA, check for proper mixed precision
                if any(fp32_name in name.lower() for fp32_name in ['layer_norm', 'softmax', 'embedding']):
                    assert module.weight.dtype == torch.float32, f"Module {name} should be float32"
                else:
                    assert module.weight.dtype == torch.float16, f"Module {name} should be float16"
            else:
                # On CPU, everything should be float32
                assert module.weight.dtype == torch.float32, f"Module {name} should be float32 on CPU"
    
    logger.debug("Module precision types verified")

def test_forward_mixed_precision(mixed_precision_model):
    """Test forward pass with mixed precision"""
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, mixed_precision_model.embeddings.vocab_size, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
    reason_token_mask = torch.zeros(batch_size, seq_len, dtype=torch.float32)
    reason_token_mask[:, [2, 5, 8]] = 1
    
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    expected_dtype = torch.float16 if device_type == 'cuda' else torch.float32
    
    outputs = mixed_precision_model(input_ids, attention_mask, reason_token_mask)
    
    # Check output properties
    assert outputs.dtype == expected_dtype
    assert torch.isfinite(outputs).all()
    logger.debug(f"Mixed precision output shape: {outputs.shape}")

def test_loss_computation(mixed_precision_model):
    """Test loss computation with mixed precision"""
    batch_size, seq_len = 2, 10
    vocab_size = mixed_precision_model.embeddings.vocab_size
    
    # Create test inputs
    old_logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float16)
    new_logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float16)
    values = torch.randn(batch_size, seq_len, dtype=torch.float16)
    rewards = torch.randn(batch_size, seq_len, dtype=torch.float16)
    attention_mask = torch.ones(batch_size, seq_len)
    
    with torch.cuda.amp.autocast():
        loss, metrics = mixed_precision_model.compute_loss(
            old_logits, new_logits, values, rewards, attention_mask
        )
    
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    logger.debug(f"Loss computed with mixed precision: {loss.item()}")

def test_gradient_scaling(mixed_precision_model):
    """Test gradient scaling in mixed precision training"""
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, mixed_precision_model.embeddings.vocab_size, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
    reason_token_mask = torch.zeros(batch_size, seq_len, dtype=torch.float32)
    reason_token_mask[:, [2, 5, 8]] = 1
    
    optimizer = torch.optim.Adam(mixed_precision_model.parameters())
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Forward pass with RL components to ensure all parts are used
    with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
        logits, values, rewards = mixed_precision_model.forward_rl(
            input_ids, attention_mask, reason_token_mask
        )
        
        # Compute PPO loss to ensure reward model gradients flow
        loss, _ = mixed_precision_model.compute_ppo_loss(
            logits.detach(),  # Old logits
            logits,           # New logits
            values,
            rewards,
            attention_mask
        )
    
    # Backward pass with scaling
    if mixed_precision_model.mixed_precision.scaler is not None:
        mixed_precision_model.mixed_precision.scaler.scale(loss).backward()
    else:
        loss.backward()
    
    # Check if gradients exist and are finite
    for name, param in mixed_precision_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Infinite gradient for {name}"
            
            # Additional check for zero gradients
            if torch.all(param.grad == 0):
                logger.warning(f"Zero gradients for {name}")
    
    logger.debug("Gradient scaling verified")

@pytest.mark.parametrize("batch_size,seq_length", [
    (1, 10),
    (4, 64),
    (8, 128)
])
def test_mixed_precision_different_sizes(mixed_precision_model, batch_size, seq_length):
    """Test mixed precision with different batch sizes and sequence lengths"""
    input_ids = torch.randint(0, mixed_precision_model.embeddings.vocab_size, (batch_size, seq_length), dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.float32)
    reason_token_mask = torch.zeros(batch_size, seq_length, dtype=torch.float32)
    reason_token_mask[:, [2, 5, 8]] = 1
    
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
        outputs = mixed_precision_model(
            input_ids, attention_mask, reason_token_mask
        )
    
    # Check output properties
    assert outputs.shape == (batch_size, seq_length, mixed_precision_model.embeddings.vocab_size)
    if device_type == 'cuda':
        assert outputs.dtype == torch.float16
    else:
        assert outputs.dtype in (torch.float16, torch.float32)
    assert torch.isfinite(outputs).all()
    
    logger.debug(f"Successfully processed batch of shape {outputs.shape}") 