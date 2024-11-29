import pytest
import torch
from src.micro_o1.config.logging import setup_logger
from src.micro_o1.main import MicroO1TransformerWithRL

# Setup logger
logger = setup_logger("rl", "tests/rl.log")

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
def rl_model(model_params):
    """Fixture for RL model"""
    return MicroO1TransformerWithRL(**model_params)

def test_reward_model(rl_model):
    """Test reward model outputs"""
    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, rl_model.embeddings.word_embeddings.embedding_dim)
    
    rewards = rl_model.reward_model(hidden_states)
    assert rewards.shape == (batch_size, seq_len)
    assert torch.isfinite(rewards).all()
    logger.debug(f"Reward shape: {rewards.shape}")

def test_critic(rl_model):
    """Test critic outputs"""
    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, rl_model.embeddings.word_embeddings.embedding_dim)
    
    values = rl_model.critic(hidden_states)
    assert values.shape == (batch_size, seq_len)
    assert torch.isfinite(values).all()
    logger.debug(f"Value shape: {values.shape}")

def test_ppo_loss(rl_model):
    """Test PPO loss computation"""
    batch_size, seq_len, vocab_size = 2, 10, rl_model.embeddings.vocab_size
    
    old_logits = torch.randn(batch_size, seq_len, vocab_size)
    new_logits = torch.randn(batch_size, seq_len, vocab_size)
    values = torch.randn(batch_size, seq_len)
    rewards = torch.randn(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len)
    
    loss, losses = rl_model.compute_ppo_loss(
        old_logits, new_logits, values, rewards, attention_mask
    )
    
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
    logger.debug(f"PPO losses: {losses}")

def test_forward_rl(rl_model):
    """Test RL forward pass"""
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, rl_model.embeddings.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    reason_token_mask = torch.zeros(batch_size, seq_len)
    reason_token_mask[:, [2, 5, 8]] = 1
    
    logits, values, rewards = rl_model.forward_rl(
        input_ids, attention_mask, reason_token_mask
    )
    
    assert logits.shape == (batch_size, seq_len, rl_model.embeddings.vocab_size)
    assert values.shape == (batch_size, seq_len)
    assert rewards.shape == (batch_size, seq_len)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(values).all()
    assert torch.isfinite(rewards).all()
    logger.debug("RL forward pass shapes verified")

@pytest.mark.parametrize("batch_size,seq_length", [
    (1, 10),
    (4, 64),
    (8, 128)
])
def test_rl_different_sizes(rl_model, batch_size, seq_length):
    """Test RL components with different batch sizes and sequence lengths"""
    input_ids = torch.randint(0, rl_model.embeddings.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    reason_token_mask = torch.zeros(batch_size, seq_length)
    reason_token_mask[:, [2, 5, 8]] = 1
    
    logits, values, rewards = rl_model.forward_rl(
        input_ids, attention_mask, reason_token_mask
    )
    
    assert logits.shape == (batch_size, seq_length, rl_model.embeddings.vocab_size)
    assert values.shape == (batch_size, seq_length)
    assert rewards.shape == (batch_size, seq_length)
    logger.debug(f"Successfully processed batch of shape {logits.shape}") 