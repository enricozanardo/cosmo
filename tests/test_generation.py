import pytest
import torch
from src.micro_o1.generation import OutputGenerator
from src.micro_o1.config.logging import setup_logger

# Setup logger
logger = setup_logger("generation", "tests/generation.log")

@pytest.fixture
def model_params():
    """Fixture for model parameters"""
    return {
        'vocab_size': 50257,
        'max_length': 1024,
        'temperature': 0.7,
        'top_k': 50,
        'top_p': 0.9,
        'repetition_penalty': 1.2
    }

@pytest.fixture
def dummy_model():
    """Fixture for dummy model"""
    class DummyModel(torch.nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.vocab_size = vocab_size
        
        def forward(self, input_ids, attention_mask=None):
            batch_size, seq_len = input_ids.shape
            # Return random logits
            return torch.randn(batch_size, seq_len, self.vocab_size)
    
    return DummyModel

def test_output_shapes(model_params, dummy_model):
    """Test output shapes from generation"""
    model = dummy_model(model_params['vocab_size'])
    generator = OutputGenerator(model, **model_params)
    
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, model_params['vocab_size'], (batch_size, seq_len))
    
    output_ids, metrics = generator.generate(input_ids, max_new_tokens=5)
    
    assert output_ids.shape[0] == batch_size
    assert output_ids.shape[1] == seq_len + 5
    assert all(key in metrics for key in ['avg_entropy', 'total_tokens', 'steps'])
    logger.debug(f"Generation metrics: {metrics}")

def test_filtering_methods(model_params, dummy_model):
    """Test top-k and top-p filtering"""
    model = dummy_model(model_params['vocab_size'])
    generator = OutputGenerator(model, **model_params)
    
    batch_size = 2
    logits = torch.randn(batch_size, model_params['vocab_size'])
    
    # Test top-k
    top_k = 10
    filtered_k = generator._top_k_filtering(logits.clone(), top_k=top_k)
    for i in range(batch_size):
        num_finite = torch.isfinite(filtered_k[i]).sum()
        assert num_finite == top_k, f"Expected {top_k} finite values, got {num_finite}"
    
    # Test top-p
    top_p = 0.9
    filtered_p = generator._top_p_filtering_cumsum(logits.clone(), top_p)
    probs = torch.softmax(filtered_p, dim=-1)
    
    for i in range(batch_size):
        valid_probs = probs[i][torch.isfinite(filtered_p[i])]
        sorted_probs, _ = torch.sort(valid_probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        
        # Debug information
        logger.debug(f"Batch {i}:")
        logger.debug(f"Number of tokens kept: {len(valid_probs)}")
        logger.debug(f"Top-5 probabilities: {sorted_probs[:5]}")
        logger.debug(f"Cumulative sum: {cumsum[-1]:.6f}")
        
        # Check that we keep enough tokens
        assert len(valid_probs) >= 1, "Must keep at least one token"
        
        # Check that removing any token would make sum < top_p
        if len(cumsum) > 1:
            assert cumsum[-2] < top_p, "Could remove more tokens while maintaining top_p"

def test_repetition_penalty(model_params, dummy_model):
    """Test repetition penalty"""
    model = dummy_model(model_params['vocab_size'])
    generator = OutputGenerator(model, **model_params)
    
    logits = torch.ones(2, model_params['vocab_size'])
    generated_tokens = [0, 1, 2]  # Some previously generated tokens
    
    penalized_logits = generator._apply_repetition_penalty(logits.clone(), generated_tokens)
    
    # Check that specified tokens are penalized
    for token in generated_tokens:
        assert torch.all(penalized_logits[:, token] < logits[:, token])
    
    logger.debug("Repetition penalty verified")

def test_generation_parameters(model_params, dummy_model):
    """Test different generation parameters"""
    model = dummy_model(model_params['vocab_size'])
    generator = OutputGenerator(model, **model_params)
    
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, model_params['vocab_size'], (batch_size, seq_len))
    
    # Test different temperatures
    for temp in [0.5, 1.0, 1.5]:
        output_ids, metrics = generator.generate(
            input_ids, 
            max_new_tokens=5,
            temperature=temp
        )
        assert output_ids.shape[1] == seq_len + 5
        logger.debug(f"Temperature {temp} metrics: {metrics}")
    
    # Test different top-k values
    for k in [10, 50, 100]:
        output_ids, metrics = generator.generate(
            input_ids,
            max_new_tokens=5,
            top_k=k
        )
        assert output_ids.shape[1] == seq_len + 5
        logger.debug(f"Top-k {k} metrics: {metrics}")

def test_attention_mask_handling(model_params, dummy_model):
    """Test handling of attention masks"""
    model = dummy_model(model_params['vocab_size'])
    generator = OutputGenerator(model, **model_params)
    
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, model_params['vocab_size'], (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[:, -2:] = 0  # Mask last two tokens
    
    output_ids, metrics = generator.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=5
    )
    
    assert output_ids.shape[1] == seq_len + 5
    logger.debug(f"Masked generation metrics: {metrics}") 