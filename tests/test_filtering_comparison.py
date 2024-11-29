import pytest
import torch
from src.micro_o1.generation import OutputGenerator
from src.micro_o1.config.logging import setup_logger

# Setup logger
logger = setup_logger("filtering_comparison", "tests/filtering_comparison.log")

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
            return torch.randn(batch_size, seq_len, self.vocab_size)
    
    return DummyModel

@pytest.fixture
def test_distributions():
    """Different probability distributions to test"""
    return {
        'peaked': torch.tensor([
            [0.6, 0.2, 0.1, 0.05, 0.05] + [0.0] * 45
        ]),
        'flat': torch.tensor([
            [0.2] * 5 + [0.0] * 45
        ]),
        'long_tail': torch.tensor([
            [0.3, 0.25, 0.2, 0.15, 0.1] + [0.0] * 45
        ]),
        'random': torch.randn(1, 50).softmax(dim=-1)
    }

def test_distribution_handling(model_params, dummy_model, test_distributions):
    """Compare how both methods handle different distributions"""
    generator = OutputGenerator(dummy_model(model_params['vocab_size']), **model_params)
    top_p = 0.9
    
    results = {}
    for dist_name, dist in test_distributions.items():
        # Convert to logits
        logits = torch.log(dist + 1e-10)
        
        # Test both methods
        mean_result = generator._top_p_filtering_mean(logits, top_p)
        cumsum_result = generator._top_p_filtering_cumsum(logits, top_p)
        
        # Get probabilities after filtering
        mean_probs = torch.softmax(mean_result, dim=-1)
        cumsum_probs = torch.softmax(cumsum_result, dim=-1)
        
        # Get detailed metrics
        results[dist_name] = {
            'original': {
                'top_5': dist[0, :5].tolist(),
                'entropy': -(dist * torch.log(dist + 1e-10)).sum().item()
            },
            'mean': {
                'tokens_kept': (mean_probs > 0).sum().item(),
                'top_5': mean_probs[0, :5].tolist(),
                'entropy': -(mean_probs * torch.log(mean_probs + 1e-10)).sum().item()
            },
            'cumsum': {
                'tokens_kept': (cumsum_probs > 0).sum().item(),
                'top_5': cumsum_probs[0, :5].tolist(),
                'entropy': -(cumsum_probs * torch.log(cumsum_probs + 1e-10)).sum().item()
            }
        }
        
        # Log detailed comparison
        logger.info(f"\n=== Distribution: {dist_name} ===")
        logger.info(f"Original top-5: {results[dist_name]['original']['top_5']}")
        logger.info(f"Original entropy: {results[dist_name]['original']['entropy']:.3f}")
        logger.info("\nMean method:")
        logger.info(f"Tokens kept: {results[dist_name]['mean']['tokens_kept']}")
        logger.info(f"Top-5 after: {results[dist_name]['mean']['top_5']}")
        logger.info(f"Entropy: {results[dist_name]['mean']['entropy']:.3f}")
        logger.info("\nCumsum method:")
        logger.info(f"Tokens kept: {results[dist_name]['cumsum']['tokens_kept']}")
        logger.info(f"Top-5 after: {results[dist_name]['cumsum']['top_5']}")
        logger.info(f"Entropy: {results[dist_name]['cumsum']['entropy']:.3f}")
    
    return results

def test_numerical_stability(model_params, dummy_model):
    """Test numerical stability with extreme probabilities"""
    generator = OutputGenerator(dummy_model(model_params['vocab_size']), **model_params)
    top_p = 0.9
    
    # Test cases
    test_cases = {
        'extreme_peaked': torch.tensor([[0.99] + [0.01/98] * 98 + [0.0]]),
        'uniform': torch.ones(1, 100) / 100,
        'exponential_decay': torch.exp(-torch.arange(100).float() / 10).unsqueeze(0)
    }
    
    for case_name, dist in test_cases.items():
        dist = dist / dist.sum()  # Normalize
        logits = torch.log(dist + 1e-10)
        
        mean_result = generator._top_p_filtering_mean(logits, top_p)
        cumsum_result = generator._top_p_filtering_cumsum(logits, top_p)
        
        mean_probs = torch.softmax(mean_result, dim=-1)
        cumsum_probs = torch.softmax(cumsum_result, dim=-1)
        
        logger.info(f"\n=== {case_name} ===")
        logger.info(f"Mean method tokens: {(mean_probs > 0).sum().item()}")
        logger.info(f"Mean method max prob: {mean_probs.max().item():.3f}")
        logger.info(f"Cumsum method tokens: {(cumsum_probs > 0).sum().item()}")
        logger.info(f"Cumsum method max prob: {cumsum_probs.max().item():.3f}")

def test_diversity(model_params, dummy_model):
    """Test output diversity"""
    generator = OutputGenerator(dummy_model(model_params['vocab_size']), **model_params)
    top_p = 0.9
    
    # Generate multiple samples
    n_samples = 1000  # Increased for better statistics
    logits = torch.randn(1, model_params['vocab_size'])
    
    mean_samples = []
    cumsum_samples = []
    
    for _ in range(n_samples):
        mean_filtered = generator._top_p_filtering_mean(logits.clone(), top_p)
        cumsum_filtered = generator._top_p_filtering_cumsum(logits.clone(), top_p)
        
        mean_probs = torch.softmax(mean_filtered, dim=-1)
        cumsum_probs = torch.softmax(cumsum_filtered, dim=-1)
        
        mean_samples.append(torch.multinomial(mean_probs, 1).item())
        cumsum_samples.append(torch.multinomial(cumsum_probs, 1).item())
    
    # Analyze diversity
    mean_unique = len(set(mean_samples))
    cumsum_unique = len(set(cumsum_samples))
    
    logger.info("\n=== Diversity Analysis ===")
    logger.info(f"Mean method unique tokens: {mean_unique}/{n_samples} ({mean_unique/n_samples:.3f})")
    logger.info(f"Cumsum method unique tokens: {cumsum_unique}/{n_samples} ({cumsum_unique/n_samples:.3f})")
    
    # Analyze distribution
    from collections import Counter
    mean_counter = Counter(mean_samples)
    cumsum_counter = Counter(cumsum_samples)
    
    logger.info("\nTop 5 most common tokens:")
    logger.info("Mean method:")
    for token, count in mean_counter.most_common(5):
        logger.info(f"Token {token}: {count/n_samples:.3f}")
    logger.info("\nCumsum method:")
    for token, count in cumsum_counter.most_common(5):
        logger.info(f"Token {token}: {count/n_samples:.3f}") 