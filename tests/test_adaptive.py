import pytest
import torch
from src.micro_o1.adaptive import AdaptiveController, AdaptiveTransformerLayer, AdaptiveInference
from src.micro_o1.config.logging import setup_logger

# Setup logger
logger = setup_logger("adaptive", "tests/adaptive.log")

@pytest.fixture
def model_params():
    """Fixture for model parameters"""
    return {
        'hidden_size': 768,
        'num_layers': 12,
        'threshold': 0.5
    }

@pytest.fixture
def dummy_transformer_layer():
    """Fixture for dummy transformer layer"""
    return torch.nn.Sequential(
        torch.nn.Linear(768, 768),
        torch.nn.GELU(),
        torch.nn.Linear(768, 768)
    )

def test_controller_output_shapes(model_params):
    """Test controller output shapes"""
    controller = AdaptiveController(**model_params)
    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, model_params['hidden_size'])
    
    halt_prob, importance_scores, metrics = controller(hidden_states)
    
    assert halt_prob.shape == (batch_size, seq_len, 1)
    assert importance_scores.shape == (batch_size, seq_len, model_params['num_layers'])
    assert 0 <= metrics['halt_prob'] <= 1
    logger.debug(f"Controller metrics: {metrics}")

def test_adaptive_layer(model_params, dummy_transformer_layer):
    """Test adaptive transformer layer"""
    layer = AdaptiveTransformerLayer(dummy_transformer_layer, layer_idx=0)
    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, model_params['hidden_size'])
    importance_score = torch.rand(batch_size, seq_len, 1)
    
    output = layer(hidden_states, importance_score)
    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()
    logger.debug(f"Layer output shape: {output.shape}")

def test_adaptive_inference(model_params, dummy_transformer_layer):
    """Test adaptive inference module"""
    num_layers = 4
    transformer_layers = torch.nn.ModuleList([
        dummy_transformer_layer for _ in range(num_layers)
    ])
    
    model = AdaptiveInference(
        transformer_layers=transformer_layers,
        hidden_size=model_params['hidden_size']
    )
    
    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, model_params['hidden_size'])
    
    output, metrics = model(hidden_states)
    assert output.shape == hidden_states.shape
    assert metrics['num_layers_used'] <= num_layers
    logger.debug(f"Adaptive inference metrics: {metrics}")

def test_early_exit(model_params, dummy_transformer_layer):
    """Test early exit mechanism"""
    num_layers = 4
    transformer_layers = torch.nn.ModuleList([
        dummy_transformer_layer for _ in range(num_layers)
    ])
    
    model = AdaptiveInference(
        transformer_layers=transformer_layers,
        hidden_size=model_params['hidden_size'],
        early_exit_threshold=0.5
    )
    model.eval()
    
    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, model_params['hidden_size'])
    
    with torch.no_grad():
        output, metrics = model(hidden_states)
    
    assert metrics['early_exit'] or metrics['num_layers_used'] == num_layers
    logger.debug(f"Early exit metrics: {metrics}")

def test_computation_tracking(model_params, dummy_transformer_layer):
    """Test computation budget tracking"""
    num_layers = 4
    transformer_layers = torch.nn.ModuleList([
        dummy_transformer_layer for _ in range(num_layers)
    ])
    
    model = AdaptiveInference(
        transformer_layers=transformer_layers,
        hidden_size=model_params['hidden_size']
    )
    
    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, model_params['hidden_size'])
    
    output, metrics = model(hidden_states)
    assert 0 <= metrics['total_computation'] <= num_layers
    logger.debug(f"Computation metrics: {metrics}") 