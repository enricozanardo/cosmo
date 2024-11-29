import pytest
import torch
import torch.nn as nn
from src.micro_o1.optimizations import HardwareOptimizer
from src.micro_o1.config.logging import setup_logger

# Setup logger
logger = setup_logger("optimizations", "tests/optimizations.log")

@pytest.fixture
def optimizer():
    """Fixture for hardware optimizer"""
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    return HardwareOptimizer(device_type=device_type)

@pytest.fixture
def simple_model():
    """Fixture for a simple test model"""
    return nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 100)
    )

def test_cpu_optimizations(optimizer, simple_model):
    """Test CPU-specific optimizations"""
    if optimizer.device_type == 'cpu':
        optimized_model = optimizer.optimize_model(simple_model)
        
        # Check thread settings
        assert torch.get_num_threads() > 0
        logger.debug(f"Using {torch.get_num_threads()} threads")

def test_gpu_optimizations(optimizer, simple_model):
    """Test GPU-specific optimizations"""
    if optimizer.device_type == 'cuda':
        optimized_model = optimizer.optimize_model(simple_model)
        
        # Check device placement
        assert next(optimized_model.parameters()).is_cuda
        
        # Check CUDA streams
        assert optimizer.main_stream is not None
        assert optimizer.memory_stream is not None
        
        logger.debug("GPU optimizations verified")
    elif optimizer.device_type == 'mps':
        optimized_model = optimizer.optimize_model(simple_model)
        assert next(optimized_model.parameters()).device.type == 'mps'
        logger.debug("MPS optimizations verified")

def test_batch_size_calculation(optimizer, simple_model):
    """Test optimal batch size calculation"""
    input_shape = (100,)
    batch_size = optimizer.get_optimal_batch_size(simple_model, input_shape)
    
    assert batch_size > 0
    assert isinstance(batch_size, int)
    logger.debug(f"Calculated optimal batch size: {batch_size}")

def test_dataloader_optimization(optimizer):
    """Test dataloader optimizations"""
    # Create dummy dataset
    dataset = torch.utils.data.TensorDataset(
        torch.randn(100, 10),
        torch.randn(100, 1)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    
    optimized_loader = optimizer.optimize_dataloader(dataloader)
    
    # Check basic properties
    assert hasattr(optimized_loader, 'batch_size')
    assert optimized_loader.batch_size == 32
    
    # Check worker settings
    if optimizer.device_type == 'cpu':
        assert optimized_loader.num_workers > 0
    elif optimizer.device_type in ['cuda', 'mps']:
        assert optimized_loader.num_workers == 2
        assert optimized_loader.pin_memory
    
    logger.debug(f"Using {optimized_loader.num_workers} workers")

def test_context_manager(optimizer):
    """Test context manager functionality"""
    with optimizer as opt:
        if optimizer.device_type == 'cuda':
            assert torch.cuda.current_stream() == opt.main_stream
        
        # Do some dummy computation
        x = torch.randn(100, 100)
        y = x @ x.t()
        
    # Context exit should handle cleanup
    if optimizer.device_type == 'cuda':
        assert torch.cuda.current_stream() != optimizer.main_stream

@pytest.mark.parametrize("batch_size", [1, 16, 32])
def test_memory_estimation(optimizer, simple_model, batch_size):
    """Test memory estimation with different batch sizes"""
    input_shape = (100,)
    mem_estimate = optimizer.estimate_memory_per_sample(simple_model, input_shape)
    
    # Basic sanity checks
    assert mem_estimate > 0
    assert isinstance(mem_estimate, int)
    
    # Check if estimate scales with model size
    total_params = sum(p.numel() * p.element_size() for p in simple_model.parameters())
    assert mem_estimate >= total_params, "Estimate should be at least as large as model parameters"
    
    logger.debug(f"Estimated memory per sample: {mem_estimate} bytes") 