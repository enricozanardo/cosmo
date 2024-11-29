import pytest
import torch
from src.micro_o1.config.logging import setup_logger
from src.micro_o1.main import MicroO1Embeddings, MicroO1Tokenizer

# Setup logger
logger = setup_logger("embeddings", "tests/embeddings.log")

@pytest.fixture(autouse=True)
def log_test(request):
    """Log test execution"""
    logger.info(f"Starting test: {request.node.name}")
    yield
    logger.info(f"Completed test: {request.node.name}")

@pytest.fixture
def embedding_params():
    """Fixture for embedding parameters"""
    return {
        'vocab_size': 50257,  # GPT-2 vocab size
        'hidden_size': 768,
        'max_position_embeddings': 1024,
        'dropout': 0.1
    }

@pytest.fixture
def embeddings(embedding_params):
    """Fixture for embeddings layer"""
    logger.debug("Creating embeddings layer")
    return MicroO1Embeddings(**embedding_params)

@pytest.fixture
def tokenizer():
    """Fixture for tokenizer"""
    return MicroO1Tokenizer()

def test_embeddings_initialization(embedding_params):
    """Test embeddings layer initialization"""
    logger.info("Testing embeddings initialization")
    embeddings = MicroO1Embeddings(**embedding_params)
    
    assert embeddings.word_embeddings.num_embeddings == embedding_params['vocab_size']
    assert embeddings.word_embeddings.embedding_dim == embedding_params['hidden_size']
    assert embeddings.position_embeddings.num_embeddings == embedding_params['max_position_embeddings']
    
    logger.debug("Embeddings initialization test passed")

def test_embeddings_forward(embeddings, tokenizer):
    """Test forward pass of embeddings layer"""
    logger.info("Testing embeddings forward pass")
    
    # Prepare input
    text = "Hello world!"
    encoded = tokenizer.encode(text)
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    
    logger.debug(f"Input shape: {input_ids.shape}")
    
    # Forward pass
    outputs = embeddings(input_ids, attention_mask)
    
    # Check output shape
    batch_size, seq_length = input_ids.shape
    assert outputs.shape == (batch_size, seq_length, embeddings.word_embeddings.embedding_dim)
    logger.debug(f"Output shape: {outputs.shape}")
    
    # Check if outputs are finite
    assert torch.isfinite(outputs).all()
    logger.debug("All output values are finite")

def test_embeddings_with_attention_mask(embeddings, tokenizer):
    """Test embeddings with attention mask"""
    logger.info("Testing embeddings with attention mask")
    
    # Create batch with different lengths
    texts = ["Hello world!", "This is a longer sequence that needs padding"]
    encoded = tokenizer.encode(texts)
    
    # Log the token IDs and vocab size for debugging
    logger.debug(f"Input IDs shape: {encoded['input_ids'].shape}")
    logger.debug(f"Input IDs: {encoded['input_ids']}")
    logger.debug(f"Vocab size: {tokenizer.vocab_size}")
    
    # Ensure token IDs are within vocab size
    if (encoded['input_ids'] >= tokenizer.vocab_size).any():
        logger.warning("Found token IDs >= vocab_size, clipping to vocab_size-1")
        encoded['input_ids'] = torch.clamp(encoded['input_ids'], max=tokenizer.vocab_size-1)
    
    outputs = embeddings(encoded['input_ids'], encoded['attention_mask'])
    
    # Check if masked positions have zero embeddings
    # Expand attention mask to match embedding dimensions
    attention_mask = encoded['attention_mask'].unsqueeze(-1).expand(-1, -1, embeddings.word_embeddings.embedding_dim)
    logger.debug(f"Attention mask shape after expansion: {attention_mask.shape}")
    logger.debug(f"Outputs shape: {outputs.shape}")
    
    # Check zeros where attention mask is 0
    masked_outputs = outputs * (1 - attention_mask)
    assert torch.all(masked_outputs == 0)
    logger.debug("Masked positions correctly zeroed")

def test_embeddings_gradient_flow(embeddings, tokenizer):
    """Test gradient flow through embeddings"""
    logger.info("Testing gradient flow")
    
    text = "Testing gradient flow"
    encoded = tokenizer.encode(text)
    
    # Enable gradient tracking
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    
    # Forward pass
    outputs = embeddings(input_ids, attention_mask)
    
    # Simulate gradient flow
    loss = outputs.sum()
    loss.backward()
    
    # Check if gradients exist
    assert embeddings.word_embeddings.weight.grad is not None
    assert embeddings.position_embeddings.weight.grad is not None
    logger.debug("Gradients properly computed")

@pytest.mark.parametrize("batch_size,seq_length", [
    (1, 10),
    (4, 64),
    (8, 128),
    (16, 256)
])
def test_embeddings_different_sizes(embeddings, batch_size, seq_length):
    """Test embeddings with different batch sizes and sequence lengths"""
    logger.info(f"Testing with batch_size={batch_size}, seq_length={seq_length}")
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    outputs = embeddings(input_ids, attention_mask)
    assert outputs.shape == (batch_size, seq_length, embeddings.word_embeddings.embedding_dim)
    logger.debug(f"Successfully processed batch of shape {outputs.shape}") 