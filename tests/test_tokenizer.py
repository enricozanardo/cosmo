import pytest
import torch
from src.micro_o1.config.logging import setup_logger
from src.micro_o1.main import MicroO1Tokenizer

# Setup logger
logger = setup_logger("tokenizer", "tests/tokenizer.log")

@pytest.fixture(autouse=True)
def log_test(request):
    """Fixture to log the start and end of each test"""
    test_name = request.node.name
    logger.info(f"Starting test: {test_name}")
    yield
    logger.info(f"Completed test: {test_name}")

@pytest.fixture
def tokenizer():
    """Fixture to create tokenizer instance for tests"""
    logger.debug("Creating tokenizer instance")
    tokenizer = MicroO1Tokenizer()
    logger.debug(f"Tokenizer created with vocab size: {tokenizer.vocab_size}")
    return tokenizer

@pytest.fixture
def test_texts():
    """Fixture providing test text samples"""
    logger.debug("Creating test text samples")
    return {
        'basic': "Solve this math problem: If John has 5 apples and gives 2 to Mary, how many does he have left?",
        'special': "[REASON] Let's solve this step by step [SOLUTION] John has 3 apples left",
        'short': "Short text",
        'long': "This is a longer text that should require more tokens",
        'very_long': "word " * 1000
    }

def test_initialization(tokenizer):
    """Test tokenizer initialization and special tokens"""
    logger.info("Testing tokenizer initialization")
    
    assert tokenizer.tokenizer.pad_token is not None
    logger.debug(f"Pad token: {tokenizer.tokenizer.pad_token}")
    
    assert tokenizer.tokenizer.pad_token_id is not None
    logger.debug(f"Pad token ID: {tokenizer.tokenizer.pad_token_id}")
    
    vocab = tokenizer.tokenizer.get_vocab()
    special_tokens = ['[PAD]', '[REASON]', '[SOLUTION]', '[SEP]']
    for token in special_tokens:
        assert token in vocab
        logger.debug(f"Special token '{token}' found in vocabulary")

def test_encode_single(tokenizer, test_texts):
    """Test encoding single text"""
    logger.info("Testing single text encoding")
    
    text = test_texts['basic']
    logger.debug(f"Input text: {text}")
    
    encoded = tokenizer.encode(text)
    logger.debug(f"Encoded shapes - input_ids: {encoded['input_ids'].shape}, "
                f"attention_mask: {encoded['attention_mask'].shape}")
    
    assert 'input_ids' in encoded
    assert 'attention_mask' in encoded
    assert isinstance(encoded['input_ids'], torch.Tensor)
    assert isinstance(encoded['attention_mask'], torch.Tensor)
    assert encoded['input_ids'].dim() == 2
    assert encoded['attention_mask'].dim() == 2

def test_encode_batch(tokenizer, test_texts):
    """Test encoding batch of texts"""
    logger.info("Testing batch text encoding")
    
    batch_texts = [test_texts['basic'], test_texts['special']]
    logger.debug(f"Batch size: {len(batch_texts)}")
    
    encoded = tokenizer.encode(batch_texts)
    logger.debug(f"Encoded batch shape: {encoded['input_ids'].shape}")
    
    assert encoded['input_ids'].shape[0] == 2
    assert encoded['attention_mask'].shape[0] == 2
    assert encoded['input_ids'].shape == encoded['attention_mask'].shape

def test_decode(tokenizer, test_texts):
    """Test decoding tokens back to text"""
    logger.info("Testing token decoding")
    
    text = test_texts['basic']
    logger.debug(f"Original text: {text}")
    
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded['input_ids'][0])
    logger.debug(f"Decoded text: {decoded}")
    
    assert "5 apples" in decoded
    assert "Mary" in decoded

def test_special_tokens(tokenizer, test_texts):
    """Test handling of special tokens"""
    logger.info("Testing special tokens handling")
    
    text = test_texts['special']
    logger.debug(f"Input text with special tokens: {text}")
    
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded['input_ids'][0])
    logger.debug(f"Decoded text: {decoded}")
    
    assert "solve this step by step" in decoded.lower()
    assert "3 apples left" in decoded.lower()

def test_max_length(tokenizer, test_texts):
    """Test max length handling"""
    logger.info("Testing max length handling")
    
    encoded = tokenizer.encode(test_texts['very_long'])
    logger.debug(f"Encoded length: {encoded['input_ids'].shape[1]}, Max length: {tokenizer.max_length}")
    
    assert encoded['input_ids'].shape[1] <= tokenizer.max_length
    assert encoded['attention_mask'].shape[1] <= tokenizer.max_length

def test_vocab_size(tokenizer):
    """Test vocab_size property"""
    logger.info("Testing vocabulary size")
    
    vocab_size = tokenizer.vocab_size
    logger.debug(f"Vocabulary size: {vocab_size}")
    
    assert isinstance(vocab_size, int)
    assert vocab_size > 0

def test_padding(tokenizer, test_texts):
    """Test padding functionality"""
    logger.info("Testing padding functionality")
    
    encoded = tokenizer.encode([test_texts['short'], test_texts['long']])
    logger.debug(f"Encoded shapes: {encoded['input_ids'].shape}")
    
    assert encoded['input_ids'][0].shape == encoded['input_ids'][1].shape
    
    non_pad_tokens = (encoded['input_ids'][0] != tokenizer.tokenizer.pad_token_id).sum()
    attention_sum = encoded['attention_mask'].sum(1)[0]
    logger.debug(f"Non-pad tokens: {non_pad_tokens}, Attention sum: {attention_sum}")
    
    assert attention_sum == non_pad_tokens

@pytest.mark.parametrize("text", [
    "",  # empty string
    "Hello",  # single word
    "Hello, world!",  # multiple words
    "A" * 2000,  # very long text
])
def test_various_inputs(tokenizer, text):
    """Test tokenizer with various input types"""
    logger.info(f"Testing input: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    encoded = tokenizer.encode(text)
    logger.debug(f"Encoded shape: {encoded['input_ids'].shape}")
    
    decoded = tokenizer.decode(encoded['input_ids'][0])
    logger.debug(f"Decoded text: {decoded[:50]}{'...' if len(decoded) > 50 else ''}")
    
    if text:  # skip empty string
        assert any(word in decoded for word in text.split() if len(word) > 3)