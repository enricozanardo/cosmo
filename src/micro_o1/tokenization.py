import torch
from typing import List, Dict, Union
from transformers import PreTrainedTokenizerFast
from loguru import logger

class MicroTokenizer:
    """Tokenizer with special handling for reasoning tokens"""
    
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        
        # Special tokens
        self.special_tokens = {
            'pad_token': '[PAD]',
            'unk_token': '[UNK]',
            'bos_token': '[BOS]',
            'eos_token': '[EOS]',
            # Reasoning-specific tokens
            'reason_start': '<reason>',
            'reason_end': 'reason>',
            'step_token': '<step>',
            'therefore_token': '<therefore>'
        }
        
        # Create base tokenizer
        self.tokenizer = PreTrainedTokenizerFast(
            vocab_size=vocab_size,
            pad_token=self.special_tokens['pad_token'],
            unk_token=self.special_tokens['unk_token'],
            bos_token=self.special_tokens['bos_token'],
            eos_token=self.special_tokens['eos_token']
        )
        
        # Add reasoning tokens
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': list(self.special_tokens.values())
        })
        
        # Cache special token IDs
        self.special_token_ids = {
            name: self.tokenizer.convert_tokens_to_ids(token)
            for name, token in self.special_tokens.items()
        }
        
        logger.info("Initialized tokenizer with reasoning tokens")
    
    def encode(self, 
               text: Union[str, List[str]], 
               add_special_tokens: bool = True,
               padding: bool = True,
               truncation: bool = True,
               max_length: int = 1024) -> Dict[str, torch.Tensor]:
        """Encode text to token IDs and attention mask"""
        # Tokenize
        encoded = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def decode(self, token_ids: torch.Tensor) -> List[str]:
        """Decode token IDs back to text"""
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    
    def is_reasoning_token(self, token_id: int) -> bool:
        """Check if token ID is a reasoning token"""
        reasoning_tokens = {
            self.special_token_ids['reason_start'],
            self.special_token_ids['reason_end'],
            self.special_token_ids['step_token'],
            self.special_token_ids['therefore_token']
        }
        return token_id in reasoning_tokens 