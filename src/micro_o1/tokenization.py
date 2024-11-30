import torch
from typing import List, Dict, Union
from transformers import GPT2TokenizerFast
from loguru import logger

class MicroTokenizer:
    """Tokenizer with special handling for reasoning tokens"""
    
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        
        # Special tokens with consistent naming
        self.special_tokens = {
            'pad_token': '[PAD]',
            'unk_token': '[UNK]',
            'bos_token': '[BOS]',
            'eos_token': '[EOS]',
            # Reasoning-specific tokens
            'reasoning_start': '<reason>',
            'reasoning_end': 'reason>',
            'step_token': '<step>',
            'therefore_token': '<therefore>',
            'reason_start': '<|reason|>',
            'reason_end': 'reason|>',
            'answer_start': '<|answer|>',
            'answer_end': 'answer|>'
        }
        
        # Initialize GPT2 tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        
        # Add special tokens
        special_tokens_dict = {
            'pad_token': self.special_tokens['pad_token'],
            'additional_special_tokens': [
                self.special_tokens['reasoning_start'],
                self.special_tokens['reasoning_end'],
                self.special_tokens['step_token'],
                self.special_tokens['therefore_token'],
                self.special_tokens['reason_start'],
                self.special_tokens['reason_end'],
                self.special_tokens['answer_start'],
                self.special_tokens['answer_end']
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)
        
        # Add special token attributes for easier access
        for name, token in self.special_tokens.items():
            setattr(self, name, token)
        
        # Also add direct token access
        self.reasoning_start = self.special_tokens['reasoning_start']
        self.reasoning_end = self.special_tokens['reasoning_end']
        
        logger.info("Initialized tokenizer with reasoning tokens")
        logger.debug(f"Special tokens: {self.special_tokens}")
    
    def encode(self, 
               text: Union[str, List[str]], 
               add_special_tokens: bool = True,
               padding: bool = True,
               truncation: bool = True,
               max_length: int = 1024) -> Dict[str, torch.Tensor]:
        """Encode text to token IDs and attention mask"""
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
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size including special tokens"""
        return len(self.tokenizer)