#!/usr/bin/env python3
"""
Medical Tokenizer Wrapper Class
Provides loading, encoding, decoding, special token management, and batch conversion
for the trained Medical BPE Tokenizer.
"""

import os
from typing import List, Union
from tokenizers import Tokenizer

class MedicalTokenizer:
    """
    Wrapper for the medical domain BPE tokenizer.
    """
    def __init__(self, tokenizer_path: str = "tokenizer/artifacts/tokenizer.json"):
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer artifact not found at {tokenizer_path}. Run tokenizer/train_tokenizer.py first."
            )
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Cache special token IDs
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.qa_token = "<med_qa>"
        self.eot_token = "<|endoftext|>"
        
        self.pad_id = self.tokenizer.token_to_id(self.pad_token)
        self.unk_id = self.tokenizer.token_to_id(self.unk_token)
        self.bos_id = self.tokenizer.token_to_id(self.bos_token)
        self.eos_id = self.tokenizer.token_to_id(self.eos_token)
        self.qa_id = self.tokenizer.token_to_id(self.qa_token)
        self.eot_id = self.tokenizer.token_to_id(self.eot_token)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encodes string into token IDs.
        """
        encoded = self.tokenizer.encode(text)
        tokens = encoded.ids
        if add_special_tokens:
            if self.bos_id is not None and self.eos_id is not None:
                tokens = [self.bos_id] + tokens + [self.eos_id]
        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """
        Decodes token IDs back into string.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> str:
        return self.tokenizer.id_to_token(token_id)

    @classmethod
    def load(cls, path: str = "tokenizer/artifacts/tokenizer.json"):
        return cls(tokenizer_path=path)
