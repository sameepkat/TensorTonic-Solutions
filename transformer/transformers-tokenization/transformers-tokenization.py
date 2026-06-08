import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        self.id_to_word[0] = self.pad_token
        self.id_to_word[1] = self.unk_token
        self.id_to_word[2] = self.bos_token
        self.id_to_word[3] = self.eos_token
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3

        self.vocab_size = 4

        text_list = [] # for unique
        for i in range(len(texts)):
            words = texts[i].lower().split()

            for word in words:
                if word in text_list:
                    pass
                else:
                    text_list.append(word)
                    self.vocab_size += 1
                

        text_list.sort()
        
        for j in range(len(text_list)):
            self.word_to_id[text_list[j]] = 4+j
            self.id_to_word[4+j] = text_list[j]
        
        pass
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        split_text = text.lower().split()
        
        # vocab_text = ["hello world", "this is a test", "hello test"]
        # vocab_phrase = " ".join(vocab_text)
        
        # self.build_vocab(vocab_phrase.split())
        # print(f"Vocab Size: {self.vocab_size}")

        ids = []
        for text in split_text:
            text_id = self.word_to_id.get(text)

            if text_id is None:
                ids.append(self.word_to_id[self.unk_token])
            else:
                ids.append(text_id)

        print(self.word_to_id)
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        text = []
        for id in ids:
            text.append(self.id_to_word.get(id, self.unk_token))

        return " ".join(text)
