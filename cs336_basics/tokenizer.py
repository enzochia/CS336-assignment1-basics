import os
from typing import BinaryIO, Dict, List, Tuple, Iterable
from collections.abc import Iterator

class Tokenizer:

    def __init__(self, 
                 vocab: Dict[int, bytes], 
                 merges: List[Tuple[bytes]], 
                 special_tokens: List[str] = None):

    @classmethod
    def from_files(self,
                   vocab_filepath: str,
                   merges_filepath: str,
                   special_tokens: List[str] = None):


    def encode(self,
               text: str) -> List[int]:


    def encode_iterable(self,
                        iterable: Iterable[str]) -> Iterator[int]:

    def decode(self,
               ids: List[int]) -> str:

