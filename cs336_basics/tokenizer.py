import os
import regex as re
from typing import BinaryIO, Dict, List, Tuple, Iterable
from collections.abc import Iterator
from cs336_basics.utils.constants import PAT_STR_GPT2, ENDOFTEXT, SPECIAL_TOKENS
from cs336_basics.train_bpe import _add_bytes_in_tuple, read_and_convert_bpe_vocab_and_merges

class Tokenizer:
    def __init__(self, 
                 vocab: Dict[int, bytes], 
                 merges: List[Tuple[bytes]], 
                 special_tokens: List[str] = None):
        self.vocab: Dict[int, bytes] = vocab
        self.inverse_vocab: Dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.merges: Dict[Tuple[bytes], int] = {pair: idx for idx, pair in enumerate(merges)}
        self.special_tokens: Set[bytes] = set([])
        self.spetial_tokens_pattern: str = None
        if special_tokens is not None:
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            for special_token_str in special_tokens:
                special_token_bytes = special_token_str.encode("utf-8")
                if special_token_bytes not in self.inverse_vocab:
                    special_token_id = len(self.vocab)
                    self.vocab[special_token_id] = special_token_bytes
                    self.inverse_vocab[special_token_bytes] = special_token_id
                self.special_tokens.add(special_token_bytes)
            self.spetial_tokens_pattern = "|".join(re.escape(token) for token in special_tokens)

    @classmethod
    def from_files(cls,
                   vocab_filepath: str,
                   merges_filepath: str,
                   special_tokens: List[str] = None):
        vocab, merges = read_and_convert_bpe_vocab_and_merges(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    def _collapse_pretoken(self,
                           bytes_list: List[bytes],
                           earliest_pair: Tuple[bytes]) -> List[bytes]:
        for idx in range(len(bytes_list) - 1):
            if (bytes_list[idx], bytes_list[idx + 1]) == earliest_pair:
                return(bytes_list[:idx] + [earliest_pair[0] + earliest_pair[1]] + bytes_list[(idx + 2):])
        raise ValueError

    def _encode_pretoken(self,
                         pretoken: str) -> List[int]:
        pretoken_bytes: bytes = pretoken.encode("utf-8")
        bytes_list: List[bytes] = [bytes([b]) for b in pretoken_bytes]
        while len(bytes_list) > 1:
            adjacent_pairs = zip(bytes_list[:-1], bytes_list[1:])
            earliest_pair = min(adjacent_pairs, key=lambda x: self.merges.get(x, float("inf")))
            if earliest_pair not in self.merges:
                break
            else:
                bytes_list = self._collapse_pretoken(bytes_list, earliest_pair)
        return [self.inverse_vocab[b] for b in bytes_list]


    def _encode_chunk(self, 
                      chunk: str) -> List[int]:
        token_id_list_in_chunk: List[int] = []
        matches = re.finditer(PAT_STR_GPT2, chunk)
        for match in matches:
            token_id_list_in_chunk.extend(self._encode_pretoken(match.group(0)))
        return(token_id_list_in_chunk)
        

    def encode(self,
               text: str) -> List[int]:
        token_id_list: List[int] = []
        if self.special_tokens:
            chunks = re.split(f"({self.spetial_tokens_pattern})", text)
            for idx, chunk in enumerate(chunks):
                if idx % 2 == 1:
                    token_id_list.append(self.inverse_vocab[chunk.encode("utf-8")])
                elif chunk:
                    token_id_list.extend(self._encode_chunk(chunk))
        else:
            token_id_list.extend(self._encode_chunk(text))
        return(token_id_list)


    def encode_iterable(self,
                        iterable: Iterable[str]) -> Iterator[int]:
        for text_chunk in iterable:
            yield from self.encode(text_chunk)

    def read_file_in_chunks(self,
                            file_handle: Iterable[str],
                            split_special_token: str,
                            chunk_size: int = 1024 * 128) -> Iterator[str]:
        buffer: str = ""
        while True:
            chunk = file_handle.read(chunk_size)
            if not chunk:
                if buffer:
                    yield buffer
                break
            buffer += chunk 

            idx_last_special_token = buffer.rfind(split_special_token)
            if idx_last_special_token != -1:
                idx_split = idx_last_special_token + len(split_special_token)
                chunk_to_yield = buffer[:idx_split]
                buffer = buffer[idx_split:]
                yield chunk_to_yield

    def decode(self,
               ids: List[int]) -> str:
        decoded_bytes: bytes = b"".join(self.vocab.get(id, b"") for id in ids)
        return decoded_bytes.decode("utf-8", errors="replace")