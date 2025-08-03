import os
import regex as re
import multiprocessing
import heapq
import time
import json
from functools import lru_cache
from tqdm import tqdm
from typing import List, Dict, Tuple, BinaryIO, Union, Any
from collections import Counter
from cs336_basics.utils.constants import PAT_STR_GPT2, ENDOFTEXT, SPECIAL_TOKENS
from cs336_basics.utils.bpe_utils import (
    bytes_to_unicode, 
    convert_and_save_bpe_vocab_and_merges,
    read_and_convert_bpe_vocab_and_merges
)


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def tokenize_chunk(args: Tuple[str, int, int]) -> Counter:
    filepath, start_byte, end_byte = args
    with open(filepath, "rb") as f:
        f.seek(start_byte)
        text_chunk = f.read(end_byte - start_byte).decode("utf-8", errors="ignore")
    escaped_tokens = [re.escape(token) for token in SPECIAL_TOKENS]
    pattern_split = "|".join(escaped_tokens)
    text_chunks = re.split(pattern_split, text_chunk)
    pretokenized_chunk = Counter()
    if text_chunks:
        for chunk in text_chunks:
            matches = re.finditer(PAT_STR_GPT2, chunk)
            pretokenized_chunk.update(match.group(0) for match in matches)
    return pretokenized_chunk


def pre_tokenize(input_path: str,
                 num_processes: int) -> Dict[Tuple[bytes], int]:
    with open(input_path, "rb") as f:
        chunk_boundaries: List[int] = find_chunk_boundaries(f, num_processes, ENDOFTEXT.encode('utf-8'))
        chunks_to_process: List[Tuple[Any]] = []
        f.seek(0)
        for idx_chunk in range(len(chunk_boundaries) - 1):
            idx_chunk_start = chunk_boundaries[idx_chunk]
            idx_chunk_end = chunk_boundaries[idx_chunk + 1]
            chunks_to_process.append((input_path, idx_chunk_start, idx_chunk_end))

    with multiprocessing.Pool(processes=num_processes) as pool:
        results: List[Dict[str, int]] = pool.map(tokenize_chunk, chunks_to_process)

    pretokenization_dict: Dict[str, int] = results[0]
    for chunk_counter in results[1:]:
        pretokenization_dict.update(chunk_counter)
    pretoken_counter: Dict[Tuple[bytes], int] = {tuple(bytes([b]) for b in key.encode("utf-8")): val for key, val in pretokenization_dict.items()}
    return pretoken_counter


def is_subtuple(subtuple: Tuple[bytes],
                fulltuple: Tuple[bytes]) -> Tuple[Union[bool, int]]:
    sub_tuple_idx_list: List[int] = []
    idx: int = 0
    while idx < (len(fulltuple) - len(subtuple) + 1):
        is_sub = True
        for idx_subtuple in range(len(subtuple)):
            if subtuple[idx_subtuple] != fulltuple[idx + idx_subtuple]:
                is_sub = False
                break
        if is_sub:
            sub_tuple_idx_list.append(idx)
            idx += len(subtuple)
        else:
            idx += 1
    return sub_tuple_idx_list


def _add_bytes_in_tuple(tuple_input: Tuple[bytes]) -> bytes:
    new_bytes: bytes = tuple_input[0]
    for b in tuple_input[1:]:
        new_bytes += b
    return new_bytes
    

def _add_and_pop_for_collapse(fulltuple: Tuple[bytes],
                              sub_tuple_idx_list: List[int],
                              new_tuple: Tuple[bytes],
                              merge_size: int = 2) -> Tuple[Dict[Tuple[bytes], int]]:
    idx_pair_to_add_in_new_tuple: Set[int] = set(idx_pair_to_add
                                                 for idx_collapse, idx_fulltuple in enumerate(sub_tuple_idx_list)
                                                 for idx_pair_to_add in [idx_fulltuple - (idx_collapse * (merge_size - 1)) - 1,
                                                                         idx_fulltuple - (idx_collapse * (merge_size - 1))])
    idx_pair_to_add_in_new_tuple = set(idx_pair_to_add for idx_pair_to_add in idx_pair_to_add_in_new_tuple
                                                       if (idx_pair_to_add >=0) and (idx_pair_to_add < (len(fulltuple) - len(sub_tuple_idx_list) * (merge_size - 1) - 1)))

    idx_pair_to_pop_in_fulltuple: Set[int] = set(idx_pair_to_pop 
                                                 for idx_collapse, idx_fulltuple in enumerate(sub_tuple_idx_list)
                                                 for idx_pair_to_pop in [idx_fulltuple - 1, idx_fulltuple + merge_size - 1])
    idx_pair_to_pop_in_fulltuple = set(idx_pair_to_pop for idx_pair_to_pop in idx_pair_to_pop_in_fulltuple
                                                       if idx_pair_to_pop >= 0 and (idx_pair_to_pop < (len(fulltuple) - 1)))

    pair_to_add_dict: Dict[Tuple[bytes], int] = {}
    pair_to_pop_dict: Dict[Tuple[bytes], int] = {}

    for idx in idx_pair_to_add_in_new_tuple:
        pair_to_add_dict.setdefault(new_tuple[idx:(idx + merge_size)], 0)
        pair_to_add_dict[new_tuple[idx:(idx + merge_size)]] += 1

    for idx in idx_pair_to_pop_in_fulltuple:
        pair_to_pop_dict.setdefault(fulltuple[idx:(idx + merge_size)], 0)
        pair_to_pop_dict[fulltuple[idx:(idx + merge_size)]] += 1 
    return(pair_to_add_dict, 
           pair_to_pop_dict)


def collapse_tuple(fulltuple: Tuple[bytes], 
                   sub_tuple_idx_list: List[int],
                   collapsed_bytes: bytes,
                   merge_size: int = 2) -> Tuple[Union[Dict[Tuple[bytes], int], Tuple[bytes]]]:
    assert len(sub_tuple_idx_list) > 0 and sub_tuple_idx_list[0] >= 0 and sub_tuple_idx_list[-1] <= len(fulltuple) - merge_size
    sub_tuple_idx_list = [-merge_size] + sub_tuple_idx_list
    new_tuple: Tuple[bytes] = ()
    for idx_subtuple in range(1, len(sub_tuple_idx_list)):
        new_tuple += fulltuple[(sub_tuple_idx_list[idx_subtuple - 1] + merge_size):sub_tuple_idx_list[idx_subtuple]] + \
                     tuple([collapsed_bytes])
    new_tuple += fulltuple[(sub_tuple_idx_list[-1] + merge_size):]
    pair_to_add_dict, pair_to_pop_dict = _add_and_pop_for_collapse(fulltuple, sub_tuple_idx_list[1:], new_tuple)
    return(new_tuple, 
           pair_to_add_dict, 
           pair_to_pop_dict)


def train_bpe(input_path: str,
              vocab_size: int,
              special_tokens: List[str]) -> Tuple[Union[Dict, List]]:
    num_processes: int = os.cpu_count()
    pretoken_counter: Dict[Tuple[bytes], int] = pre_tokenize(input_path, num_processes)

    merges: List[Tuple[bytes]] = []
    vocab: Dict[int, bytes] = {key: bytes([key])
             for key in range(256)}
    for idx, token in enumerate(special_tokens):
        vocab[idx + 256] = token.encode("utf-8")

    pair_freq: Dict[Tuple[bytes], int] = {}
    for pretoken_tuple, freq in tqdm(pretoken_counter.items()):
        for idx_pair in range(len(pretoken_tuple) - 1):
            pair_tuple = pretoken_tuple[idx_pair:(idx_pair + 2)]
            pair_freq.setdefault(pair_tuple, 0)
            pair_freq[pair_tuple] += freq

    pbar = tqdm(total=vocab_size - len(vocab))
    while len(vocab) < vocab_size:
        if not pair_freq:
            break
        max_pair = max(pair_freq, key=lambda x:(pair_freq[x], x))
        new_bytes_token = _add_bytes_in_tuple(max_pair)
        pretoken_pair_to_replace = []
        for pretoken, pretoken_freq in pretoken_counter.items():
            sub_tuple_idx_list = is_subtuple(max_pair, pretoken)
            if sub_tuple_idx_list:
                pair_freq[max_pair] -= len(sub_tuple_idx_list) * pretoken_freq
                if pair_freq[max_pair] == 0:
                    pair_freq.pop(max_pair)
                new_pretoken, \
                pair_to_add_dict, \
                pair_to_pop_dict = collapse_tuple(pretoken, sub_tuple_idx_list, new_bytes_token)
                pretoken_pair_to_replace.append([pretoken, new_pretoken])
                for pair, freq in pair_to_add_dict.items():
                    pair_freq.setdefault(pair, 0)
                    pair_freq[pair] += freq * pretoken_freq
                for pair, freq in pair_to_pop_dict.items():
                    pair_freq[pair] -= freq * pretoken_freq
                    if pair_freq[pair] == 0:
                        pair_freq.pop(pair)
        for pretoken_old, pretoken_new in pretoken_pair_to_replace:
            # setdefault here? not doing it because pretoken_new is supposed to be unique
            pretoken_counter[pretoken_new] = pretoken_counter[pretoken_old]
            pretoken_counter.pop(pretoken_old)
        merges.append(max_pair)
        vocab[len(vocab)] = new_bytes_token
        pbar.update(1)
    pbar.close()
    return vocab, merges