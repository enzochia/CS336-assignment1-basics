import os
import regex as re
import multiprocessing
import heapq
import tqdm
from typing import List, Dict, Tuple, BinaryIO, Union
from collections import Counter

PAT_STR_GPT2 = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
ENDOFTEXT = "<|endoftext|>"
SPECIAL_TOKENS = ["<|endoftext|>"]
# SPECIAL_TOKENS = ["<|endoftext|>".encode('utf-8')]
CORPUS_FILE = "data/TinyStoriesV2-GPT4-valid.txt"

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

def tokenize_chunk(text_chunk):
    escaped_tokens = [re.escape(token) for token in SPECIAL_TOKENS]
    pattern_split = "|".join(escaped_tokens)
    text_chunk = "".join(re.split(pattern_split, text_chunk))
    return Counter(re.findall(PAT_STR_GPT2, text_chunk))

# # a version using bytes rather than string
# def tokenize_chunk(text_chunk):
#     escaped_tokens = [re.escape(token) for token in SPECIAL_TOKENS]
#     pattern_split = "|".encode('utf-8').join(escaped_tokens)
#     text_chunk = "".encode('utf-8').join(re.split(pattern_split, text_chunk))
#     return Counter(re.findall(PAT_STR_GPT2.encode('utf-8'), text_chunk))


def pre_tokenize(input_path: str,
                 num_processes: int) -> Dict[Tuple[bytes], int]:
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(f, num_processes, ENDOFTEXT.encode('utf-8'))
        chunks = []
        f.seek(0)
        for idx_chunk in range(len(chunk_boundaries) - 1):
            idx_chunk_start = chunk_boundaries[idx_chunk]
            idx_chunk_end = chunk_boundaries[idx_chunk + 1]
            chunks.append(f.read(idx_chunk_end - idx_chunk_start).decode("utf-8", errors="ignore"))
            # # corresponds to the bytes version of tokenize_chunk method
            # chunks.append(f.read(idx_chunk_end - idx_chunk_start))


    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(tokenize_chunk, chunks)

    pretokenization_dict = results[0]
    for chunk_counter in results[1:]:
        pretokenization_dict.update(chunk_counter)
    pretoken_counter = {tuple(b for b in key.encode("utf-8")): val for key, val in pretokenization_dict.items()}
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
    return(sub_tuple_idx_list)


def _add_bytes_in_tuple(tuple_input: Tuple[bytes]) -> bytes:
    new_bytes: bytes = tuple_input[0]
    for b in tuple_input[1:]:
        new_bytes += b
    return(new_bytes)
    

def collapse_tuple(fulltuple: Tuple[bytes], 
                   sub_tuple_idx_list: List[int],
                   sub_tuple_len: int = 2) -> Tuple[bytes]:
    new_tuple = fulltuple[:sub_tuple_idx_list[0]]
    for idx_subtuple in range(len(sub_tuple_idx_list)):
        new_bytes = _add_bytes_in_tuple(fulltuple[sub_tuple_idx_list[idx_subtuple]:(sub_tuple_idx_list[idx_subtuple] + 2)])
        new_tuple += tuple([new_bytes])
        if idx_subtuple < (len(sub_tuple_idx_list) - 1):
            new_tuple += fulltuple[(sub_tuple_idx_list[idx_subtuple] + 2):sub_tuple_idx_list[idx_subtuple + 1]]
        else:
            new_tuple += fulltuple[(sub_tuple_idx_list[idx_subtuple] + 2):]
    return(new_tuple)


def train_bpe(input_path: str,
              vocab_size: int,
              special_tokens: List[str]) -> Tuple[Union[Dict, List]]:
    num_processes = os.cpu_count()
    pretoken_counter = pre_tokenize(input_path, num_processes)

    print("Most common pre-tokens:")
    print([(bytes(x).decode("utf-8"), pretoken_counter[x]) for x in heapq.nlargest(10, pretoken_counter, key=lambda x: pretoken_counter[x])])
    print([(x, pretoken_counter[x]) for x in heapq.nlargest(10, pretoken_counter, key=lambda x: pretoken_counter[x])])
    print("Longest pre-tokens:")
    print([bytes(x).decode("utf-8") for x in heapq.nlargest(10, pretoken_counter, key=lambda x: len(x))])

    # merges = []
    # vocab = {key: bytes([key])
    #          for key in range(256)}
    # for idx, token in enumerate(special_tokens):
    #     vocab[idx + 256] = token.encode("utf-8")
    
    # pair_freq = {}
    # for pretoken_tuple, freq in tqdm(pretoken_counter.items()):
    #     for idx_pair in range(len(pretoken_tuple) - 1):
    #         pair_tuple = pretoken_tuple[idx_pair:(idx_pair + 2)]
    #         pair_freq.setdefault(pair_tuple, 0)
    #         pair_freq[pair_tuple] += freq

    # counter_iter = 0
    # while max(counter_iter, len(vocab)) < vocab_size:
    #     counter_iter += 1
    #     max_pair = max(pair_freq, key=lambda x:(pair_freq[x], x))

            


            



    




    # return vocab, merges

if __name__ == "__main__":
    train_bpe(CORPUS_FILE,
              vocab_size=10000,
              special_tokens=SPECIAL_TOKENS)




