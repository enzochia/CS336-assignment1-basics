import os
import regex as re
import multiprocessing
import heapq
from typing import List, Dict, Tuple, BinaryIO
from collections import Counter

PAT_STR_GPT2 = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
ENDOFTEXT = "<|endoftext|>"
SPECIAL_TOKENS = ["<|endoftext|>"]
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

if __name__ == "__main__":
    num_processes = os.cpu_count()

    with open(CORPUS_FILE, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(f, num_processes, ENDOFTEXT.encode('utf-8'))
        chunks = []
        for idx_chunk in range(len(chunk_boundaries) - 1):
            idx_chunk_start = chunk_boundaries[idx_chunk]
            idx_chunk_end = chunk_boundaries[idx_chunk + 1]
            f.seek(idx_chunk_start)
            chunks.append(f.read(idx_chunk_end - idx_chunk_start).decode("utf-8", errors="ignore"))

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(tokenize_chunk, chunks)

    pretokenization_dict = results[0]
    for chunk_counter in results[1:]:
        pretokenization_dict.update(chunk_counter)
        
    print("Most common pre-tokens:")
    print(pretokenization_dict.most_common(10))
    print("Longest pre-tokens:")
    print(heapq.nlargest(10, pretokenization_dict, key=lambda x: len(x)))

# def train_bpe(input_path: str,
#               vocab_size: int,
#               special_tokens: List[str]) -> Tuple[Union[Dict, List]]:



#     return vocab, merges


