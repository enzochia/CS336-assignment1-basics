import numpy as np
from tqdm import tqdm
from cs336_basics.tokenizer import *

def get_token_count(INPUT_FILE:  str | os.PathLike,
                    tokenizer: Tokenizer,
                    SPECIAL_TOKEN_TO_SPLIT_BY: str) -> int:
    total_token_count = 0
    total_chunk_count = 0
    input_size_bytes = os.path.getsize(INPUT_FILE)
    with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as infile:
        with tqdm(total=input_size_bytes, unit_scale=True, desc="Counting tokens") as pbar:
            chunk_generator = tokenizer.read_file_in_chunks(infile, SPECIAL_TOKEN_TO_SPLIT_BY)
            for chunk in chunk_generator:
                pbar.update(len(chunk.encode('utf-8')))
                total_token_count += len(tokenizer.encode(chunk))
                total_chunk_count += 1
    return total_token_count, total_chunk_count


def encode_and_dump(INPUT_FILE:  str | os.PathLike,
                    ENCODED_OUTPUT_FILE:  str | os.PathLike,
                    tokenizer: Tokenizer,
                    SPECIAL_TOKEN_TO_SPLIT_BY: str,
                    total_token_count: int) -> None:
    
    arr = np.memmap(ENCODED_OUTPUT_FILE, dtype=np.uint16, mode='w+', shape=(total_token_count,))
    
    current_idx = 0
    with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as infile:
        with tqdm(total=total_token_count, unit_scale=True, desc="Encoding and writing") as pbar:
            chunk_generator = tokenizer.read_file_in_chunks(infile, SPECIAL_TOKEN_TO_SPLIT_BY)
            for chunk in chunk_generator:
                encoded_ids = tokenizer.encode(chunk)
                chunk_len = len(encoded_ids)
                arr[current_idx : current_idx + chunk_len] = encoded_ids
                current_idx += chunk_len
                pbar.update(chunk_len)
    arr.flush()