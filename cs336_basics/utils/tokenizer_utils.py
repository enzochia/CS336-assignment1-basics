import numpy as np
from tqdm import tqdm
from cs336_basics.tokenizer import *

def get_token_count(INPUT_FILE:  str | os.PathLike,
                    tokenizer: Tokenizer,
                    SPECIAL_TOKEN_TO_SPLIT_BY: str) -> int:
    total_token_count = 0
    with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as infile:
        chunk_generator = tokenizer.read_file_in_chunks(infile, SPECIAL_TOKEN_TO_SPLIT_BY)
        for chunk in tqdm(chunk_generator, desc="Counting tokens"):
            total_token_count += len(tokenizer.encode(chunk))
    return total_token_count


def encode_and_dump(INPUT_FILE:  str | os.PathLike,
                    ENCODED_OUTPUT_FILE:  str | os.PathLike,
                    tokenizer: Tokenizer,
                    SPECIAL_TOKEN_TO_SPLIT_BY: str) -> None:
    total_token_count = get_token_count(INPUT_FILE, tokenizer, SPECIAL_TOKEN_TO_SPLIT_BY)
    arr = np.memmap(ENCODED_OUTPUT_FILE, dtype=np.uint16, mode='w+', shape=(total_token_count,))
    
    current_idx = 0
    with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as infile:
        chunk_generator = tokenizer.read_file_in_chunks(infile, SPECIAL_TOKEN_TO_SPLIT_BY)
        for chunk in tqdm(chunk_generator, desc="Encoding and writing"):
            encoded_ids = tokenizer.encode(chunk)
            chunk_len = len(encoded_ids)
            arr[current_idx : current_idx + chunk_len] = encoded_ids
            current_idx += chunk_len
    arr.flush()