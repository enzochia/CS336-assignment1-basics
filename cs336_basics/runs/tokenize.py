import json
import numpy as np
import time
import logging
from tqdm import tqdm
from cs336_basics.tokenizer import *
from cs336_basics.utils.tokenizer_utils import get_token_count, encode_and_dump
from cs336_basics.utils.utils import log_runtime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    BPE_DIR = "data/ts/"
    INPUT_FILE = "data/TinyStoriesV2-GPT4-train.txt"
    # INPUT_FILE = "data/TinyStoriesV2-GPT4-debug.txt"
    # INPUT_FILE = "data/TinyStoriesV2-GPT4-valid.txt"

    VOCAB_PATH = os.path.join(BPE_DIR, "vocab.json")
    MERGES_PATH = os.path.join(BPE_DIR, "merges.txt")
    ENCODED_OUTPUT_FILE = os.path.join(BPE_DIR, "tokenized_corpus.bin")
    SPECIAL_TOKEN_TO_SPLIT_BY = "<|endoftext|>"

    if not os.path.exists(VOCAB_PATH) or not os.path.exists(MERGES_PATH):
        raise FileNotFoundError
    else:
        tokenizer = Tokenizer.from_files(
            vocab_filepath=VOCAB_PATH,
            merges_filepath=MERGES_PATH,
            special_tokens=SPECIAL_TOKENS
        )
    
    start_time = time.time()
    total_token_count, _ = get_token_count(INPUT_FILE, tokenizer, SPECIAL_TOKEN_TO_SPLIT_BY)
    log_runtime(start_time, time.time(), "counting total tokens in corpus")

    start_time = time.time()
    encode_and_dump(INPUT_FILE, 
                    ENCODED_OUTPUT_FILE,
                    tokenizer, 
                    SPECIAL_TOKEN_TO_SPLIT_BY,
                    total_token_count)
    log_runtime(start_time, time.time(), "actual encoding")

    logging.info(f"Tokenized input file {INPUT_FILE} with tokenizer under {BPE_DIR} and saved into {ENCODED_OUTPUT_FILE}.")

    # logging.info(f"################# Read from output .bin file and decode to validate. #################")
    # encoded_tokens_from_file = np.memmap(ENCODED_OUTPUT_FILE, dtype=np.uint16, mode='r')
    # sample_ids = encoded_tokens_from_file[:1500].tolist()
    # decoded_text_sample = tokenizer.decode(sample_ids)
    # logging.info(decoded_text_sample)
