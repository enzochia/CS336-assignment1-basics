import json
import numpy as np
from tqdm import tqdm
from cs336_basics.tokenizer import *
from cs336_basics.utils.tokenizer_utils import encode_and_dump


if __name__ == "__main__":
    BPE_DIR = "data/ts/"
    # INPUT_FILE = "data/TinyStoriesV2-GPT4-train.txt"
    INPUT_FILE = "data/TinyStoriesV2-GPT4-debug.txt"
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

    encode_and_dump(INPUT_FILE, 
                    ENCODED_OUTPUT_FILE,
                    tokenizer, 
                    SPECIAL_TOKEN_TO_SPLIT_BY)

    print(f"################# Read from output .bin file and decode to validate. #################")
    encoded_tokens_from_file = np.memmap(ENCODED_OUTPUT_FILE, dtype=np.uint16, mode='r')
    sample_ids = encoded_tokens_from_file[:500].tolist()
    decoded_text_sample = tokenizer.decode(sample_ids)
    print(decoded_text_sample)
