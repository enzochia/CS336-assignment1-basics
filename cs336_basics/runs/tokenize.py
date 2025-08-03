import json
from cs336_basics.tokenizer import *


if __name__ == "__main__":
    BPE_DIR = "data/ts/"
    # INPUT_FILE = "data/TinyStoriesV2-GPT4-train.txt"
    INPUT_FILE = "data/TinyStoriesV2-GPT4-debug.txt"
    # INPUT_FILE = "data/TinyStoriesV2-GPT4-valid.txt"

    VOCAB_PATH = os.path.join(BPE_DIR, "vocab.json")
    MERGES_PATH = os.path.join(BPE_DIR, "merges.txt")
    ENCODED_OUTPUT_FILE = os.path.join(BPE_DIR, "encoded_output.txt")

    SPECIAL_TOKEN_TO_SPLIT_BY = "<|endoftext|>"

    if not os.path.exists(VOCAB_PATH) or not os.path.exists(MERGES_PATH):
        raise FileNotFoundError
    else:
        tokenizer = Tokenizer.from_files(
            vocab_filepath=VOCAB_PATH,
            merges_filepath=MERGES_PATH,
            special_tokens=SPECIAL_TOKENS
        )

    # list_to_decode: List[int] = []
    with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as infile, \
            open(ENCODED_OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        chunk_generator = tokenizer.read_file_in_chunks(infile, SPECIAL_TOKEN_TO_SPLIT_BY)
        token_generator = tokenizer.encode_iterable(chunk_generator)
        for token_id in token_generator:
            outfile.write(str(token_id) + " ")
            # list_to_decode.append(token_id)
    print(f"Encoded file to '{ENCODED_OUTPUT_FILE}'.")

    # print(tokenizer.decode(list_to_decode))