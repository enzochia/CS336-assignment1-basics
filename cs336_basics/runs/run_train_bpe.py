import os
import time
import json
import logging
from cs336_basics.train_bpe import *
from cs336_basics.utils.utils import log_runtime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    # CORPUS_FILE = "data/TinyStoriesV2-GPT4-debug.txt"
    # CORPUS_FILE = "data/TinyStoriesV2-GPT4-valid.txt"
    CORPUS_FILE = "data/TinyStoriesV2-GPT4-train.txt"
    VOCAB_SIZE = 10000
    OUTPUT_PATH = "data/ts/"

    # CORPUS_FILE = "data/owt_valid.txt"
    # CORPUS_FILE = "data/owt_train.txt"
    # VOCAB_SIZE = 32000
    # OUTPUT_PATH = "data/owt/"

    start_time = time.time()
    vocab, merges = train_bpe(input_path=CORPUS_FILE,
                              vocab_size=VOCAB_SIZE,
                              special_tokens=SPECIAL_TOKENS)
    log_runtime(start_time, time.time(), "training the BPE tokenzier")

    convert_and_save_bpe_vocab_and_merges(vocab,
                                          merges,
                                          OUTPUT_PATH + "vocab.json",
                                          OUTPUT_PATH + "merges.txt")

    logging.info(f"Trained a BPE tokenizer on corpus file {CORPUS_FILE} with vocab size cap {VOCAB_SIZE} and got actual vocab size: {len(vocab)}.\
                 Vocab and merges file saved under {OUTPUT_PATH}")
    logging.info(f"Longest pre-tokens: {list(x.decode("utf-8") for x in heapq.nlargest(10, vocab.values(), key=len))}")