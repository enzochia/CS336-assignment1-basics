import os
import time
import json
from cs336_basics.train_bpe import *


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

    print(f"Starting BPE training with vocab size cap {VOCAB_SIZE}, on corpus file {CORPUS_FILE}")

    vocab, merges = train_bpe(input_path=CORPUS_FILE,
                              vocab_size=VOCAB_SIZE,
                              special_tokens=SPECIAL_TOKENS)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nTraining Completed, and got actual vocab size: {len(vocab)}")

    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Time Taken: {int(hours):02}:{int(minutes):02}:{seconds:05.2f}")

    print("Longest pre-tokens:")
    print(list(x.decode("utf-8") for x in heapq.nlargest(10, vocab.values(), key=len)))

    convert_and_save_bpe_vocab_and_merges(vocab,
                                          merges,
                                          OUTPUT_PATH + "vocab.json",
                                          OUTPUT_PATH + "merges.txt")