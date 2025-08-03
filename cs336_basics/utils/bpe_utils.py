import os
import json
from functools import lru_cache
from typing import List, Dict, Tuple, Any


@lru_cache
# Copied from transformers.models.bart.tokenization_bart.bytes_to_unicode
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def convert_and_save_bpe_vocab_and_merges(vocab: Dict[int, bytes],
                                          merges: List[Tuple[bytes]],
                                          vocab_path: str | os.PathLike,
                                          merges_path: str | os.PathLike) -> None:
    shift_dict: Dict[int, str] = bytes_to_unicode()
    # reverse it for compatibility with json
    vocab_reversed: Dict[str, int] = {"".join(shift_dict[one_byte] for one_byte in token): key
                                      for key, token in vocab.items()}
    # don't worry about merge pairs those already have spaces, they are shifted by the shift_dict
    merges_readable: List[str] = [" ".join(["".join(shift_dict[one_byte] for one_byte in pair[0]),
                                            "".join(shift_dict[one_byte] for one_byte in pair[1])])
                                  for pair in merges]

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_reversed, f, ensure_ascii=False)

    with open(merges_path, "w", encoding="utf-8") as f:
        for pair in merges_readable:
            f.write(pair + "\n")


def read_and_convert_bpe_vocab_and_merges(vocab_path: str | os.PathLike,
                                          merges_path: str | os.PathLike) -> Tuple[Any]:
    reverse_shift_dict: Dict[str, int] = {char: byte for byte, char in bytes_to_unicode().items()}
    with open(vocab_path, "r") as f:
        vocab_reversed: Dict[str, int] = json.load(f)
    merges_shifted: List[str] = []
    with open(merges_path, "r") as f:
        for line in f:
            token_pair = line.rstrip()
            if token_pair:
                token_pair_split = token_pair.split(" ")
                if len(token_pair_split) == 2:
                    merges_shifted.append(tuple(token_pair_split))

    # reverse it back on two dimensions: 1) now key is the interger and val is the bytes, and
    # 2) for those un-printable bytes get them back
    vocab: Dict[int, bytes] = {key: bytes([reverse_shift_dict[one_char] for one_char in val])
                               for val, key in vocab_reversed.items()}

    # shift back from printable characters
    merges: List[Tuple[bytes]] = [(bytes([reverse_shift_dict[one_char] for one_char in merge_pair_tuple[0]]),
                                   bytes([reverse_shift_dict[one_char] for one_char in merge_pair_tuple[1]]))
                                   for merge_pair_tuple in merges_shifted]

    return(vocab,
           merges)
