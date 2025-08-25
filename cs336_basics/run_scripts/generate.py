import os
import time
import logging
import torch
from dataclasses import asdict
from transformers import HfArgumentParser
from cs336_basics.utils import (
    log_runtime, 
    load_checkpoint,
    SPECIAL_TOKENS,
    ENDOFTEXT
)
from cs336_basics.nn import (
    TransformerLM, 
    softmax
)
from cs336_basics.config import Config
from cs336_basics.tokenizer import Tokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()


parser = HfArgumentParser(Config)
conf = parser.parse_args_into_dataclasses()[0]
logging.info(f"Generating on device {conf.device} with configuration:")
logging.info(f"{asdict(conf)}")
tokenizer = Tokenizer.from_files(
    vocab_filepath=os.path.join(conf.tokenizer_dir, "vocab.json"),
    merges_filepath=os.path.join(conf.tokenizer_dir, "merges.txt"),
    special_tokens=SPECIAL_TOKENS
)
logging.info(f"idx for EOS token {tokenizer.vocab[conf.eos_token_id]}: {tokenizer.inverse_vocab[ENDOFTEXT.encode("utf-8")]}")

model = TransformerLM(
    vocab_size=conf.vocab_size,
    context_length=conf.context_length,
    d_model=conf.d_model,
    num_layers=conf.num_layers,
    num_heads=conf.num_heads,
    d_ff=conf.d_ff,
    rope_theta=conf.rope_theta,
    device=conf.device
).to(conf.device)

_ = load_checkpoint(src=conf.init_from_path, 
                    model=model)

start_time = time.time()
prompt_text = "Once upon a"
prompt_tokens_list = tokenizer.encode(prompt_text)
prompt_tokens = torch.tensor(prompt_tokens_list, dtype=torch.long, device=conf.device).unsqueeze(0)
generated_tokens = model.generate(
    prompt_tokens=prompt_tokens,
    max_new_tokens=conf.max_new_tokens,
    temperature=conf.temperature,
    top_p=conf.top_p,
    eos_token_id=conf.eos_token_id
)
# TODO: Current solution only works for batch_size 1
generated_tokens_list = list(x.item() for x in generated_tokens[0])
logging.info(f"Input prompt text: {prompt_text}")
logging.info(f"Model generates: {tokenizer.decode(generated_tokens_list)}")
consumed_time = time.time() - start_time
num_tokens_generated = len(generated_tokens_list) - len(prompt_tokens_list)
logging.info(f"The model spent {consumed_time: .2f} seconds to generate {num_tokens_generated}, at a throughput of {num_tokens_generated / consumed_time: .2f} tokens/sec")