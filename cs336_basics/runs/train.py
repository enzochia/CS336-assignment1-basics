import time
import logging
import torch
import setuptools
import wandb
from dataclasses import asdict
from transformers import HfArgumentParser
from tqdm import tqdm
from cs336_basics.tokenizer import *
from cs336_basics.utils.tokenizer_utils import get_token_count, encode_and_dump
from cs336_basics.utils.utils import log_runtime, eval, load_checkpoint
from cs336_basics.utils.data import Dataset
from nn import TransformerLM, cross_entropy, gradient_clipping
from optim import AdamW, get_lr_cosine_schedule
from config import Config

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
logging.info(f"Training on device {conf.device} with configuration:")
logging.info(f"{asdict(conf)}")
if conf.wandb_logging:
    wandb.init(project=wandb_project, name=wandb_run_name)

dataset = Dataset(
    data_path=conf.data_path,
    batch_size=conf.batch_size,
    context_length=conf.context_length,
    device=conf.device,
    sampling_mode=conf.sampling_mode
)

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

optimizer = AdamW(
    params=model.parameters(), 
    lr=conf.optim_lr,
    eps=conf.optim_eps,
    betas=tuple(conf.adamw_betas),
    weight_decay=conf.optim_weight_decay
)

if conf.init_from == "pretrained":
    iter_num = load_checkpoint(src=conf.init_from_path, 
                               model=model, 
                               optimizer=optimizer)

start_time = time.time()
pbar = tqdm(range(conf.total_iters), desc=" steps")
for iter_num in pbar:
    optimizer.zero_grad()
    token_seq, next_token_seq = dataset.get_batch("train")
    loss = cross_entropy(model(token_seq), next_token_seq)
    loss.backward()
    gradient_clipping(model.parameters(), conf.grad_clip_max_l2_norm)
    lr = get_lr_cosine_schedule(
        it=iter_num,
        max_learning_rate=conf.max_learning_rate,
        min_learning_rate=conf.min_learning_rate,
        warmup_iters=conf.warmup_iters,
        cosine_cycle_iters=conf.cosine_cycle_iters,
    )
    optimizer.set_lr(lr)
    optimizer.step()
    pbar.set_postfix(loss=f"{loss.item():.2f}", lr=f"{lr:.2e}")

    if ((iter_num > 0) and 
        ((iter_num % conf.eval_every == 0) or
         (iter_num == (conf.total_iters - 1)))):
        eval(model=model,
             optimizer=optimizer,
             conf=conf,
             dataset=dataset,
             iter_num=iter_num,
             lr=lr)
pbar.close()