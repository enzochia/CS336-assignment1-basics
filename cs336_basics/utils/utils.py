import os
import logging
import torch
import setuptools
import wandb
from cs336_basics.nn.modules import TransformerLM
from cs336_basics.config import Config
from cs336_basics.dataset import Dataset
from cs336_basics.nn.functional import cross_entropy
from cs336_basics.optim import AdamW
from typing import BinaryIO, IO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_runtime(start_time: float,
                end_time: float,
                task: str) -> None:
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logging.info(f"Time Taken for {task}: {int(hours):02}:{int(minutes):02}:{seconds:05.2f}")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
    epoch: int = 0
) -> None:
    checkpoint_dir = os.path.join(out, f"iter_{iteration}")
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    os.makedirs(checkpoint_dir, exist_ok=True) 
    torch.save(
        {"model": model.state_dict(),
         "optimizer": optimizer.state_dict(),
         "iter": iteration,
         "epoch": epoch},
        checkpoint_path
    )
    logging.info(f"Saved model to {checkpoint_path}")


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None
) -> None:
    src = os.path.join(src, "checkpoint.pt")
    full_state_dict = torch.load(src)
    model.load_state_dict(full_state_dict["model"])
    if optimizer is not None:
        optimizer.load_state_dict(full_state_dict["optimizer"])
    logging.info(f"Loaded model from {src}")
    return full_state_dict["iter"]


def eval(model: torch.nn.Module,
         optimizer: torch.optim.Optimizer,
         conf: Config,
         dataset: Dataset,
         iter_num: int,
         lr: float,
         epoch: int = 0) -> None:
    total_loss = 0
    for _ in range(conf.eval_iters):
        token_seq, next_token_seq = dataset.get_batch("valid")
        token_seq = token_seq.to(conf.device)
        next_token_seq = next_token_seq.to(conf.device)
        with torch.no_grad():
            logits = model(token_seq)
            total_loss += cross_entropy(logits, next_token_seq).item()
    total_loss /= conf.eval_iters
    logging.info(f"Iter: {iter_num}, validation loss: {total_loss}, lr: {lr}.")
    if conf.wandb_logging:
        wandb.log({"iter": iter_num, "lr": lr, "val_loss": total_loss})
    if ((iter_num > 0) and 
        ((iter_num % conf.save_checkpoint_every == 0) or 
         (iter_num == (conf.total_iters - 1)))):
        save_checkpoint(model=model, 
                        optimizer=optimizer, 
                        iteration=iter_num,
                        out=conf.checkpoint_path,
                        epoch=epoch)