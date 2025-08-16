import os
import logging
import torch
from nn.modules import TransformerLM
from optim import AdamW
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
    torch.save(
        {"model": model.state_dict(),
         "optimizer": optimizer.state_dict(),
         "iter": iteration,
         "epoch": epoch},
        out
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> None:
    full_state_dict = torch.load(src)
    model.load_state_dict(full_state_dict["model"])
    optimizer.load_state_dict(full_state_dict["optimizer"])
    return full_state_dict["iter"]