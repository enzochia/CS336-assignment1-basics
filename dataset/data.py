import torch
import numpy as np
from typing import Tuple

def get_batch(x: np.ndarray,
              batch_size: int,
              context_length: int,
              device: str | None = "mps",
              dtype: torch.dtype | None = np.int64,
              start_idx: int | None = None
) -> Tuple[torch.Tensor]:
    device = torch.device(device)
    if start_idx is None:
        seq_start_idx = torch.randint(0, len(x) - context_length, (batch_size,))
    elif start_idx < len(x) - context_length * batch_size:
        seq_start_idx = torch.Tensor([start_idx + context_length * seq 
                                      for seq in range(batch_size)])
    elif len(x) - start_idx > context_length:
        seq_start_idx = torch.Tensor([start_idx + context_length * seq 
                                      for seq in range((len(x) - start_idx - 1) // context_length)])
    else:
        return None, None

    token_seq_batch = torch.stack([torch.from_numpy(x[start_idx:(start_idx + context_length)])
                                   for start_idx in seq_start_idx], dim=0)
    token_next_batch = torch.stack([torch.from_numpy(x[(start_idx + 1):(start_idx + context_length + 1)])
                                    for start_idx in seq_start_idx], dim=0)
    return token_seq_batch.to(device), token_next_batch.to(device)


class Dataset:
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        context_length: int,
        device: torch.device | None = torch.device("mps"),
        dtype: torch.dtype | None = np.int64,
        sampling_mode: str = "random"
    ) -> None:
        self.train_data = np.memmap(f"{data_path}/train.bin", dtype=np.uint16, mode="r").astype(np.int64)
        self.val_data = np.memmap(f"{data_path}/val.bin", dtype=np.uint16, mode="r").astype(np.int64)
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.dtype = dtype
        self.start_idx = 0
        self.sampling_mode = sampling_mode

    def get_batch(
        self,
        split: str
    ) -> Tuple[torch.Tensor]:
        if split == "train":
            data = self.train_data
        elif split == "valid":
            data = self.val_data
        else:
            raise ValueError(f"Wrong split string.")
        if self.sampling_mode == "random":
            return get_batch(x=data, 
                             batch_size=self.batch_size, 
                             context_length=self.context_length,
                             device=self.device)
        elif self.sampling_mode == "sequential":
            start_idx_orig = self.start_idx
            if self.start_idx < len(data) - context_length * batch_size:
                self.start_idx += self.context_length * self.batch_size
            elif len(data) - self.start_idx > context_length:
                self.start_idx += self.context_length * ((len(data) - self.start_idx) // self.batch_size)
            else:
                self.start_idx = len(data)
            return get_batch(x=data, 
                             batch_size=self.batch_size, 
                             context_length=self.context_length,
                             device=self.device,
                             start_idx=start_idx_orig)
        else:
            raise ValueError("Wrong sampling_mode string.")