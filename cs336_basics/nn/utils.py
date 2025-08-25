import math
import torch
from collections.abc import Iterable

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], 
                      max_l2_norm: float,
                      eps: float = 1e-6) -> None:
    total_norm = math.sqrt(
        sum((p.grad.data ** 2).sum() if p.grad is not None else 0 
        for p in parameters)
    )
    if total_norm >= max_l2_norm:
        coef = max_l2_norm / (total_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.detach().mul_(coef)