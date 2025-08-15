from collections.abc import Callable, Iterable
from typing import Optional, Union, Any, TypeAlias
import torch
import torch.optim as optim
import math

ParamsT: TypeAlias = Union[
    Iterable[torch.Tensor], Iterable[dict[str, Any]], Iterable[tuple[str, torch.Tensor]]
]

class AdamW(optim.Optimizer):
    def __init__(self, 
                 params: ParamsT, 
                 lr: Union[float, torch.Tensor] = 1e-3,
                 eps: float=1e-8,
                 betas: tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]] = (0.9, 0.999),
                 weight_decay: float = 1e-2):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {"lr": lr,
                    "betas": betas,
                    "weight_decay": weight_decay,
                    "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            beta_1, beta_2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] 
                t = state.get("t", 0) 
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                t += 1
                grad = p.grad
                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * (grad ** 2)
                lr_t = lr * math.sqrt(1 - (beta_2 ** t)) / (1 - (beta_1 ** t))
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t
                state["m"] = m
                state["v"] = v
        return loss