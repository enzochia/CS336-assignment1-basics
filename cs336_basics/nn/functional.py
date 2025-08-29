import torch
import math
from typing import Optional


def softmax(x: torch.Tensor,
            dim: int = -1) -> torch.Tensor:
    x = x - torch.max(x, dim=dim, keepdim=True)[0]
    x_exp = torch.exp(x)
    x_softmax = x_exp / torch.sum(x_exp, dim=dim, keepdim=True)
    return x_softmax


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] | None = None,
    is_causal: Optional[bool] = False,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None
) -> torch.Tensor:
    """
    Input
    q: [batch_size, ..., seq_len, d_k]
    k: [batch_size, ..., seq_len, d_k]
    v: [batch_size, ..., seq_len, d_v]
    attn_mask: [seq_len, seq_len]
    Output
    o: [batch_size, ..., seq_len, d_v]
    Note
    This attn_mask is not for padding
    """
    seq_len = q.size(-2)
    d_k = q.size(-1)
    # [batch_size, ..., seq_len, seq_len]
    q_k_scaled_dot_prod = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)

    if is_causal:
        if attn_mask is not None:
            raise ValueError(f"causal_mask is generated only when attn_mask is None.")
        causal_mask = torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool).tril(diagonal=0)
        q_k_scaled_dot_prod = q_k_scaled_dot_prod.masked_fill(causal_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        q_k_scaled_dot_prod = q_k_scaled_dot_prod.masked_fill(attn_mask.logical_not(), float("-inf"))
    
    # [batch_size, ..., seq_len, seq_len]
    attn_weight = softmax(q_k_scaled_dot_prod)
    # [batch_size, ..., seq_len, d_v]
    o = torch.matmul(attn_weight, v)
    return o


def _logsumexp(input: torch.Tensor) -> torch.Tensor:
    # [..., 1]
    input_max, _ = torch.max(input, dim=-1, keepdim=True)
    input = input - input_max
    return input_max + torch.log(torch.sum(torch.exp(input), dim=-1, keepdim=True))

def cross_entropy(input: torch.Tensor,
                  targets: torch.Tensor) -> torch.Tensor:
    # [batch_size, ..., 1]
    negative_log_prob = - torch.gather(input, dim=-1, index=targets.unsqueeze(-1)) + \
                        _logsumexp(input)

    return negative_log_prob.mean()
