import torch
import math
from typing import Optional


def softmax(x: torch.Tensor,
            dim: int = -1) -> torch.Tensor:
    x -= torch.max(x, dim=dim, keepdim=True)[0]
    x_exp = torch.exp(x)
    x_softmax = x_exp / torch.sum(x_exp, dim=dim, keepdim=True)
    return x_softmax


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] | None = None,
    is_causal: Optional[bool] = False
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
        causal_mask = torch.ones(seq_len, seq_len).tril(diagonal=0)
        q_k_scaled_dot_prod.masked_fill_(causal_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        q_k_scaled_dot_prod.masked_fill_(attn_mask.logical_not(), float("-inf"))
    
    # [batch_size, ..., seq_len, seq_len]
    attn_weight = softmax(q_k_scaled_dot_prod)
    # [batch_size, ..., seq_len, d_v]
    o = torch.matmul(attn_weight, v)
    return o