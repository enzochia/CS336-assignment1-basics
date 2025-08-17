import torch
import torch.nn as nn
from typing import Optional
from .linear import Linear
from .position_embeddings import RotaryPositionalEmbedding
from nn.functional import scaled_dot_product_attention

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(in_features=d_model, out_features=d_ff, **factory_kwargs)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, **factory_kwargs)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, **factory_kwargs)
        
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        # [..., d_ff]
        w1_x = self.w1(x)
        sigmoid_w1x = torch.sigmoid(w1_x)
        silu = w1_x * sigmoid_w1x
        # [..., d_ff]
        silu_w3x = silu * self.w3(x)
        # [..., d_model]
        swiglu = self.w2(silu_w3x)
        return swiglu

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_ff={self.d_ff}"


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        assert d_model > 0 and num_heads > 0
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_k = d_model // num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        if self.theta is not None:
            assert rope is None
            rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, **self.factory_kwargs)
        self.rope = rope
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model, **self.factory_kwargs)
        self.k_proj = Linear(d_model, d_model, **self.factory_kwargs)
        self.v_proj = Linear(d_model, d_model, **self.factory_kwargs)
        self.output_proj = Linear(d_model, d_model, **self.factory_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] | None = None,
        attn_mask: Optional[torch.Tensor] | None = None,
        is_causal: bool = False,
        token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        # batch_size, seq_len
        dims = x.size()[:-1]
        # [batch_size, num_heads, seq_len, d_k]
        q = self.q_proj(x).view(*dims, self.num_heads, -1).transpose(-2, -3).contiguous()
        k = self.k_proj(x).view(*dims, self.num_heads, -1).transpose(-2, -3).contiguous()
        v = self.v_proj(x).view(*dims, self.num_heads, -1).transpose(-2, -3)
        if self.rope is not None:
            # [batch_size, num_heads, seq_len, d_k]
            q = self.rope(x=q, token_positions=token_positions)
            k = self.rope(x=k, token_positions=token_positions)
        output = scaled_dot_product_attention(q=q, k=k, v=v, is_causal=True, **self.factory_kwargs)
        # [batch_size, seq_len, d_model]
        output = output.transpose(-2, -3).contiguous().view(*dims, -1)
        output = self.output_proj(output)
        return output