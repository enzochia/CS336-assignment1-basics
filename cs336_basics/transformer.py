import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
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





class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std_dev = math.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(
            self.weight, 
            mean=0.0, 
            std=std_dev, 
            a=-3 * std_dev, 
            b=3 * std_dev
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        # [..., input_dim] * [input_dim, output_dim]
        return torch.matmul(x, self.weight.transpose(0, 1)).squeeze(-1)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"



class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        _freeze: bool = False
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, **factory_kwargs),
                                   requires_grad = not _freeze)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std_dev = 1
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std_dev,
            a=-3 * std_dev,
            b=3 * std_dev
        )

    def forward(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.weight[token_ids]

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"



class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        result = x * self.weight / rms
        return result.to(in_dtype)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, eps={self.eps}"


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


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self) -> None:
        # [d // 2]
        theta = 1.0 / (self.theta ** ((torch.arange(0, self.d_k, 2)).float() / self.d_k))
        # [max_seq_len]
        seq_idx = torch.arange(self.max_seq_len, dtype=theta.dtype, device=theta.device)
        # [max_seq_len, d // 2]
        theta_mat = torch.matmul(seq_idx.view(-1, 1), theta.view(1, -1))
        # [max_seq_len, d // 2, 2]
        cos_sin_cache = torch.stack([torch.cos(theta_mat), torch.sin(theta_mat)], dim=-1)
        self.register_buffer("rope_cos_sin_cache", cos_sin_cache, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_dim_ones = [1] * (len(x.size()) - 2)
        # [batch_size, ..., seq_len, d]
        seq_len= x.size(-2)
        # [seq_len, d // 2, 2]
        cos_sin_cache = self.rope_cos_sin_cache[:seq_len] if token_positions is None \
                        else self.rope_cos_sin_cache[token_positions]
        # [batch_size, ..., seq_len, d // 2, 2]
        x_to_rotate = x.view(*x.shape[:-1], -1, 2)
        # [1, ..., seq_len, d // 2, 2]
        cos_sin_cache = cos_sin_cache.view(*batch_dim_ones, seq_len, -1, 2)
        # [batch_size, ..., seq_len, d]
        x_rotated = torch.stack(
            [
                x_to_rotate[..., 0] * cos_sin_cache[..., 0] - x_to_rotate[..., 1] * cos_sin_cache[..., 1],
                x_to_rotate[..., 0] * cos_sin_cache[..., 1] + x_to_rotate[..., 1] * cos_sin_cache[..., 0]
            ],
            dim=-1
        ).flatten(start_dim=-2)
        return x_rotated
    


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
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_k = d_model // num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        if self.theta is not None:
            assert rope is None
            rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, **factory_kwargs)
        self.rope = rope
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model, **factory_kwargs)
        self.k_proj = Linear(d_model, d_model, **factory_kwargs)
        self.v_proj = Linear(d_model, d_model, **factory_kwargs)
        self.output_proj = Linear(d_model, d_model, **factory_kwargs)

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
        output = scaled_dot_product_attention(q=q, k=k, v=v, is_causal=True)
        # [batch_size, seq_len, d_model]
        output = output.transpose(-2, -3).contiguous().view(*dims, -1)
        output = self.output_proj(output)
        return output


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: int | None = 10000,
        max_seq_len: int | None = 8192,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.max_seq_len = max_seq_len
        if self.theta is not None:
            assert rope is None
            rope = RotaryPositionalEmbedding(theta=theta, 
                                             d_k=d_model // num_heads, 
                                             max_seq_len=max_seq_len, 
                                             **factory_kwargs)
        self.rope = rope
        self.ln1 = RMSNorm(d_model=d_model)
        self.attn = (
            MultiheadAttention(d_model=d_model, num_heads=num_heads, rope=self.rope) 
            if self.rope is not None else
            MultiheadAttention(d_model=d_model, num_heads=num_heads) 
        )
        self.ln2 = RMSNorm(d_model=d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x += self.attn(x=self.ln1(x), is_causal=True)
        x += self.ffn(x=self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, theta=rope_theta, max_seq_len=context_length)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x




