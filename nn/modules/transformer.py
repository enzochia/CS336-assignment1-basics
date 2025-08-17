import torch
import torch.nn as nn
from .linear import Linear
from .normalization import RMSNorm
from .position_embeddings import RotaryPositionalEmbedding
from .activation import MultiheadAttention, SwiGLU
from .sparse import Embedding


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
            MultiheadAttention(d_model=d_model, num_heads=num_heads, rope=self.rope, **factory_kwargs) 
            if self.rope is not None else
            MultiheadAttention(d_model=d_model, num_heads=num_heads, **factory_kwargs) 
        )
        self.ln2 = RMSNorm(d_model=d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(x=self.ln1(x), is_causal=True)
        x = x + self.ffn(x=self.ln2(x))
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
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
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
            TransformerBlock(d_model=d_model, 
                             num_heads=num_heads, 
                             d_ff=d_ff, 
                             theta=rope_theta, 
                             max_seq_len=context_length,
                             device=device,
                             dtype=dtype)
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
        # [batch_size, seq_len, vocab_size]
        x = self.lm_head(x)
        return x




