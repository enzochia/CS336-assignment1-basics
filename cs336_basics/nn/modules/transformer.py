import torch
import torch.nn as nn
from .linear import Linear
from .normalization import RMSNorm
from .position_embeddings import RotaryPositionalEmbedding
from .activation import MultiheadAttention, SwiGLU
from .sparse import Embedding
from cs336_basics.nn.functional import softmax


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

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_id: int = 256
    ) -> torch.Tensor:
        # does it work for the cutomized RMSNorm?
        self.eval()
        token_seq = prompt_tokens
        for _ in range(max_new_tokens):
            token_seq = token_seq[..., -self.context_length:]
            # [batch_size, ..., vocab_size]
            logit_tensor = self(token_seq)[..., -1, :] / temperature
            prob_tensor = softmax(logit_tensor, dim=-1)
            if top_p < 1:
                # [batch_size, ..., vocab_size]
                sorted_prob_tensor, sorted_idx = torch.sort(prob_tensor, dim=-1, descending=True)
                # [batch_size, ..., vocab_size]
                cumulative_prob_tensor = torch.cumsum(sorted_prob_tensor, dim=-1)
                # [batch_size, ..., vocab_size], bool
                remove_bool_tensor = cumulative_prob_tensor > top_p
                remove_bool_tensor[..., 1:] = remove_bool_tensor[..., :-1].clone()
                remove_bool_tensor[..., 0] = False
                # [batch_size, ..., vocab_size], bool
                mask = torch.zeros_like(prob_tensor, dtype=torch.bool).scatter_(
                    dim=-1, index=sorted_idx, src=remove_bool_tensor
                )
                prob_tensor[mask] = 0
                prob_tensor /= prob_tensor.sum(dim=-1, keepdim=True)
            generated_token = torch.multinomial(prob_tensor, num_samples=1)
            token_seq = torch.cat((token_seq, generated_token), dim=-1)
            # TODO: Current solution only works for batch_size 1
            if (((len(token_seq.size()) == 1) or
                 (token_seq.size(0) == 1)) and
                generated_token.item() == eos_token_id):
                break
        return token_seq



