import torch
import torch.nn as nn


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