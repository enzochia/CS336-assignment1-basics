import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter


def softmax(x: torch.Tensor,
            dim: int = -1) -> torch.Tensor:
    x -= torch.max(x, dim=dim, keepdim=True)[0]
    x_exp = torch.exp(x)
    x_softmax = x_exp / torch.sum(x_exp, dim=dim, keepdim=True)
    return x_softmax



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
        # batch_size x seq_len x input_dim * input_dim x output_dim
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


class Swiglu(nn.Module):
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
        self.w1 = Parameter(torch.empty(d_ff, d_model, **factory_kwargs))
        self.w2 = Parameter(torch.empty(d_model, d_ff, **factory_kwargs))
        self.w3 = Parameter(torch.empty(d_ff, d_model, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std_dev = math.sqrt(2 / (self.d_model + self.d_ff))
        nn.init.trunc_normal_(
            self.w1, 
            mean=0.0, 
            std=std_dev, 
            a=-3 * std_dev, 
            b=3 * std_dev
        )
        nn.init.trunc_normal_(
            self.w2, 
            mean=0.0, 
            std=std_dev, 
            a=-3 * std_dev, 
            b=3 * std_dev
        )
        nn.init.trunc_normal_(
            self.w3, 
            mean=0.0, 
            std=std_dev, 
            a=-3 * std_dev, 
            b=3 * std_dev
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        # batch_size x seq_len x d_model * d_model x d_ff
        w1_x = torch.matmul(x, self.w1.transpose(0, 1))
        sigmoid_w1x = torch.sigmoid(w1_x)
        silu = w1_x * sigmoid_w1x
        # batch_size x seq_len x d_model * d_model x d_ff
        silu_w3x = silu * torch.matmul(x, self.w3.transpose(0, 1))
        # batch_size x seq_len x d_ff * d_ff x d_model
        swiglu = torch.matmul(silu_w3x, self.w2.transpose(0, 1))
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
        token_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        This is for single-head query and key matrices.
        """
        # [batch_size, seq_len, d]
        seq_len= x.size(1)
        # [seq_len, d // 2, 2]
        cos_sin_cache = self.rope_cos_sin_cache[:seq_len] if token_positions is None \
                        else self.rope_cos_sin_cache[token_positions]
        # [batch_size, seq_len, d // 2, 2]
        x_to_rotate = x.view(*x.shape[:-1], -1, 2)
        # [1, seq_len, 1, d // 2, 2]
        cos_sin_cache = cos_sin_cache.view(1, seq_len, -1, 2)
        # [batch_size, seq_len, d]
        x_rotated = torch.stack(
            [
                x_to_rotate[..., 0] * cos_sin_cache[..., 0] - x_to_rotate[..., 1] * cos_sin_cache[..., 1],
                x_to_rotate[..., 0] * cos_sin_cache[..., 1] + x_to_rotate[..., 1] * cos_sin_cache[..., 0]
            ],
            dim=-1
        ).flatten(start_dim=2)
        return x_rotated
    
