import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


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