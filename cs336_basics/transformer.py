import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter


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