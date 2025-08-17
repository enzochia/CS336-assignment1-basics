import torch
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


@dataclass
class Config:
    data_path: str
    batch_size: int
    device: Optional[torch.device] = field(default = torch.device("cuda") if torch.cuda.is_available() else 
                                           (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")))

    # Training parameters
    target_token_count: Optional[int] = field(default=None)
    total_iters: Optional[int] = field(default=None)
    warmup_iters: Optional[int] = field(default=None)
    cosine_cycle_iters: Optional[int] = field(default=None)
    max_learning_rate: Optional[float] = field(default=5e-3)
    min_learning_rate: Optional[float] = field(default=1e-6)
    grad_clip_max_l2_norm: Optional[float] = field(default=1.0)
    
    # Model parameters
    # https://github.com/huggingface/transformers/blob/v4.21.0/src/transformers/models/gpt2/configuration_gpt2.py#L38
    vocab_size: Optional[int] = field(default=50257)
    context_length: Optional[int] = field(default=1024)
    d_model: Optional[int] = field(default=768)
    num_layers: Optional[int] = field(default=12)
    num_heads: Optional[int] = field(default=12)
    activation_function: Optional[str] = field(default="glu_new")
    d_ff: Optional[int] = field(default=3072)
    rope_theta: Optional[float] = field(default=10000)
    bos_token_id: Optional[int] = field(default=256)
    eos_token_id: Optional[int] = field(default=256)

    # Optimizer parameters
    optim_weight_decay: Optional[float] = field(default=1e-2)
    optim_eps: Optional[float] = field(default=1e-8)
    adamw_betas: Optional[List[float]] = field(default_factory=lambda: (0.9, 0.999))
    optim_lr: Optional[float] = field(default=1e-3)

    # Logging parameters
    wandb_logging: Optional[bool] = field(default=False)
    wandb_project: Optional[str] = field(default=None)
    wandb_run_name: Optional[str] = field(default=None)
    log_every: Optional[int] = field(default=None)
    eval_every: Optional[int] = field(default=None)
    eval_iters: Optional[int] = field(default=100)

    # Checkpointing parameters
    save_checkpoint_every: Optional[int] = field(default=2000)
    checkpoint_path: Optional[str] = field(default="")
    init_from: Optional[str] = field(default="scratch")
    init_from_path: Optional[str] = field(default="")

    # Data loading parameters
    sampling_mode: Optional[str] = "random"

    def __post_init__(self):
        if self.total_iters is None:
            self.total_iters = self.target_token_count // (self.batch_size * self.context_length)
        elif self.target_token_count is None:
            self.target_token_count = self.total_iters * self.batch_size * self.context_length
        elif self.target_token_count != self.total_iters * self.batch_size * self.context_length:
            raise ValueError(f"Conflict in config: target_token_count != total_iters * batch_size * context_length")

        if self.warmup_iters is None:
            self.warmup_iters = min(int(self.total_iters * 0.01), 100)
        if self.cosine_cycle_iters is None:
            self.cosine_cycle_iters = int(self.total_iters * 0.9) if self.total_iters > 1e4 else self.total_iters
        if self.log_every is None:
            self.log_every = int(self.total_iters * 0.01)
        if self.eval_every is None:
            self.eval_every = int(self.total_iters * 0.1)
        if self.wandb_logging:
            assert self.wandb_project is not None, "wandb_project is required when wandb_logging is True."
            assert self.wandb_run_name is not None, "wandb_run_name is required when wandb_logging is True."
        
        assert self.init_from in {"scratch", "pretrained"}, "Wrong init_from string."