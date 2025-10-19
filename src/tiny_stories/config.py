"""
Training configuration for GPT model.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import torch


@dataclass
class Config:
    """Base Configuration for GPT model training."""

    # I/O
    out_dir: str = "out"
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval
    always_save_checkpoint: bool = (
        True  # if True, always save a checkpoint after each eval
    )
    init_from: str = "scratch"  # 'scratch' or 'resume' or 'gpt2*'

    # wandb logging
    wandb_log: bool = False  # disabled by default
    wandb_project: str = "tiny_stories"
    wandb_run_name: str = "gpt2"  # 'run' + str(time.time())

    # data
    dataset: str = "tiny_stories"
    train_token_path: Path = Path("train_tokens.pkl")
    valid_token_path: Path = Path("valid_tokens.pkl")
    tokeniser: str = "gpt2"
    eot_token_id: int = 50256
    # The actual number of tokens in the tokeniser is 50257, but rounding up to the nearest
    # multiple of 64 makes things faster due to CUDA things that I don't understand
    vocab_size: int = 50304
    gradient_accumulation_steps: int = 5 * 8  # used to simulate larger batch sizes
    batch_size: int = (
        12  # if gradient_accumulation_steps > 1, this is the micro-batch size
    )
    context_window: int = 1024
    num_workers: int = 4
    use_inifite_dataloader = True

    # model
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    bias: bool = False  # do we use bias inside LayerNorm and Linear layers?

    # adamw optimizer
    learning_rate: float = 6e-4  # max learning rate
    max_iters: int = 600000  # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 2000  # how many steps to warm up for
    lr_decay_iters: int = 600000  # should be ~= max_iters per Chinchilla
    min_lr: float = (
        6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    )

    # DDP settings
    backend: str = "nccl"  # 'nccl', 'gloo', etc.

    # system
    device: str = (
        "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    )
    dtype: str = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )  # 'float32', 'bfloat16', or 'float16'
    compile: bool = True  # use PyTorch 2.0 to compile the model to be faster

    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class BabyModelCPUConfig(Config):
    """Configuration for baby 49M GPT-2 model. Trained on macbooks builtin (well technically a macbook GPU)
    Parameters mostly stolen from andrej's shakespeare config
    """

    wandb_log: bool = True
    wandb_project: str = "tiny_stories"
    eval_interval: int = 500

    batch_size: int = 8
    context_window: int = 512
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.2
    gradient_accumulation_steps: int = 1
    num_workers: int = 0

    learning_rate: float = 1e-3  # with baby networks can afford to go a bit higher
    max_iters: int = 5000
    lr_decay_iters: int = 5000  # make equal to max_iters usually
    min_lr: float = 1e-4  # learning_rate / 10 usually
    beta2: float = 0.99  # make a bit bigger because number of tokens per iter is small

    warmup_iters: int = 100  # not super necessary potentially
    device: str = "mps"


@dataclass
class BabyModelGPUConfig(Config):
    """Configuration for baby 49M GPT-2 model. Trained on single GPU. Parameters mostly stolen from
    andrej's shakespeare config
    """

    wandb_log: bool = True
    wandb_project: str = "tiny_stories"
    eval_interval: int = 500

    batch_size: int = 64
    context_window: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    gradient_accumulation_steps: int = 1
    num_workers: int = 0

    learning_rate: float = 1e-3  # with baby networks can afford to go a bit higher
    max_iters: int = 10_000
    lr_decay_iters: int = 10_000  # make equal to max_iters usually
    min_lr: float = 1e-4  # learning_rate / 10 usually
    beta2: float = 0.99  # make a bit bigger because number of tokens per iter is small

    warmup_iters: int = 100  # not super necessary potentially
    device: str = "cuda"

class ConfigType(Enum):
    GPT2 = "gpt2"
    BABY_CPU = "baby_cpu"
    BABY_GPU = "baby_gpu"


def load_config(config_type: ConfigType = ConfigType.BABY_GPU) -> Config:
    match config_type:
        case ConfigType.GPT2:
            return Config()
        case ConfigType.BABY_CPU:
            return BabyModelCPUConfig()
        case ConfigType.BABY_GPU:
            return BabyModelGPUConfig()
