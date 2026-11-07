from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import torch
import tyro


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ModelConfig:
    block_size: int = 512
    dim: int = 384
    n_heads: int = 8
    n_layers: int = 4
    tie_embeddings: bool = True
    """tie the output head to tok_emb (--model.no-tie-embeddings for a separate head)"""
    vocab_size: Annotated[int, tyro.conf.Suppress] = 0
    """filled in from the tokenizer"""


@dataclass
class DataConfig:
    n_tokens: int = 500_000_000
    """train tokens from the train split, seen once (capped at what the dataset holds)"""
    vocab_size: int = 8192
    """BPE vocab size"""
    max_story_tokens: int = 1024
    """drop train stories longer than this many tokens (0 = keep all); never applied to val"""
    cache_dir: Path = Path("cache")
    rebuild_cache: bool = False
    """delete and rebuild this tokenizer's cache dir (tokenizer + tokens)"""


@dataclass
class TrainConfig:
    optimizer: Literal["muon", "adamw"] = "muon"
    lr: float = 1e-3
    """AdamW lr (also embeddings/head/norms under muon)"""
    muon_lr: float = 0.02
    """Muon lr for hidden weight matrices"""
    batch_size: int = 64
    warmup_frac: float = 0.02
    decay_frac: float = 0.2
    """fraction of steps at the end over which LR decays linearly to 0"""
    compile_mode: Literal["none", "default", "reduce-overhead", "max-autotune"] = "reduce-overhead"
    eval_every: int = 100
    """steps between subset val evals (0 disables; a full val eval always runs at the end)"""
    eval_batches: int = 20
    log_every: int = 50
    seed: int = 0


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = field(default_factory=default_device)
    out_dir: Path = Path("checkpoints")
    """runs are saved to <out_dir>/<run name>/"""
    sample_tokens: int = 800
    wandb: bool = False
    wandb_project: str = "tinystories"
    wandb_name: str | None = None
    """run name (also the output subdir); default: wandb's name, else a timestamp"""

    def to_dict(self) -> dict:
        """JSON-safe dict (Paths -> str), so checkpoints load with torch.load(weights_only=True)."""
        d = asdict(self)
        d["out_dir"] = str(self.out_dir)
        d["data"]["cache_dir"] = str(self.data.cache_dir)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        return cls(
            model=ModelConfig(**d["model"]),
            data=DataConfig(**{**d["data"], "cache_dir": Path(d["data"]["cache_dir"])}),
            train=TrainConfig(**d["train"]),
            **{k: Path(v) if k == "out_dir" else v for k, v in d.items() if k not in ("model", "data", "train")},
        )
