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
    mlp: Literal["gelu", "relu2", "swiglu"] = "gelu"
    """MLP type: GELU or ReLU^2 with 4x width, or SwiGLU with 8/3x width (same weight count)"""
    qk_norm: bool = False
    """RMS-normalize queries and keys in attention (then scale logits by 1/sqrt(head_dim))"""
    value_embeds: int = 0
    """extra token-embedding tables added to the attention values of the first and last this-many layers"""
    vocab_size: Annotated[int, tyro.conf.Suppress] = 0
    """filled in from the tokenizer"""
    eos_id: Annotated[int, tyro.conf.Suppress] = -1
    """filled in from the tokenizer (end-of-story token, used by doc_mask)"""
    doc_mask: bool = False
    """block attention across story boundaries (an EOS ends its story); uses flex attention"""


@dataclass
class DataConfig:
    n_tokens: int = 500_000_000
    """total train tokens; past what the train split holds, it repeats in reshuffled epochs"""
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
    lr: float = 0.04
    """AdamW lr (also embeddings/head/norms under muon)"""
    muon_lr: float = 0.0036
    """Muon lr for hidden weight matrices"""
    batch_size: int = 64
    warmup_frac: float = 0.0
    decay_frac: float = 0.5
    """fraction of steps at the end over which LR decays linearly to min_lr_frac"""
    min_lr_frac: float = 0.0
    """where the decay ends, as a fraction of the peak LR (0 = decay to zero)"""
    compile_mode: Literal["none", "default", "reduce-overhead", "max-autotune"] = "reduce-overhead"
    eval_every: int = 100
    """steps between subset val evals (0 disables; a full val eval always runs at the end)"""
    eval_batches: int = 20
    log_every: int = 50
    save_every: int = 1000
    """steps between resumable state saves (<run dir>/state.pt; 0 = only at the end)"""
    snapshot_steps: tuple[int, ...] = ()
    """steps at which to also keep a copy of the state (<run dir>/state-<step>.pt), to branch other runs from"""
    stop_after: int = 0
    """stop after this many steps (after saving its snapshot), without the final eval: for a constant-lr parent run
    whose rest is trained by branches (0 = run to the end)"""
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
    resume: Path | None = None
    """run dir to continue from (its state.pt): same model and data settings, --data.n-tokens is the new total"""
    branch_from: Path | None = None
    """state file to start a NEW run from (e.g. <run dir>/state-10343.pt): its weights, optimizer state and step, but
    this run's name, wandb run and lr schedule; same model, data and batch settings. For trying lr decays off one
    constant-lr run without retraining the shared steps."""

    def to_dict(self) -> dict:
        """JSON-safe dict (Paths -> str), so checkpoints load with torch.load(weights_only=True)."""
        d = asdict(self)
        d["out_dir"] = str(self.out_dir)
        d["resume"] = None if self.resume is None else str(self.resume)
        d["branch_from"] = None if self.branch_from is None else str(self.branch_from)
        d["data"]["cache_dir"] = str(self.data.cache_dir)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        return cls(
            model=ModelConfig(**d["model"]),
            data=DataConfig(**{**d["data"], "cache_dir": Path(d["data"]["cache_dir"])}),
            train=TrainConfig(**d["train"]),
            **{
                k: Path(v) if k in ("out_dir", "resume", "branch_from") and v is not None else v
                for k, v in d.items()
                if k not in ("model", "data", "train")
            },
        )
