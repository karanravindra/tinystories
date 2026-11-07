from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import torch
import tyro

from tinystories.config import default_device
from tinystories.model import GPT
from tinystories.tokenizer import BPETokenizer


@dataclass
class SampleArgs:
    ckpt_dir: Annotated[Path, tyro.conf.Positional]
    """run directory containing model.pt and tokenizer.json"""
    prompt: str = ""
    n: int = 1
    """number of samples"""
    max_new_tokens: int = 400
    temperature: float = 1.0
    """0 = greedy"""
    top_k: int | None = None
    seed: int | None = None
    device: str = field(default_factory=default_device)


def generate_text(
    model: GPT,
    tokenizer: BPETokenizer,
    prompt: str = "",
    n: int = 1,
    max_new_tokens: int = 400,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> list[str]:
    """Samples start after an EOS (the story boundary seen in training), so "" means a fresh story."""
    device = next(model.parameters()).device
    ids = [tokenizer.eos_id] + (tokenizer.encode(prompt) if prompt else [])
    idx = torch.tensor([ids] * n, dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, temperature, top_k, eos_id=tokenizer.eos_id)
    texts = []
    for row in out[:, 1:].tolist():
        if tokenizer.eos_id in row:
            row = row[: row.index(tokenizer.eos_id)]
        texts.append(tokenizer.decode(row))
    return texts


def sample(args: SampleArgs) -> list[str]:
    from tinystories.train import load_checkpoint  # keeps matplotlib out of the import path until needed

    if args.seed is not None:
        torch.manual_seed(args.seed)
    model, tokenizer, _ = load_checkpoint(args.ckpt_dir, args.device)
    texts = generate_text(model, tokenizer, args.prompt, args.n, args.max_new_tokens, args.temperature, args.top_k)
    for i, t in enumerate(texts):
        if i:
            print("\n" + "-" * 40 + "\n")
        print(t)
    return texts
