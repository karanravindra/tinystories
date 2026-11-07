import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from tinystories.config import ModelConfig, RunConfig, TrainConfig
from tinystories.data import Batcher
from tinystories.model import GPT
from tinystories.tokenizer import BPETokenizer


def build_optimizers(model: GPT, cfg: TrainConfig) -> list[torch.optim.Optimizer]:
    if cfg.optimizer == "adamw":
        return [torch.optim.AdamW(model.parameters(), lr=cfg.lr)]

    # Muon only handles 2D hidden weight matrices. Embeddings, the output head,
    # norms and biases go to AdamW.
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        is_hidden_matrix = p.ndim == 2 and not name.startswith(("tok_emb", "head"))
        (muon_params if is_hidden_matrix else adamw_params).append(p)
    return [
        torch.optim.Muon(muon_params, lr=cfg.muon_lr),
        torch.optim.AdamW(adamw_params, lr=cfg.lr),
    ]


def lr_factor(step: int, total: int, warmup_frac: float, decay_frac: float) -> float:
    """Warmup-stable-decay: linear warmup, flat, then linear decay to 0 over the last decay_frac of steps.

    `step` is 0-based (the LR used for the (step+1)-th update).
    """
    warmup = max(1, round(warmup_frac * total))
    decay = max(1, round(decay_frac * total))
    if step < warmup:
        return (step + 1) / warmup
    if step >= total - decay:
        return max(0.0, (total - step) / decay)
    return 1.0


@torch.no_grad()
def evaluate(fwd, batches, byte_lengths: torch.Tensor) -> tuple[float, float]:
    """(mean loss in nats/token, bits/byte). Bytes come from the targets, so any subset of windows works.

    No model.eval()/train() toggle: there's no dropout, and flipping it makes torch.compile recompile.
    Restore it if dropout (or anything else mode-dependent) is ever added.
    """
    nats = torch.zeros((), device=byte_lengths.device)
    n_bytes = torch.zeros((), device=byte_lengths.device, dtype=torch.long)
    n_tokens = 0
    for x, y in batches:
        _, loss = fwd(x, y)
        nats += loss.float() * y.numel()
        n_bytes += byte_lengths[y].sum()
        n_tokens += y.numel()
    total_nats, total_bytes = nats.item(), n_bytes.item()
    return total_nats / n_tokens, total_nats / math.log(2) / total_bytes


@dataclass
class TrainResult:
    losses: list[float] = field(default_factory=list)
    val: list[tuple[int, float, float]] = field(default_factory=list)  # (step, loss, bpb) on the eval subset
    final_val_loss: float = float("nan")  # full validation split
    final_val_bpb: float = float("nan")


def train(
    model: GPT,
    train_data: Batcher,
    val_data: Batcher,
    tokenizer: BPETokenizer,
    cfg: TrainConfig,
    run=None,
) -> TrainResult:
    """Single pass over the train data (no repeated epochs), then a full eval on the validation split.

    `run` is an optional wandb run.
    """
    # Optimizers use the uncompiled model (compiled wrappers prefix param names with `_orig_mod.`).
    optimizers = build_optimizers(model, cfg)
    n_steps = train_data.steps_per_epoch
    schedulers = [
        torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: lr_factor(s, n_steps, cfg.warmup_frac, cfg.decay_frac)
        )
        for opt in optimizers
    ]
    compiled = torch.compile(model, mode=cfg.compile_mode) if cfg.compile_mode != "none" else model
    device = next(model.parameters()).device
    byte_lengths = torch.from_numpy(tokenizer.byte_lengths).to(device)

    def fwd(x, y):  # bf16 autocast: enables the FlashAttention SDPA kernel and faster matmuls
        with torch.autocast(device.type, dtype=torch.bfloat16):
            return compiled(x, y)

    tokens_per_step = train_data.block_size * train_data.batch_size
    result = TrainResult()
    # Per-step logs wait here with the loss still on the GPU; they're synced once per log_every steps.
    pending: list[tuple[dict, torch.Tensor]] = []

    def flush(step: int, elapsed: float) -> None:
        values = torch.stack([loss for _, loss in pending]).tolist()
        for (log, _), v in zip(pending, values):
            log["train/loss"] = v
            result.losses.append(v)
            if run is not None:
                run.log(log)
        pending.clear()
        steps_per_sec = step / elapsed
        eta = int((n_steps - step) / steps_per_sec)
        msg = f"step {step}/{n_steps} | loss {result.losses[-1]:.4f}"
        if result.val:
            msg += f" | val {result.val[-1][1]:.4f} ({result.val[-1][2]:.4f} bpb)"
        msg += f" | {steps_per_sec:.2f} step/s | eta {eta // 3600}:{eta % 3600 // 60:02d}:{eta % 60:02d}"
        print(msg, flush=True)

    model.train()
    t0 = time.time()
    for step, (x, y) in enumerate(train_data.train_batches(), start=1):
        _, loss = fwd(x, y)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        lr = optimizers[0].param_groups[0]["lr"]
        for opt in optimizers:
            opt.step()
        for sched in schedulers:
            sched.step()
        # clone: with CUDA graphs (reduce-overhead) the output buffer is overwritten by the next step
        log = {"tokens_seen": step * tokens_per_step, "lr": lr}
        pending.append((log, loss.detach().clone()))

        if cfg.eval_every > 0 and (step % cfg.eval_every == 0 or step == n_steps):
            val_loss, val_bpb = evaluate(fwd, val_data.eval_batches(cfg.eval_batches), byte_lengths)
            result.val.append((step, val_loss, val_bpb))
            log.update({"val/loss": val_loss, "val/bpb": val_bpb})

        if step % cfg.log_every == 0 or step == n_steps:
            flush(step, time.time() - t0)

    result.final_val_loss, result.final_val_bpb = evaluate(fwd, val_data.eval_batches(), byte_lengths)
    print(f"final full-val loss {result.final_val_loss:.4f} | {result.final_val_bpb:.4f} bpb", flush=True)
    if run is not None:
        final = {"final/val_loss": result.final_val_loss, "final/val_bpb": result.final_val_bpb}
        run.log(final)
        run.summary.update(final)
    return result


def save_checkpoint(out_dir: Path, model: GPT, cfg: RunConfig, tokenizer: BPETokenizer, result: TrainResult) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    final = {"val_loss": result.final_val_loss, "val_bpb": result.final_val_bpb}
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "final": final}, out_dir / "model.pt")
    tokenizer.save(out_dir / "tokenizer.json")
    (out_dir / "config.json").write_text(json.dumps({"config": cfg.to_dict(), "final": final}, indent=2))


def load_checkpoint(ckpt_dir: str | Path, device: str = "cpu") -> tuple[GPT, BPETokenizer, RunConfig]:
    ckpt_dir = Path(ckpt_dir)
    ckpt = torch.load(ckpt_dir / "model.pt", map_location=device, weights_only=True)
    cfg = RunConfig.from_dict(ckpt["config"])
    model = GPT(ModelConfig(**ckpt["config"]["model"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, BPETokenizer.load(ckpt_dir / "tokenizer.json"), cfg


def plot_losses(result: TrainResult, tokens_per_step: int, path: Path) -> None:
    """Save train/val loss vs tokens seen to `path` (no plt.show, which blocks scripted runs)."""
    fig, ax = plt.subplots()
    ax.plot([(i + 1) * tokens_per_step for i in range(len(result.losses))], result.losses, label="train")
    if result.val:
        ax.plot([s * tokens_per_step for s, _, _ in result.val], [v for _, v, _ in result.val], "o-", label="val")
    ax.legend()
    ax.set_xlabel("tokens seen")
    ax.set_ylabel("loss (nats/token)")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
