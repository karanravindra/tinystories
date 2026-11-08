import json
import math
import signal
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from tinystories.config import ModelConfig, RunConfig, TrainConfig
from tinystories.data import Batcher
from tinystories.model import GPT
from tinystories.tokenizer import BPETokenizer


def build_optimizers(model: GPT, cfg: TrainConfig) -> list[torch.optim.Optimizer]:
    """Optimizers, each param group tagged with its peak lr ("base_lr") for the schedule."""
    if cfg.optimizer == "adamw":
        opts = [torch.optim.AdamW(model.parameters(), lr=cfg.lr)]
    else:
        opts = _muon_and_adamw(model, cfg)
    for opt in opts:
        for g in opt.param_groups:
            g["base_lr"] = g["lr"]
    return opts


def _muon_and_adamw(model: GPT, cfg: TrainConfig) -> list[torch.optim.Optimizer]:

    # Muon only handles 2D hidden weight matrices. Embeddings, the output head,
    # norms and biases go to AdamW.
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        is_hidden_matrix = p.ndim == 2 and not name.startswith(("tok_emb", "head", "value_embeds"))
        (muon_params if is_hidden_matrix else adamw_params).append(p)
    return [
        torch.optim.Muon(muon_params, lr=cfg.muon_lr),
        torch.optim.AdamW(adamw_params, lr=cfg.lr),
    ]


def lr_factor(step: int, total: int, warmup_frac: float, decay_frac: float, min_frac: float = 0.0) -> float:
    """Warmup-stable-decay: linear warmup, flat, then linear decay to min_frac (default 0) over the last decay_frac
    of steps.

    `step` is 0-based (the LR used for the (step+1)-th update).
    """
    warmup = max(1, round(warmup_frac * total))
    decay = max(1, round(decay_frac * total))
    if step < warmup:
        return (step + 1) / warmup
    if step >= total - decay:
        return min_frac + (1 - min_frac) * max(0.0, (total - step) / decay)
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


@contextmanager
def stop_requests() -> Iterator[list[int]]:
    """Turn Ctrl-C / SIGTERM into a request (the signal number, appended to the yielded list) that the training loop
    honors after the current step, so it can save a resumable state first. A second signal stops at once."""
    requested: list[int] = []

    def handler(signum, frame):
        if requested:
            raise KeyboardInterrupt
        requested.append(signum)
        print(f"\n{signal.Signals(signum).name}: stopping after this step (saving state first); again to force", flush=True)

    old = {sig: signal.signal(sig, handler) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield requested
    finally:
        for sig, h in old.items():
            signal.signal(sig, h)


@dataclass
class TrainState:
    """Where a run is, for resuming: optimizer states, steps done, history so far and train seconds so far."""

    step: int = 0
    optimizers: list[dict] = field(default_factory=list)
    result: TrainResult = field(default_factory=TrainResult)
    train_seconds: float = 0.0


def train(
    model: GPT,
    train_data: Batcher,
    val_data: Batcher,
    tokenizer: BPETokenizer,
    cfg: TrainConfig,
    n_steps: int,
    run=None,
    resume: TrainState | None = None,
    save_state: Callable[..., None] | None = None,
) -> TrainResult:
    """n_steps of training (repeating the train data in reshuffled epochs if needed), then a full val eval.

    `run` is an optional wandb run (for a resumed run, the same wandb run resumed). `resume` continues a run from its
    saved state (model weights must already be loaded); `save_state` is called every cfg.save_every steps and at
    the end.
    """
    # Optimizers use the uncompiled model (compiled wrappers prefix param names with `_orig_mod.`).
    optimizers = build_optimizers(model, cfg)
    start_step, result, prior_seconds = 0, TrainResult(), 0.0
    if resume is not None:
        for opt, sd in zip(optimizers, resume.optimizers, strict=True):
            opt.load_state_dict(sd)
            for g in opt.param_groups:  # load_state_dict restores the old lrs; the schedule uses this run's
                g["base_lr"] = cfg.muon_lr if isinstance(opt, torch.optim.Muon) else cfg.lr
        start_step, result, prior_seconds = resume.step, resume.result, resume.train_seconds
        print(f"resumed at step {start_step}/{n_steps} ({prior_seconds / 60:.1f} train minutes so far)", flush=True)
    compiled = torch.compile(model, mode=cfg.compile_mode) if cfg.compile_mode != "none" else model
    device = next(model.parameters()).device
    byte_lengths = torch.from_numpy(tokenizer.byte_lengths).to(device)

    def fwd(x, y):  # bf16 autocast: enables the FlashAttention SDPA kernel and faster matmuls
        with torch.autocast(device.type, dtype=torch.bfloat16):
            return compiled(x, y)

    tokens_per_step = train_data.block_size * train_data.batch_size
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
        steps_per_sec = (step - start_step) / elapsed
        eta = int((n_steps - step) / steps_per_sec)
        msg = f"step {step}/{n_steps} | loss {result.losses[-1]:.4f}"
        if result.val:
            msg += f" | val {result.val[-1][1]:.4f} ({result.val[-1][2]:.4f} bpb)"
        msg += f" | {steps_per_sec:.2f} step/s | eta {eta // 3600}:{eta % 3600 // 60:02d}:{eta % 60:02d}"
        print(msg, flush=True)

    def snapshot(step: int) -> TrainState:
        return TrainState(step, [opt.state_dict() for opt in optimizers], result, prior_seconds + time.time() - t0)

    model.train()
    t0 = time.time()
    batches = train_data.train_batches(n_steps, start_step)
    with stop_requests() as stop:
        for step, (x, y) in enumerate(batches, start=start_step + 1):
            _, loss = fwd(x, y)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            loss.backward()
            f = lr_factor(step - 1, n_steps, cfg.warmup_frac, cfg.decay_frac, cfg.min_lr_frac)
            for opt in optimizers:
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * f
                opt.step()
            lr = optimizers[0].param_groups[0]["lr"]
            # clone: with CUDA graphs (reduce-overhead) the output buffer is overwritten by the next step
            log = {"tokens_seen": step * tokens_per_step, "lr": lr, "train/seconds": prior_seconds + time.time() - t0}
            pending.append((log, loss.detach().clone()))

            if cfg.eval_every > 0 and (step % cfg.eval_every == 0 or step == n_steps):
                val_loss, val_bpb = evaluate(fwd, val_data.eval_batches(cfg.eval_batches), byte_lengths)
                result.val.append((step, val_loss, val_bpb))
                log.update({"val/loss": val_loss, "val/bpb": val_bpb})

            if step % cfg.log_every == 0 or step == n_steps:
                flush(step, time.time() - t0)

            if save_state is not None and step in cfg.snapshot_steps:
                if pending:
                    flush(step, time.time() - t0)
                save_state(snapshot(step), f"state-{step}.pt")

            if cfg.stop_after and step >= cfg.stop_after and step < n_steps:
                if pending:
                    flush(step, time.time() - t0)
                if save_state is not None:
                    save_state(snapshot(step), f"state-{step}.pt")
                print(f"stopped after step {step} (--train.stop-after); branch from state-{step}.pt", flush=True)
                raise SystemExit(0)

            if save_state is not None and cfg.save_every > 0 and step % cfg.save_every == 0 and step < n_steps:
                if pending:
                    flush(step, time.time() - t0)
                save_state(snapshot(step))

            if stop and step < n_steps:
                if pending:
                    flush(step, time.time() - t0)
                if save_state is not None:
                    save_state(snapshot(step))
                    print(f"saved state at step {step}; continue with --resume", flush=True)
                raise SystemExit(128 + stop[0])  # 130 for Ctrl-C, 143 for SIGTERM; wandb flushes on the way out

    if save_state is not None:
        save_state(snapshot(n_steps))
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


def write_state(
    out_dir: Path, model: GPT, state: TrainState, cfg: RunConfig, train_windows: int, wandb_id: str | None,
    name: str = "state.pt",
) -> None:
    """Save `state` with its run config to <out_dir>/<name> (via a temp file, so a crash never leaves half a file)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    r = state.result
    blob = {
        "model": model.state_dict(),
        "step": state.step,
        "optimizers": state.optimizers,
        "train_seconds": state.train_seconds,
        "losses": r.losses,
        "val": r.val,
        "config": cfg.to_dict(),
        "train_windows": train_windows,  # the shuffled pool; resuming needs the same one to continue the order
        "wandb_id": wandb_id,  # resuming continues this wandb run, so its curves (and _runtime) carry on
    }
    tmp = out_dir / f"{name}.tmp"
    torch.save(blob, tmp)
    tmp.replace(out_dir / name)


def _drop_empty_params(sd: dict) -> dict:
    """An optimizer state dict without the empty ve_gains of older runs, renumbered. That param never had a gradient,
    so it is the one param with no optimizer state (every other param has state after the first step)."""
    empty = {i for g in sd["param_groups"] for i in g["params"] if i not in sd["state"]}
    assert len(empty) <= 1, f"expected at most one param without optimizer state, found {len(empty)}"
    if not empty:
        return sd
    old_ids = [i for g in sd["param_groups"] for i in g["params"] if i not in empty]
    new_id = {old: new for new, old in enumerate(old_ids)}
    groups = [{**g, "params": [new_id[i] for i in g["params"] if i not in empty]} for g in sd["param_groups"]]
    return {"state": {new_id[i]: st for i, st in sd["state"].items() if i not in empty}, "param_groups": groups}


def read_state(path: Path, device: str) -> tuple[TrainState, dict, RunConfig, int, str | None]:
    """(train state, model weights, run config, train windows, wandb run id) saved by a run, for resuming it.

    `path` is a run dir (its state.pt) or a state file."""
    path = Path(path)
    blob = torch.load(path / "state.pt" if path.is_dir() else path, map_location=device, weights_only=True)
    result = TrainResult(losses=blob["losses"], val=[tuple(v) for v in blob["val"]])
    state = TrainState(blob["step"], blob["optimizers"], result, blob["train_seconds"])
    weights = {k: v for k, v in blob["model"].items() if not (k == "ve_gains" and v.numel() == 0)}  # see load_checkpoint
    if len(weights) < len(blob["model"]):
        state.optimizers = [_drop_empty_params(sd) for sd in state.optimizers]
    return state, weights, RunConfig.from_dict(blob["config"]), blob["train_windows"], blob["wandb_id"]


def load_checkpoint(ckpt_dir: str | Path, device: str = "cpu") -> tuple[GPT, BPETokenizer, RunConfig]:
    ckpt_dir = Path(ckpt_dir)
    ckpt = torch.load(ckpt_dir / "model.pt", map_location=device, weights_only=True)
    cfg = RunConfig.from_dict(ckpt["config"])
    model = GPT(ModelConfig(**ckpt["config"]["model"])).to(device)
    # Runs from before value embeddings were optional saved an empty ve_gains; drop it.
    weights = {k: v for k, v in ckpt["model"].items() if not (k == "ve_gains" and v.numel() == 0)}
    model.load_state_dict(weights)
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
