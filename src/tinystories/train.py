import time

import matplotlib.pyplot as plt
import torch

from tinystories.data import TinyStoriesData
from tinystories.model import GPT


def build_optimizers(model: GPT, lr: float, muon_lr: float, optimizer: str = "muon") -> list[torch.optim.Optimizer]:
    if optimizer == "adamw":
        return [torch.optim.AdamW(model.parameters(), lr=lr)]

    # Muon only handles 2D hidden weight matrices. Embeddings, the output head,
    # norms and biases go to AdamW.
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        is_hidden_matrix = p.ndim == 2 and not name.startswith(("tok_emb", "head"))
        (muon_params if is_hidden_matrix else adamw_params).append(p)
    return [
        torch.optim.Muon(muon_params, lr=muon_lr),
        torch.optim.AdamW(adamw_params, lr=lr),
    ]


@torch.no_grad()
def evaluate(model: GPT, fwd, data: TinyStoriesData, n_batches: int) -> float:
    model.eval()
    losses = [fwd(x, y)[1].item() for x, y in data.val_batches(n_batches)]
    model.train()
    return sum(losses) / len(losses)


def train(
    model: GPT,
    data: TinyStoriesData,
    lr: float = 3e-4,
    muon_lr: float = 0.02,
    optimizer: str = "muon",
    eval_every: int = 100,
    eval_batches: int = 20,
    log_every: int = 50,
    compile_mode: str | None = None,
    run=None,
) -> tuple[list[float], list[tuple[int, float]]]:
    """Single pass over the train data (no repeated epochs).

    `lr` is for AdamW (and for embeddings/head/norms/biases under Muon); `muon_lr` is for the hidden matrices.
    Returns (train losses per step, [(step, val loss)]). `run` is an optional wandb run.
    """
    # Optimizers use the uncompiled model (compiled wrappers prefix param names with `_orig_mod.`).
    optimizers = build_optimizers(model, lr, muon_lr, optimizer)
    fwd = torch.compile(model, mode=compile_mode) if compile_mode else model
    tokens_per_step = data.block_size * data.batch_size
    n_steps = data.steps_per_epoch
    losses: list[float] = []
    val_losses: list[tuple[int, float]] = []

    model.train()
    t0 = time.time()
    for x, y in data.train_batches():
        _, loss = fwd(x, y)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers:
            opt.step()
        losses.append(loss.item())
        step = len(losses)
        tokens_seen = step * tokens_per_step
        log = {"train/loss": losses[-1], "tokens_seen": tokens_seen}

        if eval_every > 0 and (step % eval_every == 0 or step == n_steps):
            val_losses.append((step, evaluate(model, fwd, data, eval_batches)))
            log["val/loss"] = val_losses[-1][1]

        if run is not None:
            run.log(log)

        if step % log_every == 0 or step == n_steps:
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            eta = int((n_steps - step) / steps_per_sec)
            msg = f"step {step}/{n_steps} | loss {losses[-1]:.4f}"
            if val_losses:
                msg += f" | val {val_losses[-1][1]:.4f}"
            msg += f" | {steps_per_sec:.2f} step/s | eta {eta // 3600}:{eta % 3600 // 60:02d}:{eta % 60:02d}"
            print(msg, flush=True)
    return losses, val_losses


def plot_losses(losses: list[float], tokens_per_step: int, val_losses: list[tuple[int, float]] | None = None) -> None:
    plt.plot([(i + 1) * tokens_per_step for i in range(len(losses))], losses, label="train")
    if val_losses:
        plt.plot([s * tokens_per_step for s, _ in val_losses], [v for _, v in val_losses], "o-", label="val")
        plt.legend()
    plt.xlabel("tokens seen")
    plt.ylabel("loss")
    plt.show()
