"""`tinystories train [--model.dim 384 ...]` and `tinystories sample <run dir>`."""

import time
from dataclasses import asdict
from pathlib import Path

import torch
import tyro
from torchinfo import summary

from tinystories.config import RunConfig
from tinystories.data import Batcher, load_or_train_tokenizer, load_tokens
from tinystories.model import GPT
from tinystories.sample import SampleArgs, generate_text, sample
from tinystories.tokenizer import BPETokenizer
from tinystories.train import (
    TrainResult,
    TrainState,
    lr_factor,
    plot_losses,
    read_state,
    save_checkpoint,
    train,
    write_state,
)


def prepare_data(cfg: RunConfig) -> tuple[BPETokenizer, Batcher, Batcher]:
    """Tokenizer, train batches (n_tokens of the train split) and val batches (the whole validation split)."""
    tokenizer, tok_dir = load_or_train_tokenizer(cfg.data)
    cfg.model.vocab_size = tokenizer.vocab_size
    cfg.model.eos_id = tokenizer.eos_id
    train_tokens = load_tokens("train", tokenizer, tok_dir, cfg.data.n_tokens, cfg.data.max_story_tokens)
    val_tokens = load_tokens("validation", tokenizer, tok_dir)
    args = (cfg.model.block_size, cfg.train.batch_size, cfg.device, cfg.train.seed)
    return tokenizer, Batcher(train_tokens, *args), Batcher(val_tokens, *args)


def build_model(cfg: RunConfig, example_input: torch.Tensor | None = None) -> GPT:
    torch.manual_seed(cfg.train.seed)
    model = GPT(cfg.model).to(cfg.device)
    if example_input is not None:
        summary(model, depth=2, input_data=example_input)
    return model


def init_wandb(cfg: RunConfig, model: GPT, resume_id: str | None = None, extra: dict | None = None):
    """A new wandb run, or with resume_id, that run continued (renamed to cfg.wandb_name, config updated)."""
    if not cfg.wandb:
        return None
    import wandb

    config = {**cfg.to_dict(), "params": sum(p.numel() for p in model.parameters()), **(extra or {})}
    if resume_id is None:
        run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_name, config=config)
    else:
        run = wandb.init(
            project=cfg.wandb_project, id=resume_id, resume="must", name=cfg.wandb_name, config=config,
            allow_val_change=True,
        )
    for metric in ("train/loss", "val/loss", "val/bpb", "lr", "train/seconds"):
        run.define_metric(metric, step_metric="tokens_seen")
    return run


def load_resume(
    cfg: RunConfig, train_data: Batcher, n_steps: int, source: Path | None = None
) -> tuple[TrainState, dict, str | None]:
    """State, weights and wandb run id to continue from `source` (default cfg.resume), after checking the run is the
    same apart from length and lrs."""
    source = Path(source if source is not None else cfg.resume)
    state, weights, old, old_windows, wandb_id = read_state(source, cfg.device)
    same = {
        "model": (asdict(old.model), asdict(cfg.model)),
        "vocab": (old.data.vocab_size, cfg.data.vocab_size),
        "max story tokens": (old.data.max_story_tokens, cfg.data.max_story_tokens),
        "optimizer": (old.train.optimizer, cfg.train.optimizer),
        "batch size": (old.train.batch_size, cfg.train.batch_size),
        "seed": (old.train.seed, cfg.train.seed),
        "train windows (data order)": (old_windows, train_data.n_windows),
    }
    diffs = [f"{k}: {a} -> {b}" for k, (a, b) in same.items() if a != b]
    if diffs:
        raise SystemExit(f"can't resume {source}, settings differ:\n  " + "\n  ".join(diffs))
    if n_steps < state.step:
        raise SystemExit(f"{source} is already at step {state.step}, past this run's {n_steps} steps")
    if cfg.branch_from is not None:
        # A branch is only the run it claims to be if both lr schedules agree on every step before the branch point.
        old_n = train_data.n_steps(old.data.n_tokens)
        t, o = cfg.train, old.train
        same_lr = (t.lr, t.muon_lr) == (o.lr, o.muon_lr) and all(
            lr_factor(i, n_steps, t.warmup_frac, t.decay_frac, t.min_lr_frac)
            == lr_factor(i, old_n, o.warmup_frac, o.decay_frac, o.min_lr_frac)
            for i in range(state.step)
        )
        if not same_lr:
            print(f"warning: this run's lr schedule differs from {source}'s before step {state.step}; the branch is "
                  "not the same as training this run from scratch")
        return state, weights, None
    for k in ("lr", "muon_lr", "warmup_frac", "decay_frac"):
        a, b = getattr(old.train, k), getattr(cfg.train, k)
        if a != b:
            print(f"warning: {k} changes on resume, {a} -> {b}")
    if cfg.wandb and wandb_id is None:
        print("warning: the run being resumed had no wandb run; logging to a new one, whose time axis starts at 0")
    return state, weights, wandb_id


def keep_old_final(run, tokens: int) -> None:
    """Before a finished run continues, copy its final full-val numbers to final_at_<tokens>M/*, so they survive."""
    for k in ("val_loss", "val_bpb"):
        v = run.summary.get(f"final/{k}")
        if v is not None:
            run.summary[f"final_at_{tokens // 1_000_000}M/{k}"] = v


def run_train(cfg: RunConfig) -> tuple[GPT, TrainResult]:
    tokenizer, train_data, val_data = prepare_data(cfg)
    n_steps = train_data.n_steps(cfg.data.n_tokens)
    model = build_model(cfg, next(val_data.eval_batches(1))[0])
    resume, resume_id, extra = None, None, None
    if cfg.resume is not None and cfg.branch_from is not None:
        raise SystemExit("use either --resume or --branch-from, not both")
    if cfg.resume is not None:
        resume, weights, resume_id = load_resume(cfg, train_data, n_steps)
        model.load_state_dict(weights)
        extra = {"resumed_from": Path(cfg.resume).name, "resumed_at_step": resume.step}
    elif cfg.branch_from is not None:
        resume, weights, _ = load_resume(cfg, train_data, n_steps, cfg.branch_from)
        model.load_state_dict(weights)
        branch = Path(cfg.branch_from)
        extra = {"branched_from": f"{branch.parent.name}/{branch.name}", "branched_at_step": resume.step}
    run = init_wandb(cfg, model, resume_id, extra)
    if run is not None and resume is not None:
        keep_old_final(run, resume.step * train_data.block_size * train_data.batch_size)
    run_name = cfg.wandb_name or (run.name if run is not None else time.strftime("%Y%m%d-%H%M%S"))
    out_dir = cfg.out_dir / run_name

    def save_state(state: TrainState, name: str = "state.pt") -> None:
        write_state(out_dir, model, state, cfg, train_data.n_windows, None if run is None else run.id, name)

    result = train(
        model, train_data, val_data, tokenizer, cfg.train, n_steps, run=run, resume=resume, save_state=save_state
    )

    save_checkpoint(out_dir, model, cfg, tokenizer, result)
    plot_losses(result, train_data.block_size * train_data.batch_size, out_dir / "loss.png")
    print(f"saved model, tokenizer, config and loss plot to {out_dir}/")

    model.eval()
    print(generate_text(model, tokenizer, max_new_tokens=cfg.sample_tokens)[0])
    if run is not None:
        run.finish()
    return model, result


def main() -> None:
    cmd = tyro.extras.subcommand_cli_from_dict({"train": RunConfig, "sample": SampleArgs})
    if isinstance(cmd, RunConfig):
        run_train(cmd)
    else:
        sample(cmd)
