"""`tinystories train [--model.dim 384 ...]` and `tinystories sample <run dir>`."""

import time

import torch
import tyro
from torchinfo import summary

from tinystories.config import RunConfig
from tinystories.data import Batcher, load_or_train_tokenizer, load_tokens
from tinystories.model import GPT
from tinystories.sample import SampleArgs, generate_text, sample
from tinystories.tokenizer import BPETokenizer
from tinystories.train import TrainResult, plot_losses, save_checkpoint, train


def prepare_data(cfg: RunConfig) -> tuple[BPETokenizer, Batcher, Batcher]:
    """Tokenizer, train batches (n_tokens of the train split) and val batches (the whole validation split)."""
    tokenizer, tok_dir = load_or_train_tokenizer(cfg.data)
    cfg.model.vocab_size = tokenizer.vocab_size
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


def init_wandb(cfg: RunConfig, model: GPT):
    if not cfg.wandb:
        return None
    import wandb

    run = wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_name,
        config={**cfg.to_dict(), "params": sum(p.numel() for p in model.parameters())},
    )
    for metric in ("train/loss", "val/loss", "val/bpb", "lr"):
        run.define_metric(metric, step_metric="tokens_seen")
    return run


def run_train(cfg: RunConfig) -> tuple[GPT, TrainResult]:
    tokenizer, train_data, val_data = prepare_data(cfg)
    model = build_model(cfg, next(val_data.eval_batches(1))[0])
    run = init_wandb(cfg, model)
    run_name = cfg.wandb_name or (run.name if run is not None else time.strftime("%Y%m%d-%H%M%S"))

    result = train(model, train_data, val_data, tokenizer, cfg.train, run=run)

    out_dir = cfg.out_dir / run_name
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
