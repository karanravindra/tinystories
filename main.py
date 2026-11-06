# %%
import argparse
from pathlib import Path

import torch
from torchinfo import summary

from tinystories.data import TinyStoriesData
from tinystories.model import GPT
from tinystories.train import plot_losses, train

default_device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Train a small GPT on TinyStories")
parser.add_argument("--device", default=default_device)
# data
parser.add_argument("--n-tokens", type=int, default=500_000_000, help="total tokens (90%% train, seen once; 10%% val)")
parser.add_argument("--vocab-size", type=int, default=8192, help="BPE vocab size")
parser.add_argument("--cache-dir", default="cache", help="cache for tokenizer + tokenized data")
parser.add_argument("--rebuild-cache", action="store_true", help="ignore and overwrite cached tokenizer/tokens")
parser.add_argument("--max-story-tokens", type=int, default=1024, help="drop stories longer than this many tokens (0 = keep all)")
parser.add_argument("--block-size", type=int, default=512)
parser.add_argument("--batch-size", type=int, default=64)
# model
parser.add_argument("--dim", type=int, default=384)
parser.add_argument("--n-heads", type=int, default=8)
parser.add_argument("--n-layers", type=int, default=4)
# train
parser.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")
parser.add_argument("--lr", type=float, default=1e-3, help="AdamW lr (also used for embeddings/head/norms under muon)")
parser.add_argument("--muon-lr", type=float, default=0.02, help="Muon lr for hidden weight matrices")
parser.add_argument(
    "--compile-mode",
    choices=["none", "default", "reduce-overhead", "max-autotune"],
    default="reduce-overhead",
    help="torch.compile mode for training/eval (none disables). reduce-overhead uses CUDA graphs (no-op on MPS)",
)
parser.add_argument("--eval-every", type=int, default=100, help="steps between val evals (0 disables)")
parser.add_argument("--eval-batches", type=int, default=20, help="val batches averaged per eval")
# sample
parser.add_argument("--out-dir", default="checkpoints", help="where to save model.pt and tokenizer.json")
parser.add_argument("--sample-tokens", type=int, default=800)
# logging
parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
parser.add_argument("--wandb-project", default="tinystories")
parser.add_argument("--wandb-name", default=None, help="run name (default: wandb picks one)")
# parse_known_args so running cells interactively (extra kernel argv) still works
args, _ = parser.parse_known_args()
device = args.device

# %% data
global_batch_size = args.block_size * args.batch_size
data = TinyStoriesData(args.block_size, args.batch_size, device, n_tokens=args.n_tokens, vocab_size=args.vocab_size,
    cache_dir=args.cache_dir, rebuild_cache=args.rebuild_cache,
    max_story_tokens=args.max_story_tokens)

# %% model
model = GPT(
    data.tokenizer.vocab_size, args.block_size, dim=args.dim, n_heads=args.n_heads, n_layers=args.n_layers
).to(device)
summary(model, depth=2, input_data=data.get_batch("train")[0])

# %% train
run = None
if args.wandb:
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config={**vars(args), "vocab_size_actual": data.tokenizer.vocab_size, "params": sum(p.numel() for p in model.parameters())},
    )
    run.define_metric("train/loss", step_metric="tokens_seen")
    run.define_metric("val/loss", step_metric="tokens_seen")

losses, val_losses = train(
    model,
    data,
    lr=args.lr,
    muon_lr=args.muon_lr,
    optimizer=args.optimizer,
    eval_every=args.eval_every,
    eval_batches=args.eval_batches,
    compile_mode=None if args.compile_mode == "none" else args.compile_mode,
    run=run,
)

# save before plotting (plt.show can block) so a crash later never loses the model
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "model": model.state_dict(),
        "config": {
            "vocab_size": data.tokenizer.vocab_size,
            "block_size": args.block_size,
            "dim": args.dim,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
        },
        "args": vars(args),
    },
    out_dir / "model.pt",
)
data.tokenizer.save(out_dir / "tokenizer.json")
print(f"saved model and tokenizer to {out_dir}/")

plot_losses(losses, global_batch_size, val_losses)

# %% sample
model.eval()
start = torch.zeros((1, 1), dtype=torch.long, device=device)
print(data.tokenizer.decode(model.generate(start, args.sample_tokens)[0].tolist()))

if run is not None:
    run.finish()
