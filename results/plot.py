"""Pull the sweeps from wandb and draw the figures used in RESULTS.md.

    uv run python results/plot.py            # fetch from wandb (cached to results/runs.json), then plot
    uv run python results/plot.py --cached   # replot from the cache
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OUT = Path(__file__).parent
CACHE = OUT / "runs.json"
GROUPS = ["vocab-d192-oldlr", "vocab-d192-tunedlr", "lr-d192-v16k-250M", "lr-d192-v16k-500M"]

# SciencePlots style (https://github.com/garrettj403/SciencePlots). "no-latex": no TeX install needed.
import scienceplots  # noqa: F401,E402  (registers the styles)

plt.style.use(["science", "no-latex", "grid"])
plt.rcParams.update({"savefig.dpi": 300, "legend.frameon": True, "legend.framealpha": 1, "legend.edgecolor": "white", "legend.fancybox": False, "axes.titlesize": 8})
SERIES = plt.rcParams["axes.prop_cycle"].by_key()["color"]
BLUE_RAMP = [plt.get_cmap("Blues")(x) for x in np.linspace(0.1, 0.95, 7)]
SURFACE, INK, INK_2 = "white", "black", "0.35"


def fetch() -> list[dict]:
    import wandb

    api = wandb.Api()
    runs = []
    for group in GROUPS:
        for r in api.runs(f"{api.default_entity}/tinystories", filters={"group": group}):
            if r.state != "finished" or any(x["id"] == r.id for x in runs):
                continue
            h = r.history(samples=100000, keys=["val/bpb", "tokens_seen", "_runtime", "_step"], pandas=True).dropna()
            c = r.config
            runs.append({
                "id": r.id, "name": r.name, "group": r.group,
                "vocab": c["data"]["vocab_size"], "n_tokens": c["data"]["n_tokens"],
                "lr": c["train"]["lr"], "muon_lr": c["train"]["muon_lr"], "seed": c["train"]["seed"],
                "final_bpb": r.summary["final/val_bpb"],
                "step": h["_step"].tolist(), "tokens": h["tokens_seen"].tolist(),
                "time": h["_runtime"].tolist(), "bpb": h["val/bpb"].tolist(),
            })
    CACHE.write_text(json.dumps(runs))
    return runs


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def vocab_runs(runs, group):
    """One run per vocab. The tuned sweep reuses the 16k run from the 500M LR check (same settings)."""
    if group == "vocab-d192-oldlr":
        sel = [r for r in runs if r["group"] == group]
    else:
        sel = [r for r in runs if r["n_tokens"] == 500_000_000 and r["lr"] == 0.02 and r["muon_lr"] == 0.0018
               and r["seed"] == 0 and r["group"] in ("vocab-d192-tunedlr", "lr-d192-v16k-500M")]
    return {r["vocab"]: r for r in sorted(sel, key=lambda r: r["vocab"])}


def plot_vocab(runs):
    old, new = vocab_runs(runs, "vocab-d192-oldlr"), vocab_runs(runs, "vocab-d192-tunedlr")
    vocabs = sorted(old)
    color = {v: SERIES[i] for i, v in enumerate(vocabs)}

    # Learning curves: old LR vs tuned LR, same axes.
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
    for ax, sweep, title in [(axes[0], old, "(a) Old lr: AdamW 0.001, Muon 0.02"),
                             (axes[1], new, "(b) Tuned lr: AdamW 0.02, Muon 0.0018")]:
        for v, r in sweep.items():
            ax.plot(np.array(r["tokens"]) / 1e6, r["bpb"], color=color[v], label=f"{v:,}")
        ax.set_title(title)
        ax.set_xlabel("Training tokens (millions)")
        ax.set_ylim(0.6, 1.2)
    axes[0].set_ylabel("Validation bpb (lower is better)")
    axes[0].legend(title="Vocab size", loc="upper right")
    save(fig, "vocab_curves.png")

    # Final score by vocab size, old vs tuned LR.
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for i, (sweep, label) in enumerate([(old, "Old learning rates"), (new, "Tuned learning rates")]):
        vs = [v for v in vocabs if v in sweep]
        ys = [sweep[v]["final_bpb"] for v in vs]
        ax.plot(vs, ys, color=SERIES[i], marker="o", markersize=4, label=label)
        for x, y in zip(vs, ys):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", color=INK_2, fontsize=6)
    ax.set_xscale("log", base=2)
    ax.set_xticks(vocabs, [f"{v:,}" for v in vocabs])
    ax.minorticks_off()
    ax.set_xlabel("Vocabulary size")
    ax.set_ylabel("Final validation bpb (lower is better)")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.01)
    ax.legend()
    save(fig, "vocab_final.png")

    # Score vs wall-clock time (old sweep has wall-clock; tuned sweep too).
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
    for ax, sweep, title in [(axes[0], old, "(a) Old lr"), (axes[1], new, "(b) Tuned lr")]:
        for v, r in sweep.items():
            ax.plot(np.array(r["time"]) / 60, r["bpb"], color=color[v], label=f"{v:,}")
        ax.set_title(title)
        ax.set_xlabel("Training time (minutes)")
        ax.set_ylim(0.6, 1.2)
    axes[0].set_ylabel("Validation bpb (lower is better)")
    axes[0].legend(title="Vocab size", loc="upper right")
    save(fig, "vocab_time.png")


def lr_grid(runs, n_tokens):
    cells = defaultdict(list)
    for r in runs:
        if r["group"].startswith("lr-d192") and r["n_tokens"] == n_tokens and r["vocab"] == 16384:
            cells[(r["lr"], r["muon_lr"])].append(r)
    return cells


def plot_heatmap(cells, title, name):
    lrs = sorted({k[0] for k in cells})
    mus = sorted({k[1] for k in cells})
    z = np.full((len(lrs), len(mus)), np.nan)
    for (lr, m), rs in cells.items():
        z[lrs.index(lr), mus.index(m)] = np.mean([r["final_bpb"] for r in rs])
    best = np.nanmin(z)
    # Darker = better. Clip the color scale so the near-optimal region still shows contrast.
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP[::-1])
    cmap.set_bad(SURFACE)
    vmax = min(np.nanmax(z), best + 0.08)
    fig, ax = plt.subplots(figsize=(0.55 * len(mus) + 1.4, 0.32 * len(lrs) + 1.0))
    ax.imshow(np.ma.masked_invalid(z), cmap=cmap, vmin=best, vmax=vmax, aspect="auto")
    for i in range(len(lrs)):
        for j in range(len(mus)):
            if np.isnan(z[i, j]):
                continue
            n = len(cells[(lrs[i], mus[j])])
            dark = (z[i, j] - best) / (vmax - best) < 0.5
            txt = f"{z[i, j]:.{4 if np.nanmax(z) - best < 0.02 else 3}f}" + ("*" if n > 1 else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=6, family="DejaVu Sans",
                    color="white" if dark else INK, fontweight="bold" if z[i, j] == best else "normal")
    ax.set_xticks(range(len(mus)), [f"{m:g}" for m in mus])
    ax.set_yticks(range(len(lrs)), [f"{lr:g}" for lr in lrs])
    ax.set_xlabel("Muon lr (hidden weight matrices)")
    ax.set_ylabel("AdamW lr (embeddings, head)")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.minorticks_off()
    ax.tick_params(length=0)
    save(fig, name)


def plot_lr_vs_length(runs):
    """With a constant LR, a run's score at T tokens equals what a T-token run would score.
    So one set of long runs shows the best LR at every shorter length."""
    cells = lr_grid(runs, 500_000_000)
    short = lr_grid(runs, 250_000_000)
    lengths = [25e6, 50e6, 100e6, 200e6, 420e6]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    for ax, axis, fixed_name in [(axes[0], 0, "Muon lr"), (axes[1], 1, "AdamW lr")]:
        for i, T in enumerate(lengths):
            src = short if T <= 245e6 else cells
            fixed = 0.0025 if (axis == 0 and src is short) else 0.0018 if axis == 0 else 0.03 if src is short else 0.02
            pts = {}
            for (lr, m), rs in src.items():
                if (m if axis == 0 else lr) != fixed:
                    continue
                x = lr if axis == 0 else m
                pts[x] = np.mean([np.interp(T, r["tokens"], r["bpb"]) for r in rs])
            xs = sorted(pts)
            if len(xs) < 3:
                continue
            ys = np.array([pts[x] for x in xs])
            ax.plot(xs, ys - ys.min(), color=SERIES[i], marker="o", markersize=3.5, label=f"{T/1e6:.0f}M")
            k = int(np.argmin(ys))
            ax.plot(xs[k], 0, marker="o", markersize=6, color=SERIES[i], markeredgecolor=SURFACE, markeredgewidth=1)
        ax.set_xscale("log")
        ax.set_ylim(-0.003, 0.05)
        ax.set_ylabel("bpb above the best at that length")
        ax.set_xlabel("AdamW lr" if axis == 0 else "Muon lr")
        ax.set_title("(a) AdamW lr sweep" if axis == 0 else "(b) Muon lr sweep")
    axes[1].legend(title="Tokens trained", loc="upper right")
    save(fig, "lr_vs_length.png")


if __name__ == "__main__":
    runs = json.loads(CACHE.read_text()) if "--cached" in sys.argv else fetch()
    plot_vocab(runs)
    plot_heatmap(lr_grid(runs, 250_000_000), "Learning-rate search at 250M tokens (final bpb)", "lr_250M.png")
    plot_heatmap(lr_grid(runs, 500_000_000), "Learning-rate check at 500M tokens (final bpb)", "lr_500M.png")
    plot_lr_vs_length(runs)
