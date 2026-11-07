import functools
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

from tinystories.config import DataConfig
from tinystories.tokenizer import BPETokenizer

TOKENIZER_TRAIN_STORIES = 200_000  # supports vocabs up to ~42k (20k stories capped out at 20,292)
TOKENIZER_VERSION = 1  # bump when BPE training settings change, so old caches aren't reused


@functools.cache
def load_split(name: str):
    return load_dataset("roneneldan/TinyStories", split=name)


def _token_dtype(vocab_size: int):
    return np.uint16 if vocab_size <= 2**16 else np.int32


def load_or_train_tokenizer(cfg: DataConfig) -> tuple[BPETokenizer, Path]:
    """BPE trained on a fixed sample of train stories (independent of n_tokens).

    Returns the tokenizer and its cache dir, which also holds the token streams made with it.
    """
    tok_dir = cfg.cache_dir / f"tok_v{cfg.vocab_size}_s{TOKENIZER_TRAIN_STORIES}_r{TOKENIZER_VERSION}"
    if cfg.rebuild_cache and tok_dir.exists():
        shutil.rmtree(tok_dir)
    tok_dir.mkdir(parents=True, exist_ok=True)
    path = tok_dir / "tokenizer.json"
    if path.exists():
        print(f"loaded cached tokenizer {path}")
        return BPETokenizer.load(path), tok_dir
    tokenizer = BPETokenizer.train(load_split("train")[:TOKENIZER_TRAIN_STORIES]["text"], cfg.vocab_size)
    tokenizer.save(path)
    return tokenizer, tok_dir


def _tokenize(
    split, tokenizer: BPETokenizer, n_tokens: int | None, max_story_tokens: int = 0, start_story: int = 0
) -> tuple[np.ndarray, int, bool]:
    """Stream stories from `start_story`, packing them (each followed by EOS) into one token stream.

    Stops once >= n_tokens (None = whole split); stories longer than max_story_tokens (excluding EOS) are
    dropped, 0 disables the filter. Returns (tokens, stories consumed, whether the split ran out).
    """
    dtype = _token_dtype(tokenizer.vocab_size)
    eos = tokenizer.eos_id
    chunks: list[np.ndarray] = []
    total = consumed = 0
    exhausted = True  # unless we stop early below
    if start_story:
        split = split.select(range(min(start_story, len(split)), len(split)))
    for batch in split.iter(batch_size=1000):
        stories = batch["text"]
        encoded = tokenizer.encode_batch(stories)
        if max_story_tokens:
            encoded = [e for e in encoded if len(e) <= max_story_tokens]
        chunk = np.fromiter(itertools.chain.from_iterable(itertools.chain(s, (eos,)) for s in encoded), dtype=dtype)
        chunks.append(chunk)
        total += len(chunk)
        consumed += len(stories)
        if n_tokens is not None and total >= n_tokens:
            exhausted = False
            break
    return (np.concatenate(chunks) if chunks else np.empty(0, dtype=dtype)), consumed, exhausted


class TokenCache:
    """Append-only token stream cache.

    Asking for more tokens than are cached tokenizes only the stories after the ones already consumed and
    appends them, so a 250M cache is extended to 500M rather than rebuilt. `<name>.json` is the source of
    truth (written after the data), so a crash mid-append leaves at most some ignored trailing bytes.
    It records the tokenizer hash (mismatch = stale cache) and whether the dataset split ran out.
    """

    def __init__(self, directory: Path, name: str, dtype, tokenizer_sha256: str) -> None:
        self.bin = directory / f"{name}.bin"
        self.meta = directory / f"{name}.json"
        self.dtype = np.dtype(dtype)
        self.sha = tokenizer_sha256

    def state(self) -> dict:
        if not (self.bin.exists() and self.meta.exists()):
            return {"tokens": 0, "stories": 0, "exhausted": False, "tokenizer_sha256": self.sha}
        m = json.loads(self.meta.read_text())
        if m.get("tokenizer_sha256") != self.sha:
            raise RuntimeError(
                f"{self.bin} was built with a different tokenizer; rerun with --data.rebuild-cache"
            )
        return m

    def append(self, tokens: np.ndarray, stories: int, exhausted: bool) -> None:
        m = self.state()
        with open(self.bin, "r+b" if self.bin.exists() else "wb") as f:
            f.truncate(m["tokens"] * self.dtype.itemsize)  # drop any orphaned bytes from an interrupted append
            f.seek(0, 2)
            tokens.astype(self.dtype, copy=False).tofile(f)
        meta = {
            "tokens": m["tokens"] + len(tokens),
            "stories": m["stories"] + stories,
            "exhausted": exhausted,
            "tokenizer_sha256": self.sha,
        }
        tmp = self.meta.with_name(self.meta.name + ".tmp")
        tmp.write_text(json.dumps(meta))
        tmp.replace(self.meta)

    def load(self) -> np.ndarray:
        return np.memmap(self.bin, dtype=self.dtype, mode="r", shape=(self.state()["tokens"],))


def load_tokens(
    split: str, tokenizer: BPETokenizer, tok_dir: Path, n_tokens: int | None = None, max_story_tokens: int = 0
) -> np.ndarray:
    """Cached token stream for a dataset split: the first n_tokens (None = the whole split)."""
    name = "val" if split == "validation" else f"{split}_max{max_story_tokens}"
    cache = TokenCache(tok_dir, name, _token_dtype(tokenizer.vocab_size), tokenizer.sha256)
    m = cache.state()
    want = n_tokens if n_tokens is not None else float("inf")
    if m["tokens"] < want and not m["exhausted"]:
        if m["tokens"]:
            print(f"extending {name} token cache from {m['tokens']:,} tokens (resuming at story {m['stories']:,})")
        need = None if n_tokens is None else n_tokens - m["tokens"]
        new, consumed, exhausted = _tokenize(load_split(split), tokenizer, need, max_story_tokens, m["stories"])
        cache.append(new, consumed, exhausted)
        print(f"cached {len(new):,} new tokens in {cache.bin}")
    data = cache.load()
    if n_tokens is not None:
        if len(data) < n_tokens:
            print(f"requested {n_tokens:,} {split} tokens, dataset has {len(data):,}; training repeats it in reshuffled epochs")
        data = data[:n_tokens]
    print(f"loaded {split} tokens {cache.bin} ({len(data):,} tokens)")
    return data


class Batcher:
    """Non-overlapping (block_size + 1)-token windows from a token stream, as (x, y) batches on `device`."""

    def __init__(self, tokens: np.ndarray, block_size: int, batch_size: int, device: str, seed: int = 0) -> None:
        self.tokens = tokens
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.n_windows = (len(tokens) - 1) // block_size
        self.pin = torch.device(device).type == "cuda"

    @property
    def steps_per_epoch(self) -> int:
        return self.n_windows // self.batch_size

    def _stack(self, starts) -> tuple[torch.Tensor, torch.Tensor]:
        w = torch.from_numpy(np.stack([self.tokens[i : i + self.block_size + 1] for i in starts.tolist()]))
        w = w.long()
        if self.pin:
            w = w.pin_memory()
        w = w.to(self.device, non_blocking=self.pin)
        return w[:, :-1], w[:, 1:]

    def n_steps(self, n_tokens: int) -> int:
        """Training steps for n_tokens: one epoch if the stream holds them all, else enough epochs to reach n_tokens."""
        tokens_per_step = self.block_size * self.batch_size
        return max(self.steps_per_epoch, n_tokens // tokens_per_step)

    def train_batches(self, n_steps: int, start_step: int = 0):
        """Batches for steps start_step..n_steps-1 (0-based).

        Each epoch is a fresh seeded shuffle of the windows, each token used once per epoch. Epoch 0 uses
        `seed` itself, so a run's order is a prefix of any longer run's, and resuming continues it exactly.
        """
        epoch = None
        for s in range(start_step, n_steps):
            e, b = divmod(s, self.steps_per_epoch)
            if e != epoch:
                epoch = e
                g = torch.Generator().manual_seed(self.seed + 1_000_003 * e)
                starts = torch.randperm(self.n_windows, generator=g) * self.block_size
            yield self._stack(starts[b * self.batch_size : (b + 1) * self.batch_size])

    def eval_batches(self, n: int | None = None):
        """Deterministic batches: the first n (None = all full batches), so evals are comparable."""
        n = self.steps_per_epoch if n is None else min(n, self.steps_per_epoch)
        for b in range(n):
            yield self._stack(torch.arange(b * self.batch_size, (b + 1) * self.batch_size) * self.block_size)
