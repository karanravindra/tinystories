import itertools
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

from tinystories.tokenizer import BPETokenizer, Tokenizer_

TOKENIZER_TRAIN_STORIES = 20_000


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    tmp.replace(path)


def _tokenize(train_split, tokenizer: Tokenizer_, n_tokens: int, max_story_tokens: int = 0) -> np.ndarray:
    """Stream stories, packing them into one token stream until we have n_tokens.

    Stories longer than max_story_tokens (excluding EOS) are dropped; 0 disables the filter.
    """
    dtype = np.uint16 if tokenizer.vocab_size <= 2**16 else np.int32
    chunks: list[np.ndarray] = []
    total = 0
    for batch in train_split.iter(batch_size=1000):
        stories = batch["text"]
        if isinstance(tokenizer, BPETokenizer):
            eos = tokenizer.eos_id
            encoded = tokenizer.encode_batch(stories)
            if max_story_tokens:
                encoded = [e for e in encoded if len(e) <= max_story_tokens]
            chunk = np.fromiter(
                itertools.chain.from_iterable(itertools.chain(s, (eos,)) for s in encoded), dtype=dtype
            )
        else:
            encoded = [tokenizer.encode(s + "\n") for s in stories]
            if max_story_tokens:
                encoded = [e for e in encoded if len(e) <= max_story_tokens]
            chunk = np.fromiter(itertools.chain.from_iterable(encoded), dtype=dtype)
        chunks.append(chunk)
        total += len(chunk)
        if total >= n_tokens:
            break
    return np.concatenate(chunks)[:n_tokens]


class TinyStoriesData:
    def __init__(
        self,
        block_size: int,
        batch_size: int,
        device: str,
        n_tokens: int = 1_000_000,
        tokenizer: Tokenizer_ | None = None,
        vocab_size: int = 4096,
        cache_dir: str | Path | None = "cache",
        rebuild_cache: bool = False,
        max_story_tokens: int = 1024,
    ) -> None:
        # Caching only applies to the built-in BPE tokenizer (its cache key is vocab_size/n_tokens).
        use_cache = tokenizer is None and cache_dir is not None
        cache = Path(cache_dir) if use_cache else None
        if cache is not None:
            cache.mkdir(parents=True, exist_ok=True)
            tok_path = cache / f"tokenizer_v{vocab_size}_s{TOKENIZER_TRAIN_STORIES}.json"
            tokens_path = cache / f"tokens_v{vocab_size}_s{TOKENIZER_TRAIN_STORIES}_n{n_tokens}_max{max_story_tokens}.npy"
            if rebuild_cache:
                tok_path.unlink(missing_ok=True)
                tokens_path.unlink(missing_ok=True)

        train_split = None

        def get_split():
            nonlocal train_split
            if train_split is None:
                train_split = load_dataset("roneneldan/TinyStories")["train"]
            return train_split

        # Train our own BPE on a fixed sample of stories unless a tokenizer is passed in.
        # It is independent of n_tokens so the vocab doesn't change with the dataset size.
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif cache is not None and tok_path.exists():
            self.tokenizer = BPETokenizer.load(tok_path)
            print(f"loaded cached tokenizer {tok_path}")
        else:
            self.tokenizer = BPETokenizer.train(get_split()[:TOKENIZER_TRAIN_STORIES]["text"], vocab_size)
            if cache is not None:
                self.tokenizer.save(tok_path)

        if cache is not None and tokens_path.exists():
            data = np.load(tokens_path, mmap_mode="r")
            print(f"loaded cached tokens {tokens_path} ({len(data):,} tokens)")
        else:
            data = _tokenize(get_split(), self.tokenizer, n_tokens, max_story_tokens)
            if cache is not None:
                _atomic_save_npy(tokens_path, data)
                print(f"cached {len(data):,} tokens to {tokens_path}")

        split = int(0.9 * len(data))
        self.train, self.val = data[:split], data[split:]  # numpy views (uint16), converted per batch

        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

    def _stack(self, d: np.ndarray, starts) -> tuple[torch.Tensor, torch.Tensor]:
        starts = [int(i) for i in starts]
        x = np.stack([d[i : i + self.block_size] for i in starts]).astype(np.int64)
        y = np.stack([d[i + 1 : i + self.block_size + 1] for i in starts]).astype(np.int64)
        return torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)

    @property
    def steps_per_epoch(self) -> int:
        return ((len(self.train) - 1) // self.block_size) // self.batch_size

    def train_batches(self):
        """One pass over the train stream: non-overlapping windows, shuffled, each token used once."""
        n_windows = (len(self.train) - 1) // self.block_size
        starts = torch.randperm(n_windows) * self.block_size
        for b in range(self.steps_per_epoch):
            yield self._stack(self.train, starts[b * self.batch_size : (b + 1) * self.batch_size])

    def val_batches(self, n: int):
        """Deterministic val batches (first n of non-overlapping windows) so val loss is comparable across evals."""
        n_windows = (len(self.val) - 1) // self.block_size
        for b in range(min(n, n_windows // self.batch_size)):
            starts = torch.arange(b * self.batch_size, (b + 1) * self.batch_size) * self.block_size
            yield self._stack(self.val, starts)

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.train if split == "train" else self.val
        ix = torch.randint(len(d) - self.block_size - 1, (self.batch_size,))
        return self._stack(d, ix)
