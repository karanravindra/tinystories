import torch
from datasets import load_dataset

from tinystories.tokenizer import BPETokenizer, Tokenizer_

TOKENIZER_TRAIN_STORIES = 20_000


class TinyStoriesData:
    def __init__(
        self,
        block_size: int,
        batch_size: int,
        device: str,
        n_tokens: int = 1_000_000,
        tokenizer: Tokenizer_ | None = None,
        vocab_size: int = 4096,
    ) -> None:
        train_split = load_dataset("roneneldan/TinyStories")["train"]

        # Train our own BPE on a fixed sample of stories unless a tokenizer is passed in.
        # It is independent of n_tokens so the vocab doesn't change with the dataset size.
        self.tokenizer = tokenizer or BPETokenizer.train(train_split[:TOKENIZER_TRAIN_STORIES]["text"], vocab_size)

        # Stream stories, packing them into one token stream until we have n_tokens.
        ids: list[int] = []
        for batch in train_split.iter(batch_size=1000):
            stories = batch["text"]
            if isinstance(self.tokenizer, BPETokenizer):
                for story in self.tokenizer.encode_batch(stories):
                    ids.extend(story)
                    ids.append(self.tokenizer.eos_id)
            else:
                for story in stories:
                    ids.extend(self.tokenizer.encode(story + "\n"))
            if len(ids) >= n_tokens:
                break
        data = torch.tensor(ids[:n_tokens], dtype=torch.long)
        split = int(0.9 * len(data))
        self.train, self.val = data[:split], data[split:]

        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

    @property
    def steps_per_epoch(self) -> int:
        return ((len(self.train) - 1) // self.block_size) // self.batch_size

    def train_batches(self):
        """One pass over the train stream: non-overlapping windows, shuffled, each token used once."""
        n_windows = (len(self.train) - 1) // self.block_size
        starts = torch.randperm(n_windows) * self.block_size
        for b in range(self.steps_per_epoch):
            ix = starts[b * self.batch_size : (b + 1) * self.batch_size]
            x = torch.stack([self.train[i : i + self.block_size] for i in ix])
            y = torch.stack([self.train[i + 1 : i + self.block_size + 1] for i in ix])
            yield x.to(self.device), y.to(self.device)

    def val_batches(self, n: int):
        """Deterministic val batches (first n of non-overlapping windows) so val loss is comparable across evals."""
        n_windows = (len(self.val) - 1) // self.block_size
        for b in range(min(n, n_windows // self.batch_size)):
            starts = torch.arange(b * self.batch_size, (b + 1) * self.batch_size) * self.block_size
            x = torch.stack([self.val[i : i + self.block_size] for i in starts])
            y = torch.stack([self.val[i + 1 : i + self.block_size + 1] for i in starts])
            yield x.to(self.device), y.to(self.device)

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.train if split == "train" else self.val
        ix = torch.randint(len(d) - self.block_size - 1, (self.batch_size,))
        x = torch.stack([d[i : i + self.block_size] for i in ix])
        y = torch.stack([d[i + 1 : i + self.block_size + 1] for i in ix])
        return x.to(self.device), y.to(self.device)
