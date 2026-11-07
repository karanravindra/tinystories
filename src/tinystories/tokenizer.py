import functools
import hashlib
from collections.abc import Iterable
from pathlib import Path

import gigatoken as gt
import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOS = "<|endoftext|>"


class BPETokenizer:
    """Byte-level BPE (GPT-2 style). Trained with HF `tokenizers`, encoded with Gigatoken."""

    def __init__(self, hf_tokenizer: Tokenizer) -> None:
        self.hf = hf_tokenizer
        self.gt = gt.Tokenizer(hf_tokenizer)
        self.vocab_size = hf_tokenizer.get_vocab_size()
        self.eos_id = hf_tokenizer.token_to_id(EOS)

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 4096) -> "BPETokenizer":
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[EOS],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tok.train_from_iterator(texts, trainer)
        return cls(tok)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: str | Path) -> None:
        self.hf.save(str(path))

    @functools.cached_property
    def sha256(self) -> str:
        """Identity of the tokenizer, stored with cached tokens so stale caches are detected."""
        return hashlib.sha256(self.hf.to_str().encode()).hexdigest()

    @functools.cached_property
    def byte_lengths(self) -> np.ndarray:
        """UTF-8 bytes each token decodes to (EOS = 0). Used to turn nats/token into bits/byte."""
        lens = np.array([len(self.gt.decode([i])) for i in range(self.vocab_size)], dtype=np.int64)
        lens[self.eos_id] = 0
        return lens

    def encode(self, s: str) -> list[int]:
        return self.gt.encode(s).tolist()

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return self.gt.encode_batch_list(texts)

    def decode(self, ids: list[int]) -> str:
        return self.gt.decode(ids).decode("utf-8", errors="replace")
