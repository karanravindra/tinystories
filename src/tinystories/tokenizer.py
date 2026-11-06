from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import gigatoken as gt
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOS = "<|endoftext|>"


class Tokenizer_(Protocol):
    vocab_size: int

    def encode(self, s: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


class CharTokenizer:
    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


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

    def encode(self, s: str) -> list[int]:
        return self.gt.encode(s).tolist()

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return self.gt.encode_batch_list(texts)

    def decode(self, ids: list[int]) -> str:
        return self.gt.decode(ids).decode("utf-8", errors="replace")
