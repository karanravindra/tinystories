import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, dim: int, m: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * m),
            nn.GELU(),
            nn.Linear(dim * m, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each (x[i], x[i + head_dim/2]) pair by an angle that grows with position (RoPE, rotate-half form)."""
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, block_size: int, rope_base: float = 10_000.0) -> None:
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        head_dim = dim // n_heads
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.n_heads = n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        inv_freq = 1.0 / rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        freqs = torch.outer(torch.arange(block_size).float(), inv_freq)  # (block_size, head_dim/2)
        angles = torch.cat((freqs, freqs), dim=-1)  # (block_size, head_dim)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, self.cos[:T], self.sin[:T]), apply_rope(k, self.cos[:T], self.sin[:T])
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, block_size: int, m: int = 4) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads, block_size)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, dim: int = 128, n_heads: int = 4, n_layers: int = 4) -> None:
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.Sequential(*[Block(dim, n_heads, block_size) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.head(self.ln_f(self.blocks(self.tok_emb(idx))))  # positions come from RoPE in attention
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size :])
            probs = F.softmax(logits[:, -1], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx
