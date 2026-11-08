import torch
import torch.nn.functional as F
from cut_cross_entropy import linear_cross_entropy
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from tinystories.config import ModelConfig


@torch.library.custom_op("tinystories::doc_ids", mutates_args=())
def doc_ids(idx: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Per position, the number of EOS tokens strictly before it. A custom op so inductor sees a realized buffer
    (a lazily fused one has a flexible layout, which flex_attention's lowering rejects when mask_mod captures it)."""
    is_eos = (idx == eos_id).to(torch.int32)
    return is_eos.cumsum(-1, dtype=torch.int32) - is_eos


@doc_ids.register_fake
def _(idx: torch.Tensor, eos_id: int) -> torch.Tensor:
    return torch.empty_like(idx, dtype=torch.int32)


class ReLU2(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()


class MLP(nn.Module):
    def __init__(self, dim: int, m: int, kind: str = "gelu") -> None:
        super().__init__()
        self.kind = kind
        if kind == "swiglu":
            hidden = round(2 * m * dim / 3 / 64) * 64  # 2/3 of the width so the weight count matches the 2-matrix MLP
            self.gate_up = nn.Linear(dim, 2 * hidden, bias=False)
            self.down = nn.Linear(hidden, dim, bias=False)
        else:
            act = {"gelu": nn.GELU(), "relu2": ReLU2()}[kind]
            self.net = nn.Sequential(nn.Linear(dim, dim * m, bias=False), act, nn.Linear(dim * m, dim, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "swiglu":
            gate, up = self.gate_up(x).chunk(2, dim=-1)
            return self.down(F.silu(gate) * up)
        return self.net(x)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each (x[i], x[i + head_dim/2]) pair by an angle that grows with position (RoPE, rotate-half form)."""
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, block_size: int, rope_base: float = 10_000.0, qk_norm: bool = False) -> None:
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        head_dim = dim // n_heads
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.n_heads = n_heads
        self.qk_norm = qk_norm
        # muP: 1/d attention scaling instead of 1/sqrt(d). With QK-norm, q and k have RMS 1, so q.k is at most d and
        # 1/sqrt(d) gives logits of the usual size.
        self.scale = head_dim**-0.5 if qk_norm else 1.0 / head_dim
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        inv_freq = 1.0 / rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        freqs = torch.outer(torch.arange(block_size).float(), inv_freq)  # (block_size, head_dim/2)
        angles = torch.cat((freqs, freqs), dim=-1)  # (block_size, head_dim)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, ve: torch.Tensor | None = None, ve_gain: torch.Tensor | None = None,
        block_mask: BlockMask | torch.Tensor | None = None, cache: list | None = None, pos: int = 0,
    ) -> torch.Tensor:
        """`block_mask`: None (causal), a flex BlockMask, or a dense boolean mask (B, 1, T, keys) that already
        includes causality. `cache` ([keys, values], both None at first) makes this a generation step: x holds the
        tokens at positions pos..pos+T-1, their keys and values are appended to the cache, and they attend to
        everything cached (or what a dense mask allows). Either the cache is empty (the prompt) or T == 1."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        if ve is not None:
            v = v + ve_gain * ve  # value embedding: a per-token vector mixed straight into the values
        q, k, v = (t.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) for t in (q, k, v))
        if self.qk_norm:
            q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.cos[pos : pos + T], self.sin[pos : pos + T]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            assert cache[0] is None or T == 1, "with a filled cache, feed one token at a time"
            if cache[0] is not None:
                k, v = torch.cat([cache[0], k], dim=2), torch.cat([cache[1], v], dim=2)
            cache[0], cache[1] = k, v
        if isinstance(block_mask, torch.Tensor):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=block_mask, scale=self.scale)
        elif block_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1, scale=self.scale)
        else:
            out = flex_attention(q, k, v, block_mask=block_mask, scale=self.scale)
        return self.proj(out.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, block_size: int, m: int = 4, mlp: str = "gelu", qk_norm: bool = False) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads, block_size, qk_norm=qk_norm)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, m, mlp)

    def forward(
        self, x: torch.Tensor, ve: torch.Tensor | None = None, ve_gain: torch.Tensor | None = None,
        block_mask: BlockMask | torch.Tensor | None = None, cache: list | None = None, pos: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), ve, ve_gain, block_mask, cache, pos)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        vocab_size, block_size, dim, n_heads, n_layers = cfg.vocab_size, cfg.block_size, cfg.dim, cfg.n_heads, cfg.n_layers
        tie_embeddings = cfg.tie_embeddings
        assert vocab_size > 0, "ModelConfig.vocab_size must be set from the tokenizer"
        assert not cfg.doc_mask or cfg.eos_id >= 0, "doc_mask needs ModelConfig.eos_id from the tokenizer"
        self.cfg = cfg
        self.block_size = block_size
        self.dim = dim
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.Sequential(*[Block(dim, n_heads, block_size, mlp=cfg.mlp, qk_norm=cfg.qk_norm) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        # Value embeddings (modded-nanogpt): k extra token-embedding tables; table i feeds the values of layer i and
        # of layer n_layers - k + i (the first k and last k layers share tables).
        k = cfg.value_embeds
        assert 2 * k <= n_layers, "value_embeds must be at most n_layers / 2"
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, dim) for _ in range(k)])
        self.ve_gains = nn.Parameter(torch.full((2 * k,), 0.5)) if k else None
        self.ve_layers = {i: i for i in range(k)} | {n_layers - k + i: i for i in range(k)}
        self.head = None if tie_embeddings else nn.Linear(dim, vocab_size, bias=False)
        self.apply(self._init_weights)
        if self.head is not None:
            nn.init.zeros_(self.head.weight)  # muP: zero-init readout
        # Residual-branch output projections get an extra 1/sqrt(2*n_layers) so the residual stream doesn't grow with depth.
        for name, p in self.named_parameters():
            if name.endswith(("attn.proj.weight", "mlp.net.2.weight", "mlp.down.weight")):
                p.data.mul_((2 * n_layers) ** -0.5)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """muP init: hidden weights ~ N(0, 1/fan_in), embedding ~ N(0, 1) (width-independent)."""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=m.in_features**-0.5)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=1.0)

    def _doc_block_mask(self, idx: torch.Tensor) -> BlockMask | torch.Tensor | None:
        """Causal AND same-story mask: None (plain causal) if doc_mask is off; a flex BlockMask for lengths that are
        a multiple of 128 (training); otherwise a dense boolean mask, so every length is masked the same way. Note
        a story's tokens can't see the end-of-story token before it: that token belongs to the previous story."""
        B, T = idx.shape
        if not self.cfg.doc_mask:
            return None
        doc = doc_ids(idx, self.cfg.eos_id)
        if T % 128 != 0:
            causal = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
            return (causal & (doc[:, :, None] == doc[:, None, :]))[:, None]

        def mask_mod(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (doc[b, q_idx] == doc[b, kv_idx])

        return create_block_mask(mask_mod, B, None, T, T, device=idx.device)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None, caches: list | None = None, pos: int = 0,
        mask: torch.Tensor | None = None,
    ):
        """`caches` (one [keys, values] per layer) and `pos` are for generation (see Attention.forward); with
        doc_mask, `mask` is then the dense mask for this step, from generate."""
        block_mask = self._doc_block_mask(idx) if caches is None or pos == 0 else mask
        x = self.tok_emb(idx)  # positions come from RoPE in attention
        ves = [e(idx) for e in self.value_embeds]
        slots = {layer: n for n, layer in enumerate(sorted(self.ve_layers))}
        for i, block in enumerate(self.blocks):
            cache = None if caches is None else caches[i]
            if i in self.ve_layers:
                x = block(x, ves[self.ve_layers[i]], self.ve_gains[slots[i]], block_mask, cache, pos)
            else:
                x = block(x, block_mask=block_mask, cache=cache, pos=pos)
        h = self.ln_f(x)
        h = h / self.dim  # muP output multiplier 1/width: logits start near 0 and are O(1/sqrt(width)) at init
        w = self.tok_emb.weight if self.head is None else self.head.weight
        if targets is None:
            return F.linear(h, w), None
        # Apple's Cut Cross-Entropy: fuses head + loss so the (B*T, vocab) logits are never materialized.
        # The fused CUDA kernel needs fp16/bf16; other devices use the pure-PyTorch (torch.compile) impl.
        on_cuda = h.is_cuda
        dtype = torch.bfloat16 if on_cuda else h.dtype
        loss = linear_cross_entropy(
            h.reshape(-1, h.size(-1)).to(dtype),
            w.to(dtype),
            targets.reshape(-1),
            impl="cce" if on_cuda else "torch_compile",
        )
        return None, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Sample continuations. With eos_id, stops once every sequence has emitted it (then pads with eos_id).

        Uses a KV cache: the prompt is run once, then each new token only computes its own keys and values. Past
        block_size tokens, the cache is rebuilt from the last block_size tokens at every step (as without a cache)."""
        done = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)

        def prefill(context):
            caches = [[None, None] for _ in self.blocks]
            return self(context, caches=caches)[0][:, -1], caches

        def step_mask(context):
            """With doc_mask: which cached positions the newest token may see (same story), as (B, 1, 1, keys)."""
            if not self.cfg.doc_mask:
                return None
            doc = doc_ids(context, self.cfg.eos_id)
            return (doc == doc[:, -1:])[:, None, None]

        logits, caches = prefill(idx[:, -self.block_size :])
        pos = min(idx.size(1), self.block_size)
        for _ in range(max_new_tokens):
            logits = logits.float()
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            if eos_id is not None:
                nxt = nxt.masked_fill(done[:, None], eos_id)
                done |= nxt[:, 0] == eos_id
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and done.all():
                break
            if pos >= self.block_size:  # context full: drop the oldest token and recompute
                logits, caches = prefill(idx[:, -self.block_size :])
            else:
                logits = self(nxt, caches=caches, pos=pos, mask=step_mask(idx[:, -(pos + 1) :]))[0][:, -1]
                pos += 1
        return idx
