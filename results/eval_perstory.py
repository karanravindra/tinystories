"""Per-story bpb for our checkpoints only, scored exactly as results/eval_official.py scores every model (section 5 of
RESULTS.md): each validation story on its own after an end-of-text token, on the stories that fit in 511 tokens of
the 16k tokenizer. Merges the scores into results/perstory.json (run name -> bpb), which plot.py reads.

    uv run python results/eval_perstory.py checkpoints/<run> [checkpoints/<run> ...]
"""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset

from tinystories.train import load_checkpoint

OUT = Path(__file__).parent / "perstory.json"
REF = "checkpoints/v16384-d192-L4-h3-muon0.0018-lr2e-2-t500M-s0-1107-1056"  # section 5's subset tokenizer


@torch.no_grad()
def score(model, seqs, batch_tokens=16 * 512):
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    total, i = 0.0, 0
    while i < len(order):
        j = i
        while j < len(order) and (j - i + 1) * len(seqs[order[j]]) <= batch_tokens:
            j += 1
        j = max(j, i + 1)
        batch = [seqs[k] for k in order[i:j]]
        T = max(len(s) for s in batch)
        x = torch.zeros(len(batch), T, dtype=torch.long)
        mask = torch.zeros(len(batch), T, dtype=torch.bool)
        for b, s in enumerate(batch):
            x[b, : len(s)] = torch.tensor(s)
            mask[b, : len(s)] = True
        x, mask = x.cuda(), mask.cuda()
        logits = model(x)[0].float()
        nll = F.cross_entropy(logits[:, :-1].transpose(1, 2), x[:, 1:], reduction="none")
        total += nll[mask[:, 1:]].sum().item()
        i = j
    return total


stories = load_dataset("roneneldan/TinyStories", split="validation")["text"]
_, tok16, _ = load_checkpoint(REF, "cpu")
keep = [s for s, ids in zip(stories, tok16.encode_batch(stories)) if len(ids) <= 511]
n_bytes = sum(len(s.encode("utf-8")) for s in keep)
print(f"{len(keep):,} stories, {n_bytes:,} bytes (section 5: 21,427 stories, 17,858,502 bytes)")
results = json.loads(OUT.read_text()) if OUT.exists() else {}
for path in sys.argv[1:]:
    model, tok, _ = load_checkpoint(path, "cuda")
    seqs = [[tok.eos_id] + ids for ids in tok.encode_batch(keep)]
    results[Path(path).name] = bpb = score(model, seqs) / math.log(2) / n_bytes
    print(f"{path}: per-story bpb {bpb:.4f}", flush=True)
    OUT.write_text(json.dumps(results, indent=1))
    del model
