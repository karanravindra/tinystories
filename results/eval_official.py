"""Compare our checkpoints with the official TinyStories models (Eldan & Li, 2023) on the validation split.

    uv run --with transformers python results/eval_official.py

Metric: bits per byte (bpb) of each validation story, scored on its own from the start of the story
(the model gets only an end-of-text token as context, then predicts every token of the story).
bpb = total nats / ln 2 / total UTF-8 bytes, so it doesn't depend on the tokenizer and every model is
scored on exactly the same text. The end-of-story token is not scored (it has 0 bytes).

Our models have a 512-token context, so we keep the stories that fit (about 97% of them) and score every
model on that same subset. This is not the same number as the training-time val bpb, which cuts the
packed stream into 512-token windows that mostly start mid-story.

Also writes sample generations for the same prompts to results/generations.md.
"""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinystories.train import load_checkpoint

OUT = Path(__file__).parent
DEVICE = "cuda"
OURS = {
    "ours-v16k": "checkpoints/v16384-d192-L4-h3-muon0.0018-lr2e-2-t500M-s0-1107-1056",
    "ours-v32k": "checkpoints/v32768-d192-L4-h3-muon0.0018-lr2e-2-t500M-s0-1107-1156",
}
OFFICIAL = [f"roneneldan/TinyStories-{s}" for s in ("1M", "3M", "8M", "28M", "33M", "1Layer-21M", "2Layers-33M")]
PROMPTS = [
    "",
    "Once upon a time, there was a little girl named Lily. She loved to",
    "Tom and his dog went to the park. Suddenly,",
    "The little bird was sad because",
]
MAX_OURS = 511  # story tokens that fit our 512 context after the leading end-of-text token


def count_params(model: torch.nn.Module, emb_names: tuple[str, ...]) -> tuple[int, int]:
    """(total, non-embedding) parameters, counting tied weights once."""
    seen, total, emb = set(), 0, 0
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
        if any(k in name for k in emb_names):
            emb += p.numel()
    return total, total - emb


@torch.no_grad()
def score(logits_fn, seqs: list[list[int]], batch_tokens: int = 16 * 512) -> float:
    """Sum of nats for predicting seq[1:] from seq[:-1], over all sequences (seq[0] is the start token)."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    total, i = 0.0, 0
    while i < len(order):
        L = len(seqs[order[i]])
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
        x, mask = x.to(DEVICE), mask.to(DEVICE)
        logits = logits_fn(x, mask).float()
        nll = F.cross_entropy(logits[:, :-1].transpose(1, 2), x[:, 1:], reduction="none")
        total += nll[mask[:, 1:]].sum().item()
        i = j
    return total


def main() -> None:
    stories = load_dataset("roneneldan/TinyStories", split="validation")["text"]
    # Subset: stories that fit our context under the larger-token-count tokenizer of the two (16k).
    _, tok16, _ = load_checkpoint(OURS["ours-v16k"], "cpu")
    keep = [s for s, ids in zip(stories, tok16.encode_batch(stories)) if len(ids) <= MAX_OURS]
    n_bytes = sum(len(s.encode("utf-8")) for s in keep)
    print(f"scoring {len(keep):,}/{len(stories):,} validation stories ({n_bytes:,} bytes)")

    results, samples = {}, {}
    torch.manual_seed(0)
    for name, path in OURS.items():
        model, tok, _ = load_checkpoint(path, DEVICE)
        seqs = [[tok.eos_id] + ids for ids in tok.encode_batch(keep)]
        assert max(map(len, seqs)) <= 512
        nats = score(lambda x, m: model(x)[0], seqs)
        total, non_emb = count_params(model, ("tok_emb", "head"))
        results[name] = {"bpb": nats / math.log(2) / n_bytes, "params": total, "non_emb_params": non_emb}
        print(name, results[name], flush=True)
        samples[name] = []
        for p in PROMPTS:
            ids = [tok.eos_id] + (tok.encode(p) if p else [])
            out = model.generate(torch.tensor([ids], device=DEVICE), 300, temperature=0.8, eos_id=tok.eos_id)[0, 1:].tolist()
            if tok.eos_id in out:
                out = out[: out.index(tok.eos_id)]
            samples[name].append(tok.decode(out)[len(p):])  # decode includes the prompt; keep only the continuation
        del model

    for repo in OFFICIAL:
        name = repo.split("/")[1]
        tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")  # the tokenizer the official models use
        model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32).to(DEVICE).eval()
        bos = tok.eos_token_id
        seqs = [[bos] + ids for ids in tok(keep)["input_ids"]]
        nats = score(lambda x, m: model(input_ids=x, attention_mask=m.long()).logits, seqs, batch_tokens=16 * 512)
        total, non_emb = count_params(model, ("wte", "wpe", "lm_head"))
        results[name] = {"bpb": nats / math.log(2) / n_bytes, "params": total, "non_emb_params": non_emb}
        print(name, results[name], flush=True)
        samples[name] = []
        for p in PROMPTS:
            ids = [bos] + (tok(p)["input_ids"] if p else [])
            out = model.generate(torch.tensor([ids], device=DEVICE), attention_mask=torch.ones(1, len(ids), device=DEVICE, dtype=torch.long),
                                 max_new_tokens=300, do_sample=True, temperature=0.8, top_k=0, pad_token_id=bos, eos_token_id=bos)
            samples[name].append(tok.decode(out[0, len(ids):], skip_special_tokens=True))
        del model
        torch.cuda.empty_cache()

    (OUT / "official_eval.json").write_text(json.dumps({"n_stories": len(keep), "n_bytes": n_bytes, "results": results}, indent=1))
    lines = ["# Sample generations", "", "Temperature 0.8, up to 300 tokens, one sample per prompt, seed 0.", ""]
    for i, p in enumerate(PROMPTS):
        lines += [f"## Prompt {i + 1}: " + (f"“{p}”" if p else "(none: start a new story)"), ""]
        for name in samples:
            lines += [f"**{name}**", "", "> " + (p + samples[name][i] if p else samples[name][i]).strip().replace("\n", "\n> "), ""]
    (OUT / "generations.md").write_text("\n".join(lines))
    print("wrote", OUT / "official_eval.json", OUT / "generations.md")


if __name__ == "__main__":
    sys.exit(main())
