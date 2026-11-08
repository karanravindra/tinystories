"""Side-by-side story samples: our checkpoints vs the official TinyStories models, on the same prompts.

    uv run --with transformers python results/compare_generations.py name=checkpoints/<run> [name=checkpoints/<run> ...]

Sampling: temperature 0.8, no top-k, up to 300 new tokens, SAMPLES per prompt. Sample i of a model is seeded with
seed(model name, i): each model gets its own seeds, because the official models share a tokenizer, so a shared seed
makes them draw the same random numbers and start their stories almost identically. The official models' samples are cached in
results/official_generations.json and only generated when missing, so later runs need no Hugging Face downloads.
Writes results/generations_compare.md.
"""

import json
import sys
import zlib
from pathlib import Path

import torch

from tinystories.train import load_checkpoint

OUT = Path(__file__).parent
CACHE = OUT / "official_generations.json"
OFFICIAL = ["roneneldan/TinyStories-33M", "roneneldan/TinyStories-28M", "roneneldan/TinyStories-8M"]
PROMPTS = [
    "",
    "Once upon a time, there was a little girl named Lily. She loved to",
    "Tom and his dog went to the park. Suddenly,",
    "The little bird was sad because",
]
SAMPLES, MAX_NEW, TEMPERATURE = 5, 300, 0.8


def seed(name: str, i: int) -> int:
    return zlib.crc32(name.encode()) + i


def official_samples() -> dict:
    """{model: {prompt: [sample, ...]}}, from the cache, generating whatever is missing."""
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    if cache.get("_seeding") != "per-model":  # older caches used seed i for every model
        cache = {"_seeding": "per-model"}
    missing = [(m, p) for m in OFFICIAL for p in PROMPTS if len(cache.get(m.split("/")[1], {}).get(p, [])) < SAMPLES]
    if missing:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")  # the tokenizer the official models use
        for repo in sorted({m for m, _ in missing}):
            name = repo.split("/")[1]
            model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32).cuda().eval()
            for p in PROMPTS:
                outs = []
                for i in range(SAMPLES):
                    torch.manual_seed(seed(name, i))
                    ids = [tok.eos_token_id] + (tok(p)["input_ids"] if p else [])
                    x = torch.tensor([ids], device="cuda")
                    out = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=MAX_NEW, do_sample=True,
                                         temperature=TEMPERATURE, top_k=0, pad_token_id=tok.eos_token_id,
                                         eos_token_id=tok.eos_token_id)
                    outs.append(p + tok.decode(out[0, len(ids):], skip_special_tokens=True))
                cache.setdefault(name, {})[p] = outs
            del model
            torch.cuda.empty_cache()
            CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
            print("cached", name)
    return {m.split("/")[1]: cache[m.split("/")[1]] for m in OFFICIAL}


def our_samples(name: str, path: str) -> dict:
    model, tok, _ = load_checkpoint(path, "cuda")
    res = {}
    for p in PROMPTS:
        outs = []
        for i in range(SAMPLES):
            torch.manual_seed(seed(name, i))
            ids = [tok.eos_id] + (tok.encode(p) if p else [])
            out = model.generate(torch.tensor([ids], device="cuda"), MAX_NEW, temperature=TEMPERATURE, eos_id=tok.eos_id)
            out = out[0, 1:].tolist()
            outs.append(tok.decode(out[: out.index(tok.eos_id)] if tok.eos_id in out else out))
        res[p] = outs
    return res


def main() -> None:
    samples = {name: our_samples(name, path) for name, path in (a.split("=", 1) for a in sys.argv[1:])}
    samples |= official_samples()
    lines = ["# Sample generations: ours vs the official models", "",
             f"Temperature {TEMPERATURE}, up to {MAX_NEW} tokens, {SAMPLES} samples per prompt, each model with its own seeds. "
             f"Made by `results/compare_generations.py`.", ""]
    for p in PROMPTS:
        lines += [f"## Prompt: " + (f"“{p}”" if p else "(none: start a new story)"), ""]
        for i in range(SAMPLES):
            lines += [f"### Sample {i + 1}", ""]
            for name, s in samples.items():
                lines += [f"**{name}**", "", "> " + s[p][i].strip().replace("\n", "\n> "), ""]
    (OUT / "generations_compare.md").write_text("\n".join(lines))
    print("wrote", OUT / "generations_compare.md")


if __name__ == "__main__":
    main()
