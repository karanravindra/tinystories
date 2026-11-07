---
name: log-results
description: Add or rewrite an experiment entry in RESULTS.md. Use after a sweep or eval finishes, or when the user says "add this to results", "write up the results", or asks to simplify existing entries.
---

# Log results

Entries in `RESULTS.md` are written for a reader with **no context**. Follow the existing sections.

## Process
1. Pull the numbers from **wandb** (group name from `scratchpad.sh`), not the terminal. `results/plot.py` caches them in `results/runs.json`.
2. Add a new numbered section at the end (or edit the existing one if the user is extending it). Date is the absolute date, `(YYYY-MM-DD)`.
3. Update the "Current best settings" paragraph at the top if the best setting changed, and make sure it matches `src/tinystories/config.py`.

## Entry format
```
## N. Plain-language title (YYYY-MM-DD)

**Question:** one sentence.

What we ran, in plain words (model size, tokens, what varied, what was held fixed).

wandb group: `<group>`

<table or figure with italic caption>

**What we learned:**
- **Bold takeaway.** One or two sentences of evidence with numbers.
```

## Style
- Plain, short sentences. Define terms on first use (bpb, Muon, AdamW lr). No jargon the reader must already know.
- Always say bpb is lower-is-better in table headers/captions.
- Report seed noise (about ±0.0015 bpb) when calling something better; call smaller gaps ties.
- State what is **not** established (single seed, one training length, untested range).
- Always include the wandb group name.
- Figures: `![alt](results/<name>.png)` followed by an italic caption saying what darker/colors mean. Generate them with the `plot-results` skill.
- Don't leave stray text in the file.
