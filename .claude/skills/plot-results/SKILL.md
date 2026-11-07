---
name: plot-results
description: Redraw or add figures for RESULTS.md from wandb data using results/plot.py. Use when the user asks for plots of experiments, a new figure, or a style fix.
---

# Plot results

All figures come from `results/plot.py`; don't make one-off plotting scripts.

## Process
1. Add any new wandb group to `GROUPS` in `results/plot.py`.
2. `uv run python results/plot.py` fetches from wandb and caches to `results/runs.json`; add `--cached` to replot without fetching (use it for style-only changes).
3. New figure = new `plot_*` function that calls `save(fig, "<name>.png")`, wired into `__main__`.
4. Look at every changed PNG (Read it) before reporting. Check legibility, not just that it rendered.
5. **Refresh the links:** the user's markdown viewer caches images. For each changed PNG linked from `RESULTS.md`, remove its `![...]` line, then re-add it, as two separate edits. Then report.

## Style (already set up in plot.py; keep it)
- SciencePlots: `plt.style.use(["science", "no-latex", "grid"])`, 300 dpi.
- Colors come from the style's cycle (`SERIES`). Series must be easy to tell apart. If lines are confusable, change markers or ordering, not just hue. Heatmaps use the blue ramp, darker = better.
- Axis labels name the quantity and units, with "(lower is better)" on bpb axes. Panels titled `(a)`, `(b)`.
- Text inside heatmap cells uses sans-serif (`family="DejaVu Sans"`), bold for the best cell, `*` for multi-seed averages. Axis labels keep the serif style font.
- Sizes: single panel 3.5x2.8 in, two panels 7x2.8 in.
