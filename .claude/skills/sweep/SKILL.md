---
name: sweep
description: Run, watch, extend and conclude a grid sweep (vocab size, learning rates, seeds, training length) with scratchpad.sh. Use when the user wants to set up a sweep, launch or watch one, extend a grid, or update defaults after one.
---

# Sweep

`scratchpad.sh` is the one place sweeps are defined. Edit its variable block; do not write new scripts.

## Setting up
1. Ask (or infer) the **one question** the sweep answers. Pick a new `WANDB_RUN_GROUP` for it, named `<what>-d<dim>-v<vocab>-<tokens>` (e.g. `lr-d192-v16k-500M`). Reuse a group only when rerunning/extending the same question.
2. Set the grid arrays (`VOCAB_SIZES`, `LRS`, `MUON_LRS`, `SEEDS`) and the fixed settings (`N_TOKENS`, `DIM`, ...). Every flag stays spelled out in the `uv run tinystories train` call.
3. Do not change run naming (`BASE` + `-MMDD-HHMM` stamp). The skip-if-finished check in wandb depends on it, so reruns and extensions only train new cells.
4. Keep the constant LR (`--train.warmup-frac 0.0 --train.decay-frac 0.0`) unless asked: it lets one long run answer "best LR at every shorter length".
5. Estimate wall time from past runs (step/s in the logs or wandb `_runtime`) and tell the user before launching.

## Launching and watching
- Run in the background (`run_in_background`) with output to a log file; find the PID with `pgrep -af tinystories`.
- Check progress from the log (step, val bpb, step/s, ETA) and wandb for finished runs. Wandb is the source of truth for scores, not the terminal (they can differ by eval batch count).
- Don't poll in a tight loop. Wait for the completion notification.

## Reading the result
- **Edge check:** if the best cell is on the edge of any grid axis, the optimum isn't bracketed. Extend that axis (add values to the array, keep the old ones, rerun; finished cells are skipped). Repeat until the best is interior.
- **Noise:** seed-to-seed spread is about ±0.0015 bpb. Differences smaller than that are ties. Prefer the wider/safer setting among ties, and add seeds if the choice matters.
- **Training length:** an LR tuned at one token count may not hold at another. Before changing defaults, check at the length we actually train (see RESULTS.md section 3) or state that this wasn't checked.
- Compare vocab sizes by **bpb**, never loss.

## Finishing
1. Update defaults in `src/tinystories/config.py` and the variable block of `scratchpad.sh` to the chosen setting.
2. Run the `plot-results` skill, then the `log-results` skill.
3. Commit atomically (code/defaults, results text and plots as separate commits).
