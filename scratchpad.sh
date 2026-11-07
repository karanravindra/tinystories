set -euo pipefail

PGID=$(ps -o pgid= $$ | tr -d ' ')
echo "scratchpad.sh PID $$, process group ${PGID}. Stop the sweep (the current run saves a resumable state first): kill -INT -- -${PGID}"

# Settings that go into the run name. Other flags are spelled out below at their defaults.
N_TOKENS=500000000    # past the 424M tokens in the train split, the data repeats in reshuffled epochs
OPTIMIZER=muon
HEAD_DIM=64           # n_heads = dim / HEAD_DIM, so head size stays fixed as width changes

# Grid: every combination of these is trained (runs already finished in wandb are skipped).
DIMS=(512 640 768)
DEPTHS=(2 4 6 8)
VOCAB_SIZES=(16384)
LRS=(2e-2)            # AdamW lr: embeddings, head, norms
MUON_LRS=(0.0018) # Muon lr: hidden weight matrices
SEEDS=(0)

# wandb group = the experiment, so every run answering one question lands in one group, however many times
# the script is rerun. Use a new name for a new question.
export WANDB_RUN_GROUP="shape-v16k-500M"

# The stamp keeps run names unique across reruns.
STAMP=$(date +%m%d-%H%M)

# The newest checkpoints/<shape, optimizer>-t*M-s<seed>-* run dir with a resumable state.pt, if any.
# Training continues from it (same data order) instead of starting over: a shorter run, or one that crashed.
latest_state() {
   # `|| true`: no match is normal, and must not trip set -e / pipefail.
   { ls -t checkpoints/"$1"-t*M-s"$2"-*/state.pt 2>/dev/null || true; } | head -1 | xargs -r dirname
}

# Exit 0 if wandb already has a finished run named <base>-MMDD-HHMM, 1 if not, 2 if the lookup failed.
run_exists() {
   uv run python - "$1" <<'EOF'
import re, sys
try:
    import wandb
    api = wandb.Api()
    pattern = re.compile(re.escape(sys.argv[1]) + r"-\d{4}-\d{4}")
    runs = api.runs(f"{api.default_entity}/tinystories", filters={"display_name": {"$regex": "^" + re.escape(sys.argv[1])}})
    match = [r.name for r in runs if r.state == "finished" and pattern.fullmatch(r.name)]
except Exception as e:
    print(f"wandb lookup failed: {e}", file=sys.stderr)
    sys.exit(2)
if match:
    print(match[0])
sys.exit(0 if match else 1)
EOF
}

for N_LAYERS in "${DEPTHS[@]}"; do
for DIM in "${DIMS[@]}"; do
for VOCAB_SIZE in "${VOCAB_SIZES[@]}"; do
for LR in "${LRS[@]}"; do
for MUON_LR in "${MUON_LRS[@]}"; do
for SEED in "${SEEDS[@]}"; do
   N_HEADS=$((DIM / HEAD_DIM))
   if [[ ${OPTIMIZER} == muon ]]; then OPT_TAG="muon${MUON_LR}-lr${LR}"; else OPT_TAG="adamw${LR}"; fi
   BASE="v${VOCAB_SIZE}-d${DIM}-L${N_LAYERS}-h${N_HEADS}-${OPT_TAG}-t$((N_TOKENS / 1000000))M-s${SEED}"
   NAME="${BASE}-${STAMP}"

   status=0
   existing=$(run_exists "${BASE}") || status=$?
   if [[ ${status} == 0 ]]; then
      echo "Skipping ${BASE}: already finished as ${existing}"
      continue
   elif [[ ${status} == 2 ]]; then
      echo "Could not check wandb for ${BASE}; stopping so nothing reruns by accident" >&2
      exit 1
   fi

   RESUME_DIR=$(latest_state "v${VOCAB_SIZE}-d${DIM}-L${N_LAYERS}-h${N_HEADS}-${OPT_TAG}" "${SEED}")
   RESUME_ARGS=()
   if [[ -n ${RESUME_DIR} ]]; then
      echo "Resuming from ${RESUME_DIR}"
      RESUME_ARGS=(--resume "${RESUME_DIR}")
   fi

   export WANDB_TAGS="vocab=${VOCAB_SIZE},optimizer=${OPTIMIZER},dim=${DIM},layers=${N_LAYERS},lr=${LR},muon_lr=${MUON_LR},tokens=$((N_TOKENS / 1000000))M"

   echo "Training ${NAME}"
   # A run stopped by Ctrl-C / kill (exit 130 / 143) stops the sweep; any other failure (e.g. out of GPU memory at
   # the big widths) is reported and the sweep moves on.
   train_status=0
   uv run tinystories train \
      --device cuda \
      --out-dir checkpoints \
      --sample-tokens 512 \
      --wandb \
      --wandb-project tinystories \
      --wandb-name "${NAME}" \
      --model.block-size 512 \
      --model.dim ${DIM} \
      --model.n-heads ${N_HEADS} \
      --model.n-layers ${N_LAYERS} \
      --model.tie-embeddings \
      --data.n-tokens ${N_TOKENS} \
      --data.vocab-size ${VOCAB_SIZE} \
      --data.max-story-tokens 512 \
      --data.cache-dir cache \
      --data.no-rebuild-cache \
      --train.optimizer ${OPTIMIZER} \
      --train.lr ${LR} \
      --train.muon-lr ${MUON_LR} \
      --train.batch-size 64 \
      --train.warmup-frac 0.0 \
      --train.decay-frac 0.0 \
      --train.compile-mode reduce-overhead \
      --train.eval-every 100 \
      --train.eval-batches 20 \
      --train.log-every 50 \
      --train.save-every 1000 \
      --train.seed ${SEED} \
      "${RESUME_ARGS[@]}" || train_status=$?
   if (( train_status == 130 || train_status == 143 )); then
      echo "Stopped during ${NAME}; rerun this script to resume it" >&2
      exit ${train_status}
   elif (( train_status != 0 )); then
      echo "FAILED ${NAME} (exit ${train_status})" >&2
   fi
done
done
done
done
done
done
