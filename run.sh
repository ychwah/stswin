#!/usr/bin/env bash
set -euo pipefail

# Configurations
TRIALS=3                 # how many runs to average (seeds 0..TRIALS-1)
EPOCHS=800               # training epochs (kept in the model config above too)
DATASET_NAME="totaltext" # for naming work_dirs only
CFG="src/baseline.py"
WORKDIR_ROOT="work_dirs/dbnet_swin_t_${DATASET_NAME}_${EPOCHS}e"


mkdir -p "${WORKDIR_ROOT}"

for SEED in $(seq 0 $((TRIALS-1))); do
  RUN_DIR="${WORKDIR_ROOT}/seed_${SEED}"
  echo "[Run] seed=${SEED} -> ${RUN_DIR}"

  # You can change --gpus or use --launcher pytorch if you want multi-GPU
  mim train mmocr "${CFG}" \
    --work-dir "${RUN_DIR}" \
    --seed "${SEED}" \
    --cfg-options train_cfg.max_epochs="${EPOCHS}"
done

echo "[Done] All ${TRIALS} trial(s) finished. Results under: ${WORKDIR_ROOT}"