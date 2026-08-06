#!/usr/bin/env bash
# Reproduces Table 3: the full model, the module ablations, and the encoder
# ablations, all under the same 10-fold splits.
set -euo pipefail

DATASET_DIR=${DATASET_DIR:-./cyp_dataset}
OUT=${OUT:-results/ablation}
EXTRA=${EXTRA:-}          # e.g. EXTRA="--log_wandb"

run () {
  local name=$1; shift
  echo "=== ${name} ==="
  python main.py \
    --dataset_dir "${DATASET_DIR}" \
    --result_dir "${OUT}/${name}" \
    --run_name "${name}" \
    ${EXTRA} "$@"
}

# --- full model ------------------------------------------------------------
run attnsom

# --- module ablations ------------------------------------------------------
run attnsom_wo_attn            --no_attention
run attnsom_wo_film            --no_film
run attnsom_wo_film_wo_attn    --no_attention --no_film

# --- encoder ablations (full ATTNSOM head, alternative backbone) -----------
for enc in chemprop gin gcn gat; do
  run "attnsom_${enc}" --encoder "${enc}"
done

# --- standalone backbones (no attention, no FiLM) --------------------------
for enc in chemprop gin gcn gat; do
  run "${enc}_only" --encoder "${enc}" --no_attention --no_film
done

echo "Done. Per-run metrics: ${OUT}/<run>/summary.json"
