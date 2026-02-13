#!/usr/bin/env bash
set -euo pipefail

# Minimal ablation launcher for Design 1' negative routing modes.
# Usage:
#   bash experiments/run_negmode_ablation.sh [DATASET]

DATASET="${1:-RGB-D}"
COMMON_ARGS=(
  --dataset "${DATASET}"
  --save_debug_npz true
  --log_dist_interval 5
  --route_uncertain_only true
  --alpha_fn 0.1
  --hn_beta 0.1
)

echo "[Ablation] neg_mode=batch"
python train.py "${COMMON_ARGS[@]}" --neg_mode batch

echo "[Ablation] neg_mode=knn, knn_neg_k=20"
python train.py "${COMMON_ARGS[@]}" --neg_mode knn --knn_neg_k 20

echo "[Ablation] neg_mode=knn, knn_neg_k=50"
python train.py "${COMMON_ARGS[@]}" --neg_mode knn --knn_neg_k 50
