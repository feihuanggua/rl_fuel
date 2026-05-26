#!/bin/bash
# PPO Fine-tuning (from BC checkpoint)
set -e
ROOT=$(dirname "$0")/..
cd "$ROOT"
exec /home/jd3/miniconda3/envs/torch/bin/python -u -m fuel_rl.train.train_ppo \
  --bc-ckpt ./fuel_rl_checkpoints/bc_v3/best_model.pth \
  --save-dir ./fuel_rl_checkpoints/ppo_v3

