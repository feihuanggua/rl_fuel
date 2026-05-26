#!/bin/bash
# BC Training
set -e
ROOT=$(dirname "$0")/..
cd "$ROOT"
exec /home/jd3/miniconda3/envs/torch/bin/python -u -m fuel_rl.train.train_bc \
  --data-path ./fuel_rl_data/expert_data_v3.pt \
  --save-dir ./fuel_rl_checkpoints/bc_v3 \
  --epochs 100

