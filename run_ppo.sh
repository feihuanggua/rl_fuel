#!/bin/bash
set -e
cd /home/jd3/FUEL/rl_fuel
exec /home/jd3/miniconda3/envs/torch/bin/python -u -m fuel_rl.train.train_ppo \
  --bc-ckpt ./fuel_rl_checkpoints/bc_v3/best_model.pth \
  --save-dir ./fuel_rl_checkpoints/ppo_v3 \
  --max-episodes 50000 \
  --update-timestep 1000
