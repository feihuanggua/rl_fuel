#!/bin/bash
cd /home/jd3/FUEL/rl_fuel
exec /home/jd3/miniconda3/envs/torch/bin/python -u -m fuel_rl.train.train_sac_seq \
  --total-steps 100000 \
  --log-every 500 \
  --gamma 0.99 \
  --save-dir ./fuel_rl_checkpoints/sac_seq4 \
  2>&1
