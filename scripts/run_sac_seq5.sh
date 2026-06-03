#!/bin/bash
cd /home/jd3/FUEL/rl_fuel
/home/jd3/miniconda3/envs/torch/bin/python -u run_sac_seq5.py \
  --total-steps 50000 \
  --buffer-size 30000 \
  --save-dir ./fuel_rl_checkpoints/sac_seq5 \
  >> fuel_rl_checkpoints/sac_seq5/train.log 2>&1
