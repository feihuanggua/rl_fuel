#!/bin/bash
set -e
cd /home/jd3/FUEL/rl_fuel
exec /home/jd3/miniconda3/envs/torch/bin/python -u -c "
import sys
sys.argv = ['sac',
    '--total-steps', '100000',
    '--log-every', '500',
    '--q-warmup-steps', '2000',
    '--num-pillars', '15']
from fuel_rl.train.train_sac import main
main()
"
