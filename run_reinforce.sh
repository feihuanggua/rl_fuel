#!/bin/bash
set -e
cd /home/jd3/FUEL/rl_fuel
exec /home/jd3/miniconda3/envs/torch/bin/python -u -c "
import sys
sys.argv = ['r',
    '--total-steps', '50000',
    '--log-every', '200',
    '--num-pillars', '15']
from fuel_rl.train.train_reinforce import main
main()
"
