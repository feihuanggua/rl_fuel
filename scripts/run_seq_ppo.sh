#!/bin/bash
cd /home/jd3/FUEL/rl_fuel
exec /home/jd3/miniconda3/envs/torch/bin/python -u -c "
import os; os.chdir('/home/jd3/FUEL/rl_fuel')
import sys; sys.argv = ['s', '--max-episodes', '10000', '--log-every', '5', '--lr', '1e-4', '--gamma', '1.0']
from fuel_rl.train.train_seq_ppo import main
main()
" > /tmp/seq_ppo8.log 2>&1
