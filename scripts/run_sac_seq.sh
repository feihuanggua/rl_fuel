#!/bin/bash
cd /home/jd3/FUEL/rl_fuel
exec /home/jd3/miniconda3/envs/torch/bin/python -u -c "
import os; os.chdir('/home/jd3/FUEL/rl_fuel')
import sys; sys.argv = ['s', '--total-steps', '100000', '--log-every', '500', '--gamma', '1.0']
from fuel_rl.train.train_sac_seq import main
main()
"
