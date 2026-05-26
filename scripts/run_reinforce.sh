#!/bin/bash
# REINFORCE Training
set -e
ROOT=$(dirname "$0")/..
cd "$ROOT"
exec /home/jd3/miniconda3/envs/torch/bin/python -u -c "
import os, sys
sys.argv = ['r', '--total-steps', '50000', '--log-every', '200']
from fuel_rl.train.train_reinforce import main
main()
"

