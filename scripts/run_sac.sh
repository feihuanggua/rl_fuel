#!/bin/bash
# SAC Training
set -e
ROOT=$(dirname "$0")/..
cd "$ROOT"
exec /home/jd3/miniconda3/envs/torch/bin/python -u -m fuel_rl.train.train_sac

