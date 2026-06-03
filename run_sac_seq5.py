#!/usr/bin/env python3
"""Standalone SAC training launcher."""
import os, sys
os.chdir("/home/jd3/FUEL/rl_fuel")
sys.path.insert(0, "/home/jd3/FUEL/rl_fuel")

from fuel_rl.train.train_sac_seq import train_sac
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--save-dir", type=str, default="./fuel_rl_checkpoints/sac_seq5")
parser.add_argument("--total-steps", type=int, default=50000)
parser.add_argument("--buffer-size", type=int, default=30000)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--log-every", type=int, default=200)
parser.add_argument("--max-env-steps", type=int, default=100)
parser.add_argument("--num-pillars", type=int, default=15)
args = parser.parse_args()
train_sac(args)
