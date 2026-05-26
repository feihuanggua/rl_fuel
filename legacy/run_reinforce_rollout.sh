#!/bin/bash
# REINFORCE + Rollout 奖励训练
# 用法: bash run_reinforce_rollout.sh [rollout_steps]
ROLLOUT=${1:-5}
echo "=== REINFORCE + Rollout (N=$ROLLOUT) ==="

cd "$(dirname "$0")"
conda run -n torch python -m fuel_rl.train.train_reinforce_rollout \
    --bc-ckpt ./fuel_rl_checkpoints/bc_v3/best_model.pth \
    --save-dir ./fuel_rl_checkpoints/reinforce_rollout \
    --total-steps 50000 \
    --lr 1e-4 \
    --std 0.15 \
    --rollout-steps $ROLLOUT \
    --log-every 200
