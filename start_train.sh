#!/bin/bash
# 启动训练 + TensorBoard
# 用法: ./start_train.sh <seq_number> [train_script.py]
# 示例: ./start_train.sh 14 train_sac_seq14.py
#
# 部署: cp到WSL的 /home/jdwsl/rl_fuel/start_train.sh

SEQ=$1
SCRIPT=${2:-train_sac_seq${SEQ}.py}
RL_ROOT="/home/jdwsl/rl_fuel"
CONDA_PYTHON="/home/jdwsl/miniconda3/envs/rl_fuel/bin/python"

if [ -z "$SEQ" ]; then
    echo "用法: $0 <seq_number> [script.py]"
    echo "示例: $0 14 train_sac_seq14.py"
    exit 1
fi

# 确保TensorBoard在运行
if ! systemctl --user is-active rl-fuel-tensorboard >/dev/null 2>&1; then
    echo "启动 TensorBoard..."
    systemctl --user start rl-fuel-tensorboard
    sleep 2
    echo "TensorBoard: http://localhost:6006"
else
    echo "TensorBoard 已在运行: http://localhost:6006"
fi

SVC="rl-fuel-train${SEQ}"

# 停止旧训练（如果有）
if systemctl --user is-active "$SVC" >/dev/null 2>&1; then
    echo "停止旧的 ${SVC}..."
    systemctl --user stop "$SVC"
fi

# 创建systemd service
SVC_FILE="$HOME/.config/systemd/user/${SVC}.service"
cat > "$SVC_FILE" << EOF
[Unit]
Description=SAC Seq${SEQ} Training
After=rl-fuel-tensorboard.service

[Service]
Type=simple
WorkingDirectory=${RL_ROOT}
ExecStart=${CONDA_PYTHON} fuel_rl/train/${SCRIPT}
Restart=no
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user start "$SVC"

echo ""
echo "================================"
echo "训练已启动: ${SVC}"
echo "TensorBoard: http://localhost:6006"
echo "查看日志: journalctl --user -u ${SVC} -f"
echo "================================"
