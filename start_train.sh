#!/bin/bash
# 启动训练 + TensorBoard
# 用法: ./start_train.sh <seq_number> [train_script.py]
# 示例: ./start_train.sh 13 train_sac_seq13.py

SEQ=$1
SCRIPT=${2:-train_sac_seq${SEQ}.py}

if [ -z "$SEQ" ]; then
    echo "用法: $0 <seq_number> [script.py]"
    echo "示例: $0 13 train_sac_seq13.py"
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

# 停止旧训练（如果有）
OLD_SVC="rl-fuel-train${SEQ}"
if systemctl --user is-active "$OLD_SVC" >/dev/null 2>&1; then
    echo "停止旧的 ${OLD_SVC}..."
    systemctl --user stop "$OLD_SVC"
fi

# 创建systemd service
SVC_FILE="$HOME/.config/systemd/user/${OLD_SVC}.service"
cat > "$SVC_FILE" << EOF
[Unit]
Description=SAC Seq${SEQ} Training
After=rl-fuel-tensorboard.service

[Service]
Type=simple
WorkingDirectory=/home/jdwsl/rl_fuel
ExecStart=/home/jdwsl/miniconda3/envs/rl_fuel/bin/python fuel_rl/train/${SCRIPT}
Restart=no
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user start "$OLD_SVC"

echo ""
echo "================================"
echo "训练已启动: ${OLD_SVC}"
echo "TensorBoard: http://localhost:6006"
echo "查看日志: journalctl --user -u ${OLD_SVC} -f"
echo "================================"
