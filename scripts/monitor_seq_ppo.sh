#!/bin/bash
# 序列 PPO 训练监控 — 每 5 分钟检查，崩溃则记录并通知
LOG=/tmp/seq_ppo3.log
SAVE_DIR=/home/jd3/FUEL/rl_fuel/fuel_rl_checkpoints/seq_ppo
REPORT=/tmp/seq_ppo_report.txt

while true; do
    if [ ! -f "$LOG" ]; then
        echo "$(date): 日志不存在" >> $REPORT
        sleep 300
        continue
    fi

    # 取最后 5 行
    LAST=$(tail -5 "$LOG")
    echo "=== $(date) ===" >> $REPORT
    echo "$LAST" >> $REPORT

    # 检查是否崩溃 (覆盖 < 15%)
    COV=$(echo "$LAST" | grep "Ep " | tail -1 | grep -oP 'cov=\K[0-9.]+')
    if [ -n "$COV" ] && [ "$(echo "$COV < 0.15" | bc)" = "1" ]; then
        echo "! 策略崩溃 cov=$COV (低于 15%)" >> $REPORT
        # 不自动重启，由人工判断
    fi

    # 检查进程是否活着
    PID=$(pgrep -f "seq_ppo" | head -1)
    if [ -z "$PID" ]; then
        echo "! 训练进程已停止" >> $REPORT
    fi

    sleep 300
done
