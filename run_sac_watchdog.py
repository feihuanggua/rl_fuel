#!/usr/bin/env python3
"""带 watchdog 的训练脚本: 如果训练卡住超过 N 分钟就自动重启."""
import os, sys, time, subprocess, signal

WATCHDOG_TIMEOUT = 300  # 5 分钟没输出就重启
MAX_RESTARTS = 20

os.chdir("/home/jd3/FUEL/rl_fuel")
sys.path.insert(0, "/home/jd3/FUEL/rl_fuel")

log_path = "./fuel_rl_checkpoints/sac_seq5/train.log"
csv_path = "./fuel_rl_checkpoints/sac_seq5/sac_log.csv"

restart_count = 0
global_step = 0

while restart_count < MAX_RESTARTS:
    print(f"[watchdog] Starting training (restart #{restart_count}, global_step ~{global_step})", flush=True)
    
    cmd = [
        sys.executable, "-u", "run_sac_seq5.py",
        "--total-steps", "50000",
        "--buffer-size", "30000",
        "--save-dir", "./fuel_rl_checkpoints/sac_seq5",
    ]
    
    proc = subprocess.Popen(cmd, stdout=open(log_path, "a"), stderr=subprocess.STDOUT)
    
    last_mtime = 0
    last_check = time.time()
    
    while proc.poll() is None:
        time.sleep(10)
        
        try:
            mtime = os.path.getmtime(csv_path)
        except FileNotFoundError:
            mtime = 0
        
        if mtime != last_mtime:
            last_mtime = mtime
            last_check = time.time()
            
            try:
                with open(csv_path) as f:
                    lines = f.readlines()
                if len(lines) > 1:
                    last_line = lines[-1].strip()
                    if last_line:
                        try:
                            global_step = int(float(last_line.split(",")[0]))
                        except:
                            pass
            except:
                pass
        else:
            elapsed = time.time() - last_check
            if elapsed > WATCHDOG_TIMEOUT:
                print(f"[watchdog] No output for {elapsed:.0f}s, killing process (step ~{global_step})", flush=True)
                proc.kill()
                proc.wait()
                break
    
    if proc.returncode == 0:
        print("[watchdog] Training completed successfully!", flush=True)
        break
    
    restart_count += 1
    print(f"[watchdog] Process died (rc={proc.returncode}). Restarting in 5s...", flush=True)
    time.sleep(5)

print(f"[watchdog] Exiting after {restart_count} restarts, step ~{global_step}", flush=True)
