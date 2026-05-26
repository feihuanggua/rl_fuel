# FUEL RL 常用命令速查

## 环境激活

```bash
# 必须使用 conda torch 环境（GPU）
conda activate torch
cd /home/jd3/FUEL/rl_fuel
```

## 训练

```bash
# 基础训练 (1M 步, GPU)
python train.py --total-timesteps 1000000

# 自定义参数训练
python train.py \
  --total-timesteps 1000000 \
  --max-steps 100 \
  --lr 1e-4 \
  --n-steps 4096 \
  --batch-size 256 \
  --ent-coef 0.1 \
  --gamma 0.995 \
  --num-pillars 15 \
  --map-size 20.0

# 后台训练（断开终端不中断）
PYTHONUNBUFFERED=1 nohup python -u train.py \
  --total-timesteps 1000000 \
  > train.log 2>&1 &
echo "PID: $!"

# 查看训练日志
tail -f train.log

# 查看训练进度
grep "total_timesteps" train.log | tail -1
grep "ep_rew_mean" train.log | tail -5
```

## TensorBoard

```bash
# 方法1: 直接用 TensorBoard（需先安装）
pip install tensorboard

# 训练时启用 TensorBoard 日志
python train.py --total-timesteps 1000000 --use-tensorboard

# 启动 TensorBoard 网页
tensorboard --logdir=./fuel_rl_tensorboard/ --port=6006 --bind_all

# 浏览器打开
# http://localhost:6006

# 方法2: 后台启动 TensorBoard
nohup tensorboard --logdir=./fuel_rl_tensorboard/ --port=6006 --bind_all > tb.log 2>&1 &
```

## 查看训练指标（无需 TensorBoard）

```bash
# 实时查看 metrics 数据
cat fuel_rl_tensorboard/metrics.json | python -m json.tool

# 快速查看 loss 和 reward 趋势
python -c "
import json
with open('fuel_rl_tensorboard/metrics.json') as f:
    d = json.load(f)
n = len(d['timesteps'])
print(f'Points: {n}, Progress: {d[\"timesteps\"][-1]:,}')
for i in range(max(0, n-10), n):
    r = d['ep_rew_mean'][i]
    l = d['loss'][i]
    r_s = f'{r:>8.1f}' if r else '    None'
    l_s = f'{l:>8.0f}' if l else '    None'
    print(f'  ts={d[\"timesteps\"][i]:>8,}  rew={r_s}  loss={l_s}')
"
```

## 可视化绘图

```bash
# 生成训练曲线图（训练结束会自动生成）
python -m fuel_rl.plot_metrics --metrics fuel_rl_tensorboard/metrics.json

# 指定输出目录和平滑系数
python -m fuel_rl.plot_metrics \
  --metrics fuel_rl_tensorboard/metrics.json \
  --output-dir /tmp/fuel_plots \
  --smooth 0.8

# 生成的文件：
#   reward_curve.png   - Episode Reward & Length
#   loss_curves.png    - Loss 曲线 (总 loss, 策略梯度, value, entropy)
#   kl_clip.png        - KL 散度 & Clip 比例

# 打开图片
xdg-open fuel_rl_tensorboard/loss_curves.png
```

## 评估（查看训练效果）

```bash
# 评估已训练模型
python eval.py \
  --model-path ./fuel_rl_tensorboard/fuel_ppo_final.zip \
  --output-dir /tmp/fuel_rl_eval \
  --animate \
  --max-steps 100

# 指定地图评估
python eval.py \
  --model-path ./fuel_rl_tensorboard/fuel_ppo_final.zip \
  --map-path ../uav_simulator/map_generator/resource/office.pcd \
  --output-dir /tmp/fuel_rl_eval \
  --animate

# 生成的文件：
#   /tmp/fuel_rl_eval/step_XXXX.png  - 每步可视化
#   /tmp/fuel_rl_eval/exploration.gif - 动画（需 imageio）
```

## 环境测试

```bash
# 快速测试环境是否正常
python test_env.py

# 测试贪心基线（零偏移策略）
python -c "
from fuel_rl.env import FuelRLEnvSingleFrontier
import numpy as np
env = FuelRLEnvSingleFrontier(num_pillars=15, max_steps=100)
for ep in range(3):
    obs, _ = env.reset()
    total = 0
    for i in range(100):
        obs, r, term, trunc, info = env.step(np.zeros(4, dtype=np.float32))
        total += r
        if term or trunc: break
    print(f'Ep {ep}: reward={total:.0f}, progress={info.get(\"exploration_progress\",0):.1%}')
"
```

## 编译 C++ 核心

```bash
# Python 环境变更后需要重新编译
python setup.py build_ext --inplace

# 验证编译
python -c "from fuel_rl_core import FuelEnvCore; print('OK')"
```
