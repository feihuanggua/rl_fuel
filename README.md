# RL-FUEL: 强化学习驱动的 UAV 探索前沿排序

基于 [FUEL](https://github.com/HKUST-Aerial-Robotics/FUEL) 的 RL 训练框架，用强化学习替代 TSP 启发式来决策前沿访问顺序，实现更高效的未知环境探索。

## 架构

```
C++ 核心 (pybind11)          Python RL 框架
┌─────────────────────┐     ┌──────────────────────┐
│ SDF Map             │     │ SequenceEnv (Gym)     │
│ Frontier Finder     │◄───►│   ├─ frontier 观测    │
│ A* Path Search      │     │   ├─ 2D 俯视图        │
│ Raycast Sensing     │     │   └─ 覆盖率奖励       │
│ Depth Rendering     │     │                      │
└─────────────────────┘     │ OrderPolicy (CNN+MLP)│
                            │   ├─ MapCNN 编码器    │
                            │   ├─ 前沿评分网络     │
                            │   └─ Value head       │
                            │                      │
                            │ 训练算法              │
                            │   ├─ PPO              │
                            │   └─ SAC              │
                            └──────────────────────┘
```

## 环境要求

- Python >= 3.8
- C++14 编译器 (GCC >= 7)
- Eigen3, PCL >= 1.10
- CUDA (可选，GPU 深度渲染加速)

### 系统依赖 (Ubuntu)

```bash
sudo apt install libeigen3-dev libpcl-dev python3-dev
```

### Python 依赖

```bash
pip install -r requirements.txt
pip install torch  # 根据 CUDA 版本自行安装
```

### 编译 C++ 核心

```bash
pip install pybind11
cd rl_fuel
python setup.py build_ext --inplace
```

## 项目结构

```
rl_fuel/
├── src/standalone/        # C++ 核心实现 (SDF地图、前沿检测、光线投射)
├── src/bindings/          # pybind11 绑定
├── fuel_rl/
│   ├── config.py          # 全局配置 (网格大小、训练超参)
│   ├── map_loader.py      # 随机地图生成
│   ├── env/
│   │   ├── sequence_env.py  # 序列级 Gym 环境
│   │   └── gpu_depth_renderer.py
│   ├── models/
│   │   └── order_policy.py  # CNN+MLP 排序策略网络
│   ├── train/             # 训练脚本 (PPO, SAC, BC)
│   ├── eval/              # 评估与可视化
│   └── data/              # 数据收集与处理
├── scripts/               # 训练启动脚本
├── tools/                 # 调试与可视化工具
├── setup.py               # 编译配置
├── pyproject.toml
└── requirements.txt
```

## 使用方法

### 训练 SAC

```bash
python run_sac_seq5.py --total-steps 50000 --num-pillars 15
```

### 训练 PPO

```bash
bash scripts/run_seq_ppo.sh
```

### 评估

```bash
python -m fuel_rl.eval.quick_eval --checkpoint fuel_rl_checkpoints/sac_seq5/best_model.pth
```

## 核心思路

1. **环境**: 每步 FUEL 原生方法检测前沿并生成视点，策略只需选择访问哪个前沿
2. **观测**: 前沿特征 (中心、大小、距离、可见性) + 2D 俯视占据图 + 全局覆盖率
3. **奖励**: 覆盖率增量，达到目标覆盖率给额外奖励
4. **网络**: MapCNN 提取地图特征，MLP 对每个前沿打分，选得分最高的访问
