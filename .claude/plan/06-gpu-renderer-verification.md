# GPU 渲染器验证与 C++ 核心健康检查方案

## 目标

确保 GPU 加速路径和 CPU 路径产生一致结果，
防止位姿估计偏差影响 RL 训练质量。

## 当前状态

已修复:
- ✅ GPU renderer 相机内参对齐 C++ (fx/fy/cx/cy)
- ✅ `validate_camera_params()` 自动检测
- ✅ `SequenceEnv.reset()` 创建 GPU renderer 时自动调用验证

仍需完成:
- [ ] 同一位姿下 GPU vs CPU 输出一致性量化
- [ ] 渲染器性能 profiling
- [ ] C++ 内存泄漏根因定位

## GPU vs CPU 一致性测试

### 测试方案
```python
# 伪代码
for yaw in test_yaws:
    core_cpu.simulate_observation(pos, yaw)
    map_cpu = core_cpu.get_occupancy_slice_2d(1.5)

    core_gpu.reset()
    hit_pts = gpu_renderer.render_with_free(pos, yaw)
    core_gpu.input_hit_points(hit_pts, pos)
    map_gpu = core_gpu.get_occupancy_slice_2d(1.5)

    diff = (map_cpu != map_gpu).mean()
    assert diff < 0.01  # 差异 < 1%
```

### 已知差异来源
1. GPU renderer 的 z-buffer 使用 min depth (保留最近点)
2. CPU raycast 使用 step-wise 采样
3. 浮点精度差异 (float32 GPU vs float64 CPU)

## 性能基准

| 操作 | CPU 时间 | GPU 时间 | 加速比 |
|------|----------|----------|--------|
| simulateObservation (C++) | ~200ms | — | — |
| GPU render_with_free | — | ~5ms | — |
| inputPointCloud + inflate + ESDF | ~30ms | ~30ms | 1× |
| 总观测模拟 | ~230ms | ~35ms | ~7× |

> GPU renderer 只加速了 raycast 部分 (200→5ms)，后续 inflate + ESDF 仍在 CPU。

## C++ 内存问题排查

### 症状
`collector.py` 注释: "使用子进程避免 C++ 内存泄漏"

### 可能原因
1. PCL (Point Cloud Library): `pcl::KdTreeFLANN` 的外部指针管理
2. SDF map 的 occupancy_buffer 分配/释放周期
3. pybind11 Eigen 类型转换

### 排查步骤
```bash
# 1. 单进程循环测试
python -c "
from fuel_rl import FuelEnvCore
# ... init + load + reset 循环 1000 次
# 监控内存使用
"

# 2. Valgrind (需要 Linux)
valgrind --leak-check=full python test_leak.py

# 3. C++ 端添加 RAII 检查
# 在所有 PCL shared_ptr 重新赋值前显式 .reset()
```

## 关键文件

| 文件 | 变更需求 |
|------|----------|
| `fuel_rl/env/gpu_depth_renderer.py` | 已修复相机参数 |
| `fuel_rl/config.py` | 已新增 validate_camera_params() |
| `fuel_rl/env/sequence_env.py` | 已集成参数验证 |
| 待新增: `tools/test_gpu_consistency.py` | GPU vs CPU 一致性测试 |
| 待新增: `tools/profile_memory.py` | 内存泄漏诊断脚本 |
