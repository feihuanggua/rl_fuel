# Skill: 编译 C++ 核心

## 触发条件
- 修改了 `src/` 下的 C++ 代码
- Python 环境变更
- 首次设置项目

## 前置条件
```bash
# Ubuntu 系统依赖
sudo apt install libeigen3-dev libpcl-dev python3-dev
```

## 执行步骤

```bash
conda activate torch
cd /home/jd3/FUEL/rl_fuel   # 改为你的路径

# 编译 C++ → Python 绑定
python setup.py build_ext --inplace

# 验证编译成功
python -c "from fuel_rl_core import FuelEnvCore; print('OK')"
```

## 常见问题

| 错误 | 原因 | 解决 |
|------|------|------|
| `fatal error: Eigen/Eigen: No such file` | Eigen3 未安装或路径不对 | `sudo apt install libeigen3-dev` |
| `fatal error: pcl/point_cloud.h` | PCL 未安装 | `sudo apt install libpcl-dev` |
| `ModuleNotFoundError: No module named 'pybind11'` | pybind11 不在当前环境 | `pip install pybind11` |
| `undefined symbol` 在 import 时 | .so 与 Python 版本不匹配 | 确认 conda 环境一致后重编 |

## 文件结构

```
src/standalone/   → C++ 实现 (SDF map, frontier, A*, raycast)
src/bindings/      → pybind11 绑定声明
setup.py           → 编译配置 (源文件列表、include/lib 路径)
```
