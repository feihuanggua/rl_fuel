# Git 提交规范

## 提交前缀 (Commit Prefixes)

所有提交必须使用以下前缀：

| 前缀 | 用途 | 示例 |
|------|------|------|
| `feat:` | 新功能 | `feat: 支持4m ray距离的大地图训练` |
| `fix:` | Bug修复 | `fix: 修复ARiADNE地图agent起点在墙内的问题` |
| `refactor:` | 重构（不改功能） | `refactor: 将map_img构建改为动态尺寸` |
| `docs:` | 文档更新 | `docs: 添加ARiADNE地图适配说明` |
| `chore:` | 构建/配置/工具 | `chore: 添加seq12训练service文件` |
| `test:` | 测试相关 | `test: 添加大地图corridor宽度分析脚本` |
| `perf:` | 性能优化 | `perf: 向量化load_ariadne_map点云生成` |
| `style:` | 代码格式 | `style: 统一缩进为4空格` |

## 格式要求

```
<前缀>: <简短描述>

<可选：详细说明>
```

- 描述用中文，简洁明了
- 一行不超过 72 字符
- 详细说明与描述之间空一行

## 分支规范

| 分支 | 用途 |
|------|------|
| `main` | 稳定版本 |
| `dev` | 开发分支 |
| `feat/*` | 功能开发分支 |
| `fix/*` | 修复分支 |

## 工作流程

1. 在 `dev` 或功能分支上开发
2. 每次有意义的改动都提交（不要积累太多改动）
3. 提交前确认 `git diff` 只包含预期改动
4. 不要提交临时文件、缓存、checkpoint模型文件

## 忽略文件规则

- `*.pth` — 模型权重文件（太大）
- `*.npz` — 快照数据文件
- `__pycache__/` — Python缓存
- `*.egg-info/` — 包信息
- `ariadne_maps/` — 原始地图副本
- 临时脚本 `tmp_*.py`, `test_*.py`, `debug_*.py`, `patch_*.py`, `check_*.py`
