---
name: 执行环境约束与可用资源
description: Docker 容器环境信息、权限约束、可用 API
type: reference
---

# 执行环境

## 容器信息

- OS：Linux 5.15.0-91-generic，Docker overlay 文件系统
- 运行用户：`hostuser`（uid=1000），无 sudo，无 CAP
- Shell 主目录：`/tmp`（hostuser 的 HOME）
- 持久化目录：`/home/yinrong/`（root 所有，需 chmod 才能写）

## 文件系统权限

| 路径 | 权限 | 说明 |
|------|------|------|
| `/home/yinrong/phase2.1/` | drwxrwxrwx（已 chmod） | **持久化，可写** |
| `/home/yinrong/`（其他） | drwxr-xr-x root | 不可直接写 |
| `/tmp/` | drwxrwxrwt | 可写但不持久化 |
| `/workspace/` | drwxr-xr-x root | 不可写 |
| `/data/` | 不存在 | 模型需要先创建此目录 |

## Python 环境

- Python 3.11（`/usr/bin/python3`）
- 已安装：numpy, scipy, scikit-learn, matplotlib, statistics
- 未安装：llamafactory, vllm, modelscope, transformers（需 pip install -r requirements.txt）

## 可用 API

```
ANTHROPIC_BASE_URL: http://model.mify.ai.srv/anthropic
ANTHROPIC_AUTH_TOKEN: sk-UFop8bGkZVJZUz1FVS9E5w20r1591Kj1d8i6i6AlI7VXkeic
模型：ppio/pa/claude-sonnet-4-6
```

可用于：真实基线评测、LLM 润色 formatter 输出

## GPU 状态

- 目标环境：4 × NVIDIA H20（96GB，NVLink，CUDA 12.6）
- 当前 session：**无 GPU**（容器内无法访问）
- 模型路径规划：`/data/models/Qwen3-32B`，`/data/models/Qwen3-14B`

## 关键操作记录

- phase2.1 目录 chmod：用户在 shell 执行 `chmod 777 /home/yinrong/phase2.1/`
- 文件从 /tmp/phase2.1/ 复制到 /home/yinrong/phase2.1/：`cp -r /tmp/phase2.1/. /home/yinrong/phase2.1/`
