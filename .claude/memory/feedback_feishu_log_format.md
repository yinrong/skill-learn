---
name: feedback-feishu-log-format
description: 只有 feishu tag 的 Langfuse 日志格式正确，其他 tag（cron/alarm/web）格式有问题
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a0e1968d-c511-4615-9dd0-dd424dd28649
---

只有 tag 为 `feishu_group` 或 `feishu_p2p` 的 Langfuse 日志格式正确，其他 tag（cron/alarm/web）日志格式损坏。

**Why:** cron/alarm 等日志中 user 消息内容包含 Python 协程对象字符串（`<coroutine object OpenAIChatModelCached.__call__ at 0x...>`），说明系统调用异步函数时忘记 await，存储的是对象引用而非实际内容。

**How to apply:**
- 所有训练/测试数据只从 feishu 标签抓取：`--langfuse_tags feishu_group feishu_p2p`
- feishu 正确格式标志：user message 末尾有 `## 用户问题` + 真实用户提问
- 当前可用的正确格式数据：`round4/data/ws_raw.jsonl`（93条）
- 重新生成的训练/测试集：`ws_feishu_train.jsonl`（84条）和 `ws_feishu_test.jsonl`（9条）
