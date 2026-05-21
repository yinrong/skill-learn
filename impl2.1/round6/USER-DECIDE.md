# Round 6 待用户决策项

| # | 问题 | 当前假设（不等待用户时用）| 状态 |
|---|------|------------------------|------|
| 1 | simulated_tools 的 metrics 应返回所有 Type A model_id，还是按 skill 返回？ | 返回全部：line 对应的 metrics 返回所有维度（12202/12198/12204/12385/12389/12194/12223/12203），模型用 skill 上下文选择正确的 | 待确认 |
| 2 | equipment-cpk 的 device 相关 model_id（12365）是否准确？ | 用 12365 作为 CPK model_id，与 R5 训练数据一致 | 待确认 |
| 3 | 分板机训练数据是否需要从 get_object_type_data 开始（而非 list）？ | 是，参照 GT 模式，分板机技能先查实体再查 metrics | 待确认 |
