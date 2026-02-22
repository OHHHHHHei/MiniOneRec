# MiniOneRec 代码审查报告

> 审查目标：核对代码实现是否正确反映论文 "MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation" 的核心机制

---

## 1. 数据预处理

### 1.1 交互次数过滤（user/item < 5）

**[Pass] ✅**

[amazon18_data_process.py](file:///e:/MiniOneRec/data/amazon18_data_process.py) 的 `k_core_filtering_json2csv_style` 函数实现了迭代式 K-core 过滤，默认 `K=5`。shell 脚本中也明确设置 `--user_k 5 --item_k 5`。

### 1.2 序列截断（最多10个item）

**[Pass] ✅**

[amazon18_data_process.py](file:///e:/MiniOneRec/data/amazon18_data_process.py) 在 `convert_to_csv_format` 函数中对用户历史交互序列进行截断处理。`rl.py` 中也设置了 `len_seq = 10`。

### 1.3 时间范围过滤

**[Fail] ❌ — 结束月份偏差**

> [!WARNING]
> Shell 脚本配置的结束月份与论文 Appendix B 不一致。

| 参数 | 论文要求 | [amazon18_data_process.sh](file:///e:/MiniOneRec/data/amazon18_data_process.sh) 实际值 |
|------|----------|-------------------------------------------------------------|
| Toys_and_Games 起始 | 2016年10月 | `--st_year 2016 --st_month 10` ✅ |
| Toys_and_Games 结束 | 2018年**11月** | `--ed_year 2018 --ed_month 10` ❌ |

Shell 脚本中 `ed_month=10`，意味着截止到2018年10月，而论文规定应为 **2018年11月**。

**修复建议**：
```diff
# amazon18_data_process.sh
- --ed_month 10 \
+ --ed_month 11 \
```

> [!NOTE]
> Industrial_and_Scientific 数据集的时间范围（Oct 1996 ~ Nov 2018）需要在单独的 shell 脚本中确认，当前仅看到 Toys_and_Games 的配置示例。

---

## 2. 全流程 SID 对齐（SFT + RL）

### 2.1 SFT 阶段的对齐任务

**[Partial Pass] ⚠️ — 存在关键任务缺失/被注释**

当前 [sft.py](file:///e:/MiniOneRec/sft.py#L215-L227) 使用三个 Dataset 拼接：

| 任务 | Dataset 类 | 状态 |
|------|-----------|------|
| SID 序列预测 → 下一个 SID | `SidSFTDataset` | ✅ 活跃 |
| SID ↔ Title 双向翻译 | `SidItemFeatDataset` | ✅ 活跃（title2sid + sid2title） |
| SID History → Title 预测 | `FusionSeqRecDataset` | ⚠️ **仅保留 title 任务，description 任务被注释** |
| Title History → SID 预测 | `TitleHistory2SidSFTDataset` | ❌ **被注释掉** |
| 用户偏好总结生成 | `PreferenceSFTDataset` | ❌ **未使用** |
| 商品描述预测 | `FusionSeqRecDataset` 的 description 分支 | ❌ **被注释** |

> [!CAUTION]
> `FusionSeqRecDataset` 中描述预测的随机分支被完全注释（[data.py:L1363-L1371](file:///e:/MiniOneRec/data.py#L1363-L1371)），当前只执行 title 预测任务。论文强调的 description prediction 和 user summary generation 两个辅助对齐任务均不活跃。

```python
# data.py FusionSeqRecDataset.pre() 第1362-1373行
# 以下代码被注释：
"""if random.random() < 0.5:
    # Title task
    prompt = self.generate_prompt_title(history_data['history_str'])
    target = history_data['target_title'] + '\n'
else:
    # Description task
    prompt = self.generate_prompt_description(history_data['history_str'])
    target = history_data['target_description'] + '\n'
"""
# 仅保留了：
prompt = self.generate_prompt_title(history_data['history_str'])
target = history_data['target_title'] + '\n'
```

**修复建议**：取消注释 description 分支，恢复随机选择 title/description 的逻辑。

### 2.2 RL 阶段的对齐任务

**[Pass] ✅**

[rl.py](file:///e:/MiniOneRec/rl.py#L89-L103) 使用三个 Dataset 拼接：

| 任务 | Dataset 类 | 状态 |
|------|-----------|------|
| SID 序列预测 | `SidDataset` | ✅ 活跃 |
| Title/Description → SID | `RLTitle2SidDataset` | ✅ 活跃 |
| Title History → SID | `RLSeqTitle2SidDataset` | ✅ 活跃 |
| SID → Title (反向) | `RLSid2TitleDataset` | 已注释（可选） |

RL 阶段的核心对齐任务基本完整。`RLSid2TitleDataset` 和 `RLSidhis2TitleDataset` 被注释掉，但这不是论文的硬性要求。

---

## 3. RL 采样策略

### 3.1 约束束搜索 (Constrained Beam Search)

**[Pass] ✅**

[minionerec_trainer.py](file:///e:/MiniOneRec/minionerec_trainer.py#L479-L495) 在 `beam_search=True` 时创建了使用 `num_beams=self.num_generations` 的 `GenerationConfig`，配合 `ConstrainedLogitsProcessor` 实现约束解码。

### 3.2 约束解码（屏蔽非法 Token）

**[Pass] ✅**

[LogitProcessor.py](file:///e:/MiniOneRec/LogitProcessor.py) 的 `ConstrainedLogitsProcessor` 使用 `prefix_allowed_tokens_fn` 构建前缀树，将非法 token 的 logit 设为 `-inf`，确保只生成合法 SID 序列。

### 3.3 束宽度配置

**[Fail] ❌ — 束宽度为 8，非论文默认 16**

| 参数 | 论文默认值 | [rl.sh](file:///e:/MiniOneRec/rl.sh#L26) 实际值 |
|------|-----------|------------------------------------------------|
| `num_generations`（= beam width） | 16 | **8** |

代码中束宽度等于 `num_generations`（[minionerec_trainer.py:L485](file:///e:/MiniOneRec/minionerec_trainer.py#L485)：`num_beams=self.num_generations`），而 `rl.sh` 设置 `--num_generations 8`。

**修复建议**：
```diff
# rl.sh
- --num_generations 8 \
+ --num_generations 16 \
```

> [!IMPORTANT]
> 修改束宽度后需同步调整 `train_batch_size` 使其能被 `num_generations` 整除。当前 `train_batch_size=64`，改为 16 后 64/16=4 仍可整除。

### 3.4 长度归一化

**[Pass] ✅**

`length_penalty` 在 [minionerec_trainer.py:L484](file:///e:/MiniOneRec/minionerec_trainer.py#L484) 设置，默认值为 `0.0`（即禁用长度归一化）。这与论文"不使用长度归一化"的要求一致。

### 3.5 避免 Dynamic Sampling / Top-k

**[Pass] ✅ — 但有一个代码异常值得注意**

`rl.sh` 设置了 `--dynamic_sampling False` 和 `--beam_search True`，不会走 top-k 采样分支。

> [!NOTE]
> 值得注意的是，beam search 的 `GenerationConfig` 中同时设置了 `do_sample=True`（[minionerec_trainer.py:L492](file:///e:/MiniOneRec/minionerec_trainer.py#L492)），这在 HuggingFace 中会产生"随机束搜索"（stochastic beam search）而非确定性束搜索。论文中使用的应是标准 beam search，建议检查是否为有意设计。
>
> ```diff
> # minionerec_trainer.py GenerationConfig (beam_search=True)
> - do_sample=True,
> + do_sample=False,
> ```

---

## 4. RL 奖励机制

### 4.1 Rule-based 奖励

**[Pass] ✅**

[rl.py](file:///e:/MiniOneRec/rl.py#L187-L198) 的 `rule_reward` 函数：正确匹配时返回 `1.0`，不匹配返回 `0.0`。

### 4.2 Rank-aware Penalty

**[Pass] ✅ — 但归一化改变了原始语义**

`rl.sh` 使用 `--reward_type ranking`，此时奖励函数为 `[rule_reward, ndcg_rule_reward]` 的组合。

[rl.py:L157-L158](file:///e:/MiniOneRec/rl.py#L157-L158) 构建惩罚值：
```python
ndcg_rewards = [-1.0/math.log2(i+2) for i in range(num_generations)]
ndcg_rewards = [-elm/sum(ndcg_rewards) for elm in ndcg_rewards]
```

这里有两个要注意的细节：

| 检查项 | 论文公式 | 代码实现 |
|--------|---------|---------|
| 基础惩罚值 | `-1/log₂(ρ_k + 1)` | `-1.0/math.log2(i+2)` ✅ 等价 |
| 归一化处理 | 未明确提及 | 第二行做了归一化 ⚠️ |

> [!NOTE]
> 代码额外做了归一化（除以所有惩罚值之和），使惩罚值总和为 `-1.0`。这是一个合理的工程选择，但会改变惩罚的绝对幅度。如果论文中绝对惩罚幅度有特殊含义，则需核实。

`ndcg_rule_reward` 函数逻辑：
- 如果一组 `num_generations` 个候选中有正确答案：正确答案 reward=0.0，错误候选按排名获得递减惩罚
- 如果一组中**没有**正确答案：所有候选 reward=0.0（不惩罚）

这种"条件惩罚"的设计合理，避免了在没有正向信号时的无意义梯度更新。

### 4.3 避免 Collaborative Filtering 奖励

**[Pass] ✅**

`rl.sh` 设置 `--reward_type ranking`，此时不会触发 `cf_reward`（仅在 `reward_type == "sasrec"` 时激活）。代码中 `cf_reward` 使用 `SASRec` 模型打分，属于论文中提到的"Reward Hacking"风险项，正确地未被使用。

---

## 5. 训练超参数

### SFT 全局 Batch Size

**[Fail] ❌ — 有效全局 batch size = 128，远低于论文的 1024**

计算过程（基于 [sft.py:L149-L156](file:///e:/MiniOneRec/sft.py#L149-L156) 和 [sft.sh](file:///e:/MiniOneRec/sft.sh)）：

```
batch_size = 128
micro_batch_size = 8
nproc_per_node = 4

gradient_accumulation_steps = batch_size // micro_batch_size = 128 // 8 = 16
# DDP 模式下再除以 world_size:
gradient_accumulation_steps = 16 // 4 = 4

有效全局 batch size = micro_batch_size × world_size × gradient_accumulation_steps
                    = 8 × 4 × 4 = 128
```

| 参数 | 论文要求 | 实际值 |
|------|---------|-------|
| 全局 batch size | **1024** | **128** |

**修复建议（两种方案任选）**：

方案 A — 增加 GPUs（推荐，如果有 8 GPU）：
```diff
# sft.sh
- torchrun --nproc_per_node 4 \
+ torchrun --nproc_per_node 8 \
# 同时修改 batch_size：
- --batch_size 128 \
+ --batch_size 1024 \
```

方案 B — 保持 4 GPU，增大 gradient accumulation：
```diff
# sft.sh
- --batch_size 128 \
+ --batch_size 1024 \
```
此时：`gradient_accumulation_steps = 1024 // 8 = 128 → 128 // 4 = 32`，有效全局 batch = 8 × 4 × 32 = **1024** ✅

---

## 总结

| # | 检查项 | 结果 | 严重度 |
|---|--------|------|--------|
| 1.1 | K-core 过滤 (K=5) | ✅ Pass | — |
| 1.2 | 序列截断 (max 10) | ✅ Pass | — |
| 1.3 | 时间范围过滤 | ❌ Fail | 🟡 低 |
| 2.1 | SFT 对齐任务 | ⚠️ Partial | 🔴 高 |
| 2.2 | RL 对齐任务 | ✅ Pass | — |
| 3.1 | 约束束搜索 | ✅ Pass | — |
| 3.2 | 约束解码 | ✅ Pass | — |
| 3.3 | 束宽度 (16) | ❌ Fail | 🟡 中 |
| 3.4 | 长度归一化禁用 | ✅ Pass | — |
| 3.5 | 避免 Dynamic Sampling | ✅ Pass (注意 do_sample) | 🟡 低 |
| 4.1 | Rule-based 奖励 | ✅ Pass | — |
| 4.2 | Rank-aware Penalty | ✅ Pass (有归一化) | — |
| 4.3 | 避免 CF 奖励 | ✅ Pass | — |
| 5.1 | SFT 全局 Batch Size 1024 | ❌ Fail | 🔴 高 |

**需要优先修复**：
1. 🔴 SFT 全局 batch size 从 128 提升到 1024
2. 🔴 恢复 `FusionSeqRecDataset` 中 description 预测任务
3. 🟡 RL beam width 从 8 调整为 16
4. 🟡 时间范围 `ed_month` 从 10 修正为 11
