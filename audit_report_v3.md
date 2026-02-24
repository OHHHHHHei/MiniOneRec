# MiniOneRec 代码库工程审计报告 (V3)

> **审查时间**：2026-02-24  
> **审查范围**：基于当前最新代码库状态，覆盖 `rl.py`, `minionerec_trainer.py`, `rl.sh`, `sft.sh`, `sft.py`, `data.py`, `amazon18_data_process.sh`  
> **审查目标**：逐项提取具体代码片段、参数值和当前生效状态，作为深度复现与系统工程审计报告的"工程实锤证据"

---

## 1. 核心解码与验证逻辑 (minionerec_trainer.py & rl.sh)

### 1.1 训练采样配置 (Generation Config)

**【当前状态】**：`do_sample` 已被显式设置为 `False`（确定性束搜索）。共显式配置了 8 个生成参数。

**【代码实锤】** `minionerec_trainer.py` 第 482~495 行：

```python
self.generation_config = GenerationConfig(
    max_new_tokens=self.max_completion_length,   # 值: 128 (GRPOConfig 中设定)
    length_penalty=self.length_penalty,           # 值: 0.0
    num_beams=self.num_generations,                # 值: 8 (来自 rl.sh --num_generations 8)
    num_return_sequences=self.num_generations,     # 值: 8
    pad_token_id=processing_class.pad_token_id,
    eos_token_id=processing_class.eos_token_id,
    top_k=None,                                    # 显式置空，屏蔽 Instruct 模型默认值
    top_p=None,                                    # 显式置空，屏蔽 Instruct 模型默认值
    temperature=self.temperature,                  # 值: 1.0 (来自 rl.sh --temperature 1.0)
    do_sample=False,                               # 确定性束搜索
)
```

* `repetition_penalty`：**未显式配置**。

---

### 1.2 前缀树状态隔离

**【当前状态】**：训练和测试**已使用独立的 `ConstrainedLogitsProcessor` 实例**（`ccc_train` 和 `ccc_test`），`count` 状态泄漏问题已修复。

**【代码实锤】** `minionerec_trainer.py` 第 689~703 行：

```python
ccc_train = ConstrainedLogitsProcessor(
        prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,
        num_beams=self.num_generations if self.beam_search else 1,  # 值: 8
        base_model=self.base_model,
        eos_token_id=self.processing_class.eos_token_id
    )
self.logits_processor = LogitsProcessorList([TemperatureLogitsWarper(temperature=self.temperature), ccc_train])

ccc_test = ConstrainedLogitsProcessor(
        prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,
        num_beams=self.test_beam,                                   # 值: 20
        base_model=self.base_model,
        eos_token_id=self.processing_class.eos_token_id
    )
self.test_lp_list = LogitsProcessorList([ccc_test])
```

附：`self.test_generation_config`（第 574~582 行）同样已包含 `top_k=None`, `top_p=None` 的显式屏蔽：

```python
self.test_generation_config = GenerationConfig(
    max_new_tokens=self.max_completion_length,
    length_penalty=self.length_penalty,
    num_beams=self.test_beam,            # 值: 20
    num_return_sequences=self.test_beam,
    top_k=None,
    top_p=None,
    do_sample=False,
    pad_token_id=self.processing_class.pad_token_id,
    eos_token_id=self.processing_class.eos_token_id,
)
```

---

### 1.3 实时验证开关

**【当前状态】**：`--test_during_training` 当前值为 **`True`**（已激活纯确定性测试分支）。

**【参数实锤】** `rl.sh` 第 31 行：

```bash
--test_during_training True \
```

---

## 2. RL 奖励函数与惩罚机制 (rl.py)

### 2.1 Rank-aware Penalty 计算

**【当前状态】**：代码中**仍然存在**除以 `sum` 的全局归一化操作。惩罚值并非论文中的绝对值 $-1/\log_2(\rho_k+1)$，而是经过归一化后的相对比例值。

**【代码实锤】** `rl.py` 第 157~158 行（惩罚池构建）：

```python
ndcg_rewards = [-1.0/math.log2(i+2) for i in range(num_generations)]
ndcg_rewards = [-elm/sum(ndcg_rewards) for elm in ndcg_rewards]    # 全局归一化
```

`rl.py` 第 169~175 行（惩罚分配逻辑）：

```python
for i, completion in enumerate(completions):
    if completion.strip("\n\" ") == targets[i].strip("\n\" "):
        flag = True
        lis.append(0.0)                                  # 命中 Ground Truth → 0 惩罚
    else:
        lis.append(ndcg_rewards[i%num_generations])       # 未命中 → 按排名位置分配归一化惩罚
```

`rl.py` 第 177~183 行（触发条件）：

```python
    if (i+1)%num_generations == 0:
        if flag:                       # 该 prompt 组内存在至少一个命中
            rewards.extend(lis)        # → 使用真实的梯度惩罚信号
        else:
            rewards.extend([0.0] * repeat)  # → 全部未命中时，整组奖励归零
        flag = False
        lis = []
```

---

### 2.2 KL 散度约束

**【当前状态】**：KL 惩罚系数 `beta` 当前**实际生效值为 `1e-3`**（由 `rl.sh` 命令行参数覆盖了 `rl.py` 中的默认值 `0.04`）。

**【参数实锤】**

`rl.py` 第 54 行（函数签名默认值）：

```python
beta: float = 0.04,
```

`rl.sh` 第 35 行（命令行传参，**覆盖了上述默认值**）：

```bash
--beta 1e-3 \
```

`rl.py` 第 278 行（注入 GRPOConfig）：

```python
beta=beta,
```

---

## 3. SFT 阶段全过程语义对齐任务 (sft.py & data.py)

### 3.1 Description Prediction（商品描述预测）

**【当前状态】**：❌ **被屏蔽**。使用 Python 多行字符串 `"""` 将整个 50% 概率执行描述预测的分支注释掉，强制 100% 只做 Title 预测。

**【代码实锤】** `data.py` 第 1362~1373 行（`FusionSeqRecDataset.pre()` 方法内）：

```python
# Randomly choose between title and description tasks
"""if random.random() < 0.5:
    # Title task
    prompt = self.generate_prompt_title(history_data['history_str'])
    target = history_data['target_title'] + '\n'
else:
    # Description task
    prompt = self.generate_prompt_description(history_data['history_str'])
    target = history_data['target_description'] + '\n'
"""
prompt = self.generate_prompt_title(history_data['history_str'])    # 硬编码仅 Title
target = history_data['target_title'] + '\n'
```

---

### 3.2 PreferenceSFTDataset（用户偏好总结生成）

**【当前状态】**：❌ **未实例化**。该类虽被 `import`，但在数据集组装流程中**从未被调用或加入训练集列表**。

**【代码实锤】** `sft.py` 第 26 行（导入语句存在）：

```python
from data import ..., PreferenceSFTDataset, UserPreference2sidSFTDataset, ...
```

`sft.py` 第 215~227 行（数据集组装，**无任何 `PreferenceSFTDataset` 的实例化代码**）：

```python
train_datasets = []
train_data1 = SidSFTDataset(...)
train_datasets.append(train_data1)
train_data2 = SidItemFeatDataset(...)
train_datasets.append(train_data2)
train_data3 = FusionSeqRecDataset(...)
train_datasets.append(train_data3)
# ← 此处无 PreferenceSFTDataset 实例
train_data = ConcatDataset(train_datasets)
```

---

### 3.3 TitleHistory2SidSFTDataset（Title 历史到 SID 映射）

**【当前状态】**：❌ **被 `#` 注释屏蔽**。

**【代码实锤】** `sft.py` 第 225~226 行：

```python
# train_data5 = TitleHistory2SidSFTDataset(train_file=train_file, item_file=item_meta_path, ...)
# train_datasets.append(train_data5)
```

---

## 4. 算力、超参数与数据截断 (sft.sh, rl.sh, amazon18_data_process.sh)

### 4.1 SFT 批次大小

**【当前状态】**：`sft.sh` 中使用自定义的 `--batch_size` 和 `--micro_batch_size`，由 `sft.py` 内部计算梯度累积步数。有效全局 Batch Size 为 **128**。

**【参数实锤】** `sft.sh` 第 10~14 行：

```bash
torchrun --nproc_per_node 4 \
        sft.py \
        --batch_size 128 \
        --micro_batch_size 8 \
```

`sft.py` 第 149~156 行（梯度累积实际计算逻辑）：

```python
gradient_accumulation_steps = batch_size // micro_batch_size        # = 128 // 8 = 16
# ...
gradient_accumulation_steps = gradient_accumulation_steps // world_size  # = 16 // 4 = 4
```

| 参数 | 值 |
|---|---|
| `micro_batch_size`（即 per_device_train_batch_size）| **8** |
| `gradient_accumulation_steps`（最终生效）| **4** |
| 有效全局 Batch Size（`8 × 4 GPU × 4 累积`）| **128** |

---

### 4.2 RL 采样束宽

**【当前状态】**：`num_generations` 当前值为 **`8`**。

**【参数实锤】** `rl.sh` 第 26 行：

```bash
--num_generations 8 \
```

---

### 4.3 数据截断时间

**【当前状态】**：Toys_and_Games 数据集的处理截止时间为 **2018 年 10 月**。

**【参数实锤】** `amazon18_data_process.sh` 第 7~8 行：

```bash
--ed_year 2018 \
--ed_month 10 \
```
