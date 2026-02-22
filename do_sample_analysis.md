# MiniOneRec 项目 `do_sample` 机制全景深度分析报告

> **审查目标**：全面梳理整个项目中所有涉及大模型生成（Generation）过程的 `do_sample`（随机采样 vs 确定性搜索）的配置及其触发条件，以明确训练、验证和最终测试时模型究竟是如何生成推荐结果的。

---

## 核心结论速览

在你的整个项目中，`do_sample` 的存在状态可以分为 **3 种完全不同的情况（场景）**：

| 场景 | 相关阶段 | 所在文件 | `do_sample` 设置 | 生成方式性质 | 备注说明 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **场景 A** | **RL 训练（采样收集经验）** | `minionerec_trainer.py` | **`True`** | 随机束搜索 或 纯随机采样 | GRPO 算法要求的勘探 (Exploration) 必须有随机性。 |
| **场景 B** | **RL 过程中的定期验证 (Eval)** | `minionerec_trainer.py` | **实际上是 `True`** | 随机生成 | **(代码逻辑漏洞)** 本该是用 `False` 跑确定性测试，但因参数开关没打开，退化成了使用训练时的 `True` 配置。 |
| **场景 C** | **最终离线系统测评测试** | `evaluate.py` | **`False`** | 确定性束搜索 | 严格的离线评测，结果具有完全的可重复性。 |

以下是详细的代码溯源和逻辑梳理：

---

## 详尽分类剖析

### 场景 A：RL 训练阶段的采样生成（必须且正确）

在特征空间的强化学习（GRPO）中，模型需要针对同一个 Prompt 生成多条不同的回复（轨迹），然后依靠 Reward 机制给它们打分。如果生成是完全确定的（`do_sample=False`），那给定同一个 Prompt 和 Beam width，模型每次都会生成完全一模一样的多条序列，就失去了"强化学习试错"（Exploration）的意义。

**代码溯源**：`minionerec_trainer.py` (大概 L479-L504)
```python
if self.beam_search:
    self.generation_config = GenerationConfig(
        # ...
        num_beams=self.num_generations,
        temperature=self.temperature,
        do_sample=True, # 开启了随机束搜索 (Stochastic Beam Search)
        # ...
    )
else:
    self.generation_config = GenerationConfig(
        # ...
        do_sample=True, # 无论是否 Beam Search，采样的底层都必须是 True
        temperature=args.temperature,
        # ...
    )
```
**影响**：这使得在 RL 训练期间，模型通过 `self.model.generate(..., generation_config=self.generation_config)` 获取经验数据时，具有正常的随机探索能力。**这是符合 RL 逻辑的正确设置。**

---

### 场景 B：RL 训练过程中的验证 (Eval / Validation)

在训练过程中，系统会根据 `eval_steps` 定期在验证集上跑验证，以输出 WandB 上的折线图。我们理想中希望这些验证指标（比如 HR / NDCG）是基于模型此时此刻“最确定的最大概率推荐”算出来的。但这部分在你的代码中存在**巨大的逻辑分歧**。

#### B.1. 理想状态（专门写了代码，但没生效）
你在 `minionerec_trainer.py` (大概 L578) 专门为”边聊边测“写了一个 `test_generation_config`：
```python
self.test_generation_config = GenerationConfig(
    # ...
    num_beams=self.test_beam,
    do_sample=False, # 明确设置为了 False (确定性的 Beam Search)
    # ...
)
```
这段代码旨在严格计算推荐指标 (HR/NDCG)。但它的触发条件，包裹在 `minionerec_trainer.py` L734 的 `if self.test_during_training:` 中。

#### B.2. 实际运行状态（跑到了 `do_sample=True`）
在你的 `rl.sh` 脚本中，你显式传递了：
```bash
--test_during_training False \
```
这导致触发专门 `test_generation_config` (`do_sample=False`) 的门被关上了。
此时，Huggingface Trainer 的原生 Eval 流程接管了。它直接复用了 `self.generation_config`。
所以你的项目现在在训练期间打出的验证指标（Loss、Reward 甚至部分的 Completion 预览日志），**全都是在 `do_sample=True` （随机采样）下算出来的。** 这会导致你的验证曲线波动可能比实际情况要剧烈。

---

### 场景 C：最终测评脚本（离线测试）

在整个模型训练结束（SFT 和 RL 都打完收工）后，你会使用独立的 `evaluate.py` 脚本来跑最终的 Test 集合，并汇报最终写到论文里的性能数据。

这一步非常关键的、用于定论的数据跑法，是**没有随机性**的。

**代码溯源**：`evaluate.py` (L186-L197)
```python
generation_config = GenerationConfig(
    num_beams=num_beams,         # 测试时传入了很大的搜索束宽（比如 50）
    length_penalty=length_penalty,
    num_return_sequences=num_beams,
    # ...
    do_sample=False,             # 明确为 False
    # ...
)
```
**影响**：最终你在 `evaluate.py` 跑出来的预测结果，是模型严谨、稳定、最大概率解（Deterministic Constraint Beam Search）。**这是符合推荐系统评测规范的正确设置。** 如果你不改代码，测上一百遍，结果也是完全一样的。

---

## 💡 给开发者的行动建议总结

梳理完这个全貌，你需要根据自己的目标做少数修改：

1. **如果你希望训练过程中的 WandB 上的指标（HR/NDCG）能更平稳、准确地反映模型真实的进展：**
   * 修改 `rl.sh`：
     ```diff
     - --test_during_training False \
     + --test_during_training True \
     ```
   * 这样就能激活你写的 `test_generation_config`，让中间验证变成 `do_sample=False`。

2. **如果 `evaluate.py` 就是你的唯一评测标准，对训练图表要求不高：**
   * 那你**完全不需要改任何一行代码**。
   * 因为 RL 吸收经验需要的 `do_sample=True` 是正确的，而最终评测脚本里的 `do_sample=False` 也是绝对正确的。中间的日志稍有随机性并不会影响最终论文的评测数据。
