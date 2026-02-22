# MiniOneRec 代码审查报告

根据《MiniOneRec》论文要求，对当前代码库的核心配置与逻辑进行审查。以下为提取的客观代码状态与偏差分析：

### 1. 核心 Bug 排查：RL 采样与验证配置 (minionerec_trainer.py & rl.sh)

- **训练采样配置 (Training Generation Config)**：[存在偏差]
  - **当前发现**：
    在 `minionerec_trainer.py` (L492) 中，`do_sample` 被显式设置为 `True`；`length_penalty` 被设置为 `0.0` (L484)。
    ```python
    # minionerec_trainer.py L482-L493
    self.generation_config = GenerationConfig(
        max_new_tokens=self.max_completion_length,
        length_penalty=self.length_penalty, # 传入值为 0.0
        num_beams=self.num_generations,
        num_return_sequences=self.num_generations,
        # ...
        temperature=self.temperature,
        do_sample=True, # 明确设置为 True
    )
    ```
  - **分析**：论文中在 RL 阶段使用的是标准的（确定性）约束束搜索 (Constrained Beam Search)。代码中开启了 `do_sample=True`，导致生成过程变为**随机束搜索** (Stochastic Beam Search)，引入了额外的随机性，这可能是导致“多样性坍缩”的原因之一。`length_penalty` 禁用（设为 0.0）与论文要求一致。

- **评估配置 (Eval Strategy)**：[存在偏差]
  - **当前发现**：
    在启动脚本 `rl.sh` (L31) 中，`--test_during_training` 被设置为 `False`。
    ```bash
    # rl.sh L31
    --test_during_training False \
    ```
  - **分析**：由于设置为 `False`，`minionerec_trainer.py` (L578) 中专门用于确定性评估的配置 (`do_sample=False`) 代码块未被触发。这导致 Hugging Face Trainer 使用训练时的采样配置（`do_sample=True`）进行周期性验证，使得输出的 Eval 指标包含随机性，产生了“验证幻觉”。

---

### 2. 语义对齐任务完整性排查 (data.py & sft.py)

- **SFT 阶段语义对齐任务**：[存在偏差]
  - **当前发现**：
    在 `sft.py` 构建训练集的数据加载代码中，部分任务被**注释掉（停用）**，同时在 `data.py` 的具体类实现中，商品描述预测分支也被**强制注释跳过**。
    ```python
    # sft.py L215-L227
    train_datasets = []
    train_data1 = SidSFTDataset(...) # 活跃：SID 序列预测
    train_datasets.append(train_data1)
    train_data2 = SidItemFeatDataset(...) # 活跃：SID <-> Title
    train_datasets.append(train_data2)
    train_data3 = FusionSeqRecDataset(...) # 活跃：目前仅实现了 Title
    train_datasets.append(train_data3)
    
    # 以下任务被注释，并未运行：
    # train_data4 = SFTData(...) 
    # train_datasets.append(train_data4)
    # train_data5 = TitleHistory2SidSFTDataset(...) # 【缺失】：文本历史到 SID 预测
    # train_datasets.append(train_data5)
    
    # PreferenceSFTDataset 完全没有在 sft.py 中被导入或实例化 【缺失】：用户偏好总结
    ```
    ```python
    # data.py L1363-L1373 (FusionSeqRecDataset 中)
    """if random.random() < 0.5:
        # Title task
        prompt = self.generate_prompt_title(history_data['history_str'])
        target = history_data['target_title'] + '\n'
    else:
        # Description task
        prompt = self.generate_prompt_description(history_data['history_str'])
        target = history_data['target_description'] + '\n'
    """
    # 强制覆盖为只有 Title 【缺失】：商品描述预测
    prompt = self.generate_prompt_title(history_data['history_str'])
    target = history_data['target_title'] + '\n'
    ```
  - **分析**：`TitleHistory2SidSFTDataset`、`PreferenceSFTDataset` 均未运行；`FusionSeqRecDataset` 中的商品描述预测逻辑被手动注释。这与论文强调的“全流程、多种类的语义对齐（SID 描述、用户摘要辅助学习）”机制**严重不符**，模型可能无法在 SFT 阶段充分获得语义到 SID 的映射能力。

---

### 3. RL 奖励函数实现细节 (rl.py)

- **Rank-aware penalty 归一化**：[存在偏差]
  - **当前发现**：
    在 `rl.py` 中，定义 `ndcg_rule_reward` 的惩罚分布时，代码对公式得出的基础值进行了**全局归一化**，强制让一个 batch 里的 penalty 总和变为 -1.0。
    ```python
    # rl.py L157-L158
    ndcg_rewards = [-1.0/math.log2(i+2) for i in range(num_generations)]
    # 额外进行了归一化操作
    ndcg_rewards = [-elm/sum(ndcg_rewards) for elm in ndcg_rewards]
    ```
  - **分析**：原论文的 Rank-aware penalty 公式为单纯的 $-1/\log(\rho_k+1)$。而在代码中，增加了一行 `[-elm/sum(ndcg_rewards)]` 的除法操作。这极大地改变了论文中建议的绝对惩罚力度幅值（使原本越靠后的较大惩罚值被强行缩放变小），削弱了对无效召回的惩罚强度。

---

### 4. 训练超参数及数据对齐 (sft.sh, rl.sh, amazon18_data_process.sh)

- **SFT 全局 Batch Size**：[存在偏差]
  - **当前发现**：
    在 `sft.sh` 中：`torchrun --nproc_per_node 4` (GPU 数量 = 4)。脚本内给定的变量：
    ```bash
    # sft.sh L10, L13-L14
    torchrun --nproc_per_node 4 \
    --batch_size 128 \
    --micro_batch_size 8 \
    ```
    配合 `sft.py` L149 的逻辑 `gradient_accumulation_steps = batch_size // micro_batch_size // world_size`，计算得出的 `gradient_accumulation_steps = 128 // 8 // 4 = 4`。
  - **分析**：最终模型见到的全局 Batch Size 实际上就是启动脚本里的 **`128`**（即 `4 (GPUs) * 8 (micro) * 4 (grad_accum)`）。这与原论文推荐的 SFT 全局 Batch Size **`1024`** 存在极大差距，可能导致梯度更新方向不稳定。

- **RL 采样束宽**：[存在偏差]
  - **当前发现**：
    在 `rl.sh` 中，`--num_generations` 被设置为 8。
    ```bash
    # rl.sh L26
    --num_generations 8 \
    ```
  - **分析**：原论文中，标准的强化学习阶段的采样束宽（即生成分支数量）应为 **`16`**。这里设置成了 `8`，导致强化学习探索空间减半。

- **Toys 数据集截断时间**：[存在偏差]
  - **当前发现**：
    在 `amazon18_data_process.sh` 中：
    ```bash
    # amazon18_data_process.sh L7-L8
    --ed_year 2018 \
    --ed_month 10 \
    ```
  - **分析**：按照原论文 Appendix B 的设定，Toys 数据集的结束时间应该为 2018 年 **11 月**，而脚本中截断在了 10 月。这会导致训练集丢失最后一个月的数据，未能完全等价于论文中汇报的实验数据拆分。
