# MiniMind 第一周计划

日期范围：2026-04-04 到 2026-04-10

## 目标

用一周的专注时间，围绕简化版的 MiniMind 风格训练栈，做出一个能写进简历、能在面试里讲清楚的小语言模型项目。优先级如下：

- 64M Dense 模型
- Pretrain
- Full SFT
- LoRA
- DPO
- GRPO
- PPO 原型或至少完成 smoke test

不要把 MoE 或完整的 Kimi 论文复现当成第一周核心范围。

## 为什么这样定范围

当前这个仓库里已经有一些重要基础：

- GPT 数据管线基础：`ch2_working_with_text_data/dataloader.py`
- GPT 模型基础：`ch4_implement_GPT/gpt.py`
- Greedy decoding：`ch4_implement_GPT/generate_text_greedy.py`
- 与预训练相关的 notebook 实践：`ch5_pretraining/ch5_main.ipynb`

这意味着第一周的重点不是重新学习 Transformer 基础原理，而是把训练链路补齐、工程化，并把项目做成一个可展示的作品。

## 硬件假设

开发机器：

- CPU：i9-14900KF
- GPU：RTX 4090 24GB

这套配置会明显缩短本地训练的墙钟时间，相比 3090 会更快，但不会同比例降低代码实现、debug 和调参的复杂度。

## 时间估算

### 专注开发时间

- 顺利情况：50-55 小时
- 更现实的情况：58-65 小时

### 额外训练墙钟时间

- 总计大约 8-12 小时，其中大部分可以和写代码、清理仓库、补文档并行进行

## 第一周预期产出

如果执行顺利，第一周结束时应当得到：

- 一个 64M Dense 小语言模型训练代码库
- 从 pretrain 到 post-training 的完整链路
- 可以运行的 checkpoint 或至少 smoke-tested 的脚本，包括：
  - pretrain
  - full SFT
  - LoRA
  - DPO
  - GRPO
  - PPO 原型
- 一套适合写简历和实习面试讲述的项目故事

## 范围规则

### 必做

- 只做 Dense 64M 主线
- 优先使用 mini 数据路径
- 先把训练链路端到端跑通
- 保留日志、checkpoint 和简单 eval 路径

### 伸展目标

- PPO 不止 smoke test，而是能有初步可解释的训练结果
- 在稳定 baseline 之上做一个极小的 Attention Residuals 实验

### 明确不属于第一周核心范围

- MoE 主线实现
- 完整的 Kimi Attention Residuals 复现
- 大规模 benchmark
- 大量超参数扫参
- 完整打磨后的 PPO

## 每日计划

## 第 1 天 - 2026-04-04

总计：8.0 小时

### 1. 确定范围和仓库结构 - 1.0h

- 冻结第一周范围为 64M Dense
- 确定仓库目录结构和命名方式
- 确定产出物：checkpoint、日志、eval 脚本、README 说明

### 2. 实现模型主干 - 3.0h

- 实现或改写：
  - RMSNorm
  - RoPE
  - Grouped-Query Attention
  - SwiGLU 前馈层
  - Decoder Block
  - Causal LM Head

### 3. 搭好核心训练接口 - 2.0h

- 增加 config 对象
- 如果计划使用 weight tying，就在这里加上
- 加入 forward 的 loss 路径
- 加入最小可用的 generate 方法

### 4. 做 smoke test - 2.0h

- 跑单 batch 的 forward/backward
- 确认 loss 在几步内能够下降
- 修 shape、mask、dtype、device 之类的问题

### 当日验收标准

- 模型能够正常 forward
- 模型能够计算 LM loss
- 模型能够 backward 而不报错
- 模型能够生成短文本

## 第 2 天 - 2026-04-05

总计：8.5 小时

### 1. 打通 pretrain 数据集路径 - 2.0h

- 实现 pretrain dataset loader
- 统一 tokenizer 输入路径
- 补齐 padding 和 label masking 逻辑

### 2. 实现 pretrain 训练循环 - 3.0h

- 实现：
  - optimizer
  - lr schedule
  - AMP
  - grad accumulation
  - grad clip
  - checkpoint 保存与恢复
  - 日志输出

### 3. 调通第一次真实训练 - 1.5h

- 修 batching、dataloader、checkpoint、OOM 等问题

### 4. 跑第一次 mini pretrain - 2.0h

- 跑一个小但真实的 pretrain 实验
- 保存第一个可用 checkpoint
- 记录基础 loss 变化趋势

### 当日验收标准

- 已有 `pretrain` checkpoint
- loss 在下降
- 训练能从 checkpoint 继续恢复

## 第 3 天 - 2026-04-06

总计：8.5 小时

### 1. 实现 SFT 数据集与 masking - 2.0h

- 打通对话数据集路径
- 加入 chat template 处理
- 如果需要，加入只监督 assistant 的 label masking

### 2. 实现 full SFT 训练循环 - 2.0h

- 创建 SFT 训练脚本
- 尽量复用 checkpoint 和 logging 基础设施

### 3. 打通 eval/chat 推理路径 - 1.5h

- 增加简单 eval 脚本或 chat 脚本
- 对比 pretrain checkpoint 和 SFT checkpoint 的输出差异

### 4. 实现 LoRA - 2.0h

- 加入 LoRA 注入逻辑
- 冻结基础模型权重
- 支持只保存 LoRA 权重

### 5. 跑 LoRA smoke test - 1.0h

- 跑一个最小 LoRA 微调实验
- 确认保存/加载路径是通的

### 当日验收标准

- 已有 `full_sft` checkpoint
- 简单交互 eval 可用
- LoRA 代码路径能够训练并保存成功

## 第 4 天 - 2026-04-07

总计：8.0 小时

### 1. 打通 DPO 数据路径 - 2.5h

- 实现 chosen/rejected 样本加载
- 确认 masking 正确
- 确认 prompt 格式与 SFT 模板保持一致

### 2. 实现 DPO 训练 - 2.0h

- 加入 reference model 加载
- 计算 per-token 或 sequence 级别 log-prob
- 实现 DPO objective

### 3. 补 DPO eval 路径 - 1.0h

- 加一个快速对比脚本，或写几个 notebook 对比单元

### 4. 跑第一次 DPO 实验并调试 - 2.5h

- 跑一个小实验
- 修不稳定、mask 错位、reference model 问题

### 当日验收标准

- DPO 训练可端到端运行
- 已保存一个 DPO checkpoint
- 至少有几组 base/SFT/DPO 的定性输出对比

## 第 5 天 - 2026-04-08

总计：9.0 小时

### 1. 搭 rollout 与 reward 基础设施 - 2.0h

- 增加最小 rollout 抽象
- 增加简单 reward function 或 reward model 接口
- 第一版尽量保持简单

### 2. 实现 GRPO - 3.0h

- 多采样生成
- grouped rewards
- advantage 标准化
- KL 项
- policy loss

### 3. 跑 GRPO smoke test - 1.0h

- 跑一个很小的实验

### 4. 稳定化和调试 GRPO - 3.0h

- 修 reward scale 问题
- 修 logging 和 rollout 问题
- 小范围调参，让训练结果至少可解释

### 当日验收标准

- GRPO 能更新参数且不崩
- reward 或 loss 指标被记录下来
- 至少有一个可复现的 GRPO 运行结果

## 第 6 天 - 2026-04-09

总计：9.0 小时

### 1. 实现 PPO - 3.0h

- Actor
- Critic
- GAE
- clipped policy loss
- value loss
- KL 处理

### 2. 调整显存和 batch 策略 - 1.5h

- 让 PPO 适配 4090 约束
- 必要时降低 batch、sequence length、update 次数

### 3. 跑 PPO smoke test - 1.0h

- 至少执行一次短训练

### 4. 预备 PPO 降级方案 - 1.5h

- 如果 PPO 很不稳定，把目标降为：
  - 可运行原型
  - 已知问题文档
  - 一次 smoke test 的日志

### 5. 清理训练基础设施 - 2.0h

- 统一 checkpoint 命名
- 统一训练命令
- 统一 eval 入口

### 当日验收标准

- 最好情况：PPO 能端到端跑通
- 最低可接受情况：PPO 原型可运行且已被记录清楚

## 第 7 天 - 2026-04-10

总计：9.0 小时

### 1. 做统一评估 - 2.0h

- 对比：
  - pretrain
  - full SFT
  - LoRA
  - DPO
  - GRPO
  - PPO（如果已完成）

### 2. 写 README 和文档 - 2.0h

- 说明项目目标
- 说明模型结构
- 说明每个训练阶段
- 加上运行命令

### 3. 准备简历和面试叙事 - 2.0h

- 写项目概述
- 写 2-3 条简历 bullet
- 写一段简短的面试讲法，解释设计选择

### 4. 清理仓库并检查可复现性 - 1.0h

- 清理明显的临时文件
- 检查仓库状态
- 验证主要命令还能正常执行

### 5. 核心完成后再做伸展实验 - 2.0h

- 尝试一个极小的 Attention Residuals 实验
- 只有在所有核心任务都稳定完成后才做

### 当日验收标准

- 项目可以演示
- 命令有文档
- 能把完整训练链路在面试里清楚讲出来

## 如果时间变紧，优先砍什么

按这个顺序砍：

1. MoE
2. 完整打磨的 PPO
3. Attention Residuals 实验
4. 额外调参和额外训练轮次

这些不能砍：

1. Dense baseline
2. Pretrain
3. Full SFT
4. 至少一条 PEFT 路线：LoRA
5. 至少一条偏好优化路线：DPO
6. 至少一条带在线 rollout 的 RL 路线：GRPO

## 主要风险

### 1. PPO 复杂度最高

PPO 很可能是第一周最难的部分，因为它引入了：

- actor model
- critic model
- GAE
- 多个 loss
- rollout 耦合
- 更大的调参空间

因此，PPO 更适合作为第一周的“原型目标”，而不是整个项目故事的唯一核心。

### 2. Reward model 带来的额外负担

即使主模型只有 64M，RL 阶段仍可能因为依赖更大的 reward model 而变重。如果出现这种情况：

- 先用 rule-based 或轻量 reward
- 先把整条 pipeline 跑通
- 后面再升级 reward 质量

### 3. Scope creep

第一周最大的失败模式，就是试图同时做完下面这些：

- Dense
- MoE
- 打磨完善的 PPO
- 完整 Kimi 复现

即使硬件很好，这个范围对一周来说也还是太大了。

## 推荐的最终项目叙事

第一周最强的项目叙事应该是：

“我从零实现了一个 64M Dense 小语言模型训练栈，覆盖 pretraining、full SFT、LoRA、DPO、GRPO，以及 PPO 原型。在搭好稳定 baseline 之后，我又为后续的结构实验（例如 Attention Residuals）预留了扩展空间。”

这个版本足够用于：

- 写简历
- 实习面试
- 技术项目 walkthrough
- 后续扩展到 MoE 或 Kimi 风格结构实验

## 第二周方向

第一周完成后，只选一个方向继续：

- 方案 A：做 MoE
- 方案 B：做一个极小的 Attention Residuals ablation

除非第一周明显提前完成，否则不要同时开两个方向。
