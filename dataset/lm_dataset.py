from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

# 禁用 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多进程环境中产生死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ──────────────────────────────────────────────────────────────────────────────
# 全局预处理 / 后处理工具函数
# ──────────────────────────────────────────────────────────────────────────────


def pre_processing_chat(conversations, add_system_ratio=0.2):
    """
    对话前处理：以一定概率随机插入 system 消息。

    特点：
    - 只有当首条消息不是 system 角色时才可能插入。
    - add_system_ratio 控制插入概率（默认 20%），引入随机性可提升模型
      对有/无 system prompt 两种情况的泛化能力。
    - system 内容从预定义的中英文 prompt 池中随机抽取，覆盖不同表达风格。
    """
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是FeiFeiMind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是FeiFeiMind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are FeiFeiMind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are FeiFeiMind, a small but useful language model.",
    ]
    if conversations and conversations[0].get("role") != "system":
        if random.random() < add_system_ratio:
            return [
                {"role": "system", "content": random.choice(SYSTEM_PROMPTS)}
            ] + conversations
        
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    """
    对话后处理：清理模板渲染后多余的空 <think> 块。

    特点：
    - 针对带 CoT（chain-of-thought）格式的模型，apply_chat_template 有时会
      渲染出 "<think>\n\n</think>\n\n" 这样的空思考块占位符。
    - 大部分情况下（概率 1 - empty_think_ratio = 95%）直接删除该空块，
      防止模型学到"无意义思考"的坏习惯。
    - 保留少量空思考块（empty_think_ratio = 5%），让模型也能处理该边界情况。
    """
    if "<think>\n\n</think>\n\n" in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace("<think>\n\n</think>\n\n", "")


# ──────────────────────────────────────────────────────────────────────────────
# 1. PretrainDataset —— 自回归预训练数据集
# ──────────────────────────────────────────────────────────────────────────────
# 训练目标：Next-Token Prediction（下一个 token 预测）
# 数据格式：{"text": "一段原始文本"}
# 训练特点：
#   - 模型对整段文本的每个位置都进行预测，没有"只学回复"的区分。
#   - 使用 BOS/EOS 标记文本边界，让模型学会文本的起止。
#   - PAD token 对应的 label 置 -100，不参与 loss 计算，节省无效梯度。
#   - labels 直接 clone 自 input_ids（即 X 和 Y 错位一格：Y[t] = X[t+1]）。
# ──────────────────────────────────────────────────────────────────────────────
class PretrainDataset(Dataset):
    """预训练阶段使用的语言模型数据集。

    每条样本会被处理成定长序列，并返回：
    input_ids、
    labels、
    attention_mask。
    """

    def __init__(self, data_path, tokenizer, max_length=512):
        """初始化预训练数据集。

        Args:
            data_path: JSON 数据文件路径。
            tokenizer: 用于分词的 tokenizer。
            max_length: 输出序列的固定长度。
        """

        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 使用 HuggingFace datasets 的惰性加载，避免一次性读入大文件
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        """返回数据集样本数。"""
        return len(self.samples)
    
    def __getitem__(self, index):
        """取出单条样本并转换成训练所需的张量。

        Args:
            index: 样本下标。

        Returns:
            input_ids: shape 为 [T] 的输入 token 序列。
            labels: shape 为 [T] 的训练标签，PAD 位置为 -100。
            attention_mask: shape 为 [T] 的 attention mask，
                有效 token 为 1，padding 为 0。
        """

        sample = self.samples[index]

        # Step 1: tokenize 原始文本，留出首尾各 1 个 token 的位置给 BOS/EOS
        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True, # 如果正文 token 数超过max_length，就直接裁掉后面的部分
        ).input_ids

        # Step 2: 拼接 BOS + token序列 + EOS，构成完整序列
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        # Step 3: 右侧用 PAD 补齐到 max_length，保证 batch 内等长
        # input_ids: [T]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Step 4: labels 与 input_ids 完全相同（见model.py的处理），
        #         但 PAD 位置置为 -100, CrossEntropyLoss 会自动忽略 -100， 不计入 loss
        # labels: [T]
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        # 返回 attention_mask，使 attention 层能屏蔽 padding token
        # attention_mask: [T]
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return input_ids, labels, attention_mask

# ──────────────────────────────────────────────────────────────────────────────
# 2. SFTDataset —— 有监督微调（Supervised Fine-Tuning）数据集
# ──────────────────────────────────────────────────────────────────────────────
# 训练目标：让模型学会"只预测 assistant 回复"，忽略 user/system 输入
# 数据格式：{"conversations": [{"role": "user"/"assistant"/"system", "content": "..."}]}
# 训练特点：
#   - 通过 generate_labels 扫描 bos_id（assistant 回复起始标记）定位每段回复，
#     仅将 assistant 回复的 token 位置设为有效 label，其余全部为 -100。
#   - 这样做的意义：让 loss 只反映模型对"正确回答"的拟合，不浪费梯度在
#     用户输入的复现上（用户输入只作为 context，不是预测目标）。
#   - 支持 function calling：若 system 消息携带 "functions" 字段，
#     会透传给 apply_chat_template，生成带工具描述的提示词。
#   - 与 PretrainDataset 的关键区别：标签是"稀疏"的，只有 assistant 部分非 -100。
# ──────────────────────────────────────────────────────────────────────────────
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=jsonl_path, split="train")
        # 预先 tokenize assistant 回复的起始标记（BOS + "assistant\n"）
        # 用于在 generate_labels 中定位每段 assistant 回复的开始位置
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n",
                                add_special_tokens=False).input_ids
        # 预先 tokenize assistant 回复的结束标记（EOS + "\n"）
        # 用于在 generate_labels 中定位每段 assistant 回复的结束位置
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n",
                                add_special_tokens=False).input_ids
        
    def __len__(self):
        return len(self.samples)
    
    def create_chat_prompt(self, conversations):
        """
        将多轮对话转换为模型输入的字符串。

        特点：
        - 复制原始 conversations，防止修改原始数据。
        - 检测 system 消息中是否携带 functions 字段（function calling 场景），
          若有则透传给 apply_chat_template，生成标准 tool-use 格式的提示词。
        - add_generation_prompt=False：不在末尾追加"请模型续写"的 prompt，
          因为训练时需要完整的 input+output 序列，而非开放续写。
        """
        messages = conversations.copy()
        tools = (conversations[0]["functions"]
                 if (
                     conversations 
                     and conversations[0]["role"] == "system" 
                     and conversations[0].get("functions")
                )
                 else None)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, tools=tools
        )
    