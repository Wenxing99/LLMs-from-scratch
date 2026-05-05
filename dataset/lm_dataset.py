from torch.utils.data import Dataset
import torch
import os
import random
import json
from datasets import load_dataset, Features, Value


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
    # tool-use 数据保持原样，避免随机插入 system 破坏工具调用模板结构
    if any(conv.get("tools") for conv in conversations):
        return conversations

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
    
    return prompt_content


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
        features = Features({
            "conversations": [{
            "role": Value("string"),
            "content": Value("string"),
            "reasoning_content": Value("string"),
            "tools": Value("string"),
            "tool_calls": Value("string"),
            }]
        })
        self.samples = load_dataset("json", data_files=jsonl_path, split="train", features=features)
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
        - 逐条拷贝 message，避免直接修改原始样本。
        - 若 system 消息中带 tools，则提取出来传给 apply_chat_template。
        - 若某条消息的 tool_calls 是字符串形式的 JSON，则先解析成 Python 对象。
        - add_generation_prompt=False：训练时需要完整 input+output，而不是给模型留一个待续写的结尾。
        """
        messages = []
        tools = None

        for message in conversations:
            message = dict(message)

            if message.get("role") == "system" and message.get("tools"):
                tools = (
                    json.loads(message["tools"])
                    if isinstance(message["tools"], str)
                    else message["tools"]
                )

            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])

            messages.append(message)

        return self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools,
    )
    
    def generate_labels(self, input_ids):
        """
        生成 SFT 训练所需的稀疏标签序列。

        算法逻辑（滑动窗口扫描）：
        1. 初始化全 -100 的 labels，默认所有位置不计算 loss。
        2. 逐位扫描 input_ids，检测是否匹配 bos_id（assistant 回复起始）。
        3. 匹配到 bos_id 后，向后扫描直到找到 eos_id（回复结束）。
        4. 将 [start, end+len(eos_id)) 区间内的 label 设为对应的 input_ids 值，
           即这段 assistant 回复参与 loss 计算。
        5. EOS token 本身也计入 label，让模型学会何时停止生成。
        6. 跳过已处理区间，继续扫描下一段 assistant 回复（支持多轮对话）。
        """
        labels = [-100] * len(input_ids)
        seq_len = len(input_ids)
        bos_len = len(self.bos_id)
        eos_len = len(self.eos_id)
        i = 0

        while i <= seq_len - bos_len:
            # 没命中 assistant 起始标记，就继续向后扫描
            if input_ids[i : i + bos_len] != self.bos_id:
                i += 1
                continue

            # 跳过 bos_id 本身，从 assistant 实际内容开始
            start = i + bos_len
            end = start

            # 向后扫描，找到 eos_id 的位置
            while end <= seq_len - eos_len:
                if input_ids[end : end + eos_len] == self.eos_id:
                    break
                end += 1

            # 找到 eos 时将 eos 一并计入 label；
            # 若本段回复因截断未包含 eos，则一直标到序列末尾
            found_eos = end <= seq_len - eos_len
            stop = end + eos_len if found_eos else seq_len

            # 用切片替代逐 token 赋值，逻辑更直接
            labels[start:stop] = input_ids[start:stop]

            # 当前这段 assistant 回复已处理完，下一轮从 stop 继续
            i = stop
        
        return labels
    
    def __getitem__(self, index):
        sample = self.samples[index]

        # Step 1：随机决定是否插入 system prompt（数据增强）
        conversations = pre_processing_chat(sample["conversations"])

        # Step 2：用 chat template 渲染完整对话字符串
        prompt = self.create_chat_prompt(conversations)

        # Step 3：清理可能出现的空 <think> 块
        prompt = post_processing_chat(prompt)

        # Step 4：tokenize 并截断到 max_length，不足则右侧 PAD 补齐
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Step 5：生成稀疏标签，只有 assistant 回复部分有有效 label
        labels = torch.tensor(self.generate_labels(input_ids.tolist()), dtype=torch.long)

        # 返回 attention_mask，使 attention 层能屏蔽 padding token
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # 再次兜底，确保 padding 位置永远不参与 loss
        labels[attention_mask == 0] = -100

        return input_ids, labels, attention_mask
    

class RLAIFDataset(Dataset):
    """
    为 PPO / GRPO 这类 on-policy 强化学习阶段准备的 prompt-only 数据集。

    设计思路：
    - 与 SFT 不同，这里不直接返回 token 级监督标签，而是只提供 prompt。
    - 真正的回答由当前 policy 在 rollout 阶段现场生成，再交给 reward model
      或规则函数打分。
    - 因此返回值里的 ``answer`` 先留空字符串，占位即可；后续训练逻辑通常
      只会消费 ``prompt``，生成出的 completion 才是真正参与优化的对象。
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        # 先保留与其他 dataset 统一的构造参数形式，后续若 rollout 前要做长度裁剪，
        # trainer 可以直接复用这个配置。
        self.max_length = max_length
        # 按概率打开 thinking，让 RL 阶段同时覆盖“显式思考”和“空 think 占位”
        # 两种 prompt 形态，避免 rollout 只适配单一模板。
        self.thinking_ratio = thinking_ratio
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        # 这两个标记当前类里还没直接用到，但先保留，方便后面如果要在
        # PPO / GRPO 中做 response span 定位或调试时复用。
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)
    
    def create_chat_prompt(self, conversations):
        """
        只渲染“待模型回答之前”的上下文 prompt。

        注意：
        - 这里对 conversations 做 ``[:-1]``，默认假设样本最后一条不是要喂给
          policy 的上下文，而是应被截掉的目标答案 / 占位尾项。
        - add_generation_prompt=True 会在末尾补上 assistant 起始模板，让 rollout
          引擎从这里继续生成。
        """
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )
    
    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])

        return {
            # rollout 阶段真正需要的是 prompt；回答由当前策略在线生成。
            'prompt': prompt,
            # 先保留一个统一字段占位，便于后续 trainer / reward pipeline
            # 如果想额外塞参考答案或调试信息时不必改 collate 形状。
            'answer': ""
        }
