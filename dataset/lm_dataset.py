from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

# 禁用 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多进程环境中产生死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
