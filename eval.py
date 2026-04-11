import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import random
import warnings
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model import FeiFeiMindConfig, FeiFeiMindForCausalLM
from trainer.trainer_utils import setup_seed

warnings.filterwarnings("ignore")

def init_model(args):
    """根据命令行参数初始化模型和 tokenizer。"""

    # tokenizer 统一从 args.load_from 指向的目录 / 模型名加载
    # 本地自定义模型和 Hugging Face 模型都会先走这一步
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)

    # 如果 load_from 里包含 "model"，就认为当前走的是：
    # 1. 本地自定义的 FeiFeiMind 架构
    # 2. 再从 save_dir 里的 .pth 权重加载 state_dict
    if "model" in args.load_from:
        model = FeiFeiMindForCausalLM(
            FeiFeiMindConfig(
                hidden_size = args.hidden_size,
                num_hidden_layers = args.num_hidden_layers,
                use_moe = bool(args.use_moe),
                inference_rope_scaling = args.inference_rope_scaling
            )
        )
        moe_suffix = "_moe" if hasattr(args, "use_moe") and args.use_moe else ""
        ckp = f"./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth"
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)

        # todo: 写好lora之后加载lora
    else:
        # 否则就按 Hugging Face 标准格式模型来加载：
        # AutoModelForCausalLM 会自动读取 config / 权重并组装模型
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)

    print(f"加载的模型参数: {sum(p.numel() for p in model.parameters())/ 1e6:.2f} M.")

    return model.eval().to(args.device), tokenizer


def main():
    parser = argparse.ArgumentParser(description="FeiFeiMind reasoning and conversation")
    parser.add_argument("--load_from", default="model", type=str,
        help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument("--save_dir", default="out", type=str, help="模型权重目录")
    parser.add_argument("--weight", default="full_sft", type=str,
        help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument("--lora_weight", default="None", type=str,
        help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument("--hidden_size", default=1024, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], 
                        help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--inference_rope_scaling", default=False, action="store_true",
        help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument("--max_new_tokens", default=8192, type=int,
        help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument("--temperature", default=0.85, type=float,
        help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument("--top_p", default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument("--historys", default=0, type=int,
        help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
        type=str, help="运行设备")
    args = parser.parse_args()

    prompts = [
        "你能做什么？",
        "为什么大海是蓝色的？",
        "请用Python写一个计算等比数列的函数",
        '解释一下"转录"的基本过程',
        "如果明天下雪，我应该如何出门",
        "比较一下猫和狗作为宠物的优缺点",
        "解释什么是AI Agents",
        "推荐一些中国广州的美食",
    ]

    # conversation 保存多轮对话历史，后面会一起喂给 chat template
    conversation = []
    model, tokenizer = init_model(args)

    # input() 返回的是字符串，所以先检查是否合法，再转成 int
    choice = input("[0] 自动测试\n[1] 手动输入\n")
    if choice not in {"0", "1"}:
        raise ValueError("请输入 0 或 1")
    input_mode = int(choice)   

    # TextStreamer 用来边生成边打印；
    # skip_prompt=True 表示不重复打印输入 prompt，
    # skip_special_tokens=True 表示不把 <|im_start|> 之类特殊 token 显示出来
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 自动测试模式：直接遍历 prompts 列表
    # 手动模式：不断执行 input("👶: ")，直到用户输入空字符串 "" 才停止
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input("👶: "), "")
    for prompt in prompt_iter:
        setup_seed(2026)
        if input_mode == 0:
            print(f"👶: {prompt}")

        # 只保留最近 historys 轮历史，再把当前用户输入拼进去
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        # 这里先把 apply_chat_template 需要的参数收进一个字典 templates，
        # 后面通过 **templates 解包，相当于：
        # tokenizer.apply_chat_template(conversation=..., tokenize=False, add_generation_prompt=True, ...)
        templates = {
            "conversation": conversation,
            "tokenize": False,
            "add_generation_prompt": True,
        }

        # 当前 tokenizer 的 chat_template 里检查的是 open_thinking，
        # 所以这里要传同名参数，模板里才会真的走到打开 thinking 的分支
        if args.weight == "reason":
            templates["open_thinking"] = True  # 仅Reason模型使用
        inputs = (
            # SFT / reason / chat 模型走 chat template，
            # 会把 conversation 渲染成模型习惯的对话格式字符串
            tokenizer.apply_chat_template(**templates)
            # pretrain 模型不走 chat template，直接 BOS + prompt 即可
            if args.weight != "pretrain"
            else (tokenizer.bos_token + prompt)
        )

        # 再把字符串输入编码成 PyTorch 张量，并直接搬到目标设备上
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        # end="" 的意思是打印后不换行，
        # 这样 streamer 输出的内容会直接接在 "🤖️: " 后面
        print("🤖️: ", end="")
        generate_ids = model.generate(
            # generate() 会返回“原始输入 + 新生成 token”的完整序列
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=1.0,
        )

        # len(inputs["input_ids"][0]) 是当前 prompt 的 token 长度
        # 这里把前面的输入部分切掉，只保留模型新生成的回答
        response = tokenizer.decode(
            generate_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True
        )

        # streamer 负责实时显示；这里再 decode 一次，
        # 是为了把完整回答保存进 conversation，供下一轮对话继续使用
        conversation.append({"role": "assistant", "content": response})
        print("\n\n")

if __name__ == "__main__":
    main()
