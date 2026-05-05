import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import random
import warnings
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

from model.model import FeiFeiMindConfig, FeiFeiMindForCausalLM
from model.model_lora import apply_lora, load_lora
from trainer.trainer_utils import setup_seed

warnings.filterwarnings("ignore")


def init_model(args):
    """根据命令行参数初始化模型和 tokenizer。"""

    tokenizer = AutoTokenizer.from_pretrained(args.load_from)

    if "model" in args.load_from:
        model = FeiFeiMindForCausalLM(
            FeiFeiMindConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                use_moe=bool(args.use_moe),
                inference_rope_scaling=args.inference_rope_scaling,
            )
        )
        moe_suffix = "_moe" if hasattr(args, "use_moe") and args.use_moe else ""
        ckp = f"./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth"
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)

        if args.lora_weight != "None":
            apply_lora(model)
            load_lora(
                model,
                f"./{args.save_dir}/{args.lora_weight}_{args.hidden_size}{moe_suffix}.pth",
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.load_from, trust_remote_code=True
        )

    print(f"加载的模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M.")

    return model.eval().to(args.device), tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="FeiFeiMind reasoning and conversation"
    )
    parser.add_argument(
        "--load_from",
        default="model",
        type=str,
        help="模型加载路径（model=原生torch权重，其他路径=transformers格式）",
    )
    parser.add_argument("--save_dir", default="out", type=str, help="模型权重目录")
    parser.add_argument(
        "--weight",
        default="full_sft",
        type=str,
        help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）",
    )
    parser.add_argument(
        "--lora_weight",
        default="None",
        type=str,
        help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）",
    )
    parser.add_argument("--hidden_size", default=1024, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )
    parser.add_argument(
        "--inference_rope_scaling",
        default=False,
        action="store_true",
        help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）",
    )
    parser.add_argument(
        "--max_new_tokens",
        default=8192,
        type=int,
        help="最大生成长度（注意：并非模型实际长文本能力）",
    )
    parser.add_argument(
        "--temperature",
        default=0.85,
        type=float,
        help="生成温度，控制随机性（0-1，越大越随机）",
    )
    parser.add_argument(
        "--top_p", default=0.85, type=float, help="nucleus采样阈值（0-1）"
    )
    parser.add_argument(
        "--historys",
        default=0,
        type=int,
        help="携带历史对话轮数（需为偶数，0表示不携带历史）",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
        help="运行设备",
    )
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

    conversation = []
    model, tokenizer = init_model(args)

    choice = input("[0] 自动测试\n[1] 手动输入\n")
    if choice not in {"0", "1"}:
        raise ValueError("请输入 0 或 1")
    input_mode = int(choice)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    prompt_iter = prompts if input_mode == 0 else iter(lambda: input("👶: "), "")
    for prompt in prompt_iter:
        setup_seed(2026)
        if input_mode == 0:
            print(f"👶: {prompt}")

        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        templates = {
            "conversation": conversation,
            "tokenize": False,
            "add_generation_prompt": True,
        }

        if args.weight == "reason":
            templates["open_thinking"] = True

        inputs = (
            tokenizer.apply_chat_template(**templates)
            if args.weight != "pretrain"
            else (tokenizer.bos_token + prompt)
        )

        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(
            args.device
        )

        print("🤖️: ", end="")
        generate_ids = model.generate(
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

        response = tokenizer.decode(
            generate_ids[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
        )

        conversation.append({"role": "assistant", "content": response})
        print("\n\n")


if __name__ == "__main__":
    main()
