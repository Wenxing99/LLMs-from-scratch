import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import math
import re
import time
import warnings
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dataset.lm_dataset import RLAIFDataset
from model.model import FeiFeiMindConfig
from trainer.rollout_engine import create_rollout_engine
from trainer.trainer_utils import (
    LMForRewardModel,
    Logger,
    SkipBatchSampler,
    get_lr,
    init_distributed_mode,
    init_model,
    is_main_process,
    lm_checkpoint,
    setup_seed,
)

warnings.filterwarnings("ignore")


def normalize_resume_path(path):
    return os.path.normcase(os.path.abspath(path))


def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    return (
        min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams))
        if grams
        else 0.0
    )


def extract_messages_from_prompt(prompt):
    """
    从 chat template 渲染后的 prompt 中反解析出历史消息。

    这条接口约束对当前 GRPO 链路很关键：
    - RLAIFDataset 负责把 conversations 渲染成 prompt 字符串
    - reward 侧再从该字符串反解析回 messages，交给外部 reward model 打分
    - 如果模板和正则不兼容，训练不会直接报 shape/device 错，而是会静默喂错上下文
    """

    pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
    matches = re.findall(pattern, prompt, re.DOTALL)
    if not matches:
        raise ValueError("无法从当前 prompt 中解析出 reward model 所需的 messages")

    messages = [{"role": role, "content": content.strip()} for role, content in matches]
    if messages[-1]["role"] != "user":
        raise ValueError(
            f"reward prompt 反解析后的最后一条消息必须是 user，当前为 {messages[-1]['role']}"
        )
    return messages


def inspect_rlaif_dataset_contract(train_ds, sample_limit=8):
    """
    在真正训练前先检查 RLAIF 数据契约，避免 `conversations[:-1]` 假设失效后静默训练错误 prompt。
    """

    num_to_check = min(len(train_ds), sample_limit)
    if num_to_check == 0:
        raise ValueError("RLAIFDataset 为空，无法启动 GRPO 训练")

    for idx in range(num_to_check):
        sample = train_ds.samples[idx]
        conversations = sample["conversations"]
        if not isinstance(conversations, list) or not conversations:
            raise ValueError(f"RLAIF sample[{idx}] 的 conversations 不是非空 list")
        if not isinstance(conversations[-1], dict):
            raise ValueError(f"RLAIF sample[{idx}] 的最后一条消息不是 dict")
        if conversations[-1].get("role") != "assistant":
            raise ValueError(
                f"RLAIF sample[{idx}] 不满足当前 `conversations[:-1]` 契约：最后一条 role={conversations[-1].get('role')}"
            )

        prompt = train_ds.create_chat_prompt(conversations)
        extract_messages_from_prompt(prompt)


def gather_completion_logps(logits, output_ids, prompt_lens, completion_len):
    """
    从完整序列 logits 中取出 completion 对应 token 的 logprob。

    当前仓库这条实现保留 full forward + 显式索引的路线：
    - 不回退到 upstream 那种更简化的 shortcut
    - 明确依赖 `prompt_lens` / `completion_len` 和左侧 padding 的对齐关系
    """

    token_logps = F.log_softmax(logits[:, :-1, :], dim=-1)
    token_logps = token_logps.gather(2, output_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(
        completion_len, device=output_ids.device
    ).unsqueeze(0)
    return token_logps.gather(1, logp_pos)


def calculate_rewards(prompts, responses, reward_model):
    rewards = torch.zeros(len(responses), device=args.device, dtype=torch.float32)

    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)

        for i in range(batch_size):
            prompt = prompts[i]
            messages = extract_messages_from_prompt(prompt)
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]

                answer = response
                rewards[response_idx] += (
                    0.5 if 20 <= len(response.strip()) <= 800 else -0.5
                )

                if "</think>" in response:
                    thinking_content, answer_content = response.split("</think>", 1)
                    rewards[response_idx] += (
                        1.0
                        if 20 <= len(thinking_content.strip()) <= 300
                        else -0.5
                    )
                    rewards[response_idx] += (
                        0.25 if response.count("</think>") == 1 else -0.25
                    )
                    answer = answer_content.strip()

                rewards[response_idx] -= rep_penalty(answer)
                reward_model_scores.append(float(reward_model.get_score(messages, answer)))

        reward_model_scores = torch.tensor(
            reward_model_scores, device=args.device, dtype=torch.float32
        )
        rewards += reward_model_scores

    return rewards


def save_train_state(
    epoch,
    step,
    update_step,
    total_update_steps,
    world_size,
    dataset_num_samples,
    wandb=None,
    completed_epoch=False,
):
    model.eval()
    moe_suffix = "_moe" if lm_config.use_moe else ""
    ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"

    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, "_orig_mod", raw_model)
    state_dict = raw_model.state_dict()
    torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)

    lm_checkpoint(
        lm_config,
        weight=args.save_weight,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=epoch + 1 if completed_epoch else epoch,
        step=0 if completed_epoch else step,
        wandb=wandb,
        save_dir=CHECKPOINT_DIR,
        update_step=update_step,
        total_update_steps=total_update_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        world_size=world_size,
        data_path=args.data_path,
        dataset_num_samples=dataset_num_samples,
        num_generations=args.num_generations,
        reward_model_path=args.reward_model_path,
        rollout_engine=args.rollout_engine,
        loss_type=args.loss_type,
        max_seq_len=args.max_seq_len,
        max_gen_len=args.max_gen_len,
        thinking_ratio=args.thinking_ratio,
        beta=args.beta,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        use_moe=args.use_moe,
    )
    model.train()


def train_epoch(
    epoch,
    loader,
    iters,
    total_update_steps,
    rollout_engine,
    ref_model,
    reward_model,
    start_step=0,
    start_update_step=0,
    world_size=1,
    dataset_num_samples=0,
    wandb=None,
):
    start_time = time.time()
    last_step = start_step
    last_saved_step = start_step
    update_step = start_update_step

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]
        last_step = step

        prompt_inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
            padding_side="left",
            add_special_tokens=False,
        ).to(args.device)
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][
                :, -args.max_seq_len :
            ]

        lr = get_lr(update_step, total_update_steps, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )

        outputs = rollout_result.output_ids.to(args.device)
        completion_ids = rollout_result.completion_ids.to(args.device)
        completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        completions = rollout_result.completions
        old_per_token_logps = rollout_result.per_token_logps.to(args.device)
        prompt_lens = rollout_result.prompt_lens.to(args.device)
        full_mask = (outputs != tokenizer.pad_token_id).long()

        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        with autocast_ctx:
            res = model_unwrapped(outputs, attention_mask=full_mask)
            aux_loss = (
                res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            )
            per_token_logps = gather_completion_logps(
                res.logits,
                outputs,
                prompt_lens,
                completion_ids.size(1),
            )

        with torch.no_grad():
            ref_res = ref_model(outputs, attention_mask=full_mask)
            ref_per_token_logps = gather_completion_logps(
                ref_res.logits,
                outputs,
                prompt_lens,
                completion_ids.size(1),
            )

        rewards = calculate_rewards(prompts, completions, reward_model)

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger("-" * 100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    Logger(f"{'=' * 28} [DEBUG] gen[{j}] RESPONSE_BEGIN {'=' * 28}")
                    Logger(completions[idx])
                    Logger(f"{'=' * 29} [DEBUG] gen[{j}] RESPONSE_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{j}] reward={rewards[idx].item():.4f}")
                Logger("=" * 100)

        grouped_rewards = rewards.view(-1, args.num_generations)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(
            args.num_generations
        )
        # GRPO 不额外训练 value/critic，而是在同一 prompt 的多条回答内部做相对标准化。
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        # 训练只看真正的 completion 区间：先屏蔽 padding，再在第一个 EOS 后停止。
        is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask
        eos_idx = torch.full(
            (is_eos.size(0),),
            is_eos.size(1) - 1,
            dtype=torch.long,
            device=args.device,
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (
            torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1)
            <= eos_idx.unsqueeze(1)
        ) & completion_pad_mask

        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1
        ratio = torch.exp(per_token_logps - old_per_token_logps)

        if args.loss_type == "cispo":
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(
                clamped_ratio * advantages.unsqueeze(1) * per_token_logps
                - args.beta * per_token_kl
            )
        else:
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(
                torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl
            )

        policy_loss = (
            (per_token_loss * completion_mask).sum(dim=1)
            / completion_mask.sum(dim=1).clamp(min=1)
        ).mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps

        should_update = step % args.accumulation_steps == 0
        sync_ctx = nullcontext()
        if (
            dist.is_initialized()
            and isinstance(model, DistributedDataParallel)
            and not should_update
            and step != iters
        ):
            sync_ctx = model.no_sync()

        with sync_ctx:
            scaler.scale(loss).backward()

        if should_update:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            update_step += 1

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            processed_steps = step - start_step
            remaining_steps = iters - step
            eta_min = spend_time / max(processed_steps, 1) * remaining_steps / 60
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            kl_ref_val = (
                ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item()
                / max(completion_mask.sum().item(), 1)
            )
            advantages_mean_val = advantages.mean().item()
            advantages_std_val = advantages.std().item()
            current_lr = optimizer.param_groups[0]["lr"]

            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                f"Reward: {avg_reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, "
                f"Adv Std: {advantages_std_val:.4f}, Adv Mean: {advantages_mean_val:.4f}, "
                f"Actor Loss: {policy_loss_val:.4f}, Aux Loss: {current_aux_loss:.4f}, "
                f"Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}, "
                f"epoch_time: {eta_min:.1f}min"
            )

            if wandb and is_main_process():
                wandb.log(
                    {
                        "reward": avg_reward_val,
                        "kl_ref": kl_ref_val,
                        "advantages_std": advantages_std_val,
                        "advantages_mean": advantages_mean_val,
                        "policy_loss": policy_loss_val,
                        "aux_loss": current_aux_loss,
                        "avg_response_len": avg_len_val,
                        "learning_rate": current_lr,
                        "epoch_time": eta_min,
                    }
                )

        if (
            should_update
            and (step - last_saved_step) >= args.save_interval
            and is_main_process()
        ):
            save_train_state(
                epoch,
                step,
                update_step,
                total_update_steps,
                world_size,
                dataset_num_samples,
                wandb=wandb,
                completed_epoch=False,
            )
            if args.rollout_engine != "torch":
                rollout_engine.update_policy(model)
            last_saved_step = step

        del (
            prompt_inputs,
            outputs,
            completion_ids,
            completion_pad_mask,
            completions,
            old_per_token_logps,
            prompt_lens,
            full_mask,
            res,
            ref_res,
            ref_per_token_logps,
            rewards,
            grouped_rewards,
            mean_r,
            std_r,
            advantages,
            completion_mask,
            per_token_logps,
            per_token_kl,
            ratio,
            policy_loss,
            loss,
        )

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        update_step += 1

    if is_main_process():
        save_train_state(
            epoch,
            iters,
            update_step,
            total_update_steps,
            world_size,
            dataset_num_samples,
            wandb=wandb,
            completed_epoch=True,
        )
        if args.rollout_engine != "torch":
            rollout_engine.update_policy(model)

    return update_step


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

    parser = argparse.ArgumentParser(
        description="FeiFeiMind GRPO (Group Relative Policy Optimization)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "out"),
        help="模型保存目录",
    )
    parser.add_argument("--save_weight", default="grpo", type=str, help="保存权重前缀")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="初始学习率")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="保存间隔")
    parser.add_argument("--hidden_size", default=1024, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用 MoE 架构（0=否，1=是）",
    )
    parser.add_argument("--max_seq_len", default=768, type=int, help="Prompt 最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="最大生成长度")
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "dataset", "rlaif.jsonl"),
        help="RLAIF 数据路径",
    )
    parser.add_argument(
        "--num_generations", type=int, default=6, help="每个 prompt 采样的回答数"
    )
    parser.add_argument("--beta", type=float, default=0.1, help="KL 惩罚系数")
    parser.add_argument(
        "--loss_type",
        type=str,
        default="grpo",
        choices=["grpo", "cispo"],
        help="loss 类型",
    )
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="CISPO ratio 上界")
    parser.add_argument("--from_weight", default="full_sft", type=str, help="基座权重名")
    parser.add_argument(
        "--reward_model_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "internlm2-1_8b-reward"),
        help="外部 reward model 路径",
    )
    parser.add_argument(
        "--from_resume",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否自动检测并续训（0=否，1=是）",
    )
    parser.add_argument("--use_wandb", action="store_true", help="是否使用 swanlab")
    parser.add_argument(
        "--wandb_project", type=str, default="FeiFeiMind-GRPO", help="swanlab 项目名"
    )
    parser.add_argument(
        "--use_compile",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用 torch.compile 加速（0=否，1=是）",
    )
    parser.add_argument("--debug_mode", action="store_true", help="是否打印调试采样")
    parser.add_argument(
        "--debug_interval", type=int, default=20, help="调试采样打印间隔"
    )
    parser.add_argument(
        "--thinking_ratio", type=float, default=0.9, help="按概率打开 thinking"
    )
    parser.add_argument(
        "--rollout_engine",
        type=str,
        default="torch",
        choices=["torch", "sglang"],
        help="rollout 引擎类型",
    )
    parser.add_argument(
        "--sglang_base_url",
        type=str,
        default="http://localhost:8998",
        help="SGLang 服务地址",
    )
    parser.add_argument(
        "--sglang_model_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "model"),
        help="SGLang tokenizer 路径",
    )
    parser.add_argument(
        "--sglang_shared_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "sglang_ckpt_grpo"),
        help="SGLang 权重共享目录",
    )
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = FeiFeiMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len + args.max_gen_len,
        use_moe=bool(args.use_moe),
    )

    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir=CHECKPOINT_DIR)
        if args.from_resume == 1
        else None
    )
    if args.from_resume == 1 and ckp_data is None:
        raise FileNotFoundError(
            f"from_resume=1 但未找到 GRPO checkpoint: {CHECKPOINT_DIR}"
        )

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.cuda.amp.autocast(dtype=dtype)
    )

    model, tokenizer = init_model(
        lm_config, args.from_weight, save_dir=args.save_dir, device=args.device
    )
    ref_model, _ = init_model(
        lm_config, args.from_weight, save_dir=args.save_dir, device=args.device
    )
    ref_model = ref_model.eval().requires_grad_(False)

    reward_dtype = torch.float16 if device_type == "cuda" else torch.float32
    reward_model = LMForRewardModel(
        args.reward_model_path, device=args.device, dtype=reward_dtype
    )

    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )

    train_ds = RLAIFDataset(
        args.data_path,
        tokenizer,
        max_length=lm_config.max_seq_len,
        thinking_ratio=args.thinking_ratio,
    )
    inspect_rlaif_dataset_contract(train_ds)

    dataset_num_samples = len(train_ds)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    global_effective_batch = world_size * args.batch_size * args.accumulation_steps
    updates_per_epoch = (
        dataset_num_samples + global_effective_batch - 1
    ) // global_effective_batch
    total_update_steps = updates_per_epoch * args.epochs
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(
        enabled=(device_type == "cuda" and args.dtype == "float16")
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step, start_update_step = 0, 0, 0
    if ckp_data:
        required_resume_fields = [
            "update_step",
            "total_update_steps",
            "learning_rate",
            "batch_size",
            "accumulation_steps",
            "world_size",
            "data_path",
            "dataset_num_samples",
            "num_generations",
            "reward_model_path",
            "rollout_engine",
            "loss_type",
            "max_seq_len",
            "max_gen_len",
            "thinking_ratio",
            "beta",
            "epsilon",
            "epsilon_high",
            "use_moe",
        ]
        missing_fields = [field for field in required_resume_fields if field not in ckp_data]
        if missing_fields:
            raise KeyError(
                f"resume checkpoint 缺少严格续训字段: {', '.join(missing_fields)}"
            )
        if ckp_data["total_update_steps"] != total_update_steps:
            raise ValueError(
                f"resume total_update_steps 不一致: checkpoint={ckp_data['total_update_steps']}, current={total_update_steps}"
            )
        scalar_fields = [
            ("learning_rate", args.learning_rate),
            ("batch_size", args.batch_size),
            ("accumulation_steps", args.accumulation_steps),
            ("world_size", world_size),
            ("dataset_num_samples", dataset_num_samples),
            ("num_generations", args.num_generations),
            ("rollout_engine", args.rollout_engine),
            ("loss_type", args.loss_type),
            ("max_seq_len", args.max_seq_len),
            ("max_gen_len", args.max_gen_len),
            ("thinking_ratio", args.thinking_ratio),
            ("beta", args.beta),
            ("epsilon", args.epsilon),
            ("epsilon_high", args.epsilon_high),
            ("use_moe", args.use_moe),
        ]
        for field_name, current_value in scalar_fields:
            if ckp_data[field_name] != current_value:
                raise ValueError(
                    f"resume {field_name} 不一致: checkpoint={ckp_data[field_name]}, current={current_value}"
                )
        if normalize_resume_path(ckp_data["data_path"]) != normalize_resume_path(
            args.data_path
        ):
            raise ValueError(
                f"resume data_path 不一致: checkpoint={ckp_data['data_path']}, current={args.data_path}"
            )
        if normalize_resume_path(
            ckp_data["reward_model_path"]
        ) != normalize_resume_path(args.reward_model_path):
            raise ValueError(
                f"resume reward_model_path 不一致: checkpoint={ckp_data['reward_model_path']}, current={args.reward_model_path}"
            )

        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)
        start_update_step = ckp_data["update_step"]

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = (
            f"FeiFeiMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        )
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            id=wandb_id,
            resume=resume,
        )

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger("torch.compile enabled")
        rollout_engine.update_policy(model)

    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    rollout_engine.update_policy(model)

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        if skip > 0:
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前 {start_step} 个 step，从 step {start_step + 1} 开始"
            )
            start_update_step = train_epoch(
                epoch,
                loader,
                len(loader) + skip,
                total_update_steps,
                rollout_engine,
                ref_model,
                reward_model,
                start_step,
                start_update_step,
                world_size,
                dataset_num_samples,
                wandb,
            )
        else:
            start_update_step = train_epoch(
                epoch,
                loader,
                len(loader),
                total_update_steps,
                rollout_engine,
                ref_model,
                reward_model,
                0,
                start_update_step,
                world_size,
                dataset_num_samples,
                wandb,
            )

    if dist.is_initialized():
        dist.destroy_process_group()
