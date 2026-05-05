import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model import FeiFeiMindConfig
from dataset.lm_dataset import SFTDataset
from model.model_lora import save_lora, apply_lora
from trainer.trainer_utils import (
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler,
)

warnings.filterwarnings("ignore")


def normalize_resume_path(path):
    return os.path.normcase(os.path.abspath(path))


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
    lora_save_path = f"{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}{moe_suffix}.pth"
    save_lora(model, lora_save_path)
    lm_checkpoint(
        lm_config,
        weight=args.lora_name,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=epoch + 1 if completed_epoch else epoch,
        step=0 if completed_epoch else step,
        wandb=wandb,
        save_dir=CHECKPOINT_DIR,
        update_step=update_step,
        total_update_steps=total_update_steps,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        world_size=world_size,
        data_path=args.data_path,
        dataset_num_samples=dataset_num_samples,
    )
    model.train()


def train_epoch(
    epoch,
    loader,
    iters,
    total_update_steps,
    lora_params,
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

    for step, (input_ids, labels, attention_mask) in enumerate(
        loader, start=start_step + 1
    ):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        attention_mask = attention_mask.to(args.device)
        last_step = step

        lr = get_lr(update_step, total_update_steps, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels, attention_mask=attention_mask)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

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
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            update_step += 1

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]["lr"]
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min"
            )
            if wandb:
                wandb.log(
                    {
                        "loss": current_loss,
                        "logits_loss": current_logits_loss,
                        "aux_loss": current_aux_loss,
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
            last_saved_step = step

        del input_ids, labels, attention_mask, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
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

    return update_step


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

    parser = argparse.ArgumentParser(description="MiniMind LoRA Fine-tuning")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "out"),
        help="模型保存目录",
    )
    parser.add_argument(
        "--lora_name",
        type=str,
        default="lora_medical",
        help="LoRA权重名称(如lora_identity/lora_medical等)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument(
        "--accumulation_steps", type=int, default=1, help="梯度累积步数"
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument("--hidden_size", default=1024, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument(
        "--max_seq_len",
        default=340,
        type=int,
        help="训练的最大截断长度（中文1token≈1.5~1.7字符）",
    )
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "dataset", "lora_medical.jsonl"),
        help="LoRA训练数据路径（默认使用 dataset/lora_medical.jsonl）",
    )
    parser.add_argument(
        "--from_weight",
        default="full_sft",
        type=str,
        help="基于哪个权重训练，默认full_sft",
    )
    parser.add_argument(
        "--from_resume",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否自动检测&续训（0=否，1=是）",
    )
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="MiniMind-LoRA",
        help="wandb项目名",
    )
    parser.add_argument(
        "--use_compile",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用torch.compile加速（0=否，1=是）",
    )
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = FeiFeiMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.lora_name, save_dir=CHECKPOINT_DIR)
        if args.from_resume == 1
        else None
    )
    if args.from_resume == 1 and ckp_data is None:
        moe_suffix = "_moe" if lm_config.use_moe else ""
        raise FileNotFoundError(
            f"未找到 resume checkpoint: {CHECKPOINT_DIR}/{args.lora_name}_{lm_config.hidden_size}{moe_suffix}_resume.pth"
        )

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.cuda.amp.autocast(dtype=dtype)
    )

    # ========== 4. 定义模型、应用LoRA、冻结非LoRA参数 ==========
    model, tokenizer = init_model(
        lm_config, args.from_weight, save_dir=args.save_dir, device=args.device
    )
    apply_lora(model)

    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(
        p.numel() for name, p in model.named_parameters() if "lora" in name
    )
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")

    lora_params = []
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False

    # ========== 5. 定义数据和优化器 ==========
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    dataset_num_samples = len(train_ds)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    global_effective_batch = world_size * args.batch_size * args.accumulation_steps
    updates_per_epoch = (
        dataset_num_samples + global_effective_batch - 1
    ) // global_effective_batch
    total_update_steps = updates_per_epoch * args.epochs
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)

    # ========== 6. 严格校验resume配置并恢复状态 ==========
    start_epoch, start_step, start_update_step = 0, 0, 0
    if ckp_data:
        required_resume_fields = [
            "update_step",
            "total_update_steps",
            "batch_size",
            "accumulation_steps",
            "world_size",
            "data_path",
            "dataset_num_samples",
        ]
        missing_fields = [
            field for field in required_resume_fields if field not in ckp_data
        ]
        if missing_fields:
            raise KeyError(
                f"resume checkpoint缺少严格续训字段: {', '.join(missing_fields)}"
            )
        if ckp_data["total_update_steps"] != total_update_steps:
            raise ValueError(
                f"resume total_update_steps不一致: checkpoint={ckp_data['total_update_steps']}, current={total_update_steps}"
            )
        if ckp_data["batch_size"] != args.batch_size:
            raise ValueError(
                f"resume batch_size不一致: checkpoint={ckp_data['batch_size']}, current={args.batch_size}"
            )
        if ckp_data["accumulation_steps"] != args.accumulation_steps:
            raise ValueError(
                f"resume accumulation_steps不一致: checkpoint={ckp_data['accumulation_steps']}, current={args.accumulation_steps}"
            )
        if ckp_data["world_size"] != world_size:
            raise ValueError(
                f"resume world_size不一致: checkpoint={ckp_data['world_size']}, current={world_size}"
            )
        if normalize_resume_path(ckp_data["data_path"]) != normalize_resume_path(
            args.data_path
        ):
            raise ValueError(
                f"resume data_path不一致: checkpoint={ckp_data['data_path']}, current={args.data_path}"
            )
        if ckp_data["dataset_num_samples"] != dataset_num_samples:
            raise ValueError(
                f"resume dataset_num_samples不一致: checkpoint={ckp_data['dataset_num_samples']}, current={dataset_num_samples}"
            )

        model.load_state_dict(ckp_data["model"], strict=False)
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)
        start_update_step = ckp_data["update_step"]

    # ========== 7. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = (
            f"MiniMind-LoRA-{args.lora_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        )
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            id=wandb_id,
            resume=resume,
        )

    # ========== 8. 编译和分布式包装 ==========
    if args.use_compile == 1:
        args.use_compile = 0
        Logger("[LoRA] monkey-patch forward 与 torch.compile 不兼容，use_compile 已自动关闭")
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 9. 开始训练 ==========
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
                f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始"
            )
            start_update_step = train_epoch(
                epoch,
                loader,
                len(loader) + skip,
                total_update_steps,
                lora_params,
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
                lora_params,
                0,
                start_update_step,
                world_size,
                dataset_num_samples,
                wandb,
            )

    # ========== 10. 清理分布进程 ==========
    if dist.is_initialized():
        dist.destroy_process_group()
