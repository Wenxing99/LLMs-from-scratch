import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

import argparse  # 命令行参数解析
import time  # 时间统计
import warnings  # 警告控制
import torch
import torch.distributed as dist  # 分布式训练支持
from contextlib import nullcontext  # 上下文管理器
from torch import optim  # 优化器
from torch.nn.parallel import DistributedDataParallel  # 分布式数据并行
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载器

from model.model import FeiFeiMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import (  # 训练工具函数
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler,
)

# 忽略警告信息，保持输出清洁
warnings.filterwarnings("ignore")

def train_epoch(
    epoch,
    loader,
    iters,
    total_update_steps,
    start_step=0,
    start_update_step=0,
    wandb=None,
):
    """运行一个 epoch 的预训练循环。"""
    start_time = time.time()
    last_step = start_step
    update_step = start_update_step

    # input_ids, labels, attention_mask: [B, T]
    for step, (input_ids, labels, attention_mask) in enumerate(loader, start=start_step+1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        attention_mask = attention_mask.to(args.device) 

        lr = get_lr(update_step, total_update_steps, args.learning_rate)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        should_update = (step % args.accumulation_steps == 0)

        sync_ctx = nullcontext()
        if (
            dist.is_initialized()
            and isinstance(model, DistributedDataParallel)
            and not should_update
        ):
            sync_ctx = model.no_sync()

        with sync_ctx:
            with autocast_ctx:
                # 前向传播时交给 autocast 自动选择精度；
                # 这样大部分算子会用半精度执行，通常更省显存也更快
                res = model(input_ids, labels=labels, attention_mask=attention_mask)
                # dense 路径下 aux_loss 通常为 0；MoE 路径下会额外加上路由辅助损失
                loss = res.loss + res.aux_loss

                # 为梯度累积服务，可以理解为想模拟更大的 batch size
                # 因为我们连续 args.accumulation_steps 步才更新参数，
                # 但每次都做反向传播，如果每次都传原始 loss，不做缩放，那么累积的梯度会是
                # 原来的 args.accumulation_steps 倍
                loss = loss / args.accumulation_steps

            # 反向传播
            # scaler 为“梯度缩放器”，因为混合精度里很多梯度是 fp16，数值范围小，容易出现：
            # 梯度太小，直接下溢成 0
            # 所以做法是：先把 loss 放大很多倍，再 backward，更新参数前再把梯度缩回真实大小
            scaler.scale(loss).backward()

        if should_update:
            # 还原梯度的真实值
            scaler.unscale_(optimizer)

            # 梯度裁剪
            # 如果模型所有参数梯度的全局范数太大，就按比例缩小到 args.grad_clip 这个上限。
            # 目的是：防止梯度爆炸，提高训练稳定性，在大模型、长序列、混合精度里很常见
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # optimizer 更新
            scaler.step(optimizer) # 执行参数更新
            scaler.update() # 更新scaler的缩放因子

            optimizer.zero_grad(set_to_none=True) # 清空梯度
            update_step += 1

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps # 恢复真实 loss
            current_lr = optimizer.param_groups[-1]["lr"] 

            # 这里是当前 epoch 剩余时间的一个粗略估计，单位为分钟
            processed_steps = max(step - start_step, 1)
            remaining_steps = max(iters - step, 0)
            eta_min = spend_time / processed_steps * remaining_steps / 60

            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:"
            )

            # 记录到实验跟踪系统
            if wandb:
                wandb.log(
                    {"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min}
                )

        if (step % args.save_interval == 0) and is_main_process():
            model.eval()

            # 构建保存路径
            moe_suffix = (
                "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
            )
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"

            # DDP 模型需要通过 .module 访问真正的模型本体
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            # 纯权重文件保存成 half，主要是为了减小磁盘占用
            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)

            # 保存完整训练状态
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                update_step=update_step,
                batch_size=args.batch_size,
                accumulation_steps=args.accumulation_steps,
                wandb=wandb,
                save_dir=CHECKPOINT_DIR,
            )

            model.train()  # 恢复训练模式
            del state_dict

        del input_ids, labels, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        update_step += 1

    if is_main_process():
        model.eval()

        moe_suffix = (
            "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
        )
        ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        state_dict = {k: v.half() for k, v in state_dict.items()}
        torch.save(state_dict, ckp)

        lm_checkpoint(
            lm_config,
            weight=args.save_weight,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch + 1,
            step=0,
            update_step=update_step,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps,
            wandb=wandb,
            save_dir=CHECKPOINT_DIR,
        )

        model.train()
        del state_dict

    return update_step

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FeiFeiMind Pretraining")

    # ========== 基础训练参数 ==========
    parser.add_argument(
        "--save_dir", type=str, default=os.path.join(PROJECT_ROOT, "out"), help="模型保存目录"
    )  
    parser.add_argument(
        "--save_weight", default="pretrain", type=str, help="保存权重的前缀名"
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="训练轮数（建议1轮zero或2-6轮充分训练）"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")

    # ========== 硬件和性能参数 ==========
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载线程数")

    # ========== 训练策略参数 ==========
    parser.add_argument(
        "--accumulation_steps", type=int, default=2, help="梯度累积步数"
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")

    # ========== 模型架构参数 ==========
    parser.add_argument("--hidden_size", default=1024, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument(
        "--max_seq_len", default=512, type=int, help="训练的最大截断长度"
    )
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )

    # ========== 数据和恢复参数 ==========
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "dataset", "pretrain_t2t.jsonl"),
        help="预训练数据路径",
    )
    parser.add_argument(
        "--from_weight",
        default="none",
        type=str,
        help="基于哪个权重训练，为none则从头开始",
    )
    parser.add_argument(
        "--from_resume",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否自动检测&续训（0=否，1=是）",
    )

    # ========== 实验跟踪参数 ==========
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument(
        "--wandb_project", type=str, default="FeiFeiMind-Pretrain", help="wandb项目名"
    )

    # 解析命令行参数
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    """
    📚 分布式训练初始化知识点：
    - local_rank: 当前进程在本机上的GPU编号
    - 随机种子: 确保不同进程有不同但可复现的随机序列
    - 这样既保证了随机性，又保证了可复现性
    """
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"  # 分布式训练时使用对应的GPU

    # 📚 随机种子设置知识点
    # 不同进程使用不同的种子，避免数据采样完全相同
    # 42是基础种子，每个进程加上自己的rank保证不同
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查点 ==========
    """
    模型配置和检查点管理：
    - 创建保存目录
    - 构建模型配置对象
    - 尝试加载断点续训数据
    """
    os.makedirs(args.save_dir, exist_ok=True)  # 确保保存目录存在

    # 创建 FeiFeiMind 模型配置
    lm_config = FeiFeiMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )

    # 如果开启了断点续训，尝试加载之前的训练状态
    ckp_data = (
        lm_checkpoint(
            lm_config, weight=args.save_weight, save_dir=CHECKPOINT_DIR
        )  
        if args.from_resume == 1
        else None
    )
    if args.from_resume == 1 and ckp_data is None:
        moe_suffix = "_moe" if lm_config.use_moe else ""
        raise FileNotFoundError(
            f"未找到 resume checkpoint: {CHECKPOINT_DIR}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}_resume.pth"
        )

    # ========== 3. 设置混合精度 ==========
    """
    混合精度训练：
    - bfloat16: Google开发，数值范围大，更稳定
    - float16: 标准半精度，节省内存但可能溢出
    - autocast: 自动选择精度，关键运算用float32
    """
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # CPU不支持autocast，使用nullcontext作为空操作
    autocast_ctx = (
        nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    )

    # ========== 4. 配置WandB实验跟踪 ==========
    """
    实验跟踪系统：
    - WandB: 实验管理平台，记录训练过程
    - SwanLab: 国产替代方案
    - 支持断点续训时恢复到同一个实验
    """
    wandb = None
    if args.use_wandb and is_main_process():
        # 这里用 swanlab 提供和 wandb 类似的日志接口
        import swanlab as wandb

        # 如果有检查点数据，获取之前的wandb_id来恢复实验
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None  # 有历史 run id 时，强制续接同一个实验

        # 构建实验名称，包含关键超参数
        wandb_run_name = f"FeiFeiMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume
        )

    # ========== 5. 定义模型、数据、优化器 ==========
    """
    训练组件初始化：
    - 模型: 根据配置创建 FeiFeiMind 模型
    - 数据集: 加载预训练数据
    - 采样器: 分布式训练的数据分配
    - 优化器: AdamW优化器
    - 缩放器: 混合精度训练的梯度缩放
    """
    # 初始化模型和分词器
    model, tokenizer = init_model(
        lm_config,
        args.from_weight,
        save_dir=args.save_dir,
        device=args.device,
    )

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    global_effective_batch = world_size * args.batch_size * args.accumulation_steps
    updates_per_epoch = (len(train_ds) + global_effective_batch - 1) // global_effective_batch
    total_update_steps = updates_per_epoch * args.epochs

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step, start_update_step = 0, 0, 0
    if ckp_data:
        # 恢复模型参数
        model.load_state_dict(ckp_data["model"])
        # 恢复优化器状态（动量、方差估计等）
        optimizer.load_state_dict(ckp_data["optimizer"])
        # 恢复梯度缩放器状态
        scaler.load_state_dict(ckp_data["scaler"])
        # 恢复训练进度
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)
        start_update_step = ckp_data.get("update_step")
        if start_update_step is None:
            saved_accum = ckp_data.get("accumulation_steps") or args.accumulation_steps
            start_update_step = ckp_data["step"] // saved_accum

        saved_ws = ckp_data.get("world_size", 1)
        saved_bs = ckp_data.get("batch_size") or args.batch_size
        saved_accum = ckp_data.get("accumulation_steps") or args.accumulation_steps
        current_ws = dist.get_world_size() if dist.is_initialized() else 1

        layout_changed = (
            saved_ws != current_ws
            or saved_bs != args.batch_size
            or saved_accum != args.accumulation_steps
        )

        if layout_changed and start_step != 0:
            raise RuntimeError(
                "跨机器/跨卡数/跨 batch 配置恢复时，只支持在 epoch 边界恢复（step 必须为 0）。"
            )

    if dist.is_initialized():
        # freqs_cos / freqs_sin 是 RoPE 的预计算缓存，不参与训练；
        # DDP 不需要对这两个 buffer 做同步
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        # 每个epoch设置不同的随机种子，确保数据顺序随机化
        if train_sampler:
            train_sampler.set_epoch(epoch)

        if epoch == start_epoch and start_step > 0:  # 第一个epoch且存在检查点
            # 使用跳批采样器，跳过已训练的数据
            setup_seed(42 + epoch)
            indices = torch.randperm(len(train_ds)).tolist()
            batch_sampler = SkipBatchSampler(
                train_sampler or indices, args.batch_size, start_step
                )
            loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始"
            )
            start_update_step = train_epoch(
                epoch,
                loader,
                len(loader) + start_step,
                total_update_steps,
                start_step,
                start_update_step,
                wandb,
            )
        else:  # 默认从头开始
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            start_update_step = train_epoch(
                epoch,
                loader,
                len(loader),
                total_update_steps,
                0,
                start_update_step,
                wandb,
            )

