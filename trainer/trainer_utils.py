import os
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler

# 检查当前进程是否应该被当作“主进程”
# 1. 如果分布式还没初始化，说明当前是普通单进程训练，此时返回 True
# 2. 如果已经初始化分布式，只有 rank == 0 的进程才是主进程
# 这个判断通常用于控制日志打印、保存模型等只需要执行一次的操作
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


# 日志
def Logger(content):
    if is_main_process():
        print(content)

# 动态学习率计算
def get_lr(current_step, total_steps, lr):
    # 一个简单的 cosine decay:
    # 1. 起始学习率为 lr
    # 2. 最低衰减到 0.1 * lr
    # 3. current_step 从 0 逐步走到 total_steps
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))

def init_distributed_mode():
    # 如果环境变量里没有 RANK，说明当前不是 torchrun/DDP 启动
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    # torchrun 会通过环境变量传入当前进程的 local rank
    # 每个进程只绑定到自己负责的那张 GPU 上
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

# 设置种子
def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置检查点
def lm_checkpoint(
    lm_config,
    weight="full_sft",
    model=None,
    optimizer=None,
    epoch=0,
    step=0,
    wandb=None,
    save_dir="checkpoints",
    **kwargs,
):
    os.makedirs(save_dir, exist_ok=True)

    moe_path = "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
    ckp_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth"
    resume_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth"

    if model is not None:
        from torch.nn.parallel import DistributedDataParallel

        # DDP 包裹过的模型需要先取出真正的 module 再拿 state_dict
        if isinstance(model, DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        # 纯权重文件用于推理/发布，保存成 half 可以省空间
        ckp_tmp = ckp_path + ".tmp"
        torch.save({k: v.half() for k, v in state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        wandb_id = None
        if wandb:
            if hasattr(wandb, "get_run"):
                run = wandb.get_run()
                wandb_id = getattr(run, "id", None) if run else None
            else:
                wandb_id = getattr(wandb, "id", None)

        # resume 文件保留完整训练现场，后续继续训练时优先读这个
        resume_data = {
            "model": state_dict,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "wandb_id": wandb_id,
        }

        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, "state_dict"):
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + ".tmp"
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)

    else:  # 加载模式
        if os.path.exists(resume_path):
            # 恢复训练时统一先从 resume 文件读，里面信息最完整
            ckp_data = torch.load(resume_path, map_location="cpu")
            saved_ws = ckp_data.get("world_size", 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1

            if saved_ws != current_ws:
                # world size 变化后，按比例折算已经走过的 global step
                ckp_data["step"] = ckp_data["step"] * saved_ws // current_ws
                Logger(
                    f"GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data['step']}"
                )

            return ckp_data
        return None
    
# 初始化模型
def init_model(
    lm_config,
    from_weight="pretrain",
    tokenizer_path=None,
    save_dir="../out",
    device="cuda",
):
    from transformers import AutoTokenizer
    from model.model import FeiFeiMindForCausalLM

    # 如果没有指定 tokenizer_path，使用项目根目录下的 model 文件夹
    if tokenizer_path is None:
        # 获取当前文件所在目录的父目录（项目根目录）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        tokenizer_path = os.path.join(project_root, "model")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 这里先构建一份随机初始化模型；如果指定 from_weight，再覆盖权重
    model = FeiFeiMindForCausalLM(lm_config)

    if from_weight != "none":
        moe_suffix = (
            "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
        )
        weight_path = (
            f"{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
        )

        weights = torch.load(weight_path, map_location=device)

        model.load_state_dict(weights, strict=False)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f"所加载Model可训练参数：{total_params / 1e6:.3f} 百万")

    return model.to(device), tokenizer

class SkipBatchSampler(Sampler):
    # 这个 batch sampler 会在原 sampler 基础上，直接跳过前 skip_batches 个 batch
    # 常用于断点续训时快速对齐 dataloader 的读取位置
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler  #
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        # yield 出去的是一个 batch 对应的样本索引列表，而不是单个样本索引
        batch = []  # 当前批次
        skipped = 0  # 已跳过的批次数

        for idx in self.sampler:
            batch.append(idx)  # 添加样本到当前批次

            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1  # 增加跳过计数
                    batch = []  # 清空批次，不返回
                    continue  # 跳过这个批次

                yield batch
                batch = []  # 重置批次

        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size

        return max(0, total_batches - self.skip_batches)

