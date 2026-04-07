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
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step // total_steps)))

