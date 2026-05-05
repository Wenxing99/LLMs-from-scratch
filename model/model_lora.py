import torch
from torch import optim, nn


class LoRA(nn.Module):
    """LoRA 低秩适配层。

    对原始线性层的增量写成：
        DeltaW = B @ A
    前向时等价于：
        x -> B(A(x))

    这里把 B 初始化为全 0，因此刚注入 LoRA 时，
    模型输出与原模型完全一致，只会在训练后逐步偏离。
    """

    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        # A 先把高维特征压到低秩空间
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # B 全 0 初始化，保证初始增量为 0
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model, rank=16):
    """给目标 Linear 层打上 LoRA 补丁。

    当前实现只给“方阵 Linear”注入 LoRA：
    - weight.shape[0] == weight.shape[1]
    - 更接近给 hidden_size -> hidden_size 的层加增量

    注入方式：
    1. 给原层挂一个 `module.lora`
    2. monkey-patch 原层 forward，变成 原输出 + LoRA 增量
    """

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank).to(model.device)

            setattr(module, "lora", lora)
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora


def load_lora(model, path):
    """从磁盘加载 LoRA 参数到已经 apply_lora 的模型中。

    约定：
    - 先构建原模型
    - 先调用 apply_lora()
    - 再把保存好的 LoRA 增量权重灌回各层的 module.lora
    """

    state_dict = torch.load(path, map_location=model.device)
    # 兼容 DDP 保存时可能带上的 module. 前缀
    state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            # 从总 state_dict 中切出当前层对应的 lora.A / lora.B
            lora_state = {
                k.replace(f"{name}.lora.", ""): v
                for k, v in state_dict.items()
                if f"{name}.lora." in k
            }
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    """只保存 LoRA 分支参数，不保存原模型全量权重。

    保存流程：
    1. 如果模型经过 torch.compile，优先取回原始模型 `_orig_mod`
    2. 遍历所有带 `.lora` 的层
    3. 只收集 module.lora.state_dict()
    4. 键名统一存成 `<layer_name>.lora.<param_name>`
    5. 参数搬到 CPU 并转成 fp16，减小保存体积

    因此保存出来的是“LoRA 增量权重包”，不是完整模型 checkpoint。
    后续加载时，需要先构建原模型，再 apply_lora()，最后 load_lora()。
    """

    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            # DDP 包装下 named_modules() 可能出现 module.xxx，这里统一去掉最外层 module.
            clean_name = name[7:] if name.startswith("module.") else name
            # 保存的键会长成：layers.0.attn.q_proj.lora.A.weight
            lora_state = {
                f"{clean_name}.lora.{k}": v.cpu().half()
                for k, v in module.lora.state_dict().items()
            }
            state_dict.update(lora_state)
    torch.save(state_dict, path)


def merge_lora(model, lora_path, save_path):
    """把 LoRA 增量真正并回原始线性层权重，并导出合并后的模型权重。

    数学上等价于：
        W_merged = W_base + B @ A

    导出的 save_path 是“合并后的完整模型权重”，
    之后推理时不再需要 apply_lora()/load_lora()。
    """

    load_lora(model, lora_path)
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if ".lora." not in k}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and ".lora." not in name:
            state_dict[f"{name}.weight"] = module.weight.data.clone().cpu().half()
            if hasattr(module, "lora"):
                state_dict[f"{name}.weight"] += (
                    module.lora.B.weight.data @ module.lora.A.weight.data
                ).cpu().half()
    torch.save(state_dict, save_path)
