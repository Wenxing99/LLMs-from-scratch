import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Union
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class FeiFeiMindConfig(PretrainedConfig):
    """FeiFeiMind 的配置类。

    负责保存模型结构相关超参数，并兼容 Hugging Face
    `PretrainedConfig` 的保存 / 加载接口。
    """

    # model_type 会写进 config.json，后面和 AutoConfig/AutoModel 对接时会用到
    model_type = "feifeimind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        # 把 kwargs 里 transformers 通用配置字段交给基类处理
        # 例如 pad_token_id、tie_word_embeddings、name_or_path 等
        super().__init__(**kwargs)

        # 下面这些字段主要描述模型结构本身
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        # rope_scaling 不为 None 时，表示在推理阶段启用 YaRN 风格的长上下文外推
        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

class RMSNorm(nn.Module):
    """RMSNorm.

    数学形式：
        y = x / sqrt(mean(x^2) + eps) * gamma
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 只做 RMS 归一化本体，不乘可学习参数 gamma
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        # 先转成 float32 做归一化，数值更稳
        # 最后再转回输入 dtype，避免在混合精度训练里把激活一直留在 float32
        return (self.gamma.float() * self._norm(x.float())).type_as(x)
    
def precompute_freqs(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None
):
    """预计算 RoPE / YaRN 的 cos 和 sin 查表。

    标准 RoPE 会直接生成基础频率；当 `rope_scaling` 不为 None 且
    `end` 超过原始训练长度时，会按 YaRN 的规则对频率做分段缩放：
    高频不缩放，低频全量缩放，中频平滑过渡。

    当前实现采用 split-half 的维度配对方式：
    第 i 维和第 i + D/2 维共享同一组旋转角度。

    Args:
        dim: 单个 attention head 的维度，通常应为偶数。
        end: 需要预计算到的最大序列长度。
        rope_base: RoPE 的 base theta。
        rope_scaling: YaRN 缩放配置；为 None 时表示使用标准 RoPE。

    Returns:
        freqs_cos: shape 为 [end, dim] 的 cos 查表。
        freqs_sin: shape 为 [end, dim] 的 sin 查表。
    """

    # freqs 的 shape 是 [dim // 2]
    # 每个 freqs[i] 对应一组旋转角速度，
    # 供第 i 维和第 i + D/2 维这一对特征共享
    # 初始化标准 RoPE 频率
    # torch.arange(0, dim, 2) 生成 [0, 2, 4, ... dim-2]
    # 计算出的 freqs 就是标准的 1 / (base ** (2i / d))
    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim)),
        1.0
    )

    if rope_scaling is not None:
        # 2. 从配置字典中提取YaRN超参
        # orig_max: 模型预训练时的原始最大长度
        # factor: 要扩展的倍数s（比如从2k拓展到32k，factor就是16）
        # beta_fast: 高频边界。
        #            当 orig_max / wavelength > beta_fast 时，
        #            说明该 RoPE 分量在原始上下文窗口内包含很多个周期，属于高频，不做缩放。
        # beta_slow: 低频边界。
        #            当 orig_max / wavelength < beta_slow 时，
        #            说明该 RoPE 分量在原始上下文窗口内不足一个周期，属于低频，做全量缩放。
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )

        # 只有当要推断的长度大于原始训练长度时，才应用缩放
        if end / orig_max > 1.0:
            # 3. b 到 i 的映射：b 就是 RoPE 分量在原始上下文窗口内包含多少个周期，输入 b，得到b对应的index i
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))

            # 4. 计算高频区和低频区的维度切分点
            # low: 不需要缩放的高频部分的最高索引
            # high: 需要完全缩放的低频部分的最低索引
            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 -1)
            )

            # 5. 计算混合因子 γ (Ramp)
            # 在 low 之前，ramp 为 0；在 high 之后，ramp 为 1；在 low 和 high 之间，线性过渡。
            # clamp 函数限制了数值只能在 [0, 1] 之间。
            ramp = torch.clamp(
                (torch.arange(dim // 2, device = freqs.device).float() - low) / max(high - low, 0.001),
                0, 
                1
            )

            # 6. 频率融合公式：freq' = (1 - γ) * freq + γ * freq/factor
            # 当 ramp=0 时（高频）：系数为 1，保持原频率不变。
            # 当 ramp=1 时（低频）：系数为 1/factor，即对频率进行线性插值缩放。
            # ramp 在 0-1 之间时：平滑过渡。
            freqs = freqs * (1 - ramp + ramp / factor)

    # 7. 根据目标长度 end，生成位置索引向量 t
    t = torch.arange(end, device = freqs.device)

    # 8. 计算外积：将位置 t 与处理好的频率 freqs 相乘，得到每个位置的旋转角度 θ
    # 这里相当于做了一张角度表。维度：[end, d / 2]
    freqs = torch.outer(t, freqs).float()

    # 9. 计算 Cos 和 Sin，并应用注意力补偿系数 (attn_factor)
    # 这里把 cos / sin 复制两份，是因为当前实现里：
    # 第 i 维和第 i + D/2 维共用同一个 theta
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """把预计算好的 RoPE cos / sin 应用到 q 和 k 上。

    当前实现采用 split-half 配对：
    第 i 维和第 i + D/2 维作为一对做二维旋转。
    旋转完成后，q / k 的张量布局保持不变，只是数值被更新。

    Args:
        q: query tensor。
        k: key tensor。
        cos: 预计算好的 cos 查表。
        sin: 预计算好的 sin 查表。
        position_ids: 预留参数，当前实现里没有用到。
        unsqueeze_dim: 在哪一维插入广播维，以对齐 q / k 的 shape。

    Returns:
        q_embed: 施加 RoPE 之后的 query。
        k_embed: 施加 RoPE 之后的 key。
    """

    # q, k 的最后一维是 head_dim
    # 这里默认 RoPE 的配对方式是：
    # 前半维和后半维配成一组，而不是相邻两维配成一组
    def rotate_half(x):
        # 如果 x = [x1, x2]
        # 其中 x1 是前半维，x2 是后半维，
        # 那么 rotate_half(x) = [-x2, x1]
        # 这相当于把每一对 (x_i, x_{i + D/2}) 变成旋转公式中的正交项
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    # cos / sin 的原始 shape 通常是 [T, D_head] 或 [B, T, D_head]
    # 通过 unsqueeze_dim 在 head 维上插一个维度，方便后面和 q / k 广播
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # 这里不是按相邻两维做旋转，而是按第 i 维和第 i + D/2 维配对
    # q_embed / k_embed 的 shape 和 q / k 完全一致，
    # 只是每一对 (x_i, x_{i + D/2}) 都已经完成了二维旋转
    # 工程实现上可以写成：
    # x_rot = x * cos(theta) + rotate_half(x) * sin(theta)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """把 KV head 复制到和 Q head 数量一致。"""

    # x: [B, T, H_kv, D_head]
    B, T, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    
    # x[:, :, :, None, :]: [B, T, H_kv, 1, D_head]
    # expand 之后: [B, T, H_kv, n_rep, D_head]
    # reshape 之后: [B, T, H_kv * n_rep, D_head]
    return (
        x[:, :, :, None, :]
        .expand(B, T, num_key_value_heads, n_rep, head_dim)
        .reshape(B, T, num_key_value_heads*n_rep, head_dim)
    )

class GroupQueryAttention(nn.Module):
    """Grouped-Query Attention.

    Q 保持完整的 attention heads 数量；
    K / V 只保留更少的 KV heads，再通过 repeat_kv 扩展回去和 Q 对齐。
    """

    def __init__(self, args: FeiFeiMindConfig):
        super().__init__()

        self.num_key_value_heads = (
            args.num_attention_heads 
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )

        # hidden_size 必须能均匀分给所有 Q heads
        assert args.hidden_size % args.num_attention_heads == 0
        # Q heads 数量必须能被 KV heads 数量整除，这样 repeat_kv 才能整齐复制
        assert args.num_attention_heads % self.num_key_value_heads == 0

        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        # q_proj 输出 [B, T, H_q * D_head]
        # k_proj / v_proj 输出 [B, T, H_kv * D_head]
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attention
        )

    def forward(
            self,
            x: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache = False,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        """执行一层 GQA，返回注意力输出和可选的 KV cache。"""

        # x: [B, T, D_model]
        B, T, _ = x.shape

        # xq: [B, T, H_q * D_head]
        # xk, xv: [B, T, H_kv * D_head]
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # xq: [B, T, H_q, D_head]
        # xk, xv: [B, T, H_kv, D_head]
        xq = xq.view(B, T, self.n_local_heads, self.head_dim)
        xk = xk.view(B, T, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(B, T, self.n_local_kv_heads, self.head_dim)

        # cos, sin: [T, D_head] 或 [B, T, D_head]
        cos, sin = position_embeddings

        # RoPE 只改 q / k 的数值，不改 shape
        # xq: [B, T, H_q, D_head]
        # xk: [B, T, H_kv, D_head]
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # past_key_value[0], past_key_value[1]: [B, T_cache, H_kv, D_head]
        # 拼接之后：
        # xk, xv: [B, T_k, H_kv, D_head]
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # repeat_kv 之后，K / V 的 head 数从 H_kv 扩展到 H_q
        # transpose 后统一成注意力计算更常见的布局 [B, H, T, D_head]
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )
        # xq: [B, H_q, T_q, D_head]
        # xk, xv: [B, H_q, T_k, D_head]

        if (
            self.flash 
            and (T > 1) 
            and (past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            # output: [B, H_q, T_q, D_head]
            output = F.scaled_dot_product_attention(
                xq, xk, xv, 
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # xk.transpose(-2, -1): [B, H_q, D_head, T_k]
            # scores: [B, H_q, T_q, T_k]
            scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # 只对当前这一步对应的最后 T 列施加 causal mask
            scores[..., -T:] += torch.triu(
                torch.full((T, T), float("-inf"), device = scores.device),
                diagonal = 1
            )

            if attention_mask is not None:
                # attention_mask: [B, T_k]
                # 1 代表可以被attend，0 代表需要被mask掉
                # scores: [B, H_q, T_q, T_k]
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * (-1e9)
                scores += extended_attention_mask

            # scores: [B, H_q, T_q, T_k]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)

            # output: [B, H_q, T_q, D_head]
            output = torch.matmul(scores, xv)

        # output.transpose(1, 2): [B, T, H_q, D_head]
        # reshape 之后: [B, T, H_q * D_head]
        # o_proj 之后: [B, T, D_model]
        # 这里用了 reshape，可以不用 contiguous()；如果用 view，一定要 contiguous()
        output = output.transpose(1, 2).reshape(B, T, -1)
        output = self.resid_dropout(self.o_proj(output))

        return output, past_kv


class FeedForward(nn.Module):
    """Transformer 中的前馈网络。

    当前实现采用 SwiGLU 风格：
    先分别做 gate_proj 和 up_proj，
    再用 act(gate_proj(x)) * up_proj(x) 做门控，
    最后通过 down_proj 投回 hidden_size。
    """

    def __init__(self, config: FeiFeiMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            # 向上取整到64的倍数
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        # gate_proj / up_proj: [B, T, D_model] -> [B, T, D_ff]
        # down_proj: [B, T, D_ff] -> [B, T, D_model]
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """执行一层前馈网络。

        Args:
            x: 输入 hidden states，shape 为 [B, T, D_model].

        Returns:
            shape 为 [B, T, D_model] 的输出张量。
        """

        # gate_proj(x): [B, T, D_ff]
        # up_proj(x): [B, T, D_ff]
        # gated: [B, T, D_ff]
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)

        # down_proj(gated): [B, T, D_model]
        return self.dropout(self.down_proj(gated))
    
class MoEGate(nn.Module):
    """MoE 路由门控模块。"""

    def __init__(self, config: FeiFeiMindConfig):
        super().__init__()
        self.config = config

    def forward(self, x):
        raise NotImplementedError("MoEGate.forward is not implemented yet.")


class MoEFeedForward(nn.Module):
    """MoE 版前馈网络。"""

    def __init__(self, config: FeiFeiMindConfig):
        super().__init__()
        self.config = config

    def forward(self, x):
        raise NotImplementedError("MoEFeedForward.forward is not implemented yet.")

    
class FeiFeiMindBlock(nn.Module):
    """一个标准的 Transformer block。

    结构上采用 pre-norm：
    attention 前先做一次 RMSNorm，
    attention 后再做一次 RMSNorm 接 MLP，
    两个子层外面都带残差连接。
    """

    def __init__(self, layer_id: int, config: FeiFeiMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attention = GroupQueryAttention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.mlp = FeedForward(config) if not config.use_moe else MoEFeedForward(config)

    def forward(self, 
                hidden_states,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache = False,
                attention_mask: Optional[torch.Tensor] = None,
                ):
        """执行一个 Transformer block。

        Args:
            hidden_states: 输入 hidden states，shape 为 [B, T, D_model].
            position_embeddings: 预计算好的 (cos, sin).
            past_key_value: 可选的 KV cache.
            use_cache: 是否返回新的 KV cache.
            attention_mask: 可选的 attention mask.

        Returns:
            hidden_states: shape 为 [B, T, D_model] 的 block 输出。
            present_key_value: 当前层新的 KV cache；如果 use_cache=False，则为 None。
        """

        # hidden_states: [B, T, D_model]
        res = hidden_states

        # pre-norm 之后仍然是 [B, T, D_model]
        # self_attention 返回：
        # hidden_states: [B, T, D_model]
        # present_key_value: 可选的 KV cache
        hidden_states, present_key_value = self.self_attention(
            self.input_layernorm(hidden_states), # pre-norm
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )

        # attention 残差连接
        # hidden_states: [B, T, D_model]
        hidden_states = res + hidden_states

        # 先做 post-attention layernorm，再过 MLP，最后做第二次残差连接
        # self.mlp(...) 的输出 shape 仍然是 [B, T, D_model]
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )

        return hidden_states, present_key_value




       
       
