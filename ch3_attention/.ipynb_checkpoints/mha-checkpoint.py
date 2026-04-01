import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout_p, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_o = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout_p)

        self.register_buffer("mask",
                            torch.triu(torch.ones(context_length, context_length), diagonal=1))


    def forward(self, x):

        B, T, D_in = x.shape

        # 三个矩阵  
        # [B, T, D_in] -> [B, T, D_out]
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
            
        # 分头
        # [B, T, D] -> [B, T, H, d] -> [B, H, T, d]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # scores: [B, H, T_q, T_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # mask
        mask_bool = self.mask.bool()[:T, :T]
        scores.masked_fill_(mask_bool, - torch.inf)

        # attn (softmax and dropout)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
 
        # weighted sum
        # out: [B, H, T, d]
        out = torch.matmul(attn, v)

        # 拼接多头
        # out: [B, H, T, d] -> [B, T, H, d] -> [B, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_out)

        # out
        out = self.W_o(out)

        return out