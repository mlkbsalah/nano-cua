import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_heads == 0

        self.is_causal = config.is_causal
        self.n_embed = config.n_embed
        self.n_heads = config.n_heads


        self.att = nn.Linear(self.n_embed, 3 * self.n_embed)
        self.proj = nn.Linear(self.n_embed, self.n_embed)

    def _init_weights(self, ):
        nn.init.xavier_uniform_(self.att.weight)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor):
        """X : (batch, seq, emb)"""
        B, T, C = x.shape
        qkv = self.att(x) # (B, T, 3 * C)
        q, k, v = torch.split(qkv, split_size_or_sections=self.n_embed, dim=-1)

        q = q.view(B, T, self.n_heads, self.n_embed // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.n_embed // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.n_embed // self.n_heads).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)

        return out
