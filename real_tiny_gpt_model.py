import math
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """Embedding with per-token log10 scale factors (to create subnormal-heavy paths)."""
    def __init__(self, num_embeddings, embedding_dim,
                 min_log10_scale=-6, max_log10_scale=0, seed=0):
        super().__init__(num_embeddings, embedding_dim)
        rng = random.Random(seed)
        log_scales = [rng.uniform(min_log10_scale, max_log10_scale)
                      for _ in range(num_embeddings)]
        self.register_buffer("token_log10_scales",
                             torch.tensor(log_scales, dtype=torch.float32))

    def forward(self, idx):
        base = super().forward(idx)            # [B,T,D]
        scales = 10.0 ** self.token_log10_scales[idx]  # [B,T]
        return base * scales.unsqueeze(-1)


class TinyBlockReal(nn.Module):
    """
    Tiny transformer block.
    All ops run in whatever dtype the module parameters are in *inside autocast*.
    """
    def __init__(self, d_model, d_ff, n_heads, sub_scale: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.sub_scale = sub_scale

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def _split_heads(self, x):
        # x: [B,T,C] -> [B, nH, T, Hd]
        B, T, C = x.shape
        x = x.view(B, T, self.n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        # x: [B, nH, T, Hd] -> [B,T,C]
        B, nH, T, Hd = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, T, nH * Hd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        h = self.ln1(x)
        h = h * self.sub_scale

        qkv = self.qkv(h)                  # [B,T,3C]
        q, k, v = qkv.chunk(3, dim=-1)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        # [B, nH, T, Hd]

        d = q.size(-1)
        # [B,nH,T,T]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        attn_weights = F.softmax(attn_scores, dim=-1)
        # [B,nH,T,Hd]
        attn_out = torch.matmul(attn_weights, v)
        attn_out = self._merge_heads(attn_out)

        x = x + self.out_proj(attn_out)

        # MLP
        h2 = self.ln2(x)
        h2 = h2 * self.sub_scale
        h2 = self.act(self.fc1(h2))
        h2 = self.fc2(h2)
        x = x + h2
        return x


class TinyGPTReal(nn.Module):
    """
    Tiny GPT-like model for real hardware.
    The dtype is controlled by AMP (autocast); parameters can stay in fp32.
    """
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 256,
                 n_layer: int = 4,
                 n_heads: int = 4,
                 d_ff: int = 1024,
                 block_size: int = 64,
                 sub_scale: float = 1e-4):
        super().__init__()
        self.block_size = block_size
        self.sub_scale = sub_scale

        self.tok_emb = ScaledEmbedding(
            vocab_size, d_model,
            min_log10_scale=-6,
            max_log10_scale=0,
            seed=0,
        )
        self.pos_emb = nn.Embedding(block_size, d_model)

        self.blocks = nn.ModuleList(
            [TinyBlockReal(d_model, d_ff, n_heads, sub_scale)
             for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.block_size
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        x = x * self.sub_scale
        logits = self.head(x)
        return logits

