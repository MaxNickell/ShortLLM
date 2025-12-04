import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ShortGPTConfig
from model.rope import RotaryEmbedding


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention for Short-GPT.
    Includes:
        - Q, K, V projections
        - RoPE for Q and K
        - causal masking
        - output projection
    """

    def __init__(self, config: ShortGPTConfig):
        super().__init__()
        self.config = config

        assert config.d_model % config.n_heads == 0, \
            "d_model must be divisible by n_heads."

        self.head_dim = config.d_model // config.n_heads
        self.n_heads = config.n_heads

        # QKV projection
        self.qkv_proj = nn.Linear(
            config.d_model,
            3 * config.d_model,    # Q, K, V
            bias=False
        )

        # Output projection
        self.out_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False
        )

        # Rotary embeddings
        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base
        )

        # Casual attention mask
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("causal_mask", mask, persistent=False)

        # Dropout
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        x: (B, T, d_model)
        returns: (B, T, d_model)
        """
        B, T, C = x.shape   # C = d_model

        # 1. QKV projection
        qkv = self.qkv_proj(x)  # (B, T, 3*d_model)
        q, k, v = qkv.split(C, dim=2)

        # 2. Reshape into heads (B, T, n_heads, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)

        # 3. Apply RoPE to Q and K
        q = self.rope(q, seq_len=T)
        k = self.rope(k, seq_len=T)

        # 4. Move heads before time dim (B, n_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 5. Compute scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # shape: (B, n_heads, T, T)

        # 6. Causal mask
        causal = self.causal_mask[:T, :T]  # (T, T)
        att = att.masked_fill(causal == 0, float('-inf'))

        # 7. Softmax over last dim
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # 8. Weighted sum over V
        out = att @ v  # (B, n_heads, T, head_dim)

        # 9. Recombine heads
        out = out.transpose(1, 2).contiguous()  # (B, T, n_heads, head_dim)
        out = out.view(B, T, C)                 # (B, T, d_model)

        # 10. Output projection + residual dropout
        out = self.out_proj(out)
        out = self.resid_drop(out)

        return out
