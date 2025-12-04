import torch.nn as nn
from src.config import ShortGPTConfig
from model.attention import MultiHeadSelfAttention
from model.feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    A single Transformer block used in Short-GPT.
    Uses Pre-Norm architecture:
        x = x + Attention(LN(x))
        x = x + MLP(LN(x))
    """

    def __init__(self, config: ShortGPTConfig):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config)

    def forward(self, x):
        # Attention sublayer
        x = x + self.attn(self.ln1(x))

        # FeedForward sublayer
        x = x + self.ff(self.ln2(x))

        return x
