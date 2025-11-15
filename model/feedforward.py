import torch.nn as nn
from config import ShortGPTConfig


class FeedForward(nn.Module):
    """
    Simple 2-layer MLP used inside each Transformer block.
    Expands d_model → d_ff → d_model.
    """

    def __init__(self, config: ShortGPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)
