"""Efficient Channel Attention (ECA) for Ultralytics YAML models."""

from __future__ import annotations

import torch
from torch import nn


class ECA(nn.Module):
    """Efficient Channel Attention with a lightweight 1D channel interaction.

    The module keeps the input channel count unchanged. It is intentionally
    implemented without a channel-reduction MLP so it can be inserted after a
    YOLO feature map with minimal parameter and latency overhead.
    """

    def __init__(self, k_size: int = 3):
        super().__init__()
        k_size = max(1, int(k_size))
        if k_size % 2 == 0:
            k_size += 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


def register_custom_modules() -> None:
    """Register custom modules in the Ultralytics parser namespace."""

    from ultralytics.nn import tasks

    tasks.ECA = ECA
