"""ConvNeXt-tiny architecture matching facebookresearch/ConvNeXt layout.

Reproduced from the original ConvNeXt paper (Liu et al., 2022) and the
reference implementation at github.com/facebookresearch/ConvNeXt — this
matches the state-dict layout used by boomb0om/watermark-detectors
(downsample_layers.*, stages.*, norm.*, head.*).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm supporting channels_first (NCHW) layout used inside the network."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if data_format not in ("channels_last", "channels_first"):
            raise ValueError(data_format)
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # channels_first
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class Block(nn.Module):
    """ConvNeXt residual block: depthwise 7x7 conv → LN → 1x1 → GELU → 1x1 → gamma."""

    def __init__(self, dim: int, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,))) if layer_scale_init_value > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return residual + x


class ConvNeXt(nn.Module):
    """ConvNeXt with state-dict keys matching facebookresearch/ConvNeXt:

      downsample_layers.0..3, stages.0..3, norm, head
    """

    def __init__(self, in_chans: int = 3, num_classes: int = 1000,
                 depths=(3, 3, 9, 3), dims=(96, 192, 384, 768),
                 layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            ))

        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(
                *[Block(dims[i], layer_scale_init_value) for _ in range(depths[i])]
            ))

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for ds, stage in zip(self.downsample_layers, self.stages):
            x = ds(x)
            x = stage(x)
        return self.norm(x.mean([-2, -1]))  # GAP + LN

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def convnext_tiny(num_classes: int = 1000) -> ConvNeXt:
    return ConvNeXt(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), num_classes=num_classes)
