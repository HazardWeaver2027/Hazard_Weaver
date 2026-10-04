"""HW-owned wildfire predictors (inference). No external research-package imports."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class DangerMLP(nn.Module):
    """Tabular wildfire-danger classifier."""

    def __init__(self, in_dim: int = 8, hidden: int = 64, n_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DangerForecastMLP(nn.Module):
    """Weekly danger/activity head: flattens (B, T, F) → (B, out_dim)."""

    def __init__(self, lookback: int = 12, n_features: int = 7, hidden: int = 64, out_dim: int = 5):
        super().__init__()
        self.lookback = lookback
        self.n_features = n_features
        self.net = nn.Sequential(
            nn.Linear(lookback * n_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        return self.net(x)


class SpreadASPPLite(nn.Module):
    """Lightweight multi-scale spread segmenter for HW-owned checkpoints."""

    def __init__(self, in_ch: int = 12, mid: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.branch1 = nn.Conv2d(mid, mid // 2, 1)
        self.branch3 = nn.Conv2d(mid, mid // 2, 3, padding=1)
        self.branch5 = nn.Conv2d(mid, mid // 2, 5, padding=2)
        self.fuse = nn.Sequential(
            nn.Conv2d((mid // 2) * 3, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        parts = [self.branch1(h), self.branch3(h), self.branch5(h)]
        return self.fuse(torch.cat(parts, dim=1))


class SpreadUNetLite(nn.Module):
    """Tiny encoder–decoder spread segmenter."""

    def __init__(self, in_ch: int = 12, mid: int = 24):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, mid, 3, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(mid, mid * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(mid * 2, mid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        out = self.dec(e2)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


class TemporalSpreadCNN(nn.Module):
    """Uses last timestep of (B, T, C, H, W) with a spatial CNN head."""

    def __init__(self, in_ch: int = 6, mid: int = 24):
        super().__init__()
        self.head = SpreadUNetLite(in_ch=in_ch, mid=mid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            x = x[:, -1]
        return self.head(x)


ArchName = Literal[
    "danger_mlp",
    "danger_forecast_mlp",
    "spread_aspp_lite",
    "spread_unet_lite",
    "temporal_spread_cnn",
]


def build_arch(arch: str, **kwargs) -> nn.Module:
    if arch == "danger_mlp":
        return DangerMLP(**kwargs)
    if arch == "danger_forecast_mlp":
        return DangerForecastMLP(**kwargs)
    if arch == "spread_aspp_lite":
        return SpreadASPPLite(**kwargs)
    if arch == "spread_unet_lite":
        return SpreadUNetLite(**kwargs)
    if arch == "temporal_spread_cnn":
        return TemporalSpreadCNN(**kwargs)
    raise ValueError(f"unknown arch: {arch}")
