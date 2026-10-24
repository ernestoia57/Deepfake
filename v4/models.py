"""Minimal, ablatable RGB and pixel-wise temporal-frequency classifiers.

No diffusion prior or implicit time-order-invariant ViT masquerading as a
temporal transformer. Every model returns LOGITS (no sigmoid in forward).
"""
import torch
from torch import nn


class ConvEncoder(nn.Module):
    def __init__(self, channels, dim=128):
        super().__init__()
        blocks = []
        c = channels
        for width in (32, 64, 96, 128):
            blocks.extend([
                nn.Conv2d(c, width, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(width), nn.SiLU(),
                nn.Conv2d(width, width, kernel_size=3, padding=1),
                nn.BatchNorm2d(width), nn.SiLU(),
            ])
            c = width
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(c, dim)

    def forward(self, x):
        z = self.pool(self.features(x)).flatten(1)
        return self.proj(z)


class RGBEncoder(nn.Module):
    def __init__(self, dim=128, temporal="gru"):
        super().__init__()
        if temporal not in ("gru", "mean"):
            raise ValueError("temporal must be gru or mean")
        self.backbone = ConvEncoder(3, dim)
        self.temporal = temporal
        self.gru = nn.GRU(dim, dim, batch_first=True) if temporal == "gru" else None

    def forward(self, rgb):
        if rgb.ndim != 5 or rgb.shape[2] != 3:
            raise ValueError("RGB input must have shape [B,T,3,H,W]")
        B, T, C, H, W = rgb.shape
        embeddings = self.backbone(rgb.reshape(B*T, C, H, W)).reshape(B, T, -1)
        if self.gru is None:
            return embeddings.mean(dim=1)
        _, hidden = self.gru(embeddings)
        return hidden[-1]


class Detector(nn.Module):
    def __init__(self, mode="fusion", frames=32, dim=128,
                 temporal="gru", frequency="fft", cwt_bands=8):
        super().__init__()
        if mode not in ("rgb", "freq", "fusion"):
            raise ValueError("mode must be rgb, freq or fusion")
        self.mode = mode
        self.frequency = frequency
        if frequency not in ("fft", "cwt"):
            raise ValueError("frequency must be fft or cwt")
        self.rgb_branch = RGBEncoder(dim, temporal) if mode in ("rgb", "fusion") else None
        nch = frames // 2 if frequency == "fft" else cwt_bands
        self.freq_branch = ConvEncoder(nch, dim) if mode in ("freq", "fusion") else None
        self.classifier = nn.Linear(dim * (2 if mode == "fusion" else 1), 1)

    def forward(self, rgb, freq):
        features = []
        if self.rgb_branch is not None:
            features.append(self.rgb_branch(rgb))
        if self.freq_branch is not None:
            features.append(self.freq_branch(freq))
        return self.classifier(torch.cat(features, dim=1)).squeeze(1)
