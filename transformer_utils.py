
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Patch Embedding (ViT style)
# -----------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 64, patch_size: int = 8, in_chans: int = 1, embed_dim: int = 384):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):  # x: [B, C, H, W]
        x = self.proj(x)   # [B, D, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


# -----------------------------
# Transformer Encoder (minimal)
# -----------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=384, depth=6, num_heads=6, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm1": nn.LayerNorm(embed_dim),
                "attn": nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_drop, batch_first=True),
                "norm2": nn.LayerNorm(embed_dim),
                "mlp": nn.Sequential(
                    nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                    nn.Dropout(drop),
                ),
            }) for _ in range(depth)
        ])
        self.dropout = nn.Dropout(drop)

    def forward(self, x):  # [B, N, D]
        for blk in self.layers:
            h = blk["norm1"](x)
            attn_out, _ = blk["attn"](h, h, h, need_weights=False)
            x = x + self.dropout(attn_out)
            h = blk["norm2"](x)
            x = x + self.dropout(blk["mlp"](h))
        return x


# -----------------------------
# Simple CNN for CWT stack
# -----------------------------
class CWTEncoder(nn.Module):
    """Tiny CNN to encode a stack of CWT/Gabor maps (channels-first)."""
    def __init__(self, in_chans: int = 8, out_dim: int = 256):
        super().__init__()
        c = 48
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, c, 3, padding=1), nn.BatchNorm2d(c), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.SiLU(),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(c, 2*c, 3, padding=1), nn.BatchNorm2d(2*c), nn.SiLU(),
            nn.Conv2d(2*c, 2*c, 3, padding=1), nn.BatchNorm2d(2*c), nn.SiLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(2*c, 4*c, 3, padding=1), nn.BatchNorm2d(4*c), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(4*c, out_dim)

    def forward(self, x):  # [B*T, Cg, H, W]
        z = self.net(x).flatten(1)
        return self.head(z)  # [B*T, out_dim]


# -----------------------------
# Quality conditioning
# -----------------------------
class QualityCond(nn.Module):
    """Maps per-frame ELBO (scalar) to an embedding that conditions the ViT."""
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
    def forward(self, e_bpd):  # [B, T]
        q = self.mlp(e_bpd.unsqueeze(-1))  # [B, T, d_model]
        return q


# -----------------------------
# ViT over frames with quality token, + CWT fusion
# -----------------------------
class ViTWithQualityAndCWT(nn.Module):
    """
    - Processes per-frame images with ViT (shared) and injects a quality token
      derived from ELBO-bpd.
    - Encodes per-frame CWT/Gabor stacks via a small CNN.
    - Temporal pooling across frames, then late fusion (concat) + head.
    """
    def __init__(
        self,
        img_size: int = 64,
        in_chans: int = 1,
        patch_size: int = 8,
        d_model: int = 384,
        depth: int = 6,
        heads: int = 6,
        cwt_in_chans: int = 8,
        cwt_dim: int = 256,
        num_classes: int = 2,
    ):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch_size, in_chans, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.tr = TransformerEncoder(embed_dim=d_model, depth=depth, num_heads=heads)
        self.norm = nn.LayerNorm(d_model)
        self.qcond = QualityCond(d_model)

        # CWT branch
        self.cwt_enc = CWTEncoder(in_chans=cwt_in_chans, out_dim=cwt_dim)

        # Fusion + head
        fusion_dim = d_model + cwt_dim + 3  # ELBO stats: mean/median/min
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(
        self,
        frames: torch.Tensor,     # [B, T, C, H, W]
        e_bpd: torch.Tensor,      # [B, T]
        cwt_stack: torch.Tensor,  # [B, T, Cg, H, W]
    ):
        B, T, C, H, W = frames.shape

        # ---- ViT branch (per frame, shared) ----
        frames_2d = frames.view(B*T, C, H, W)
        x = self.patch(frames_2d)  # [B*T, N, D]
        N = x.size(1)
        cls = self.cls_token.expand(B*T, -1, -1)  # [B*T, 1, D]

        # Quality token (additive into cls)
        q = self.qcond(e_bpd).view(B*T, 1, -1)    # [B*T, 1, D]
        cls = cls + q

        x = torch.cat([cls, x], dim=1) + self.pos_embed[:, : N+1, :]
        x = self.tr(x)               # [B*T, 1+N, D]
        x = self.norm(x[:, 0])       # cls-token per frame -> [B*T, D]
        z_vit = x.view(B, T, -1)     # [B, T, D]

        # Temporal pooling (mean)
        z_vit_clip = z_vit.mean(dim=1)  # [B, D]

        # ---- CWT branch ----
        cwt_2d = cwt_stack.view(B*T, *cwt_stack.shape[2:])  # [B*T, Cg, H, W]
        z_cwt = self.cwt_enc(cwt_2d).view(B, T, -1).mean(dim=1)  # [B, cwt_dim]

        # ---- ELBO stats (mean, median, min) ----
        e_mean = e_bpd.mean(dim=1, keepdim=True)
        e_med  = e_bpd.median(dim=1).values.unsqueeze(-1)
        e_min  = e_bpd.min(dim=1).values.unsqueeze(-1)
        e_stats = torch.cat([e_mean, e_med, e_min], dim=1)  # [B, 3]

        # ---- Fusion ----
        z = torch.cat([z_vit_clip, z_cwt, e_stats], dim=1)  # [B, D + cwt_dim + 3]
        logits = self.head(z)
        aux = {"z_vit": z_vit_clip, "z_cwt": z_cwt, "e_stats": e_stats}
        return logits, aux
