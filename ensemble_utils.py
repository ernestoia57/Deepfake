
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_blend(p_vit: torch.Tensor, p_cwt: torch.Tensor, p_diff: torch.Tensor,
                   weights: Tuple[float, float, float] = (0.6, 0.25, 0.15)) -> torch.Tensor:
    """
    Simple convex combination of probabilities for the FAKE class.
    Inputs are tensors of shape [B], values in [0,1].
    """
    w1, w2, w3 = weights
    s = w1 * p_vit + w2 * p_cwt + w3 * p_diff
    return torch.clamp(s, 0.0, 1.0)


class StackingHead(nn.Module):
    """
    Tiny logistic stacker over features [logit_vit, prob_cwt, elbo_mean, elbo_med, elbo_min].
    """
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(5, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, 5]
        logit = self.lin(feats).squeeze(1)  # [B]
        return torch.sigmoid(logit)


def make_stacking_features(
    logits_vit: torch.Tensor,  # [B, 2]
    prob_cwt: torch.Tensor,    # [B] prob FAKE from CWT-only head or model
    e_stats: torch.Tensor,     # [B, 3] mean/median/min ELBO-bpd
) -> torch.Tensor:
    prob_vit = F.softmax(logits_vit, dim=-1)[:, 1]  # FAKE prob
    feat = torch.stack([logits_vit[:, 1], prob_cwt, e_stats[:, 0], e_stats[:, 1], e_stats[:, 2]], dim=1)
    return feat  # [B, 5]
