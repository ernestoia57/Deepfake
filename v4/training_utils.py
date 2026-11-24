"""v4.10: auditable epoch LR schedule and exponential moving averages.

Training-only helpers: no modification of inference architecture,
held-out test, manifest, window index, or labels.
"""
from copy import deepcopy
import math

import torch


def epoch_learning_rate(base_lr, epoch, total_epochs, schedule="constant",
                        warmup_epochs=0, min_lr_ratio=0.05):
    """Epoch-based LR, unchanged constant default; epoch is 1-indexed.

    Warmup increases linearly to base_lr. Cosine starts at base_lr right
    after warmup and reaches base_lr*min_lr_ratio on the final epoch.
    """
    if not (math.isfinite(base_lr) and base_lr > 0):
        raise ValueError("base_lr must be finite and > 0")
    if not (isinstance(total_epochs, int) and total_epochs >= 1):
        raise ValueError("total_epochs must be positive")
    if not (isinstance(epoch, int) and 1 <= epoch <= total_epochs):
        raise ValueError("epoch must be within 1..total_epochs")
    if schedule not in ("constant", "cosine"):
        raise ValueError("schedule must be constant or cosine")
    if not (isinstance(warmup_epochs, int) and 0 <= warmup_epochs < total_epochs):
        raise ValueError("warmup_epochs must be within 0..total_epochs-1")
    if not (math.isfinite(min_lr_ratio) and 0 <= min_lr_ratio <= 1):
        raise ValueError("min_lr_ratio must lie within [0, 1]")
    if warmup_epochs and epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    if schedule == "constant":
        return base_lr
    decay_epochs = total_epochs - warmup_epochs
    # With a single decay epoch there is no interpolation interval.
    if decay_epochs == 1:
        return base_lr
    progress = (epoch - warmup_epochs - 1) / (decay_epochs - 1)
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) *
                      (1 + math.cos(math.pi * progress)) / 2)


class ModelEMA:
    """Shadow model with EMA weights, live BatchNorm running buffers.

    The first optimizer step initializes EMA to the updated weights,
    avoiding overweighting random initialization. Non-floating state
    buffers (such as BN counters) are always copied; all BN buffers
    are synced from the live model instead of averaged.

    AMP skipped optimizer steps must NOT call update().
    """

    def __init__(self, model, decay):
        if not (math.isfinite(decay) and 0 < decay < 1):
            raise ValueError("EMA decay must be in (0,1)")
        self.decay = float(decay)
        self.module = deepcopy(model).eval()
        self.module.requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        if self.updates == 1:
            self.module.load_state_dict(model.state_dict())
            return
        ema_parameters = dict(self.module.named_parameters())
        model_parameters = dict(model.named_parameters())
        if ema_parameters.keys() != model_parameters.keys():
            raise ValueError("EMA architecture parameter mismatch")
        for name, param in model_parameters.items():
            target = ema_parameters[name]
            if target.is_floating_point():
                target.mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)
            else:
                target.copy_(param.detach())
        ema_buffers = dict(self.module.named_buffers())
        model_buffers = dict(model.named_buffers())
        if ema_buffers.keys() != model_buffers.keys():
            raise ValueError("EMA architecture buffer mismatch")
        for name, buffer in model_buffers.items():
            ema_buffers[name].copy_(buffer.detach())
