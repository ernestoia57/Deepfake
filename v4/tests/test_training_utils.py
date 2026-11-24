"""v4.10 scheduler/EMA deterministic tests without FF++ data."""
import math

import pytest
import torch
from torch import nn

from v4.training_utils import ModelEMA, epoch_learning_rate


def test_legacy_learning_rate_is_constant_by_default():
    values = [epoch_learning_rate(1e-4, e, 15) for e in range(1, 16)]
    assert values == [1e-4] * 15


def test_warmup_and_cosine_have_expected_endpoints():
    vals = [epoch_learning_rate(1e-4, e, 30, schedule="cosine",
                                warmup_epochs=2, min_lr_ratio=0.05)
            for e in range(1, 31)]
    assert vals[0] == pytest.approx(0.5e-4)
    assert vals[1] == pytest.approx(1e-4)
    assert vals[2] == pytest.approx(1e-4)
    assert vals[-1] == pytest.approx(5e-6)
    assert all(vals[i+1] <= vals[i] for i in range(2, 29))
    assert all(v > 0 for v in vals)


def test_schedule_rejects_bad_controls():
    for options in (
        dict(epoch=0, total_epochs=15),
        dict(epoch=16, total_epochs=15),
        dict(epoch=1, total_epochs=0),
        dict(epoch=1, total_epochs=15, warmup_epochs=15),
        dict(epoch=1, total_epochs=15, min_lr_ratio=-0.1),
        dict(epoch=1, total_epochs=15, min_lr_ratio=1.1),
        dict(epoch=1, total_epochs=15, schedule="triangular"),
    ):
        with pytest.raises(ValueError):
            epoch_learning_rate(1e-4, **options)
    with pytest.raises(ValueError):
        epoch_learning_rate(float("nan"), 1, 15)


def test_model_ema_copies_first_step_then_smooths_weights_and_syncs_bn():
    torch.manual_seed(5)
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))
    ema = ModelEMA(model, decay=0.5)
    assert ema.updates == 0
    assert not any(p.requires_grad for p in ema.module.parameters())
    first = next(model.parameters())
    with torch.no_grad():
        first.fill_(2)
        model[1].running_mean.fill_(9)
        model[1].num_batches_tracked.fill_(7)
    ema.update(model)
    assert ema.updates == 1
    assert torch.allclose(next(ema.module.parameters()), torch.full_like(first, 2))
    assert torch.allclose(ema.module[1].running_mean, model[1].running_mean)
    with torch.no_grad():
        first.fill_(4)
        model[1].running_mean.fill_(11)
        model[1].num_batches_tracked.fill_(8)
    ema.update(model)
    assert ema.updates == 2
    assert torch.allclose(next(ema.module.parameters()), torch.full_like(first, 3))
    assert torch.equal(ema.module[1].num_batches_tracked,
                       model[1].num_batches_tracked)
    assert torch.allclose(ema.module[1].running_mean, model[1].running_mean)
    assert all(math.isfinite(float(v)) for v in ema.module[1].running_mean)


def test_ema_rejects_invalid_decay_and_unchanged_model():
    model = nn.Linear(2, 1)
    for decay in (0.0, 1.0, -0.5, float("nan")):
        with pytest.raises(ValueError):
            ModelEMA(model, decay)
    ema = ModelEMA(model, 0.9)
    before = [p.detach().clone() for p in ema.module.parameters()]
    assert ema.updates == 0
    for old, new in zip(before, ema.module.parameters()):
        assert torch.equal(old, new)
