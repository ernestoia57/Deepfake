"""Unit tests: paired 2-vs-4-clip evaluation and strict checkpoint discovery."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from v4.multiclip_eval import (
    _bootstrap_comparisons, _match_config, _valid_prediction_file,
    discover_checkpoints,
)


def _config(manifest, baseline, mode, seed):
    return dict(manifest=str(manifest), window_index=str(baseline),
                mode=mode, seed=seed, frames=16, size=64,
                frequency="fft", fft_control="original",
                filter_mode="median", face="haar", temporal="gru")




def _write_training_run(checkpoint, model_config, epochs):
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model_config, "epoch": min(epochs, 10)}, checkpoint)
    (checkpoint.parent / "config.json").write_text(
        json.dumps({"config": model_config, "train": {"epochs": epochs}}),
        encoding="utf8")

def test_match_only_original_protocol(tmp_path):
    source, baseline = tmp_path / "dataset.csv", tmp_path / "old.json"
    valid = _config(source, baseline, "freq", 42)
    assert _match_config(valid, source, baseline) == "freq_s42"
    for name, change in (
        ("fft_control", "permute"), ("frames", 32),
        ("size", 128), ("mode", "other"), ("face", "center"),
        ("frequency", "cwt"), ("window_index", str(tmp_path / "new.json")),
    ):
        modified = dict(valid, **{name: change})
        assert _match_config(modified, source, baseline) is None


def test_discover_checkpoint_configs_without_guessing_names(tmp_path):
    source, baseline = tmp_path / "manifest.csv", tmp_path / "old.json"
    root = tmp_path / "runs"
    for mode in ("rgb", "freq", "fusion"):
        location = root / f"training_unrelated_name_{mode}" / "best.pt"
        _write_training_run(location, _config(source, baseline, mode, 42), 15)
    unrelated = root / "permuted" / "best.pt"
    _write_training_run(
        unrelated, dict(_config(source, baseline, "freq", 42),
                        fft_control="permute"), 15)
    # Same model config as the real checkpoint, but only 1-epoch smoke.
    smoke = root / "freq_smoke" / "best.pt"
    _write_training_run(smoke, _config(source, baseline, "freq", 42), 1)
    found = discover_checkpoints(root, source, baseline, (42,))
    assert set(found) == {"rgb_s42", "freq_s42", "fusion_s42"}
    duplicate = root / "copy" / "best.pt"
    _write_training_run(duplicate, _config(source, baseline, "freq", 42), 15)
    with pytest.raises(ValueError, match="duplicate candidates"):
        discover_checkpoints(root, source, baseline, (42,))


def test_paired_bootstrap_is_deterministic_and_grouped():
    ids = [f"vid{i}" for i in range(24)]
    groups = np.asarray([f"g{i//2}" for i in range(24)])
    labels = np.asarray([0, 1] * 12)
    p2 = np.asarray([
        (0.84 if (i//2) % 3 == 0 else 0.20) if labels[i] == 0
        else (0.16 if (i//2) % 4 == 0 else 0.78)
        for i in range(24)
    ])
    p4 = np.where(labels == 1, 0.86, 0.12).astype(float)
    sample = lambda p: {
        "ids": ids, "labels": labels, "groups": groups,
        "scores": {f"{mode}_s42": p for mode in ("rgb", "freq", "fusion")},
    }
    a, valid_a = _bootstrap_comparisons(
        sample(p2), sample(p4), (42,), draws=120, rng_seed=2026)
    b, valid_b = _bootstrap_comparisons(
        sample(p2), sample(p4), (42,), draws=120, rng_seed=2026)
    assert a == b and valid_a == valid_b == 120
    assert a["all_mean"]["delta_auc_4_minus_2"] > 0
    lo, hi = a["all_mean"]["cluster_bootstrap_ci95_delta_conditional"]
    assert lo <= hi
    assert "fusion_s44" not in a


def test_refuse_mismatched_existing_predicted_videos(tmp_path):
    csvfile = tmp_path / "predictions.csv"
    with csvfile.open("w", newline="", encoding="utf8") as h:
        writer = csv.DictWriter(h, fieldnames=[
            "video_id", "label", "probability_fake", "clips"])
        writer.writeheader()
        writer.writerow(dict(video_id="val_a", label=1,
                             probability_fake=0.5, clips=4))
    assert _valid_prediction_file(csvfile, {"val_a": 1}, 4)
    with pytest.raises(ValueError, match="label/clips mismatch"):
        _valid_prediction_file(csvfile, {"val_a": 0}, 4)
    with pytest.raises(ValueError, match="unexpected validation video IDs"):
        _valid_prediction_file(csvfile, {"val_a": 1, "val_b": 0}, 4)


def test_refuses_missing_or_inconsistent_training_metadata(tmp_path):
    source, baseline = tmp_path / "manifest.csv", tmp_path / "old.json"
    root = tmp_path / "runs"
    checkpoint = root / "freq" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    config = _config(source, baseline, "freq", 42)
    torch.save({"config": config}, checkpoint)
    with pytest.raises(ValueError, match="Missing training metadata"):
        discover_checkpoints(root, source, baseline, (42,))
    (checkpoint.parent / "config.json").write_text(
        json.dumps({"config": dict(config, size=128),
                    "train": {"epochs": 15}}), encoding="utf8")
    with pytest.raises(ValueError, match="differs from training metadata"):
        discover_checkpoints(root, source, baseline, (42,))


def test_rejects_other_training_budgets_when_no_full_run(tmp_path):
    source, baseline = tmp_path / "manifest.csv", tmp_path / "old.json"
    root = tmp_path / "runs"
    for mode in ("rgb", "freq", "fusion"):
        _write_training_run(
            root / mode / "best.pt", _config(source, baseline, mode, 42), 1)
    with pytest.raises(ValueError, match="excluded_other_epochs"):
        discover_checkpoints(root, source, baseline, (42,), expected_epochs=15)
    selected = discover_checkpoints(root, source, baseline, (42,), expected_epochs=1)
    assert set(selected) == {"rgb_s42", "freq_s42", "fusion_s42"}
