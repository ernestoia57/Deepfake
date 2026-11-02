"""Data-independent tests for paired validation AUC and source-group bootstrap."""
import csv
import json
from pathlib import Path

import pytest

from v4.paired_stats import analyze, load_validation


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def fixture_data(tmp_path, flip=False):
    # A repeated video in the same group is deliberately not independent.
    # Two labels are present in each of 6 source groups.
    m = []
    for split in ("train", "val", "test"):
        for group in range(6):
            for label in (0, 1):
                vid = f"{split}_group{group}_class{label}"
                m.append({"path": str(tmp_path / f"{vid}.mp4"),
                          "video_id": vid, "label": label,
                          "manipulation": "synthetic",
                          "source_group": f"{split}_group{group}",
                          "split": split})
    manifest = tmp_path / "manifest.csv"
    _csv(manifest, m)
    root = tmp_path / "predictions"
    for seed in (42, 43, 44):
        for model in ("rgb", "freq", "fusion"):
            rows = []
            for r in m:
                if r["split"] != "val":
                    continue
                label = int(r["label"])
                if model == "rgb":
                    # constant predictions -> AUC = 0.5
                    prob = 0.5
                else:
                    # Strong but identical FFT/Fusion -> paired delta = 0 exactly.
                    prob = 0.2 if label == 0 else 0.8
                rows.append({"video_id": r["video_id"], "label": label,
                             "probability_fake": prob, "clips": 2})
            _csv(root / f"{model}_s{seed}" / "predictions.csv", rows)
    if flip:
        p = root / "freq_s43" / "predictions.csv"
        with p.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rows[0]["label"] = "1" if rows[0]["label"] == "0" else "0"
        _csv(p, rows)
    return manifest, root


def test_paired_cluster_bootstrap_and_determinism(tmp_path):
    m, root = fixture_data(tmp_path)
    r = analyze(m, root, bootstrap=100, rng_seed=11)
    assert r["data"]["videos"] == 12
    assert r["data"]["source_groups"] == 6
    assert r["data"]["real"] == r["data"]["fake"] == 6
    assert r["models"]["freq"]["mean_auc"] == 1
    assert r["models"]["rgb"]["mean_auc"] == 0.5
    assert r["comparisons"]["freq_minus_rgb"]["mean_delta"] == 0.5
    assert r["comparisons"]["fusion_minus_freq"]["mean_delta"] == 0
    assert r["comparisons"]["fusion_minus_freq"]["cluster_bootstrap_ci95_mean_delta_conditional"] == [0, 0]
    assert r == analyze(m, root, bootstrap=100, rng_seed=11)
    assert r["bootstrap"]["draws_valid"] == 100


def test_reject_label_mismatch(tmp_path):
    m, root = fixture_data(tmp_path, flip=True)
    with pytest.raises(ValueError, match="label disagreement"):
        load_validation(m, root)


def test_reject_missing_validation_video(tmp_path):
    m, root = fixture_data(tmp_path)
    p = root / "rgb_s42" / "predictions.csv"
    with p.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    _csv(p, rows[1:])
    with pytest.raises(ValueError, match="expected exactly val videos"):
        load_validation(m, root)


def test_reject_clip_count_mismatch(tmp_path):
    m, root = fixture_data(tmp_path)
    p = root / "fusion_s44" / "predictions.csv"
    with p.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows[0]["clips"] = "1"
    _csv(p, rows)
    with pytest.raises(ValueError, match="expected 2 clips"):
        load_validation(m, root)
