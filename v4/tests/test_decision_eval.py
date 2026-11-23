"""Dataset-independent tests for v4.9 validation decision analysis."""
import csv

import numpy as np
import pytest

from v4.decision_eval import (
    _confusion_metrics, _select_threshold, analyze, load_predictions,
)


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path):
    manifest = tmp_path / "manifest.csv"
    rows = []
    for split in ("train", "val", "test"):
        for source in range(12):
            for label in (0, 1):
                rows.append({
                    "path": str(tmp_path / "no_video_needed.mp4"),
                    "video_id": f"{split}_source{source}_label{label}",
                    "label": label, "source_group": f"{split}_source{source}",
                    "split": split, "manipulation": "synthetic",
                })
    _write(manifest, rows)
    root = tmp_path / "predictions"
    for mode in ("rgb", "freq", "fusion"):
        for seed in (42, 43, 44):
            preds = []
            for row in rows:
                if row["split"] != "val":
                    continue
                label = int(row["label"])
                source = int(row["video_id"].split("_source")[1].split("_")[0])
                if mode == "rgb":
                    # A controlled error among the fake labels on two groups.
                    p = 0.28 if label == 0 else (0.18 if source in (0, 1) else 0.72)
                elif mode == "freq":
                    p = (0.80 if source in (2, 3) else 0.12) if label == 0 else 0.88
                else:
                    p = (0.79 if source == 4 else 0.15) if label == 0 else 0.89
                preds.append({
                    "video_id": row["video_id"], "label": label,
                    "probability_fake": p + (seed - 43) * 0.005,
                    "clips": 2,
                })
            _write(root / f"{mode}_s{seed}" / "predictions.csv", preds)
    return manifest, root


def test_confusion_metrics_and_precision_denominator():
    y = np.asarray([0, 0, 1, 1])
    pred = np.asarray([0, 1, 0, 1])
    m = _confusion_metrics(y, pred)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 1)
    assert m["accuracy"] == m["balanced_accuracy"] == 0.5
    assert m["precision_fake"] == m["recall_fake"] == m["f1_fake"] == 0.5
    assert m["specificity_real"] == m["fpr"] == 0.5


def test_threshold_selected_from_separate_training_labels():
    y = np.asarray([0, 0, 1, 1])
    p = np.asarray([0.1, 0.4, 0.6, 0.9])
    t = _select_threshold(y, p, "balanced_accuracy")
    assert 0.4 < t < 0.6
    assert _confusion_metrics(y, p >= t)["accuracy"] == 1.0
    assert _select_threshold(y, p, "balanced_accuracy") == t


def test_complete_ensembles_are_deterministic(tmp_path):
    manifest, root = _fixture(tmp_path)
    a = analyze(manifest, root, folds=4)
    b = analyze(manifest, root, folds=4)
    assert a == b
    assert a["data"]["split"] == "val"
    assert a["data"]["videos"] == 24
    assert a["data"]["source_groups"] == 12
    assert len(a["candidates"]) == 15
    assert "freq_fusion_mean" in a["candidates"]
    assert "all_mean" in a["candidates"]
    assert len(a["candidates"]["freq_mean"]["thresholds_by_fold"]) == 4
    for val in a["candidates"].values():
        m = val["crossfit_metrics"]
        assert sum(m[k] for k in ("tn", "fp", "fn", "tp")) == 24
        assert 0 <= m["accuracy"] <= 1
        assert 0 <= m["roc_auc"] <= 1
        assert 0 <= val["threshold_for_future_independent_evaluation"] <= 1.01


def test_rejects_misaligned_validation_ids(tmp_path):
    manifest, root = _fixture(tmp_path)
    path = root / "freq_s42" / "predictions.csv"
    with path.open(newline="", encoding="utf8") as file:
        rows = list(csv.DictReader(file))
    _write(path, rows[1:])
    with pytest.raises(ValueError, match="expected exactly val videos"):
        load_predictions(manifest, root)


def test_rejects_label_or_clip_mismatch(tmp_path):
    manifest, root = _fixture(tmp_path)
    path = root / "fusion_s44" / "predictions.csv"
    with path.open(newline="", encoding="utf8") as file:
        rows = list(csv.DictReader(file))
    rows[0]["label"] = "1" if rows[0]["label"] == "0" else "0"
    _write(path, rows)
    with pytest.raises(ValueError, match="label disagreement"):
        load_predictions(manifest, root)
    rows[0]["label"] = "1" if rows[0]["label"] == "0" else "0"
    rows[0]["clips"] = "3"
    _write(path, rows)
    with pytest.raises(ValueError, match="expected 2 clips"):
        load_predictions(manifest, root)


def test_rejects_impossible_folds_or_duplicate_seeds(tmp_path):
    manifest, root = _fixture(tmp_path)
    with pytest.raises(ValueError, match="folds"):
        analyze(manifest, root, folds=99)
    with pytest.raises(ValueError, match="Seeds"):
        analyze(manifest, root, seeds=(42, 42, 43))
