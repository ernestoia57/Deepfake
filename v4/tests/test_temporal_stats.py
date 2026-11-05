"""Dataset-independent tests for v4.5 temporal ablation statistics."""
import csv

import pytest

from v4.temporal_stats import analyze_temporal, load_temporal_validation


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def synthetic_inputs(tmp_path):
    rows = [
        dict(path=str(tmp_path / "unused.mp4"),
             video_id=f"{split}-{group}-label{label}",
             label=str(label), manipulation="synthetic",
             source_group=f"{split}-{group}", split=split)
        for split in ("train", "val", "test")
        for group in range(6) for label in (0, 1)
    ]
    manifest = tmp_path / "manifest.csv"
    save_csv(manifest, rows)
    ref = tmp_path / "old"
    ablations = tmp_path / "new"
    for seed in (42, 43, 44):
        for name in ("original", "permute", "static_middle"):
            out = ref / f"freq_s{seed}" if name == "original" else ablations / f"{name}_s{seed}"
            entries = []
            for r in reversed(rows):
                if r["split"] != "val":
                    continue
                y = int(r["label"])
                if name == "original":
                    prob = 0.1 if y == 0 else 0.9
                elif name == "permute":
                    prob = 0.4 if y == 0 else 0.6
                else:
                    prob = 0.5
                entries.append(dict(
                    video_id=r["video_id"], label=y, clips=2,
                    probability_fake=prob))
            save_csv(out / "predictions.csv", entries)
    return manifest, ref, ablations


def test_temporal_stats_group_paired_deterministic(tmp_path):
    manifest, ref, abl = synthetic_inputs(tmp_path)
    result = analyze_temporal(manifest, ref, abl, bootstrap=100, rng_seed=12)
    assert result["data"]["videos"] == 12
    assert result["data"]["source_groups"] == 6
    assert result["data"]["real"] == result["data"]["fake"] == 6
    assert result["variants"]["original"]["mean_auc"] == 1
    assert result["variants"]["permute"]["mean_auc"] == 1
    assert result["variants"]["static_middle"]["mean_auc"] == 0.5
    assert result["comparisons"]["original_minus_permute"]["mean_delta"] == 0
    assert result["comparisons"]["original_minus_static_middle"]["mean_delta"] == 0.5
    assert result["comparisons"]["original_minus_static_middle"]["cluster_bootstrap_ci95_mean_delta_conditional"] == [0.5, 0.5]
    assert result == analyze_temporal(manifest, ref, abl, bootstrap=100, rng_seed=12)


def test_temporal_stats_rejects_label_mismatch(tmp_path):
    manifest, ref, abl = synthetic_inputs(tmp_path)
    path = abl / "permute_s43" / "predictions.csv"
    with path.open(newline="", encoding="utf8") as f:
        entries = list(csv.DictReader(f))
    entries[0]["label"] = "1" if entries[0]["label"] == "0" else "0"
    save_csv(path, entries)
    with pytest.raises(ValueError, match="label disagreement"):
        load_temporal_validation(manifest, ref, abl)


def test_temporal_stats_rejects_misaligned_validation_ids(tmp_path):
    manifest, ref, abl = synthetic_inputs(tmp_path)
    path = abl / "static_middle_s42" / "predictions.csv"
    with path.open(newline="", encoding="utf8") as f:
        entries = list(csv.DictReader(f))
    save_csv(path, entries[1:])
    with pytest.raises(ValueError, match="expected exactly validation videos"):
        load_temporal_validation(manifest, ref, abl)


def test_temporal_stats_rejects_clip_count_mismatch(tmp_path):
    manifest, ref, abl = synthetic_inputs(tmp_path)
    path = ref / "freq_s44" / "predictions.csv"
    with path.open(newline="", encoding="utf8") as f:
        entries = list(csv.DictReader(f))
    entries[0]["clips"] = "1"
    save_csv(path, entries)
    with pytest.raises(ValueError, match="expected 2 clips"):
        load_temporal_validation(manifest, ref, abl)
