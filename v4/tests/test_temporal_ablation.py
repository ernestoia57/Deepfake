"""Synthetic, dataset-independent validation of v4.5 temporal controls."""
import csv

import cv2
import numpy as np
import pytest

from v4.dataset import VideoClipDataset
from v4.frequency import fft_temporal_ablation, temporal_fft
from v4.window_index import build_index


def _video():
    rng = np.random.default_rng(123)
    x = rng.integers(20, 200, (16, 32, 32, 3), dtype=np.uint8)
    for t in range(16):
        x[t, 8:18, 10:20] = (10 + 12 * t) % 256
    return x


def test_fft_original_is_identical_to_v43():
    x = _video()
    assert np.array_equal(fft_temporal_ablation(x), temporal_fft(x))
    assert np.array_equal(
        fft_temporal_ablation(x, variant="original", filter_mode="none"),
        temporal_fft(x, filter_mode="none"))


def test_temporal_permutation_changes_spectrum_but_not_shape():
    x = _video()
    perm = np.array([15, 4, 1, 8, 0, 2, 5, 9, 14, 11, 6, 12, 7, 13, 10, 3])
    baseline = fft_temporal_ablation(x, filter_mode="none")
    scrambled = fft_temporal_ablation(
        x, variant="permute", filter_mode="none", permutation=perm)
    assert scrambled.shape == baseline.shape == (8, 32, 32)
    assert np.isfinite(scrambled).all()
    assert not np.allclose(baseline, scrambled)
    # Circular shifts are NOT a valid time disruption for FFT magnitude.
    shifted = temporal_fft(np.roll(x, 3, axis=0), filter_mode="none")
    np.testing.assert_allclose(baseline, shifted, atol=2e-6)


def test_static_middle_ignores_all_other_frames():
    x = _video()
    y = x.copy()
    y[:8] = 0
    y[9:] = 250
    a = fft_temporal_ablation(x, variant="static_middle")
    b = fft_temporal_ablation(y, variant="static_middle")
    assert a.shape == b.shape == (8, 32, 32)
    assert np.array_equal(a, b)
    assert np.array_equal(a[0], a[7])
    assert np.isfinite(a).all()


@pytest.mark.parametrize("variant,permutation", [
    ("permute", None),
    ("permute", list(range(15)) + [14]),
    ("original", list(range(16))),
    ("static_middle", list(range(16))),
    ("not_a_control", None),
])
def test_invalid_temporal_ablation_rejected(variant, permutation):
    with pytest.raises(ValueError):
        fft_temporal_ablation(_video(), variant=variant,
                              permutation=permutation)


def _synthetic_manifest(tmp_path):
    path = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             25.0, (64, 64))
    if not writer.isOpened():
        pytest.skip("OpenCV video writer missing")
    rng = np.random.default_rng(33)
    for _ in range(48):
        writer.write(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
    writer.release()
    rows = [
        dict(path=str(path), video_id=f"{split}-label{lab}", label=lab,
             manipulation="unit", source_group=f"{split}-{lab}", split=split)
        for split in ("train", "val", "test") for lab in ("0", "1")
    ]
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    idx = tmp_path / "windows.json"
    report = build_index(manifest, idx, frames=16, size=32,
                         face="center", max_candidates=8,
                         train_windows=2, eval_windows=2)
    assert report["incomplete_videos"] == 0
    return manifest, idx


def test_val_permutation_is_deterministic_and_seed_independent(tmp_path):
    manifest, idx = _synthetic_manifest(tmp_path)
    common = dict(manifest=manifest, split="val", frames=16, size=32,
                  face="center", frequency="fft", window_index=idx,
                  clips_per_video=2)
    a = VideoClipDataset(**common, seed=42, fft_control="permute")
    b = VideoClipDataset(**common, seed=44, fft_control="permute")
    c = VideoClipDataset(**common, seed=42, fft_control="original")
    static = VideoClipDataset(**common, seed=42, fft_control="static_middle")
    for i in range(len(a)):
        x, y, z, w = a[i], b[i], c[i], static[i]
        assert x["start"] == y["start"] == z["start"] == w["start"]
        assert np.array_equal(x["rgb"], z["rgb"])
        assert np.array_equal(x["freq"], y["freq"])
        assert np.array_equal(x["freq"], a[i]["freq"])
        assert x["freq"].shape == z["freq"].shape == w["freq"].shape


def test_train_epoch_token_is_repeatable(tmp_path):
    manifest, idx = _synthetic_manifest(tmp_path)
    ds = VideoClipDataset(manifest, "train", frames=16, size=32,
                          face="center", frequency="fft", window_index=idx,
                          fft_control="permute", seed=43)
    ds.set_epoch(2)
    a = ds[0]
    b = ds[0]
    assert a["start"] == b["start"]
    assert np.array_equal(a["freq"], b["freq"])


def test_reject_fft_control_for_cwt(tmp_path):
    manifest, idx = _synthetic_manifest(tmp_path)
    with pytest.raises(ValueError, match="FFT only"):
        VideoClipDataset(manifest, "val", frames=16, size=32,
                         face="center", frequency="cwt", window_index=idx,
                         fft_control="permute")
