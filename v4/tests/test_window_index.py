"""Synthetic regressions for v4.3 fixed, prevalidated video windows."""
import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from v4.dataset import SampleError, StableFaceCrop, VideoClipDataset
from v4.window_index import candidate_starts, build_index, load_index


def _video(path, n=32):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             25.0, (64, 64))
    if not writer.isOpened():
        pytest.skip("No FFmpeg MP4 writer available")
    for i in range(n):
        writer.write(np.full((64, 64, 3), i*4 % 255, dtype=np.uint8))
    writer.release()


def _fixture(tmp_path, n=32):
    mp4 = tmp_path / "tiny.mp4"
    _video(mp4, n=n)
    records = []
    for split in ("train", "val", "test"):
        for label in ("0", "1"):
            records.append(dict(path=str(mp4), video_id=f"{split}-{label}",
                                label=label, manipulation="dummy",
                                source_group=f"{split}-{label}", split=split))
    manifest = tmp_path / "input.csv"
    with manifest.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader()
        w.writerows(records)
    return manifest


def test_farthest_first_window_starts_are_deterministic():
    a = candidate_starts(100, 16, maximum=9)
    assert a == candidate_starts(100, 16, maximum=9)
    assert a[0] == 42
    assert len(set(a)) == len(a) == 9
    assert 0 in a and 84 in a
    assert candidate_starts(8, 8, maximum=9) == [0]
    assert candidate_starts(7, 8, maximum=9) == []


def test_cached_face_bbox_reused_across_models_and_epochs(monkeypatch, tmp_path):
    manifest = _fixture(tmp_path)
    monkeypatch.setattr(StableFaceCrop, "locate",
                        lambda self, frames: [12, 12, 32])
    index_path = tmp_path / "valid.json"
    result = build_index(manifest, index_path, frames=8, size=32, face="haar",
                         max_candidates=7, train_windows=3, eval_windows=2)
    assert result["incomplete_videos"] == 0
    indexed = load_index(index_path, manifest, frames=8, size=32, face="haar")
    assert len(indexed) == 6
    # Runtime must not call Haar when reusing cached crop from the index.
    def no_detection(self, frames):
        raise AssertionError("Haar rerun despite cached crop")
    monkeypatch.setattr(StableFaceCrop, "locate", no_detection)
    rgb = VideoClipDataset(manifest, "val", frames=8, size=32, face="haar",
                           frequency="fft", window_index=index_path,
                           clips_per_video=2)
    freq = VideoClipDataset(manifest, "val", frames=8, size=32, face="haar",
                            frequency="fft", window_index=index_path,
                            clips_per_video=2)
    assert rgb[0]["start"] == freq[0]["start"]
    assert np.array_equal(rgb[0]["rgb"], freq[0]["rgb"])
    assert rgb[0]["rgb"].shape == (8, 3, 32, 32)
    train = VideoClipDataset(manifest, "train", frames=8, size=32,
                             face="haar", window_index=index_path, seed=99)
    train.set_epoch(2)
    assert train[0]["start"] == train[0]["start"]
    assert train[0]["start"] in {x["start"] for x in indexed["train-0"]}
    assert rgb[1]["start"] != rgb[0]["start"]


def test_index_rejects_stale_manifest_and_wrong_size(monkeypatch, tmp_path):
    manifest = _fixture(tmp_path)
    monkeypatch.setattr(StableFaceCrop, "locate",
                        lambda self, frames: [12, 12, 32])
    dest = tmp_path / "index.json"
    build_index(manifest, dest, frames=8, size=32, face="haar",
                train_windows=2, eval_windows=2)
    with pytest.raises(ValueError, match="setting size"):
        load_index(dest, manifest, 8, 64, "haar")
    with manifest.open("a", encoding="utf8") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_index(dest, manifest, 8, 32, "haar")


def test_incomplete_index_is_kept_as_report_but_rejected(monkeypatch, tmp_path):
    manifest = _fixture(tmp_path, n=8)
    monkeypatch.setattr(StableFaceCrop, "locate",
                        lambda self, frames: [12, 12, 32])
    dest = tmp_path / "partial.json"
    report = build_index(manifest, dest, frames=8, size=32, face="haar",
                         max_candidates=12, train_windows=2, eval_windows=2)
    assert report["incomplete_videos"] == 6
    assert dest.exists()
    with pytest.raises(ValueError, match="incomplete"):
        load_index(dest, manifest, 8, 32, "haar")
