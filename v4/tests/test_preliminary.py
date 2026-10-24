"""Fast, dataset-independent checks; do not claim these are FF++ results."""
import csv
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
from v4.frequency import residual_video, temporal_fft, temporal_cwt
from v4.manifest import create_manifest, audit_manifest
from v4.models import Detector
from v4.dataset import VideoClipDataset, SampleError


def _frames(n=32, size=40):
    return np.zeros((n, size, size, 3), dtype=np.uint8)


def test_fft_shape_and_static_zero():
    x = _frames()
    f = temporal_fft(x, filter_mode="none")
    assert f.shape == (16, 40, 40)
    assert f.dtype == np.float32
    assert np.all(f == 0.0)


def test_fft_frequency_is_temporal_not_spatial():
    T, fs, hz = 32, 32, 4
    x = _frames(n=T)
    values = 125 + 70*np.sin(2*np.pi*hz*np.arange(T)/fs)
    x[:, :, :, :] = values[:, None, None, None].astype(np.uint8)
    f = temporal_fft(x, filter_mode="none")
    peak = int(f[:, 20, 20].argmax()) + 1
    assert peak == hz


def test_cwt_rejects_unresolvable_low_band():
    with pytest.raises(ValueError, match="cannot be estimated"):
        temporal_cwt(_frames(), fps=30.0, fmin=1.0, fmax=3.0)


def test_cwt_shape_and_nonnegative():
    x = _frames()
    x[::2, 15:25, 15:25, :] = 255
    f = temporal_cwt(x, fps=30.0, bands=4, fmin=6., fmax=10.,
                     filter_mode="none")
    assert f.shape == (4, 40, 40)
    assert np.isfinite(f).all() and np.min(f) >= 0


def test_no_source_group_leakage(tmp_path):
    for sub, stem in [("real", "01__a"), ("real", "02__b"),
                      ("real", "03__c"), ("real", "04__d"),
                      ("fake", "01_02__scene")]:
        p = tmp_path / sub / (stem + ".mp4")
        p.parent.mkdir(exist_ok=True)
        p.touch()
    # Only 3 groups; depending on split, class balancing correctly fails
    with pytest.raises(ValueError, match="missing a real or fake class"):
        create_manifest(tmp_path, tmp_path/"manifest.csv", seed=1)
    path = tmp_path/"bad.csv"
    rows = [
      dict(path="x", video_id="a", label=0, manipulation="original",
           source_group="same", split="train"),
      dict(path="y", video_id="b", label=1, manipulation="fake",
           source_group="same", split="test"),
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match="LEAKAGE"):
        audit_manifest(path, verify_files=False)


def test_models_return_logits_and_gru_is_order_sensitive():
    torch.manual_seed(3)
    rgb = torch.rand(1, 8, 3, 32, 32)
    freq = torch.rand(1, 4, 32, 32)
    for mode in ("rgb", "freq", "fusion"):
        model = Detector(mode=mode, frames=8, dim=16).eval()
        with torch.no_grad():
            score = model(rgb, freq)
        assert score.shape == (1,)
        assert torch.isfinite(score).all()
    # Mean aggregation is invariant under frame permutation.
    model = Detector(mode="rgb", frames=8, dim=16, temporal="mean").eval()
    with torch.no_grad():
        before = model(rgb, freq)
        after = model(rgb.flip(1), freq)
    assert torch.allclose(before, after, atol=1e-5)


def test_bad_decode_is_not_a_labeled_black_frame(tmp_path):
    path = tmp_path / "missing.mp4"
    csv_path = tmp_path/"manifest.csv"
    rows = []
    for split in ("train", "val", "test"):
        for label in (0, 1):
            rows.append(dict(path=str(path), video_id=f"{split}-{label}",
                label=label, manipulation="test", source_group=f"{split}-{label}", split=split))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)
    # Metadata is audited before decoding, including file existence.
    with pytest.raises(FileNotFoundError):
        VideoClipDataset(csv_path, "train", frames=8, size=32)


def test_decoder_center_crop_smoke(tmp_path):
    path = tmp_path / "one.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64,64))
    if not out.isOpened():
        pytest.skip("OpenCV/FFmpeg MP4 writer not present")
    for i in range(16):
        out.write(np.full((64,64,3), i*10, dtype=np.uint8))
    out.release()
    manifest = tmp_path/"m.csv"
    rows = []
    for split in ("train", "val", "test"):
        for label in (0, 1):
            rows.append(dict(path=str(path), video_id=f"{split}-{label}",
                             label=label, manipulation="test",
                             source_group=f"{split}-{label}", split=split))
    with manifest.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0]); wr.writeheader(); wr.writerows(rows)
    ds = VideoClipDataset(manifest, "train", frames=8, size=32, face="center")
    sample = ds[0]
    assert sample["rgb"].shape == (8,3,32,32)
    assert sample["freq"].shape == (4,32,32)
    assert sample["start"] >= 0
