"""Domain-correction regression tests: synthetic manifest only."""
import csv
from pathlib import Path
import pytest

from v4.protocol import describe_video, prepare, report
from v4.manifest import read_manifest, audit_manifest


def _row(split, label, manipulation, quality, serial, root):
    kind = "original_sequences" if label == 0 else "manipulated_sequences"
    name = f"{serial:03d}.mp4"
    path = root / kind / manipulation / quality / "videos" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return dict(path=str(path), video_id=f"{kind}/{manipulation}/{quality}/videos/{name}",
                label=str(label), manipulation="unknown_fake" if label else "original",
                source_group=f"src-{split}-{label}-{serial}", split=split)


def fixture_manifest(tmp_path):
    rows = []
    for i, split in enumerate(("train", "val", "test")):
        rows.append(_row(split, 0, "youtube", "raw", i*10+1, tmp_path))
        rows.append(_row(split, 1, "Deepfakes", "raw", i*10+2, tmp_path))
        rows.append(_row(split, 1, "Face2Face", "c23", i*10+3, tmp_path))
        rows.append(_row(split, 1, "DeepFakeDetection", "raw", i*10+4, tmp_path))
    path = tmp_path / "source.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader();w.writerows(rows)
    return path


def test_directory_labels_and_qualities(tmp_path):
    original = _row("train", 0, "youtube", "raw", 1, tmp_path)
    detection = _row("val", 1, "DeepFakeDetection", "raw", 2, tmp_path)
    assert describe_video(original) == ("original", "raw")
    assert describe_video(detection) == ("DeepFakeDetection", "raw")
    detection["label"] = "0"
    with pytest.raises(ValueError, match="Label/path mismatch"):
        describe_video(detection)


def test_raw_profile_preserves_splits_and_excludes_compression_confounds(tmp_path):
    inp = fixture_manifest(tmp_path)
    out = tmp_path / "raw_deepfakes.csv"
    result = prepare(inp, out)
    data = read_manifest(out)
    assert result["videos"] == 6
    assert {r["compression"] for r in data} == {"raw"}
    assert {r["manipulation"] for r in data} == {"original", "Deepfakes"}
    expected = {r["video_id"]: r["split"] for r in read_manifest(inp)}
    assert all(expected[r["video_id"]] == r["split"] for r in data)
    assert audit_manifest(out)["videos"] == 6


def test_annotation_keeps_all_videos_and_makes_dfd_explicit(tmp_path):
    inp = fixture_manifest(tmp_path)
    out = tmp_path / "annotated.csv"
    result = prepare(inp, out, profile="annotated_all")
    assert result["videos"] == 12
    assert sum(r["manipulation"] == "DeepFakeDetection" for r in read_manifest(out)) == 3
    summary = report(out)
    assert summary["quality_by_method"]["Face2Face"] == ["c23"]
    assert summary["quality_by_method"]["DeepFakeDetection"] == ["raw"]


def test_never_overwrites_input_and_rejects_unknown_domain(tmp_path):
    inp = fixture_manifest(tmp_path)
    with pytest.raises(ValueError, match="overwrite"):
        prepare(inp, inp)
    fake = _row("train", 1, "UnknownGenerator", "raw", 98, tmp_path)
    with pytest.raises(ValueError, match="unsupported"):
        describe_video(fake)
