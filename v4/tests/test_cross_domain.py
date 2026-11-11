"""Synthetic C23 cross-manipulation protocol and provenance checks."""
import csv
from pathlib import Path

import pytest

from v4.cross_domain import _replace_raw_quality, inspect
from v4.manifest import read_manifest


def _fixture(tmp_path, c23_ready=False, wrong_group=False, missing_parent=False):
    root = tmp_path / "ffpp"
    rows = []
    for split, start in (("train", 100), ("val", 200), ("test", 300)):
        ids = [str(start), str(start + 1)]
        for id_ in ids:
            p = root / "original_sequences" / "youtube" / "raw" / "videos" / f"{id_}.mp4"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
            rows.append({
                "path": str(p),
                "video_id": str(p.relative_to(root)).replace("\\", "/"),
                "label": "0", "manipulation": "original",
                "source_group": split, "split": split,
            })
            if c23_ready:
                cp = Path(_replace_raw_quality(str(p)))
                cp.parent.mkdir(parents=True, exist_ok=True)
                cp.touch()
        fake_ids = [f"{ids[0]}_{ids[1]}", f"{ids[1]}_{ids[0]}"]
        if missing_parent and split == "val":
            fake_ids[-1] = "999_998"
        for method, quality in (("Deepfakes", "raw"), ("Face2Face", "c23")):
            for i, stem in enumerate(fake_ids):
                p = root / "manipulated_sequences" / method / quality / "videos" / f"{stem}.mp4"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()
                rows.append({
                    "path": str(p),
                    "video_id": str(p.relative_to(root)).replace("\\", "/"),
                    "label": "1", "manipulation": method,
                    "source_group": (
                        split + "_unrelated" if wrong_group and split == "val"
                        and method == "Face2Face" and i == 0 else split),
                    "split": split,
                })
                if c23_ready and method == "Deepfakes":
                    cp = Path(_replace_raw_quality(str(p)))
                    cp.parent.mkdir(parents=True, exist_ok=True)
                    cp.touch()
    manifest = tmp_path / "ffpp.csv"
    with manifest.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_inspect_is_read_only_and_reports_missing_c23(tmp_path):
    manifest = _fixture(tmp_path)
    report_file = tmp_path / "report.json"
    result = inspect(manifest, output=report_file)
    assert result["candidate_videos"] == 18
    assert result["groups"] == 3
    assert not result["all_c23_files_present"]
    assert sum(r["missing"] for r in result["expected_by_split_method"]) == 12
    assert report_file.is_file()
    assert not (tmp_path / "prepared.csv").exists()


def test_prepare_only_when_all_files_and_provenance_valid(tmp_path):
    manifest = _fixture(tmp_path, c23_ready=True)
    out = tmp_path / "prepared.csv"
    result = inspect(manifest, output=out, prepare=True)
    assert result["all_c23_files_present"]
    assert result["prepared_manifest_audit"]["videos"] == 18
    rows = read_manifest(out)
    assert all("/c23/" in r["video_id"] for r in rows)
    assert set(r["manipulation"] for r in rows) == {"original", "Deepfakes", "Face2Face"}
    assert len({r["video_id"] for r in rows}) == 18
    assert all(Path(r["path"]).is_file() for r in rows)
    assert len({r["source_group"] for r in rows}) == 3


def test_prepare_refuses_missing_and_does_not_write_partial(tmp_path):
    manifest = _fixture(tmp_path)
    out = tmp_path / "prepared.csv"
    with pytest.raises(ValueError, match="videos missing"):
        inspect(manifest, output=out, prepare=True)
    assert not out.exists()


def test_rejects_source_mismatch(tmp_path):
    manifest = _fixture(tmp_path, c23_ready=True, wrong_group=True)
    with pytest.raises(ValueError, match="source_group"):
        inspect(manifest)


def test_rejects_missing_parent(tmp_path):
    manifest = _fixture(tmp_path, c23_ready=True, missing_parent=True)
    with pytest.raises(ValueError, match="missing original YouTube parent"):
        inspect(manifest)
