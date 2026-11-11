"""Fail-closed C23 matched-manipulation preparation for FF++.

Audits only paths, labels, provenance, fixed splits and source groups.
Never downloads, transcodes, resplits, reads test predictions or changes videos.

Expected original data:
  original_sequences/youtube/raw/videos/<id>.mp4
  manipulated_sequences/Deepfakes/raw/videos/<target>_<source>.mp4
  manipulated_sequences/Face2Face/c23/videos/<target>_<source>.mp4
C23 copies for originals and Deepfakes must exist at corresponding paths.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .manifest import audit_manifest, read_manifest, source_ids
from .protocol import describe_video

SPLITS = ("train", "val", "test")


def _parts(row):
    value = row["video_id"].replace("\\", "/").lower()
    parts = value.split("/")
    if "original_sequences" not in parts and "manipulated_sequences" not in parts:
        parts = row["path"].replace("\\", "/").lower().split("/")
    return parts


def _youtube(row):
    p = _parts(row)
    return "original_sequences" in p and "youtube" in p


def _replace_raw_quality(path, c23_root=None):
    """Map a RAW input to the corresponding C23 path, on the same or another disk.

    c23_root is the directory under which official FF++ original_sequences/
    and manipulated_sequences/ folders are downloaded, e.g.
    C:\\Deepfake\\FFPP_C23. This does not alter existing E: paths.
    """
    p = Path(path)
    parts = list(p.parts)
    positions = [i for i, part in enumerate(parts) if part.lower() == "raw"]
    if len(positions) != 1:
        raise ValueError(f"Expected exactly one raw path component: {path}")
    parts[positions[0]] = "c23"
    if c23_root is None:
        return str(Path(*parts))
    data_roots = [i for i, part in enumerate(parts)
                  if part.lower() in ("original_sequences", "manipulated_sequences")]
    if len(data_roots) != 1:
        raise ValueError("Cannot identify FF++ original_sequences/ or "
                         f"manipulated_sequences/ in RAW path: {path}")
    return str(Path(c23_root).expanduser() / Path(*parts[data_roots[0]:]))


def _verify_parents(originals, fakes, method):
    owners = {}
    for row in originals:
        ids = source_ids(Path(row["video_id"].replace("\\", "/")))
        if len(ids) != 1:
            raise ValueError("Original YouTube videos must have a single numeric source ID")
        key = (row["split"], ids[0])
        if key in owners:
            raise ValueError(f"Duplicate YouTube parent in split: {key}")
        owners[key] = row["source_group"]
    for row in fakes:
        parts = source_ids(Path(row["video_id"].replace("\\", "/")))
        if len(parts) != 2:
            raise ValueError(f"{method} fake lacks target/source parent IDs: {row['video_id']}")
        for sid in parts:
            key = (row["split"], sid)
            if key not in owners:
                raise ValueError(
                    f"{method} missing original YouTube parent {sid} "
                    f"in {row['split']}: {row['video_id']}")
            if owners[key] != row["source_group"]:
                raise ValueError(
                    f"{method} source_group disagrees with original parent "
                    f"{sid}: {row['video_id']}")
    return {"videos": len(fakes), "parent_references_verified": 2 * len(fakes)}


def _collect(manifest, c23_root=None):
    audit_manifest(manifest)
    rows = read_manifest(manifest)
    originals, deepfakes, face2face = [], [], []
    for r in rows:
        method, quality = describe_video(r)
        if method == "original" and quality == "raw" and _youtube(r):
            originals.append(r)
        elif method == "Deepfakes" and quality == "raw":
            deepfakes.append(r)
        elif method == "Face2Face" and quality == "c23":
            face2face.append(r)
    if not (originals and deepfakes and face2face):
        raise ValueError("Need YouTube RAW originals, Deepfakes RAW and Face2Face C23")
    for split in SPLITS:
        counts = [sum(row["split"] == split for row in group)
                  for group in (originals, deepfakes, face2face)]
        if not all(counts):
            raise ValueError(f"Missing method in split={split}: {counts}")
        if len(set(counts)) != 1:
            raise ValueError(
                f"Original/Deepfakes/Face2Face counts differ in {split}: {counts}")
    df_parents = _verify_parents(originals, deepfakes, "Deepfakes")
    f2f_parents = _verify_parents(originals, face2face, "Face2Face")

    candidates = []
    for method, source_rows in (
        ("original", originals), ("Deepfakes", deepfakes), ("Face2Face", face2face)
    ):
        for r in source_rows:
            target = dict(r)
            if method == "Face2Face":
                location = r["path"]
            else:
                location = _replace_raw_quality(r["path"], c23_root=c23_root)
            target["path"] = location
            target["compression"] = "c23"
            target["manipulation"] = method
            # video_id must distinguish c23 rows and represent the path content;
            # replace the exact raw segment in the logical ID as well.
            if method != "Face2Face":
                target["video_id"] = r["video_id"].replace("\\", "/")
                tokens = target["video_id"].split("/")
                indices = [i for i, x in enumerate(tokens) if x.lower() == "raw"]
                if len(indices) != 1:
                    raise ValueError(f"Unexpected RAW video_id: {r['video_id']}")
                tokens[indices[0]] = "c23"
                target["video_id"] = "/".join(tokens)
            candidates.append(target)
    ids = [r["video_id"] for r in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate predicted video IDs in C23 manifest")
    groups = defaultdict(set)
    for r in candidates:
        groups[r["source_group"]].add(r["split"])
    if any(len(s) != 1 for s in groups.values()):
        raise ValueError("C23 source_group leakage across splits")
    return candidates, {"Deepfakes": df_parents, "Face2Face": f2f_parents}


def inspect(manifest, output=None, prepare=False, c23_root=None):
    candidates, provenance = _collect(manifest, c23_root=c23_root)
    counts = Counter((r["split"], r["manipulation"]) for r in candidates)
    missing = [r for r in candidates if not Path(r["path"]).is_file()]
    absent = Counter((r["split"], r["manipulation"]) for r in missing)
    summary = {
        "protocol": "c23_matched_youtube_Deepfakes_Face2Face",
        "manifest": str(manifest),
        "c23_root": str(Path(c23_root).expanduser()) if c23_root else None,
        "candidate_videos": len(candidates),
        "groups": len({r["source_group"] for r in candidates}),
        "expected_by_split_method": [
            {"split": split, "method": method,
             "expected": counts[(split, method)],
             "present": counts[(split, method)] - absent[(split, method)],
             "missing": absent[(split, method)]}
            for split in SPLITS
            for method in ("original", "Deepfakes", "Face2Face")
        ],
        "provenance_check": provenance,
        "all_c23_files_present": not missing,
        "missing_examples": [
            {"method": r["manipulation"], "split": r["split"],
             "expected_path": r["path"]}
            for r in missing[:12]
        ],
        "test_untouched": True,
        "notes": [
            "Read-only dataset-file presence audit, not a decode/face validity certification.",
            "Existing split and source_group assignments preserved; no test model evaluation.",
            "C23 quality is matched at nominal setting only; codec/frame-rate artifacts still need auditing.",
            "Train and select checkpoints on Deepfakes+YouTube C23; Face2Face val is exploratory stress only, never used to select checkpoints.",
            "DeepFakeDetection actors excluded: different identity/provenance domain and imbalanced split.",
            "Acquire FF++ data under official access and usage terms; do not redistribute videos.",
        ],
    }
    if prepare:
        if output is None:
            raise ValueError("prepare requires --out")
        if Path(output).resolve() == Path(manifest).resolve():
            raise ValueError("Refusing to overwrite source manifest")
        if missing:
            raise ValueError(
                f"Cannot prepare C23 manifest: {len(missing)} of "
                f"{len(candidates)} videos missing. Run inspect first.")
        dest = Path(output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", newline="", encoding="utf8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(set().union(
                *(r.keys() for r in candidates))))
            writer.writeheader()
            writer.writerows(candidates)
        summary["output"] = str(dest)
        summary["prepared_manifest_audit"] = audit_manifest(dest)
    elif output:
        dest = Path(output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf8")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Fail-closed FF++ same-compression cross-manipulation dataset audit")
    sub = parser.add_subparsers(dest="command", required=True)
    for cmd in ("inspect", "prepare"):
        p = sub.add_parser(cmd)
        p.add_argument("--manifest", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--c23-root", default=None,
                       help="Alternate FF++ root for new original/Deepfakes C23 files, e.g. C:\\Deepfake\\FFPP_C23")
    args = parser.parse_args()
    result = inspect(args.manifest, output=args.out,
                     prepare=args.command == "prepare", c23_root=args.c23_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
