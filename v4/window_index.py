"""v4.3 reproducible per-video window prevalidation and cached face crops.

Conservative by design:
* never inserts fabricated frames, silently drops samples, or changes splits;
* indexes fixed frame starts and stable crop squares;
* fails closed if a video cannot supply all required windows;
* provenance bound to SHA256 of the original manifest CSV;
* shared verbatim by RGB/FFT/fusion across matched seeds.

Only one frame window per indexed candidate is checked. Videos may have other
unvalidated windows, which are intentionally not sampled during training.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from .manifest import audit_manifest, read_manifest, SPLITS
from .dataset import SampleError, StableFaceCrop, video_metadata, read_video_window


SCHEMA = 1


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_starts(frame_count, frames, maximum=16):
    """Spread fixed candidate starts over the clip, middle first then farthest.

    Candidate ordering never depends on class labels, model predictions, or
    test performance. First attempt is the middle window used in the v4 probe.
    """
    if not (frames >= 8 and frames % 2 == 0 and maximum >= 1):
        raise ValueError("Expected even frames >= 8 and maximum >= 1")
    if frame_count < frames:
        return []
    max_start = frame_count - frames
    positions = sorted({int(round(max_start*i/max(1,maximum-1)))
                        for i in range(maximum)})
    if maximum == 1:
        positions = [max_start//2]
    ordered = [min(positions, key=lambda x: (abs(x-max_start/2), x))]
    positions.remove(ordered[0])
    while positions:
        next_start = max(positions, key=lambda x: (
            min(abs(x-p) for p in ordered), -abs(x-max_start/2), -x))
        ordered.append(next_start)
        positions.remove(next_start)
    return ordered


def build_index(manifest, output, frames=16, size=64, face="haar",
                max_candidates=16, train_windows=4, eval_windows=3):
    """Create an auditable JSON index and return summary; never skip a video.

    Result is incomplete if any video cannot supply required valid windows;
    train and evaluate refuse incomplete indexes. Increase max_candidates
    or improve face detection; do NOT remove failures without group review.
    """
    if frames < 8 or frames % 2 or size < 32:
        raise ValueError("Require even frames >= 8 and size >= 32")
    if max_candidates < 1 or train_windows < 1 or eval_windows < 1:
        raise ValueError("Candidate/window counts must be positive")
    if face not in ("haar", "center"):
        raise ValueError("face must be haar or center")
    source = Path(manifest)
    dest = Path(output)
    if source.resolve() == dest.resolve():
        raise ValueError("Never overwrite input manifest")
    audit_manifest(source)
    rows = read_manifest(source)
    cropper = StableFaceCrop() if face == "haar" else None
    records = []
    summary_counts = Counter()
    failure_counts = Counter()
    unrecovered = []
    all_errors = Counter()
    for row in rows:
        split = row["split"]
        required = train_windows if split == "train" else eval_windows
        key = (split, row["label"])
        summary_counts[key] += 1
        valid = []
        examples = []
        attempted = 0
        count = None
        fps = None
        try:
            count, fps = video_metadata(row["path"])
            starts = candidate_starts(count, frames, maximum=max_candidates)
            if not starts:
                raise SampleError(f"Short video: {row['video_id']}, frames={count}")
            for start in starts:
                if len(valid) >= required:
                    break
                attempted += 1
                try:
                    clip, clip_fps = read_video_window(row["path"], frames, start)
                    if cropper is not None:
                        bbox = cropper.locate(clip)
                        # Crop exactly as the runtime will, verifying bounds.
                        cropper(clip, size=size, bbox=bbox)
                    else:
                        bbox = None  # Debug-only center crop, not a face
                    valid.append({"start": int(start), "bbox": bbox})
                except (SampleError, ValueError, OSError, RuntimeError) as exc:
                    all_errors[type(exc).__name__ + ": " + str(exc).split(":")[0]] += 1
                    if len(examples) < 3:
                        examples.append({"start": int(start), "error": str(exc)})
        except (SampleError, ValueError, OSError, RuntimeError) as exc:
            all_errors[type(exc).__name__ + ": " + str(exc).split(":")[0]] += 1
            examples.append({"start": None, "error": str(exc)})
        valid.sort(key=lambda w: w["start"])
        complete_video = len(valid) >= required
        if not complete_video:
            failure_counts[key] += 1
            unrecovered.append({"video_id": row["video_id"], "split": split,
                                "label": row["label"], "needed": required,
                                "found": len(valid), "attempted": attempted,
                                "errors": examples})
        records.append({"video_id": row["video_id"], "split": split,
                        "label": row["label"], "source_group": row["source_group"],
                        "frame_count": count, "fps": fps,
                        "windows": valid, "attempted": attempted})
    summary = {
        "videos": len(records),
        "complete_videos": len(records) - len(unrecovered),
        "incomplete_videos": len(unrecovered),
        "by_split_label": [
            {"split": split, "label": label, "videos": summary_counts[(split, label)],
             "incomplete": failure_counts[(split, label)]}
            for split in SPLITS for label in ("0", "1")
        ],
        "error_counts": dict(all_errors.most_common(20)),
        "incomplete_examples": unrecovered,
        "warning": (
            "Fixed-window Haar/center prevalidation, not verified face identity "
            "or robustness under unseen encoding/decoders. Incomplete indexes "
            "are deliberately rejected by train/evaluate."
        ),
    }
    index = {
        "schema": SCHEMA,
        "manifest_sha256": sha256_file(source),
        "settings": {"frames": frames, "size": size, "face": face,
                     "max_candidates": max_candidates,
                     "train_windows": train_windows,
                     "eval_windows": eval_windows},
        "complete": not unrecovered,
        "summary": summary,
        "records": records,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def load_index(index_path, manifest, frames, size, face):
    """Return verified video_id -> indexed windows; reject mismatches."""
    audit_manifest(manifest)
    data = json.loads(Path(index_path).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError("Unsupported window-index schema")
    if not data.get("complete"):
        raise ValueError("Window index incomplete; see summary.incomplete_examples")
    if data.get("manifest_sha256") != sha256_file(manifest):
        raise ValueError("Manifest/index SHA256 mismatch; rebuild the index")
    expected = {"frames": frames, "size": size, "face": face}
    for name, value in expected.items():
        if data.get("settings", {}).get(name) != value:
            raise ValueError(f"Window index setting {name} mismatch; rebuild")
    rows = read_manifest(manifest)
    observed = {}
    for rec in data.get("records", []):
        vid = rec["video_id"]
        if vid in observed:
            raise ValueError(f"Duplicate index video_id: {vid}")
        observed[vid] = rec
    if set(observed) != {r["video_id"] for r in rows}:
        raise ValueError("Window index records do not match manifest videos")
    for row in rows:
        rec = observed[row["video_id"]]
        for key in ("label", "split", "source_group"):
            if rec.get(key) != row[key]:
                raise ValueError(f"Window index {key} mismatch: {row['video_id']}")
        starts = set()
        for w in rec["windows"]:
            start = w["start"]
            if not isinstance(start, int) or start < 0 or start in starts:
                raise ValueError(f"Invalid or duplicate window start for {row['video_id']}")
            starts.add(start)
            bbox = w.get("bbox")
            if face == "haar" and not (isinstance(bbox, list) and len(bbox) == 3
                                       and all(isinstance(v, int) for v in bbox)):
                raise ValueError(f"Missing indexed face bounding box for {row['video_id']}")
    return {vid: rec["windows"] for vid, rec in observed.items()}


def main():
    p = argparse.ArgumentParser(description="v4.3 validated fixed-window index")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--face", choices=("haar", "center"), default="haar")
    p.add_argument("--max-candidates", type=int, default=16)
    p.add_argument("--train-windows", type=int, default=4)
    p.add_argument("--eval-windows", type=int, default=3)
    args = p.parse_args()
    summary = build_index(args.manifest, args.out, args.frames, args.size,
                          args.face, args.max_candidates,
                          args.train_windows, args.eval_windows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["incomplete_videos"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
