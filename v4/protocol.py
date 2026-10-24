"""v4.1 domain, compression and crop-audit controls.

Nothing here modifies, recompresses or relabels video content. The split and
source_group assignments are copied from the input manifest *unchanged*.
The RAW/Deepfakes subset is a diagnostic baseline, not a cross-synthesis study.
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .manifest import REQUIRED, SPLITS, audit_manifest, read_manifest, source_ids

METHODS = {
    "deepfakes": "Deepfakes",
    "face2face": "Face2Face",
    "faceswap": "FaceSwap",
    "neuraltextures": "NeuralTextures",
    "deepfakedetection": "DeepFakeDetection",
}
QUALITIES = {"raw", "c23", "c40"}


def describe_video(row):
    """Return (manipulation, compression) using exact path components.

    The class must agree with the directory convention; surprising paths
    fail closed instead of being guessed. Only path *components* are checked,
    avoiding accidental substring matches like /foo/raw_export/.
    """
    name = (row.get("video_id") or row.get("path") or "")
    parts = [p.lower() for p in name.replace("\\", "/").split("/") if p]
    if not (any(p in ("original_sequences", "manipulated_sequences",
                      "real", "fake") for p in parts)):
        parts = [p.lower() for p in row["path"].replace("\\", "/").split("/") if p]
    label = str(row["label"])
    original = any(p in ("original_sequences", "real") for p in parts)
    manipulated = any(p in ("manipulated_sequences", "fake") for p in parts)
    if original == manipulated:
        raise ValueError(f"Ambiguous REAL/FAKE source directories in {name}")
    if original:
        method = "original"
    else:
        candidates = {METHODS[p] for p in parts if p in METHODS}
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous/unsupported manipulation for {name}: {candidates}")
        method = candidates.pop()
    expected = "0" if original else "1"
    if label != expected:
        raise ValueError(f"Label/path mismatch for {name}: label={label}, expected={expected}")
    quality_hits = {p for p in parts if p in QUALITIES}
    if len(quality_hits) != 1:
        raise ValueError(f"Expected exactly one raw/c23/c40 path component in {name}")
    return method, quality_hits.pop()


def normalized_rows(manifest):
    # Checks that existing provenance groups and per-split class presence
    # satisfy the invariants before applying any subset filter.
    audit_manifest(manifest)
    records = []
    for r in read_manifest(manifest):
        method, quality = describe_video(r)
        new_row = dict(r)
        new_row["manipulation"] = method
        new_row["compression"] = quality
        records.append(new_row)
    return records


def summarize(rows):
    return {
        "videos": len(rows),
        "source_groups": len({r["source_group"] for r in rows}),
        "by_split_label": {
            s: dict(Counter(r["label"] for r in rows if r["split"] == s))
            for s in SPLITS
        },
        "by_split_method_quality": [
            {"split": s, "manipulation": method, "compression": quality, "videos": count}
            for (s, method, quality), count in sorted(Counter(
                (r["split"], r["manipulation"], r["compression"])
                for r in rows
            ).items())
        ],
    }


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_raw_youtube_deepfakes_pairs(rows):
    """Ensure every selected Deepfakes parent ID is a RAW YouTube original.

    Same totals by class are NOT evidence of pairing. Verify filenames,
    split and source_group explicitly. FF++ two-parent fake names such as
    559_543 must resolve both 559.mp4 and 543.mp4 within the same split.
    """
    originals = {}
    origins_across_splits = {}
    for row in rows:
        if row["manipulation"] != "original":
            continue
        identifier = source_ids(Path(row["video_id"].replace("\\", "/")))
        if len(identifier) != 1:
            raise ValueError("Expected one source ID for a YouTube original: "
                             + row["video_id"])
        sid = identifier[0]
        key = (row["split"], sid)
        if key in originals:
            raise ValueError(f"Duplicate YouTube source in split {key}")
        if sid in origins_across_splits and origins_across_splits[sid] != row["split"]:
            raise ValueError(f"YouTube source ID appears in multiple splits: {sid}")
        origins_across_splits[sid] = row["split"]
        originals[key] = row["source_group"]

    failures = []
    parent_refs = 0
    for row in rows:
        if row["manipulation"] != "Deepfakes":
            continue
        source_parts = source_ids(Path(row["video_id"].replace("\\", "/")))
        for sid in source_parts:
            parent_refs += 1
            expected_group = originals.get((row["split"], sid))
            if expected_group is None:
                failures.append(f"missing RAW YouTube original {sid} in {row['split']}: "
                                + row["video_id"])
            elif row["source_group"] != expected_group:
                failures.append(f"source_group mismatch for {sid}: "
                                + row["video_id"])
    if failures:
        raise ValueError("Unpaired fake/original provenance (first 8): "
                         + " | ".join(failures[:8]))

    for split in SPLITS:
        counts = Counter(r["label"] for r in rows if r["split"] == split)
        if counts["0"] != counts["1"]:
            raise ValueError(f"RAW paired split {split} not balanced: {dict(counts)}")

    return {"verified_parent_references": parent_refs,
            "unique_youtube_sources": len(originals),
            "all_parent_ids_match_split_and_source_group": True}


def prepare(input_manifest, output, profile="raw_deepfakes"):
    inp, out = Path(input_manifest), Path(output)
    if inp.resolve() == out.resolve():
        raise ValueError("Refusing to overwrite source manifest")
    rows = normalized_rows(inp)
    if profile == "raw_deepfakes":
        selected = [r for r in rows if r["compression"] == "raw"
                    and r["manipulation"] in ("original", "Deepfakes")]
    elif profile == "raw_youtube_deepfakes":
        selected = [
            r for r in rows if r["compression"] == "raw" and (
                r["manipulation"] == "Deepfakes" or (
                    r["manipulation"] == "original" and
                    "youtube" in r["video_id"].replace("\\", "/").lower().split("/")
                )
            )
        ]
    elif profile == "annotated_all":
        selected = rows
    else:
        raise ValueError("Unknown profile")
    if not selected:
        raise ValueError(f"No videos selected by profile={profile}")
    # Check BEFORE writing. No resplitting or group changes.
    for split in SPLITS:
        label_counts = Counter(r["label"] for r in selected if r["split"] == split)
        if not label_counts.get("0") or not label_counts.get("1"):
            raise ValueError(f"{profile}: {split} has missing class: {label_counts}")
    owners = defaultdict(set)
    for r in selected:
        owners[r["source_group"]].add(r["split"])
    leakage = [g for g, s in owners.items() if len(s) != 1]
    if leakage:
        raise ValueError(f"Source-group leakage in selected rows: {leakage[:5]}")
    paired = (verify_raw_youtube_deepfakes_pairs(selected)
              if profile == "raw_youtube_deepfakes" else None)
    _write_csv(out, selected)
    check = audit_manifest(out)
    report = summarize(selected)
    if paired is not None:
        report["pair_verification"] = paired
    report.update({"profile": profile, "output": str(out),
                   "partitions_preserved": True, "audit": check,
                   "warning": ("Exploratory RAW within-domain subset. "
                               "Not a Face2Face vs Deepfakes cross-synthesis study; "
                               "not an official FF++ split reproduction.")})
    return report


def report(manifest):
    rows = normalized_rows(manifest)
    result = summarize(rows)
    qualities = {r["compression"] for r in rows}
    methods = sorted({r["manipulation"] for r in rows})
    warnings = []
    if len(qualities) > 1:
        warnings.append(
            "Multiple compression domains present. Avoid attributing quality "
            "differences to manipulation effects; compare matched qualities."
        )
    quality_by_method = defaultdict(set)
    for r in rows:
        quality_by_method[r["manipulation"]].add(r["compression"])
    result["quality_by_method"] = {m: sorted(v) for m, v in sorted(quality_by_method.items())}
    if "Face2Face" in methods and quality_by_method["Face2Face"] == {"c23"}:
        warnings.append("Face2Face is C23-only here; it is not matched to the RAW domains.")
    result["warnings"] = warnings
    return result


def probe(manifest, samples_per_stratum=20, frames=16, size=64,
          face="haar", seed=42):
    """Reproducible RANDOM samples per (split, method, label) stratum.

    This addresses v4.audit's bias toward the first N sorted videos.
    Reports decode/crop failures; NEVER inserts black labeled substitutes.
    """
    if samples_per_stratum < 1:
        raise ValueError("samples_per_stratum must be >= 1")
    from .dataset import VideoClipDataset
    records = normalized_rows(manifest)
    rng = random.Random(seed)
    outcomes = []
    totals = Counter()
    failures = Counter()
    for split in SPLITS:
        ds = VideoClipDataset(manifest, split=split, frames=frames,
                              size=size, face=face, seed=seed)
        by_stratum = defaultdict(list)
        for idx, row in enumerate(ds.rows):
            method, quality = describe_video(row)
            by_stratum[(method, quality, row["label"])].append(idx)
        for (method, quality, label), candidates in sorted(by_stratum.items()):
            sample = rng.sample(candidates, min(samples_per_stratum, len(candidates)))
            for idx in sample:
                key = (split, method, quality, label)
                totals[key] += 1
                try:
                    ds[idx]
                except Exception as exc:
                    failures[key] += 1
                    outcomes.append(dict(split=split, method=method,
                        compression=quality, label=label,
                        video_id=ds.rows[idx]["video_id"],
                        error_type=type(exc).__name__, error=str(exc)))
    stats = []
    for key in sorted(totals):
        n = totals[key]
        k = failures[key]
        stats.append(dict(split=key[0], method=key[1], compression=key[2],
                          label=key[3], audited=n, failures=k,
                          failure_rate=k/n))
    return {"seed": seed, "face": face, "clips": sum(totals.values()),
            "failures": sum(failures.values()),
            "by_stratum": stats, "failed_videos": outcomes,
            "warning": "Stratified sample probe, not full-dataset validity certification."}


def main():
    ap = argparse.ArgumentParser(description="v4.2 provenance-aware FF++ dataset protocol")
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("report", "prepare", "probe"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True)
        p.add_argument("--out")
        if name == "prepare":
            p.add_argument("--profile", choices=("raw_deepfakes","raw_youtube_deepfakes","annotated_all"),
                           default="raw_deepfakes")
        elif name == "probe":
            p.add_argument("--samples-per-stratum", type=int, default=20)
            p.add_argument("--frames", type=int, default=16)
            p.add_argument("--size", type=int, default=64)
            p.add_argument("--face", choices=("haar","center"), default="haar")
            p.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.command == "report":
        result = report(args.manifest)
    elif args.command == "prepare":
        if not args.out:
            ap.error("prepare requires --out")
        result = prepare(args.manifest, args.out, args.profile)
    else:
        result = probe(args.manifest, samples_per_stratum=args.samples_per_stratum,
                       frames=args.frames, size=args.size,
                       face=args.face, seed=args.seed)
    output = json.dumps(result, indent=2, ensure_ascii=False)
    print(output)
    if args.out and args.command != "prepare":
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
