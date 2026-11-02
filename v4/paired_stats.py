"""Exploratory paired validation AUC with FF++ source-group bootstrap.

Input: v4.pipeline evaluate predictions.csv for all 3 models x 3 seeds.
This estimates uncertainty from validation source groups conditional on already
trained and validation-selected checkpoints. It is NOT a confirmatory test;
no inference of generalization, and no use of test split here.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .manifest import read_manifest


MODELS = ("rgb", "freq", "fusion")
PAIRS = (("freq", "rgb"), ("fusion", "rgb"), ("fusion", "freq"))


def _read_predictions(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"video_id", "label", "probability_fake", "clips"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing columns {required - set(reader.fieldnames or [])}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty predictions")
    out = {}
    for row in rows:
        video_id = row["video_id"]
        if video_id in out:
            raise ValueError(f"{path}: repeated video_id {video_id}")
        label, clips = int(row["label"]), int(row["clips"])
        probability = float(row["probability_fake"])
        if label not in (0, 1) or clips < 1 or not np.isfinite(probability) or not (0 <= probability <= 1):
            raise ValueError(f"{path}: invalid label, clip count or probability for {video_id}")
        out[video_id] = (label, probability, clips)
    return out


def load_validation(manifest_path, predictions_root, seeds=(42, 43, 44), clips=2):
    """Return sorted labels/groups, model probability vectors for each seed."""
    manifest = Path(manifest_path)
    rows = read_manifest(manifest)
    validation = {r["video_id"]: r for r in rows if r["split"] == "val"}
    if not validation:
        raise ValueError("Manifest contains no validation videos")
    if len(validation) != sum(r["split"] == "val" for r in rows):
        raise ValueError("Repeated video_id in manifest validation split")
    ids = sorted(validation)
    labels = np.asarray([int(validation[v]["label"]) for v in ids], dtype=np.int64)
    groups = np.asarray([validation[v]["source_group"] for v in ids])
    if len(np.unique(labels)) != 2:
        raise ValueError("Validation must contain both labels")
    predictions = {}
    for seed in seeds:
        for mode in MODELS:
            csv_path = Path(predictions_root) / f"{mode}_s{seed}" / "predictions.csv"
            data = _read_predictions(csv_path)
            if set(data) != set(ids):
                missing = sorted(set(ids) - set(data))
                extra = sorted(set(data) - set(ids))
                raise ValueError(f"{csv_path}: expected exactly val videos; missing={missing[:3]}, extra={extra[:3]}")
            for vid in ids:
                lab, _, nclips = data[vid]
                if lab != int(validation[vid]["label"]):
                    raise ValueError(f"{csv_path}: label disagreement for {vid}")
                if nclips != clips:
                    raise ValueError(f"{csv_path}: expected {clips} clips per video, got {nclips} for {vid}")
            predictions[(int(seed), mode)] = np.asarray([data[v][1] for v in ids], dtype=np.float64)
    return {"ids": ids, "labels": labels, "groups": groups,
            "predictions": predictions, "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}


def _ci(values):
    if len(values) < 20:
        return [None, None]
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def analyze(manifest_path, predictions_root, seeds=(42, 43, 44), clips=2,
            bootstrap=2000, rng_seed=2026):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique and nonempty")
    if bootstrap < 100:
        raise ValueError("At least 100 bootstrap iterations required")
    sample = load_validation(manifest_path, predictions_root, seeds, clips)
    y, groups, preds = sample["labels"], sample["groups"], sample["predictions"]
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    by_group = [np.flatnonzero(inverse == g) for g in range(len(unique_groups))]
    rng = np.random.default_rng(rng_seed)
    # Keep all observations in a source group together. Use the same sampled
    # groups for every architecture and seed, enabling paired differences.
    point = {(seed, model): float(roc_auc_score(y, preds[(seed, model)]))
             for seed in seeds for model in MODELS}
    boot = defaultdict(list)
    skipped = 0
    for _ in range(bootstrap):
        group_sample = rng.integers(0, len(by_group), size=len(by_group))
        indices = np.concatenate([by_group[g] for g in group_sample])
        labels = y[indices]
        if len(np.unique(labels)) < 2:
            skipped += 1
            continue
        for seed in seeds:
            for model in MODELS:
                boot[(seed, model)].append(float(roc_auc_score(
                    labels, preds[(seed, model)][indices])))
    valid = bootstrap - skipped
    if valid < 100:
        raise ValueError(f"Too few valid cluster bootstrap draws: {valid}")
    report = {
        "method": "paired_source_group_cluster_bootstrap_on_validation",
        "data": {
            "split": "val",
            "videos": len(y),
            "real": int(np.sum(y == 0)),
            "fake": int(np.sum(y == 1)),
            "source_groups": int(len(unique_groups)),
            "clips_per_video": clips,
            "manifest_sha256": sample["manifest_sha256"],
            "seeds": list(seeds),
        },
        "bootstrap": {
            "draws_requested": bootstrap, "draws_valid": valid,
            "draws_skipped_single_class": skipped,
            "seed": rng_seed,
            "unit": "source_group",
        },
        "models": {},
        "comparisons": {},
        "limitations": [
            "Exploratory, post-selection validation: best checkpoints were selected on this same split, so uncertainty intervals do not correct selection optimism.",
            "Bootstrap resamples validation source groups, not training runs; only three observed seeds are used and confidence intervals are conditional on those checkpoints.",
            "Model-family comparisons and bootstrap intervals are descriptive, not multiplicity-adjusted confirmatory hypothesis tests.",
            "Only matched FF++ RAW YouTube/Deepfakes; no out-of-method, compression or dataset generalization claim.",
            "Test split was not read or evaluated.",
        ],
    }
    for model in MODELS:
        scores = [point[(s, model)] for s in seeds]
        mean_boot = np.mean(np.array([boot[(s, model)] for s in seeds]), axis=0)
        report["models"][model] = {
            "auc_by_seed": {str(s): point[(s, model)] for s in seeds},
            "mean_auc": float(np.mean(scores)),
            "seed_std_sample": float(np.std(scores, ddof=1)) if len(seeds) > 1 else None,
            "cluster_bootstrap_ci95_mean_auc_conditional": _ci(mean_boot),
        }
    for better, baseline in PAIRS:
        name = f"{better}_minus_{baseline}"
        diffs = {str(s): point[(s, better)] - point[(s, baseline)] for s in seeds}
        draws = np.mean(np.array([
            np.asarray(boot[(s, better)]) - np.asarray(boot[(s, baseline)])
            for s in seeds]), axis=0)
        report["comparisons"][name] = {
            "auc_delta_by_seed": diffs,
            "mean_delta": float(np.mean(list(diffs.values()))),
            "cluster_bootstrap_ci95_mean_delta_conditional": _ci(draws),
            "fraction_cluster_bootstrap_delta_gt_zero": float(np.mean(draws > 0)),
            "warning": "Descriptive bootstrap probability, NOT a p-value.",
        }
    return report


def main():
    p = argparse.ArgumentParser(description="Paired cluster bootstrap on v4 validation predictions")
    p.add_argument("--manifest", required=True)
    p.add_argument("--predictions-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--clips", type=int, default=2)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--rng-seed", type=int, default=2026)
    args = p.parse_args()
    result = analyze(args.manifest, args.predictions_root, seeds=args.seeds,
                     clips=args.clips, bootstrap=args.bootstrap,
                     rng_seed=args.rng_seed)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
