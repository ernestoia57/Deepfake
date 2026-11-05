"""Exploratory paired AUC comparisons for v4.5 temporal FFT ablations.

This does not read test data. It shares bootstrap source-group draws across all
variants and seeds, retaining the same validation population. The resulting
confidence intervals are post-checkpoint-selection descriptive intervals, not
confirmatory inference or generalization evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .manifest import read_manifest
from .paired_stats import _ci, _read_predictions


VARIANTS = ("original", "permute", "static_middle")
PAIRS = (
    ("original", "permute"),
    ("original", "static_middle"),
    ("permute", "static_middle"),
)


def load_temporal_validation(manifest_path, reference_root, ablations_root,
                             seeds=(42, 43, 44), clips=2):
    manifest = Path(manifest_path)
    rows = [r for r in read_manifest(manifest) if r["split"] == "val"]
    if not rows:
        raise ValueError("No val videos")
    ids_to_row = {}
    for row in rows:
        vid = row["video_id"]
        if vid in ids_to_row:
            raise ValueError(f"Duplicate validation video_id: {vid}")
        ids_to_row[vid] = row
    ids = sorted(ids_to_row)
    y = np.asarray([int(ids_to_row[vid]["label"]) for vid in ids])
    groups = np.asarray([ids_to_row[vid]["source_group"] for vid in ids])
    if len(np.unique(y)) != 2:
        raise ValueError("Validation must contain both labels")
    if np.any(groups == ""):
        raise ValueError("Empty source_group")
    predictions = {}
    for seed in seeds:
        for variant in VARIANTS:
            directory = (Path(reference_root) / f"freq_s{seed}"
                         if variant == "original"
                         else Path(ablations_root) / f"{variant}_s{seed}")
            csv_path = directory / "predictions.csv"
            data = _read_predictions(csv_path)
            if set(data) != set(ids):
                missing = sorted(set(ids) - set(data))
                extra = sorted(set(data) - set(ids))
                raise ValueError(f"{csv_path}: expected exactly validation videos; "
                                 f"missing={missing[:3]}, extra={extra[:3]}")
            for vid in ids:
                label, _, nclips = data[vid]
                if label != int(ids_to_row[vid]["label"]):
                    raise ValueError(f"{csv_path}: label disagreement for {vid}")
                if nclips != clips:
                    raise ValueError(f"{csv_path}: expected {clips} clips, "
                                     f"got {nclips} for {vid}")
            predictions[(seed, variant)] = np.asarray(
                [data[vid][1] for vid in ids], dtype=np.float64)
    return dict(labels=y, groups=groups, preds=predictions,
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())


def analyze_temporal(manifest_path, reference_root, ablations_root,
                     seeds=(42, 43, 44), clips=2,
                     bootstrap=2000, rng_seed=2026):
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be nonempty and distinct")
    if bootstrap < 100:
        raise ValueError("At least 100 resamples required")
    if clips < 1:
        raise ValueError("Clips must be positive")
    sample = load_temporal_validation(
        manifest_path, reference_root, ablations_root, seeds=seeds, clips=clips)
    y, groups, preds = sample["labels"], sample["groups"], sample["preds"]
    group_ids, group_index = np.unique(groups, return_inverse=True)
    units = [np.flatnonzero(group_index == i) for i in range(len(group_ids))]
    point = {
        (seed, variant): float(roc_auc_score(y, preds[(seed, variant)]))
        for seed in seeds for variant in VARIANTS
    }
    curves = {(seed, variant): [] for seed in seeds for variant in VARIANTS}
    skipped = 0
    rng = np.random.default_rng(rng_seed)
    for _ in range(bootstrap):
        sampled = rng.integers(0, len(units), len(units))
        indices = np.concatenate([units[i] for i in sampled])
        sampled_y = y[indices]
        if len(np.unique(sampled_y)) < 2:
            skipped += 1
            continue
        for seed in seeds:
            for variant in VARIANTS:
                curves[(seed, variant)].append(
                    float(roc_auc_score(sampled_y, preds[(seed, variant)][indices])))
    valid = bootstrap - skipped
    if valid < 100:
        raise ValueError(f"Too few valid bootstrap draws: {valid}")
    result = {
        "method": "v4_5_temporal_paired_source_group_cluster_bootstrap_on_validation",
        "data": {
            "split": "val",
            "videos": len(y),
            "real": int((y == 0).sum()),
            "fake": int((y == 1).sum()),
            "source_groups": len(group_ids),
            "clips_per_video": clips,
            "seeds": list(seeds),
            "manifest_sha256": sample["manifest_sha256"],
        },
        "bootstrap": {
            "draws_requested": bootstrap, "draws_valid": valid,
            "draws_skipped_single_class": skipped,
            "seed": rng_seed, "unit": "source_group",
        },
        "variants": {},
        "comparisons": {},
        "limitations": [
            "Exploratory, post-selection validation: all best.pt checkpoints were selected on val; confidence intervals do not correct this selection.",
            "Bootstrap varies validation source groups, conditional on these trained checkpoints and the three observed seeds; it does not include retraining variability.",
            "No multiplicity correction or confirmatory hypothesis testing; fractions of positive bootstrap differences are NOT p-values.",
            "The time permutation changes the temporal spectrum and may create artificial discontinuities, although both training and evaluation use the same deterministic control.",
            "The static_middle control removes frames, so an AUC decrease cannot be attributed solely to temporal order.",
            "The original fixed 15-epoch budget is used for all variants; some runs peaked at epoch 15, so convergence equivalence is not established.",
            "Only FF++ RAW YouTube + Deepfakes validation; no out-of-method or cross-dataset generalization test.",
            "Test split was not accessed.",
        ],
    }
    for variant in VARIANTS:
        aucs = [point[(s, variant)] for s in seeds]
        mean_draws = np.mean(np.asarray([curves[(s, variant)] for s in seeds]), axis=0)
        result["variants"][variant] = {
            "auc_by_seed": {str(s): point[(s, variant)] for s in seeds},
            "mean_auc": float(np.mean(aucs)),
            "seed_std_sample": float(np.std(aucs, ddof=1)) if len(seeds) > 1 else None,
            "cluster_bootstrap_ci95_mean_auc_conditional": _ci(mean_draws),
        }
    for better, worse in PAIRS:
        delta = {str(s): point[(s, better)] - point[(s, worse)] for s in seeds}
        paired_draws = np.mean(np.asarray([
            np.asarray(curves[(s, better)]) - np.asarray(curves[(s, worse)])
            for s in seeds
        ]), axis=0)
        result["comparisons"][f"{better}_minus_{worse}"] = {
            "auc_delta_by_seed": delta,
            "mean_delta": float(np.mean(list(delta.values()))),
            "cluster_bootstrap_ci95_mean_delta_conditional": _ci(paired_draws),
            "fraction_cluster_bootstrap_delta_gt_zero": float(np.mean(paired_draws > 0)),
            "warning": "Descriptive bootstrap fraction, NOT a p-value.",
        }
    return result


def main():
    p = argparse.ArgumentParser(
        description="Exploratory paired temporal FFT ablation AUC on validation")
    p.add_argument("--manifest", required=True)
    p.add_argument("--reference-root", required=True,
                   help="v4.3 prediction directories freq_s42/freq_s43/freq_s44")
    p.add_argument("--ablations-root", required=True,
                   help="v4.5 prediction directories permute_s42 and static_middle_s42 etc")
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--clips", type=int, default=2)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--rng-seed", type=int, default=2026)
    args = p.parse_args()
    results = analyze_temporal(
        args.manifest, args.reference_root, args.ablations_root,
        seeds=tuple(args.seeds), clips=args.clips,
        bootstrap=args.bootstrap, rng_seed=args.rng_seed)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
