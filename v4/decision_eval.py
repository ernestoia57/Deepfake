"""v4.9: full validation metrics, grouped threshold cross-fitting and ensembles.

Uses saved v4.3 validation predictions only; no model retraining or test
access. Source-group held-out folds limit direct threshold overfitting,
but do NOT remove prior checkpoint selection on this validation split.
This is exploratory and not an independent performance estimate.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from .manifest import read_manifest
from .paired_stats import _read_predictions


MODES = ("rgb", "freq", "fusion")
GOALS = {
    "accuracy": 0.95,
    "precision_fake": 0.95,
    "recall_fake": 0.95,
    "f1_fake": 0.95,
    "specificity_real": 0.95,
    "roc_auc": 0.97,
}


def load_predictions(manifest, predictions_root, seeds=(42, 43, 44), clips=2):
    """Align saved predictions exactly to the original validation video IDs.

    Refuses IDs, labels, or clip counts inconsistent with the validation
    split. No test predictions are accepted.
    """
    source = Path(manifest)
    rows = [r for r in read_manifest(source) if r["split"] == "val"]
    if not rows:
        raise ValueError("Manifest contains no validation videos")
    records = {}
    for row in rows:
        vid = row["video_id"]
        if vid in records:
            raise ValueError(f"Repeated validation video_id: {vid}")
        if not row["source_group"]:
            raise ValueError(f"Empty source_group for {vid}")
        records[vid] = row
    ids = sorted(records)
    y = np.asarray([int(records[k]["label"]) for k in ids], dtype=np.int64)
    groups = np.asarray([records[k]["source_group"] for k in ids])
    if set(y) != {0, 1}:
        raise ValueError("Validation requires real and fake classes")
    scores = {}
    for mode in MODES:
        for seed in seeds:
            csv_path = Path(predictions_root) / f"{mode}_s{seed}" / "predictions.csv"
            data = _read_predictions(csv_path)
            if set(data) != set(ids):
                missing = sorted(set(ids) - set(data))
                extra = sorted(set(data) - set(ids))
                raise ValueError(
                    f"{csv_path}: expected exactly val videos; "
                    f"missing={missing[:3]} extra={extra[:3]}")
            for vid in ids:
                label, _, nclips = data[vid]
                if label != int(records[vid]["label"]):
                    raise ValueError(f"{csv_path}: label disagreement for {vid}")
                if nclips != clips:
                    raise ValueError(
                        f"{csv_path}: expected {clips} clips per video, got "
                        f"{nclips} for {vid}")
            scores[f"{mode}_s{seed}"] = np.asarray(
                [data[k][1] for k in ids], dtype=np.float64)
    return dict(ids=ids, labels=y, groups=groups, scores=scores,
                manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest())


def _confusion_metrics(y, binary):
    y = np.asarray(y, dtype=np.int64)
    binary = np.asarray(binary, dtype=np.int64)
    if len(y) != len(binary) or len(y) == 0:
        raise ValueError("Expected aligned nonempty labels and decisions")
    if not set(np.unique(y)).issubset({0, 1}) or not set(np.unique(binary)).issubset({0, 1}):
        raise ValueError("Expected binary labels and decisions")
    tp = int(np.sum((y == 1) & (binary == 1)))
    tn = int(np.sum((y == 0) & (binary == 0)))
    fp = int(np.sum((y == 0) & (binary == 1)))
    fn = int(np.sum((y == 1) & (binary == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision_fake": precision,
        "recall_fake": recall,
        "specificity_real": specificity,
        "fpr": fp / (tn + fp) if tn + fp else 0.0,
        "f1_fake": f1,
    }


def _select_threshold(y, probs, objective):
    """Select one deterministic threshold; break ties toward 0.5.

    The threshold must be selected on DIFFERENT source groups from the
    observations on which the corresponding decisions are scored.
    """
    y = np.asarray(y, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if set(y) != {0, 1}:
        raise ValueError("Threshold optimization needs both classes")
    if objective not in ("balanced_accuracy", "accuracy", "f1_fake"):
        raise ValueError("Unsupported threshold objective")
    values = np.unique(probs)
    # All-positive at 0; all-negative at >1; and transitions between
    # neighboring score levels. No threshold candidate uses labels.
    mids = (values[:-1] + values[1:]) / 2
    candidates = np.unique(np.r_[0.0, mids, np.nextafter(1.0, 2.0)])
    best = None
    for threshold in candidates:
        m = _confusion_metrics(y, probs >= threshold)
        # Exact deterministic preference: high objective, then closest to
        # default 0.5, then smaller threshold.
        key = (m[objective], -abs(float(threshold)-0.5), -float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold))
    return best[1]


def _ensembles(scores, seeds):
    result = dict(scores)
    mode_means = {}
    for mode in MODES:
        mode_means[mode] = np.mean(
            np.stack([scores[f"{mode}_s{s}"] for s in seeds]), axis=0)
        result[f"{mode}_mean"] = mode_means[mode]
    result["rgb_freq_mean"] = (mode_means["rgb"] + mode_means["freq"]) / 2
    result["freq_fusion_mean"] = (mode_means["freq"] + mode_means["fusion"]) / 2
    result["all_mean"] = np.mean(np.stack(list(mode_means.values())), axis=0)
    return result


def analyze(manifest, predictions_root, seeds=(42, 43, 44), clips=2,
            folds=5, objective="balanced_accuracy"):
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be nonempty and distinct")
    if clips < 1:
        raise ValueError("clips must be >= 1")
    if objective not in ("balanced_accuracy", "accuracy", "f1_fake"):
        raise ValueError("Unsupported threshold objective")
    source = load_predictions(manifest, predictions_root, seeds, clips)
    y, groups = source["labels"], source["groups"]
    unique_groups = np.unique(groups)
    if not (2 <= folds <= len(unique_groups)):
        raise ValueError("folds must be between 2 and the number of source groups")
    fold_indices = list(GroupKFold(n_splits=folds).split(
        np.zeros(len(y)), y, groups))
    # In unusual datasets, grouped folds might be single-class.
    for train_idx, held_idx in fold_indices:
        if set(y[train_idx]) != {0, 1} or set(y[held_idx]) != {0, 1}:
            raise ValueError("Group fold missing class; use fewer folds or revise groups")
        if set(groups[train_idx]) & set(groups[held_idx]):
            raise ValueError("Source group leakage in cross-fitting")
    result = {
        "method": "validation_grouped_threshold_crossfit_and_equal_weight_ensembles",
        "data": {
            "split": "val", "videos": len(y), "real": int(np.sum(y == 0)),
            "fake": int(np.sum(y == 1)), "source_groups": len(unique_groups),
            "seeds": list(seeds), "clips_per_video": clips,
            "manifest_sha256": source["manifest_sha256"],
        },
        "thresholding": {
            "objective": objective, "source_group_folds": folds,
            "decision_protocol": "For each fold, optimize threshold only on other validation source groups; score on held-out fold. The final threshold is fitted on all validation groups for possible future independent test use, not for the reported crossfit metrics.",
        },
        "goals": GOALS,
        "candidates": {},
        "limitations": [
            "Exploratory validation: checkpoints were already selected using this same validation split; cross-fitting thresholds does not correct that selection optimism.",
            "All candidate ensembles use fixed, equal weights rather than tuning weights on validation; no claim of cross-method generalization.",
            "Cross-fitted operating-point metrics combine held-out group decisions; their fold thresholds differ. AUC/AP/Brier are descriptive for the fixed probability scores.",
            "The full-validation fitted threshold is a deployment candidate, not an unbiased metric estimate. Do not score the same val split with this threshold as independent confirmation.",
            "No C23 or extra videos, no model retraining, and no test split used.",
        ],
    }
    for name, probs in _ensembles(source["scores"], seeds).items():
        hard = np.zeros(len(y), dtype=np.int64)
        fold_thresholds = []
        for train_idx, held_idx in fold_indices:
            t = _select_threshold(y[train_idx], probs[train_idx], objective)
            hard[held_idx] = (probs[held_idx] >= t).astype(np.int64)
            fold_thresholds.append(t)
        metrics = _confusion_metrics(y, hard)
        roc = float(roc_auc_score(y, probs))
        ap = float(average_precision_score(y, probs))
        metrics["roc_auc"] = roc
        metrics["average_precision_fake"] = ap
        metrics["brier_score"] = float(brier_score_loss(y, probs))
        final_threshold = _select_threshold(y, probs, objective)
        result["candidates"][name] = {
            "crossfit_metrics": metrics,
            "thresholds_by_fold": fold_thresholds,
            "threshold_for_future_independent_evaluation": final_threshold,
            "goals_met_on_exploratory_crossfit": {
                k: bool(metrics[k] >= v) for k, v in GOALS.items()
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="v4.9: grouped threshold cross-fitting and ensembles from FF++ validation predictions")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--clips", type=int, default=2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--objective", choices=["balanced_accuracy", "accuracy", "f1_fake"],
                        default="balanced_accuracy")
    args = parser.parse_args()
    report = analyze(args.manifest, args.predictions_root,
                     seeds=tuple(args.seeds), clips=args.clips,
                     folds=args.folds, objective=args.objective)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({
        "data": report["data"],
        "thresholding": report["thresholding"],
        "candidate_summary": {
            key: {
                "accuracy": value["crossfit_metrics"]["accuracy"],
                "precision_fake": value["crossfit_metrics"]["precision_fake"],
                "recall_fake": value["crossfit_metrics"]["recall_fake"],
                "f1_fake": value["crossfit_metrics"]["f1_fake"],
                "roc_auc": value["crossfit_metrics"]["roc_auc"],
                "final_val_threshold": value["threshold_for_future_independent_evaluation"],
            } for key, value in report["candidates"].items()
        },
        "out": str(dest),
    }, indent=2))


if __name__ == "__main__":
    main()
