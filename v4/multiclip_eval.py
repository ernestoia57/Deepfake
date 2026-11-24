"""v4.9 extension: paired 2-vs-4-clip validation experiment, no test inference.

Finds only original v4.3 compatible checkpoints (by saved config),
evaluates them with *one common* new fixed-window index, and compares
nine models and fixed-weight ensembles using source-group bootstrap.

Trusted local PyTorch checkpoints only. No changes to training data,
parameters, model weights, splits or test predictions.
"""
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .decision_eval import MODES, _ensembles, analyze, load_predictions
from .manifest import read_manifest
from .paired_stats import _ci, _read_predictions
from .window_index import load_index, sha256_file


def _expected_keys(seeds):
    return [f"{mode}_s{seed}" for mode in MODES for seed in seeds]


def _match_config(config, manifest, baseline_index):
    """Require exact original study settings, not only a convenient filename."""
    if not isinstance(config, dict):
        return None
    mode = config.get("mode")
    seed = config.get("seed")
    if mode not in MODES or not isinstance(seed, int):
        return None
    if config.get("frames") != 16 or config.get("size") != 64:
        return None
    if config.get("frequency", "fft") != "fft":
        return None
    if config.get("fft_control", "original") != "original":
        return None
    if config.get("filter_mode", "median") != "median":
        return None
    if config.get("face") != "haar":
        return None
    if config.get("temporal", "gru") != "gru":
        return None
    if not config.get("manifest") or not config.get("window_index"):
        return None
    if Path(config["manifest"]).resolve() != Path(manifest).resolve():
        return None
    if Path(config["window_index"]).resolve() != Path(baseline_index).resolve():
        return None
    return f"{mode}_s{seed}"


def discover_checkpoints(runs_root, manifest, baseline_index, seeds,
                         expected_epochs=15):
    """Find exactly one FULL-TRAINING best.pt per v4.3 model and seed.

    Require checkpoint config AND sibling config.json train.epochs.
    The training budget is not checkpoint['epoch'], which stores the
    validation-selected best epoch and may be < expected_epochs.
    Smoke tests (epochs=1) are deliberately excluded without deletion.
    Reject duplicates among fully qualified candidates.
    """
    root = Path(runs_root)
    if expected_epochs < 1:
        raise ValueError("expected_epochs must be positive")
    if not root.is_dir():
        raise FileNotFoundError(f"Missing runs root: {runs_root}")
    matched = defaultdict(list)
    excluded_short_runs = defaultdict(list)
    for checkpoint in sorted(root.rglob("best.pt")):
        # torch.load(..., weights_only=False) is reserved for checkpoints
        # already created on this user's trusted local machine.
        try:
            data = torch.load(checkpoint, map_location="cpu", weights_only=False)
            config = data.get("config") if isinstance(data, dict) else None
            key = _match_config(config, manifest, baseline_index)
        except (ValueError, KeyError, RuntimeError, EOFError, OSError,
                AttributeError, TypeError) as error:
            raise ValueError(
                f"Could not inspect local checkpoint {checkpoint}: {error}") from error
        if key and key in _expected_keys(seeds):
            # train.epochs is in the run's config.json, not the checkpoint
            # model config. Read ONLY after the checkpoint matches protocol.
            run_config_path = checkpoint.parent / "config.json"
            if not run_config_path.is_file():
                raise ValueError(
                    f"Missing training metadata for {checkpoint}: "
                    f"{run_config_path}. Refusing to guess epochs.")
            try:
                run_config = json.loads(run_config_path.read_text(encoding="utf8"))
                epochs = run_config["train"]["epochs"]
            except (OSError, UnicodeError, ValueError, KeyError, TypeError) as error:
                raise ValueError(
                    f"Invalid training metadata at {run_config_path}: {error}"
                ) from error
            if type(epochs) is not int or epochs < 1:
                raise ValueError(f"{run_config_path}: invalid train.epochs={epochs!r}")
            if run_config.get("config") != config:
                raise ValueError(
                    f"{run_config_path}: model checkpoint config differs from "
                    "training metadata; refusing ambiguous provenance")
            if epochs == expected_epochs:
                matched[key].append(checkpoint)
            else:
                excluded_short_runs[key].append({
                    "checkpoint": str(checkpoint), "epochs": epochs})
    required = _expected_keys(seeds)
    missing = [name for name in required if not matched.get(name)]
    ambiguous = {name: [str(p) for p in matched[name]]
                 for name in required if len(matched[name]) > 1}
    if missing or ambiguous:
        raise ValueError(
            f"Expected exactly one original v4.3 checkpoint trained for "
            f"{expected_epochs} epochs per model/seed. "
            f"Missing={missing}; duplicate candidates={ambiguous}; "
            f"excluded_other_epochs={dict(excluded_short_runs)}. "
            "Check checkpoint config.json training metadata; "
            "do not delete smoke runs.")
    return {name: matched[name][0] for name in required}


def _valid_prediction_file(path, expected_rows, clips):
    observed = _read_predictions(path)
    if set(observed) != set(expected_rows):
        raise ValueError(f"Existing {path} has unexpected validation video IDs")
    for vid, (label, _, number) in observed.items():
        if label != expected_rows[vid] or number != clips:
            raise ValueError(
                f"Existing {path} label/clips mismatch for {vid}; "
                "do not reuse this output.")
    return True


def run_evaluations(manifest, index, checkpoints, out_dir, batch=4,
                    workers=0, resume=False):
    out_dir = Path(out_dir)
    rows = {r["video_id"]: int(r["label"]) for r in read_manifest(manifest)
            if r["split"] == "val"}
    if not rows:
        raise ValueError("No validation rows")
    manifest_digest = sha256_file(manifest)
    index_digest = sha256_file(index)
    for clips in (2, 4):
        for key, ckpt in checkpoints.items():
            dest = out_dir / f"c{clips}" / key
            csv_path = dest / "predictions.csv"
            metadata_path = dest / "experiment_meta.json"
            expected_meta = {
                "manifest_sha256": manifest_digest,
                "index_sha256": index_digest,
                "checkpoint_sha256": sha256_file(ckpt),
                "checkpoint_path": str(Path(ckpt).resolve()),
                "clips": clips,
                "split": "val",
                "candidate": key,
            }
            if dest.exists():
                if not resume or not csv_path.is_file() or not metadata_path.is_file():
                    raise FileExistsError(
                        f"Output exists or incomplete, refusing to overwrite: {dest}. "
                        "Use --resume only for complete, fingerprinted outputs.")
                saved_meta = json.loads(metadata_path.read_text(encoding="utf8"))
                if saved_meta != expected_meta:
                    raise ValueError(
                        f"{dest}: checkpoint, index or manifest changed; "
                        "refusing to reuse stale predictions")
                _valid_prediction_file(csv_path, rows, clips)
                print(f"[skip] verified existing {csv_path}", flush=True)
                continue
            command = [
                sys.executable, "-m", "v4.pipeline", "evaluate",
                "--checkpoint", str(ckpt),
                "--window-index", str(index),
                "--split", "val",
                "--clips", str(clips),
                "--batch", str(batch),
                "--workers", str(workers),
                "--bootstrap", "0",
                "--out", str(dest),
            ]
            print(f"[evaluate c{clips}] {key} <- {ckpt}", flush=True)
            subprocess.run(command, check=True)
            _valid_prediction_file(csv_path, rows, clips)
            metadata_path.write_text(
                json.dumps(expected_meta, ensure_ascii=False, indent=2),
                encoding="utf8")
    return out_dir


def _bootstrap_comparisons(source2, source4, seeds, draws=2000, rng_seed=2026):
    if draws < 100:
        raise ValueError("At least 100 source-group bootstrap draws required")
    if source2["ids"] != source4["ids"] or not np.array_equal(
            source2["labels"], source4["labels"]) or not np.array_equal(
            source2["groups"], source4["groups"]):
        raise ValueError("c2 and c4 validation IDs, labels or groups do not match")
    y = source2["labels"]
    groups = source2["groups"]
    unique, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(unique))]
    p2 = _ensembles(source2["scores"], seeds)
    p4 = _ensembles(source4["scores"], seeds)
    names = ("all_mean", "fusion_mean", "freq_mean",
             "rgb_freq_mean", "fusion_s44", "freq_s44")
    names = [k for k in names if k in p2 and k in p4]
    rng = np.random.default_rng(rng_seed)
    delta_draws = defaultdict(list)
    valid = 0
    for _ in range(draws):
        picked = rng.integers(0, len(members), size=len(members))
        inds = np.concatenate([members[g] for g in picked])
        labels = y[inds]
        if len(np.unique(labels)) != 2:
            continue
        valid += 1
        for name in names:
            delta_draws[name].append(float(
                roc_auc_score(labels, p4[name][inds]) -
                roc_auc_score(labels, p2[name][inds])))
    if valid < 100:
        raise ValueError("Too few valid grouped bootstrap samples")
    result = {}
    for name in names:
        auc2 = float(roc_auc_score(y, p2[name]))
        auc4 = float(roc_auc_score(y, p4[name]))
        result[name] = {
            "auc_2clips": auc2,
            "auc_4clips": auc4,
            "delta_auc_4_minus_2": auc4 - auc2,
            "cluster_bootstrap_ci95_delta_conditional": _ci(delta_draws[name]),
            "fraction_bootstrap_delta_positive_not_p_value": float(
                np.mean(np.asarray(delta_draws[name]) > 0)),
        }
    return result, valid


def compare(manifest, out_dir, index, seeds=(42, 43, 44),
            folds=5, bootstrap=2000, rng_seed=2026):
    root = Path(out_dir)
    sub2, sub4 = root / "c2", root / "c4"
    report2 = analyze(manifest, sub2, seeds=seeds, clips=2, folds=folds)
    report4 = analyze(manifest, sub4, seeds=seeds, clips=4, folds=folds)
    source2 = load_predictions(manifest, sub2, seeds, clips=2)
    source4 = load_predictions(manifest, sub4, seeds, clips=4)
    pairs, valid = _bootstrap_comparisons(
        source2, source4, seeds, draws=bootstrap, rng_seed=rng_seed)
    decision_names = ("all_mean", "fusion_mean", "freq_mean", "rgb_freq_mean")
    decisions = {}
    for name in decision_names:
        if name not in report2["candidates"] or name not in report4["candidates"]:
            raise ValueError(f"Missing ensemble {name}")
        m2 = report2["candidates"][name]["crossfit_metrics"]
        m4 = report4["candidates"][name]["crossfit_metrics"]
        decisions[name] = {
            "metrics_2clips": m2,
            "metrics_4clips": m4,
            "delta_crossfit_accuracy_4_minus_2": m4["accuracy"]-m2["accuracy"],
            "delta_crossfit_f1_4_minus_2": m4["f1_fake"]-m2["f1_fake"],
            "delta_crossfit_recall_4_minus_2": m4["recall_fake"]-m2["recall_fake"],
        }
    result = {
        "protocol": "matched-index_2_vs_4_clip_validation_only",
        "data": {
            "split": "val", "videos": len(source2["ids"]),
            "source_groups": int(len(np.unique(source2["groups"]))),
            "seeds": list(seeds), "index_sha256": sha256_file(index),
            "manifest_sha256": source2["manifest_sha256"],
        },
        "bootstrap": {
            "unit": "source_group", "requested": bootstrap,
            "valid": valid, "seed": rng_seed,
            "unadjusted_exploratory": True,
        },
        "paired_auc": pairs,
        "crossfit_operating_points": decisions,
        "limitations": [
            "Exploratory post-checkpoint-selection validation only; no test data were evaluated.",
            "AUC on the same new window index is the primary isolated comparison of 2 versus 4 clips.",
            "Crossfit accuracy/F1 comparisons refit operating thresholds in each clip condition; those differences include threshold adaptation.",
            "A different window index from v4.3 may alter the 2-clip baseline; compare c2 and c4 ONLY from this experiment.",
            "Paired bootstrap is conditional on validation-selected checkpoints and does not correct multiple exploratory comparisons.",
            "A complete Haar window index does not certify facial identity, correct frames or crop quality.",
        ],
    }
    dest = root / "paired_multiclip_summary.json"
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    return result, dest


def main():
    parser = argparse.ArgumentParser(
        description="v4.9 paired 2-vs-4 clip experiment, validation only")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", required=True,
                        help="New complete 4-eval-window index")
    parser.add_argument("--baseline-index", required=True,
                        help="Original 2-clip v4.3 window index to select checkpoints")
    parser.add_argument("--search-root", default="runs",
                        help="Search trusted local best.pt checkpoints")
    parser.add_argument("--expected-epochs", type=int, default=15,
                        help="Require exactly this original training budget "
                             "from each checkpoint's sibling config.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--rng-seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true",
                        help="Reuse only already-complete verified CSVs")
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)) or not args.seeds:
        parser.error("Seeds must be unique and nonempty")
    if args.batch < 1 or args.workers < 0:
        parser.error("Invalid batch/workers")
    if Path(args.index).resolve() == Path(args.baseline_index).resolve():
        parser.error("New 4-window index must be separate from original 2-window index")
    # Validates all rows, manifest hash, fixed clips/face/size, and completeness.
    load_index(args.index, args.manifest, frames=16, size=64, face="haar")
    checkpoints = discover_checkpoints(
        args.search_root, args.manifest, args.baseline_index, args.seeds,
        expected_epochs=args.expected_epochs)
    print(json.dumps(
        {"index": args.index, "checkpoints": {
            key: str(value) for key, value in checkpoints.items()}},
        indent=2), flush=True)
    run_evaluations(args.manifest, args.index, checkpoints, args.out,
                    batch=args.batch, workers=args.workers,
                    resume=args.resume)
    result, dest = compare(args.manifest, args.out, args.index,
                           seeds=tuple(args.seeds), folds=args.folds,
                           bootstrap=args.bootstrap, rng_seed=args.rng_seed)
    print(json.dumps({
        "out": str(dest), "data": result["data"],
        "paired_auc": result["paired_auc"],
        "crossfit_operating_points": {
            key: {
                "accuracy_2clips": val["metrics_2clips"]["accuracy"],
                "accuracy_4clips": val["metrics_4clips"]["accuracy"],
                "recall_2clips": val["metrics_2clips"]["recall_fake"],
                "recall_4clips": val["metrics_4clips"]["recall_fake"],
            } for key, val in result["crossfit_operating_points"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
