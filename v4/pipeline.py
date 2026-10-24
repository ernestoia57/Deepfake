"""v4 command line: manifest, audit, train, evaluate, teardown.

Examples:
 python -m v4.pipeline manifest --root /data/FF++ --out manifests/ffpp_v4.csv
 python -m v4.pipeline audit --manifest manifests/ffpp_v4.csv
 python -m v4.pipeline train --manifest manifests/ffpp_v4.csv --mode rgb --out runs/v4_rgb
 python -m v4.pipeline evaluate --checkpoint runs/v4_rgb/best.pt --split test
 python -m v4.pipeline teardown --mode fusion --out teardown.json
"""
import argparse
import csv
import json
import math
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import VideoClipDataset, SampleError
from .manifest import audit_manifest, create_manifest
from .models import Detector


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def settings(args):
    return {name: getattr(args, name) for name in (
        "manifest", "frames", "size", "mode", "temporal", "frequency", "filter_mode",
        "face", "dim", "seed", "cwt_bands", "cwt_fmin", "cwt_fmax",
    )}


def build_model(config):
    return Detector(mode=config["mode"], frames=config["frames"],
                    dim=config["dim"], temporal=config["temporal"],
                    frequency=config["frequency"], cwt_bands=config["cwt_bands"])


def loader(config, split, batch_size, workers, clips_per_video):
    ds = VideoClipDataset(
        manifest=config["manifest"], split=split, frames=config["frames"],
        size=config["size"], frequency=config["frequency"],
        filter_mode=config["filter_mode"], face=config["face"],
        clips_per_video=clips_per_video, seed=config["seed"],
        cwt_bands=config["cwt_bands"], cwt_fmin=config["cwt_fmin"],
        cwt_fmax=config["cwt_fmax"])
    dl = DataLoader(ds, batch_size=batch_size,
                    shuffle=(split == "train"), drop_last=False,
                    num_workers=workers,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=False)
    return ds, dl


def metrics(labels, probs):
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score, roc_curve
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    if len(np.unique(y)) < 2:
        raise ValueError("Metrics require both REAL and FAKE classes")
    pred = (p >= 0.5).astype(int)
    auc = float(roc_auc_score(y, p))
    balanced = float(balanced_accuracy_score(y, pred))
    fpr, tpr, _ = roc_curve(y, p)
    fnr = 1.0 - tpr
    eer = float((fpr[np.argmin(np.abs(fnr - fpr))] +
                  fnr[np.argmin(np.abs(fnr - fpr))]) / 2)
    return {"video_auc": auc, "balanced_accuracy_at_0_5": balanced,
            "eer_approx": eer, "videos": int(len(y))}


def ci_auc(labels, probs, repeats=500, seed=99):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    aucs = []
    for _ in range(repeats):
        idx = rng.integers(0, len(labels), size=len(labels))
        if len(np.unique(labels[idx])) == 2:
            aucs.append(roc_auc_score(labels[idx], probs[idx]))
    return (list(map(float, np.quantile(aucs, [0.025, 0.975])))
            if len(aucs) >= 20 else [None, None])


@torch.no_grad()
def evaluate_epoch(model, data_loader, device):
    model.eval()
    video_predictions = defaultdict(list)
    video_labels = {}
    for batch in data_loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        freq = batch["freq"].to(device, non_blocking=True)
        logits = model(rgb, freq)
        probs = torch.sigmoid(logits).cpu().numpy()
        labs = batch["label"].cpu().numpy()
        for vid, label, prob in zip(batch["video_id"], labs, probs):
            if vid in video_labels and video_labels[vid] != int(label):
                raise ValueError("Conflicting labels for a video")
            video_labels[vid] = int(label)
            video_predictions[vid].append(float(prob))
    ids = sorted(video_labels)
    labels = [video_labels[x] for x in ids]
    probabilities = [float(np.mean(video_predictions[x])) for x in ids]
    return metrics(labels, probabilities), [
        dict(video_id=vid, label=video_labels[vid],
             probability_fake=float(np.mean(video_predictions[vid])),
             clips=len(video_predictions[vid]))
        for vid in ids
    ]


def revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def train(args):
    seed_everything(args.seed)
    info = audit_manifest(args.manifest)
    config = settings(args)
    run_dir = Path(args.out); run_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "rgb" and args.frequency == "cwt":
        # Frequency still computed by data loader. Prefer fft to save work.
        raise ValueError("RGB-only runs must use --frequency fft")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_train, dl_train = loader(config, "train", args.batch, args.workers, 1)
    _, dl_val = loader(config, "val", args.batch, args.workers, args.val_clips)
    model = build_model(config).to(device)
    labels = [int(x["label"]) for x in ds_train.rows]
    n_positive = sum(labels)
    pos_weight = (len(labels)-n_positive) / n_positive
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda" and args.amp))
    metadata = dict(config=config, train=dict(epochs=args.epochs, batch=args.batch,
        lr=args.lr, weight_decay=args.weight_decay, pos_weight=pos_weight,
        workers=args.workers, val_clips=args.val_clips), audit=info, git_sha=revision())
    (run_dir/"config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    best_auc = float("-inf")
    history = []
    for epoch in range(1, args.epochs+1):
        ds_train.set_epoch(epoch)
        model.train()
        total_loss, seen = 0., 0
        for batch in dl_train:
            rgb = batch["rgb"].to(device, non_blocking=True)
            freq = batch["freq"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,
                                enabled=(device.type=="cuda" and args.amp)):
                logits = model(rgb, freq)
                loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * y.numel()
            seen += y.numel()
        val_metrics, _ = evaluate_epoch(model, dl_val, device)
        result = dict(epoch=epoch, train_loss=total_loss/max(1,seen), **val_metrics)
        history.append(result)
        (run_dir/"history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
        if val_metrics["video_auc"] > best_auc:
            best_auc = val_metrics["video_auc"]
            torch.save(dict(state_dict=model.state_dict(), config=config,
                            epoch=epoch, val_metrics=val_metrics,
                            git_sha=revision()), run_dir/"best.pt")
    print(f"Saved best checkpoint: {run_dir/'best.pt'}; test set untouched.")


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    if args.manifest is not None:
        config["manifest"] = args.manifest
    audit_manifest(config["manifest"])
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    _, dl = loader(config, args.split, args.batch, args.workers, args.clips)
    result, records = evaluate_epoch(model, dl, device)
    result["auc_95ci_bootstrap_video"] = ci_auc(
        [r["label"] for r in records], [r["probability_fake"] for r in records],
        repeats=args.bootstrap)
    result["split"] = args.split
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    with (output/"predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0])
        writer.writeheader(); writer.writerows(records)
    (output/"metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def audit(args):
    result = audit_manifest(args.manifest)
    if args.inspect_clips:
        # Probe only known indexed samples; failures are counted, not replaced.
        failures = []
        config = dict(manifest=args.manifest, frames=args.frames, size=args.size,
            frequency="fft", filter_mode="median", face=args.face, seed=42,
            cwt_bands=8, cwt_fmin=6., cwt_fmax=10.)
        for split in ("train", "val", "test"):
            ds, _ = loader(config, split, 1, 0, 1)
            for i in range(min(args.inspect_clips, len(ds))):
                try:
                    ds[i]
                except (SampleError, ValueError, OSError, RuntimeError) as exc:
                    failures.append(dict(split=split, video_id=ds.rows[i]["video_id"],
                                         label=ds.rows[i]["label"], error=str(exc)))
        result["sample_probe"] = dict(clips_per_split=args.inspect_clips,
                                       failures=failures)
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


def teardown(args):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = Detector(mode=args.mode, frames=args.frames, dim=args.dim,
        temporal=args.temporal, frequency=args.frequency,
        cwt_bands=args.cwt_bands).to(device).eval()
    rgb = torch.zeros(1,args.frames,3,args.size,args.size, device=device)
    channels = args.frames//2 if args.frequency == "fft" else args.cwt_bands
    freq = torch.zeros(1,channels,args.size,args.size,device=device)
    ops = {}
    hooks = []
    def record(name, layer):
        def fn(_module, inp, out):
            if isinstance(layer, nn.Conv2d):
                b, oc, oh, ow = out.shape
                macs = b*oc*oh*ow*(layer.in_channels//layer.groups) * (
                    layer.kernel_size[0]*layer.kernel_size[1])
            elif isinstance(layer, nn.Linear):
                macs = out.numel()*layer.in_features
            else:
                b,t,d = inp[0].shape
                h = layer.hidden_size
                macs = b*t*3*(d*h + h*h)  # GRU gate matmuls, approximate
            if name not in ops:
                ops[name] = dict(layer=type(layer).__name__,
                    output_shape=list(out[0].shape) if isinstance(out, tuple)
                    else list(out.shape), estimated_macs=int(macs),
                    parameters=sum(p.numel() for p in layer.parameters(recurse=False)))
            else:
                ops[name]["estimated_macs"] += int(macs)
        return fn
    for name, layer in m.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.GRU)):
            hooks.append(layer.register_forward_hook(record(name,layer)))
    with torch.no_grad():
        m(rgb,freq)
    for h in hooks: h.remove()
    for _ in range(2):
        with torch.no_grad(): m(rgb,freq)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    samples = []
    for _ in range(args.iterations):
        if device.type == "cuda": torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad(): m(rgb,freq)
        if device.type == "cuda": torch.cuda.synchronize()
        samples.append((time.perf_counter()-start)*1000)
    total = sum(p.numel() for p in m.parameters())
    details = dict(model=args.mode, device=str(device), frames=args.frames,
        image_size=args.size, trainable_parameters=total,
        estimated_macs_per_clip=sum(v["estimated_macs"] for v in ops.values()),
        estimated_flops_per_clip=None,
        latency_ms_median=float(np.median(samples)),
        peak_cuda_memory_mb=(torch.cuda.max_memory_allocated()/2**20
                             if device.type == "cuda" else None),
        limitations="MAC count approximates Conv/Linear/GRU matmuls only; excludes activations, preprocessing, FFT and face detection. FLOPs are not asserted.",
        blocks=ops)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in details.items() if k!="blocks"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Research prototype v4")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("manifest")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p = sub.add_parser("audit")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out")
    p.add_argument("--inspect-clips", type=int, default=0)
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--face", choices=["haar","center"], default="haar")
    p = sub.add_parser("train")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["rgb","freq","fusion"], required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--val-clips", type=int, default=3)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--temporal", choices=["gru","mean"], default="gru")
    p.add_argument("--frequency", choices=["fft","cwt"], default="fft")
    p.add_argument("--filter-mode", choices=["median","mean","none"], default="median")
    p.add_argument("--face", choices=["haar","center"], default="haar")
    p.add_argument("--cwt-bands", type=int, default=8)
    p.add_argument("--cwt-fmin", type=float, default=6.)
    p.add_argument("--cwt-fmax", type=float, default=10.)
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest")
    p.add_argument("--split", choices=["val","test"], default="test")
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--clips", type=int, default=3)
    p.add_argument("--bootstrap", type=int, default=500)
    p = sub.add_parser("teardown")
    p.add_argument("--mode", choices=["rgb","freq","fusion"], default="fusion")
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--temporal", choices=["gru","mean"], default="gru")
    p.add_argument("--frequency", choices=["fft","cwt"], default="fft")
    p.add_argument("--cwt-bands", type=int, default=8)
    p.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.command == "manifest":
        print(json.dumps(create_manifest(args.root,args.out,seed=args.seed,
            val_fraction=args.val_fraction,test_fraction=args.test_fraction), indent=2))
    elif args.command == "audit": audit(args)
    elif args.command == "train": train(args)
    elif args.command == "evaluate": evaluate(args)
    elif args.command == "teardown": teardown(args)


if __name__ == "__main__":
    main()
