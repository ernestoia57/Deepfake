# quality_utils.py
# Scoring de calidad con ELBO de tu ε-UNet y cache a disco (por frame y por video)

import os, json, math
from pathlib import Path
from typing import Dict, Optional, Sequence

import cv2
import numpy as np
import torch

import diff_utils as DU
from train_utils import default_preprocess, sample_uniform_frames, face_crop_fn

def load_diffusion_model(ckpt_path: str, channels: int = 1, t_dim: int = 128, base: int = 32, device=None):
    """
    Carga el ε-UNet con 'channels' que correspondan a tu checkpoint.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DU.TinyUNet(in_ch=channels, base=base, t_dim=t_dim).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, device

@torch.no_grad()
def score_video_elbo(
    video_path: str,
    model: torch.nn.Module,
    sched,
    frames_per_video: int = 8,
    image_size: int = 64,
    channels: int = 1,
    use_face_crop: bool = True,
) -> Dict[int, float]:
    """
    Devuelve dict {frame_idx: elbo} para un video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    idxs = sample_uniform_frames(cap, want=frames_per_video)
    elbos: Dict[int, float] = {}
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        if use_face_crop:
            try:
                frame = face_crop_fn(frame)
            except Exception:
                pass
        x = default_preprocess(frame, out_size=image_size, to_gray=(channels==1), norm="tanh")
        x = x.unsqueeze(0).to(next(model.parameters()).device)
        e = DU.estimate_elbo_mc(model, x, sched).mean().item()
        elbos[int(idx)] = float(e)
    cap.release()
    return elbos

def cache_quality_scores(
    videos: Sequence[str],
    ckpt_path: str,
    out_json: str,
    frames_per_video: int = 8,
    image_size: int = 64,
    channels: int = 1,
    use_face_crop: bool = True,
    T_steps: int = 300,
):
    """
    Procesa una lista de videos y guarda:
      - <out_json>: { video_path: {frame_idx: elbo, ...}, ... }
      - también un archivo hermano *_video_mean.json con el promedio por video
    """
    model, device = load_diffusion_model(ckpt_path, channels=channels)
    sched = DU.build_schedule(T=T_steps, device=device)

    cache: Dict[str, Dict[int, float]] = {}
    means: Dict[str, float] = {}

    for vp in videos:
        elbos = score_video_elbo(
            video_path=vp, model=model, sched=sched,
            frames_per_video=frames_per_video, image_size=image_size,
            channels=channels, use_face_crop=use_face_crop
        )
        if elbos:
            cache[vp] = elbos
            means[vp] = float(np.mean(list(elbos.values())))

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(cache, f)

    with open(out_json.with_name(out_json.stem + "_video_mean.json"), "w") as f:
        json.dump(means, f)

    return str(out_json), str(out_json.with_name(out_json.stem + "_video_mean.json"))
