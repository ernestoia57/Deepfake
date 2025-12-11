# data_utils.py
# Índices DFDC (REAL/FAKE) + DataSet mixto con ganchos para face-crop y CWT

import os, json, random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass

# Reutiliza funciones que ya tienes:
from train_utils import (
    read_dfdc_metadata,
    sample_uniform_frames,
    default_preprocess,
    face_crop_fn as _default_face_crop_fn,
)

# ------------------------------
# Construcción de índice DFDC
# ------------------------------

def build_dfdc_index(
    root: str | Path,
    parts: Sequence[int] = tuple(range(50)),
    only_split: Optional[str] = "train",
) -> List[Dict]:
    """
    Devuelve una lista de dicts:
      {"path": <Path a mp4>, "label": 0/1, "label_str": "REAL"/"FAKE", "part": int}
    label=1 -> FAKE, label=0 -> REAL (útil para BCE)
    """
    root = Path(root)
    items: List[Dict] = []
    for p in parts:
        part_dir = root / f"dfdc_train_part_{p}"
        if not part_dir.is_dir():
            continue
        meta = read_dfdc_metadata(part_dir)
        for fname, rec in meta.items():
            if only_split and str(rec.get("split","")).lower() != only_split:
                continue
            lab = str(rec.get("label","")).upper()
            if lab not in ("REAL","FAKE"):
                continue
            vpath = part_dir / fname
            if not vpath.is_file():
                continue
            items.append({
                "path": str(vpath),
                "label": 1 if lab == "FAKE" else 0,
                "label_str": lab,
                "part": p,
            })
    return items


# ------------------------------
# Dataset mixto con CWT + Calidad
# ------------------------------

@dataclass
class DFDCMixedSample:
    x: torch.Tensor          # [C,H,W] imagen preprocesada
    y: int                   # 0/1 (REAL/FAKE)
    path: str                # ruta video
    frame_idx: int           # índice de frame
    elbo: Optional[float]    # calidad (si existe cache)
    cwt: Optional[torch.Tensor]  # tokens CWT/feature opcional

class DFDCMixedFramesDataset(Dataset):
    """
    Entrega muestras de frames tanto REAL como FAKE.
    - Usa face crop (opcional) y el mismo preprocess del training.
    - Puede traer un score de calidad por frame desde un cache (dict[path][frame_idx] = elbo)
    - Puede adjuntar features/tokens CWT via un callback (cwt_fn)
    - Permite filtrado por calidad (min_elbo, max_elbo) y/o muestreo ponderado vía sampler externo.
    """
    def __init__(
        self,
        index: List[Dict],
        samples_per_video: int = 4,
        image_size: int = 64,
        channels: int = 1,
        use_face_crop: bool = True,
        face_crop_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        quality_cache: Optional[Dict[str, Dict[int, float]]] = None,  # {video_path: {frame_idx: elbo}}
        min_elbo: Optional[float] = None,
        max_elbo: Optional[float] = None,
        cwt_fn: Optional[Callable[[np.ndarray], torch.Tensor]] = None,  # recibe crop BGR -> tokens/features (Tensor)
        seed: int = 0,
    ):
        super().__init__()
        self.index = index
        self.samples_per_video = int(samples_per_video)
        self.image_size = int(image_size)
        self.channels = int(channels)
        self.use_face_crop = bool(use_face_crop)
        self.face_crop_fn = face_crop_fn or _default_face_crop_fn
        self.quality_cache = quality_cache or {}
        self.min_elbo = min_elbo
        self.max_elbo = max_elbo
        self.cwt_fn = cwt_fn
        random.seed(seed)

        # expandimos (video, y, rep)
        self.expanded: List[Tuple[str, int]] = []
        for rec in self.index:
            for _ in range(self.samples_per_video):
                self.expanded.append((rec["path"], rec["label"]))

    def __len__(self) -> int:
        return len(self.expanded)

    def _passes_quality(self, path: str, frame_idx: int) -> bool:
        if not self.quality_cache:
            return True
        vdict = self.quality_cache.get(path)
        if vdict is None:
            return True
        elbo = vdict.get(frame_idx)
        if elbo is None:
            return True
        if (self.min_elbo is not None) and (elbo < self.min_elbo):
            return False
        if (self.max_elbo is not None) and (elbo > self.max_elbo):
            return False
        return True

    def __getitem__(self, i: int) -> DFDCMixedSample:
        path, y = self.expanded[i]

        # hasta 5 intentos para encontrar un frame válido (y que pase el filtro de calidad si existe)
        for _try in range(5):
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                break
            idxs = sample_uniform_frames(cap, want=8)
            random.shuffle(idxs)
            ok_any = False
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    continue
                # filtro por calidad (si hay cache y thresholds)
                if not self._passes_quality(path, idx):
                    continue
                # face crop (fallback silencioso)
                if self.use_face_crop:
                    try:
                        frame = self.face_crop_fn(frame)
                    except Exception:
                        pass
                # preprocess
                x = default_preprocess(
                    frame_bgr=frame,
                    out_size=self.image_size,
                    to_gray=(self.channels == 1),
                    norm="tanh"
                )
                # cwt opcional
                cwt_tokens = None
                if self.cwt_fn is not None:
                    try:
                        cwt_tokens = self.cwt_fn(frame)  # espera BGR crop
                    except Exception:
                        cwt_tokens = None

                # elbo opcional desde cache
                elbo = None
                if self.quality_cache:
                    vdict = self.quality_cache.get(path)
                    if vdict is not None:
                        elbo = vdict.get(idx)

                cap.release()
                return DFDCMixedSample(
                    x=x, y=int(y), path=path, frame_idx=int(idx),
                    elbo=(None if elbo is None else float(elbo)),
                    cwt=cwt_tokens
                )
            cap.release()

        # fallback inocuo
        C = 1 if self.channels == 1 else 3
        return DFDCMixedSample(
            x=torch.zeros(C, self.image_size, self.image_size, dtype=torch.float32),
            y=int(y), path=path, frame_idx=-1, elbo=None, cwt=None
        )
