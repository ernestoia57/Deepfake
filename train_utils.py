# train_utils.py
# Entrenamiento de ε-UNet con frames REALES del DFDC.
# Requiere diff_utils.py en el PYTHONPATH.

import os, json, math, csv, random
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils as nn_utils

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Importa tu toolbox de difusión
import diff_utils as DU


# -------------------------------
# Face crop con OpenCV Haar (rápido, sin deps extra)
# -------------------------------
try:
    HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    FACE_CASCADE = cv2.CascadeClassifier(HAAR_PATH)
except Exception:
    FACE_CASCADE = None  # fallback si no está disponible

def face_crop_fn(frame_bgr: np.ndarray, enlarge: float = 1.25, min_size: int = 64) -> np.ndarray:
    """
    Recorta la cara más grande de la imagen y devuelve un BGR crop.
    - enlarge: factor para agrandar el bbox (evita cortar frente/barbilla).
    - min_size: tamaño mínimo del bbox para aceptar la detección.
    Si no detecta, levanta excepción para que el dataset haga fallback a center-crop.
    """
    if FACE_CASCADE is None:
        raise RuntimeError("No Haar cascade available")

    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, flags=cv2.CASCADE_SCALE_IMAGE
    )
    if faces is None or len(faces) == 0:
        raise RuntimeError("No face")

    # cara más grande
    x, y, fw, fh = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
    if fw < min_size or fh < min_size:
        raise RuntimeError("Face too small")

    # agrandar bbox y cuadrar
    cx, cy = x + fw/2.0, y + fh/2.0
    side = int(max(fw, fh) * float(enlarge))
    side = int(min(side, max(h, w)))
    x0 = int(max(0, cx - side/2)); y0 = int(max(0, cy - side/2))
    x1 = int(min(w, x0 + side));   y1 = int(min(h, y0 + side))
    side = min(x1 - x0, y1 - y0);  x1 = x0 + side; y1 = y0 + side
    if side <= 0: raise RuntimeError("Invalid crop")

    crop = frame_bgr[y0:y1, x0:x1].copy()
    if crop.size == 0: raise RuntimeError("Empty crop")
    return crop


# -------------------------------
# Utilidades DFDC
# -------------------------------

def find_part_json(part_dir: Path) -> Path:
    """Devuelve el .json de metadatos dentro de un dfdc_train_part_X."""
    cands = list(part_dir.glob("*.json"))
    if not cands:
        raise FileNotFoundError(f"No JSON found in {part_dir}")
    # DFDC suele tener 1 json por carpeta
    return cands[0]


def read_dfdc_metadata(part_dir: Path) -> Dict[str, Dict]:
    """Lee el JSON de la carpeta dfdc_train_part_X y retorna el dict de metadatos."""
    meta_path = find_part_json(part_dir)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return meta  # { "video.mp4": {"label":"REAL/FAKE","split":"train", ...}, ... }


def build_real_index(
    root: Path,
    parts: Sequence[int] = tuple(range(50)),
    only_split: str = "train",
) -> List[Tuple[Path, str]]:
    """
    Recorre dfdc_train_part_X y devuelve lista [(video_path, label)].
    Aquí filtramos a REAL (para entrenar el prior de reales).
    """
    items: List[Tuple[Path, str]] = []
    for p in parts:
        part_dir = root / f"dfdc_train_part_{p}"
        if not part_dir.is_dir():
            continue
        meta = read_dfdc_metadata(part_dir)
        for fname, rec in meta.items():
            if (only_split is None) or (rec.get("split", "").lower() == only_split):
                label = rec.get("label", "").upper()
                if label == "REAL":
                    vpath = part_dir / fname
                    if vpath.is_file():
                        items.append((vpath, label))
    return items


# -------------------------------
# Preprocesado de frames
# -------------------------------

def center_square_crop(img: np.ndarray, ratio: float = 0.9) -> np.ndarray:
    """Recorte cuadrado centrado (ratio del lado menor)."""
    h, w = img.shape[:2]
    side = int(min(h, w) * float(ratio))
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0:y0 + side, x0:x0 + side]


def default_preprocess(
    frame_bgr: np.ndarray,
    out_size: int = 64,
    crop_ratio: float = 0.9,
    to_gray: bool = True,
    norm: str = "tanh",  # "tanh" -> [-1,1], "unit" -> [0,1]
) -> torch.Tensor:
    """
    Preprocesa un frame: BGR->RGB, center-crop, resize, (opcional) gray, normaliza.
    Devuelve tensor [1,64,64] float32 (o [3,64,64] si to_gray=False).
    """
    # BGR->RGB
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame = center_square_crop(frame, ratio=crop_ratio)
    frame = cv2.resize(frame, (out_size, out_size), interpolation=cv2.INTER_AREA)

    if to_gray:
        r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
        y = 0.2989 * r + 0.5870 * g + 0.1140 * b
        x = y.astype(np.float32) / 255.0  # [0,1]
        if norm == "tanh":
            x = x * 2.0 - 1.0             # [-1,1]
        x = x[None, ...]                  # [1,H,W]
    else:
        x = frame.astype(np.float32) / 255.0
        if norm == "tanh":
            x = x * 2.0 - 1.0
        x = np.transpose(x, (2, 0, 1))    # [3,H,W]
    return torch.from_numpy(x).contiguous()


def sample_uniform_frames(
    cap: cv2.VideoCapture,
    want: int,
    stride_hint: Optional[int] = None,
) -> List[int]:
    """
    Índices uniformes de frames a muestrear (sin cargar todo el video).
    Si stride_hint no viene, se reparte en [0..N-1].
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        return []
    if want <= 1:
        return [total // 2]
    idxs = np.linspace(0, max(0, total - 1), num=want, dtype=int).tolist()
    return idxs

# ====== NUEVO: índice genérico por etiqueta ======
def build_index_by_label(
    root: Path,
    parts: Sequence[int] = tuple(range(50)),
    only_split: str = "train",
    allowed_labels: Optional[Sequence[str]] = None,  # p.ej. ["REAL"] o ["FAKE"]
) -> List[Tuple[Path, str]]:
    items: List[Tuple[Path, str]] = []
    allowed = None if allowed_labels is None else {lab.upper() for lab in allowed_labels}
    for p in parts:
        part_dir = root / f"dfdc_train_part_{p}"
        if not part_dir.is_dir():
            continue
        meta = read_dfdc_metadata(part_dir)
        for fname, rec in meta.items():
            if (only_split is None) or (rec.get("split", "").lower() == only_split):
                label = str(rec.get("label", "")).upper()
                if (allowed is None) or (label in allowed):
                    vpath = part_dir / fname
                    if vpath.is_file():
                        items.append((vpath, label))
    return items

def build_fake_index(root: Path, parts: Sequence[int] = tuple(range(50)), only_split: str = "train"):
    return build_index_by_label(root, parts=parts, only_split=only_split, allowed_labels=["FAKE"])


# -------------------------------
# Dataset de frames REALES (DFDC)
# -------------------------------

class DFDCFramesDataset(Dataset):
    """
    Cada muestra = 1 frame de un video DFDC (muestreo uniforme) con filtro de etiqueta.
    allowed_labels: ["REAL"], ["FAKE"] o None (ambos).
    """
    def __init__(
        self,
        root: str,
        parts: Sequence[int] = tuple(range(50)),
        allowed_labels: Optional[Sequence[str]] = ("REAL",),  # por defecto REAL
        samples_per_video: int = 2,
        out_size: int = 64,
        crop_ratio: float = 0.9,
        to_gray: bool = True,
        norm: str = "tanh",
        max_videos: Optional[int] = None,
        face_crop_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        seed: int = 0,
    ):
        super().__init__()
        self.root = Path(root)
        self.items = build_index_by_label(self.root, parts=parts, only_split="train",
                                          allowed_labels=allowed_labels)
        if max_videos is not None:
            self.items = self.items[:max_videos]
        self.samples_per_video = int(samples_per_video)
        self.out_size = int(out_size)
        self.crop_ratio = float(crop_ratio)
        self.to_gray = bool(to_gray)
        self.norm = norm
        self.face_crop_fn = face_crop_fn
        random.seed(seed)

        self.expanded: List[Tuple[Path, int]] = []
        for vpath, _ in self.items:
            for _k in range(self.samples_per_video):
                self.expanded.append((vpath, -1))

    def __len__(self) -> int:
        return len(self.expanded)

    def __getitem__(self, i: int) -> torch.Tensor:
        vpath, _ = self.expanded[i]
        for _try in range(5):
            cap = cv2.VideoCapture(str(vpath))
            if not cap.isOpened():
                break
            idxs = sample_uniform_frames(cap, want=8)
            if not idxs:
                cap.release(); break
            random.shuffle(idxs)
            for idx in idxs[:4]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    cap.release()
                    if self.face_crop_fn is not None:
                        try:   frame = self.face_crop_fn(frame)
                        except Exception:  pass
                    x = default_preprocess(frame_bgr=frame, out_size=self.out_size,
                                           crop_ratio=self.crop_ratio, to_gray=self.to_gray, norm=self.norm)
                    return x
            cap.release()
        c = 1 if self.to_gray else 3
        return torch.zeros(c, self.out_size, self.out_size, dtype=torch.float32)



# -------------------------------
# Entrenamiento (ε-MSE + ELBO) con reanudación
# -------------------------------

def make_dataloader_by_label(
    root: str,
    parts: Sequence[int] = tuple(range(50)),
    allowed_labels: Optional[Sequence[str]] = ("REAL",),
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    **ds_kwargs,
) -> DataLoader:
    ds = DFDCFramesDataset(root=root, parts=parts, allowed_labels=allowed_labels, **ds_kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                      num_workers=num_workers, pin_memory=pin_memory)



def train_eps_unet_on_dfdc_label(
    root: str,
    outdir: str,
    label: str = "REAL",                 # <<< "REAL" o "FAKE"
    parts: Sequence[int] = tuple(range(50)),
    image_size: int = 64,
    channels: int = 1,
    T: int = 300,
    batch_size: int = 64,
    steps_per_epoch: int = 1000,
    epochs: int = 2,
    lr: float = 2e-4,
    weight_decay: float = 1e-2,
    samples_per_video: int = 2,
    max_videos: Optional[int] = None,
    seed: int = 0,
    use_face_crop: bool = True,
    resume_from: Optional[str] = None,
    resume_optimizer: bool = True,
    amp: bool = True,
    grad_clip: Optional[float] = 1.0,
    warmup_steps: int = 0,
) -> Dict[str, str]:
    """
    Entrena ε-UNet sobre DFDC filtrando por etiqueta (REAL/FAKE).
    Guarda checkpoints dentro de outdir/<label>/...
    """
    label = str(label).upper()
    assert label in ("REAL","FAKE")
    outdir = Path(outdir) / label.lower()
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl = make_dataloader_by_label(
        root=root,
        parts=parts,
        allowed_labels=(label,),
        batch_size=batch_size,
        samples_per_video=samples_per_video,
        out_size=image_size,
        to_gray=(channels == 1),
        max_videos=max_videos,
        seed=seed,
        face_crop_fn=(face_crop_fn if use_face_crop else None),
    )

    model = DU.TinyUNet(in_ch=channels, base=32, t_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = DU.build_schedule(T=T, device=device)

    # reanudar (igual que tu versión)
    start_epoch, global_step = 1, 0
    if resume_from:
        state = torch.load(resume_from, map_location=device)
        if "model" in state: model.load_state_dict(state["model"])
        if resume_optimizer and ("optimizer" in state):
            try: opt.load_state_dict(state["optimizer"])
            except: pass
        start_epoch = int(state.get("epoch", 0)) + 1
        global_step = int(state.get("global_step", 0))
        print(f"[Resume:{label}] start_epoch={start_epoch} global_step={global_step}")

    csv_path = outdir / "training_log.csv"
    write_header = not (resume_from and csv_path.exists())
    f_csv = open(csv_path, "a" if resume_from else "w", newline="")
    writer = csv.writer(f_csv)
    if write_header:
        writer.writerow(["step","epoch","loss_simple","ELBO_est"])

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda") and amp)

    from tqdm import tqdm
    import matplotlib.pyplot as plt
    steps_hist, loss_hist, elbo_hist = [], [], []

    model.train()
    for epoch in range(start_epoch, epochs+1):
        pbar = tqdm(total=steps_per_epoch, desc=f"[{label}] Epoch {epoch}/{epochs}", ncols=0)
        it = 0
        for x0 in dl:
            it += 1
            if it > steps_per_epoch: break
            global_step += 1

            if warmup_steps and global_step <= warmup_steps:
                scale = float(global_step) / float(max(1, warmup_steps))
                for pg in opt.param_groups: pg["lr"] = lr * scale

            x0 = x0.to(device, non_blocking=True)
            B = x0.size(0)
            t_idx = torch.randint(1, T+1, (B,), device=device)
            xt, eps_true = DU.sample_xt(x0, t_idx, sched)

            with torch.cuda.amp.autocast(enabled=(device.type=="cuda") and amp):
                eps_hat = model(xt, t_idx)
                loss = torch.nn.functional.mse_loss(eps_hat, eps_true)

            opt.zero_grad(set_to_none=True)
            if (device.type=="cuda") and amp:
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(opt)
                    nn_utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    nn_utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                opt.step()

            with torch.no_grad():
                elbo_est = DU.estimate_elbo_mc(model, x0, sched).mean().item()

            lv = float(loss.item())
            steps_hist.append(global_step); loss_hist.append(lv); elbo_hist.append(float(elbo_est))
            writer.writerow([global_step, epoch, lv, float(elbo_est)])

            pbar.set_postfix(loss=f"{lv:.4f}", elbo=f"{float(elbo_est):.4f}",
                             lr=f"{opt.param_groups[0]['lr']:.2e}")
            pbar.update(1)
        pbar.close()

        ckpt = outdir / f"eps_unet_{label.lower()}_epoch{epoch}.pt"
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "epoch": epoch, "global_step": global_step}, ckpt)
        print(f"[{label}] saved: {ckpt}")

    f_csv.close()
    final_ckpt = outdir / f"eps_unet_{label.lower()}_final.pt"
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "epoch": epochs, "global_step": global_step}, final_ckpt)
    print(f"[{label}] CSV: {csv_path}")
    print(f"[{label}] Final: {final_ckpt}")

    # plot rápido
    try:
        plt.figure(figsize=(8,4))
        ax1 = plt.gca()
        ax1.plot(steps_hist, loss_hist, label="loss_simple")
        ax1.set_xlabel("step"); ax1.set_ylabel("loss_simple"); ax1.grid(True, alpha=0.3)
        ax2 = ax1.twinx(); ax2.plot(steps_hist, elbo_hist, label="ELBO_est", linestyle="--")
        ax2.set_ylabel("ELBO_est")
        ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
        plt.title(f"Training {label} (loss & ELBO)")
        fig_path = outdir / "training_progress.png"
        plt.tight_layout(); plt.savefig(fig_path, dpi=150)
        print(f"[{label}] Plot: {fig_path}")
    except Exception as e:
        print(f"({label}) No plot: {e}")

    return {"csv": str(csv_path), "ckpt": str(final_ckpt)}



# -------------------------------
# Probe de ELBO en un subset
# -------------------------------

@torch.no_grad()
def elbo_probe_on_dfdc(
    ckpt_path: str,
    root: str,
    parts: Sequence[int] = (0,),
    n_videos: int = 16,
    frames_per_video: int = 8,
    image_size: int = 64,
    channels: int = 1,
    T: int = 300,
    use_face_crop: bool = False,  # activa face crop en probe
) -> List[Tuple[str, float]]:
    """
    Calcula ELBO_est promedio por video REAL en un subset (sanity check).
    Devuelve lista [(video_path, elbo_mean), ...].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DU.TinyUNet(in_ch=channels, base=32, t_dim=128).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    sched = DU.build_schedule(T=T, device=device)

    # Index videos REALES
    vids = build_real_index(Path(root), parts=parts)
    vids = vids[:n_videos]

    results = []
    for vpath, _ in vids:
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            continue
        idxs = sample_uniform_frames(cap, want=frames_per_video)
        elbos = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue

            # face-crop opcional (fallback silencioso a center-crop)
            if use_face_crop:
                try:
                    frame = face_crop_fn(frame)
                except Exception:
                    pass

            x = default_preprocess(frame, out_size=image_size, to_gray=(channels == 1), norm="tanh")
            x = x.unsqueeze(0).to(device)  # [1,C,H,W]
            e = DU.estimate_elbo_mc(model, x, sched).mean().item()
            elbos.append(e)
        cap.release()
        if elbos:
            results.append((str(vpath), float(np.mean(elbos))))
    return results
