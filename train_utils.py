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

# Importa tu toolbox de difusión
import diff_utils as DU


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
    Devuelve tensor [1,64,64] float32.
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
        x = np.transpose(x, (2, 0, 1))  # [3,H,W]
        # Para ε-UNet ligera usamos 1 canal por defecto, pero puedes cambiarlo.

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


# -------------------------------
# Dataset de frames REALES (DFDC)
# -------------------------------

class DFDCRealFramesDataset(Dataset):
    """
    Cada muestra = 1 frame REAL de un video DFDC (muestreo uniforme).
    - Puedes controlar cuántas muestras por video con 'samples_per_video'.
    - Hook 'face_crop_fn' si quieres recorte facial (opcional).
    """
    def __init__(
        self,
        root: str,
        parts: Sequence[int] = tuple(range(50)),
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
        self.items = build_real_index(self.root, parts=parts, only_split="train")
        if max_videos is not None:
            self.items = self.items[:max_videos]
        self.samples_per_video = int(samples_per_video)
        self.out_size = int(out_size)
        self.crop_ratio = float(crop_ratio)
        self.to_gray = bool(to_gray)
        self.norm = norm
        self.face_crop_fn = face_crop_fn
        random.seed(seed)

        # Expandimos a (video_path, frame_index_placeholder) * samples_per_video
        self.expanded: List[Tuple[Path, int]] = []
        for vpath, _ in self.items:
            for _k in range(self.samples_per_video):
                # Frame index se decide en __getitem__ con muestreo uniforme
                self.expanded.append((vpath, -1))

    def __len__(self) -> int:
        return len(self.expanded)

    def __getitem__(self, i: int) -> torch.Tensor:
        vpath, _ = self.expanded[i]
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            raise RuntimeError(f"No pude abrir el video: {vpath}")

        # Selecciona un frame uniformemente al azar entre un pool uniforme
        idxs = sample_uniform_frames(cap, want=8)
        if not idxs:
            cap.release()
            raise RuntimeError(f"Video vacío o ilegible: {vpath}")
        idx = random.choice(idxs)

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"No pude leer el frame {idx} de {vpath}")

        if self.face_crop_fn is not None:
            try:
                frame = self.face_crop_fn(frame)  # Debe devolver un recorte BGR
            except Exception:
                # Si falla el detector, fallback a center-crop
                pass

        x = default_preprocess(
            frame_bgr=frame,
            out_size=self.out_size,
            crop_ratio=self.crop_ratio,
            to_gray=self.to_gray,
            norm=self.norm,
        )
        # Salida [C,H,W], por defecto [1,64,64] en [-1,1]
        return x


# -------------------------------
# Entrenamiento (ε-MSE + ELBO)
# -------------------------------

def make_dataloader(
    root: str,
    parts: Sequence[int] = tuple(range(50)),
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    **ds_kwargs,
) -> DataLoader:
    ds = DFDCRealFramesDataset(root=root, parts=parts, **ds_kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                      num_workers=num_workers, pin_memory=pin_memory)


def train_eps_unet_on_dfdc(
    root: str,
    outdir: str = "./checkpoints_dfdc_eps",
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
) -> Dict[str, str]:
    """
    Entrena TinyUNet (ε-pred) sobre frames REALES del DFDC.
    Devuelve rutas a: checkpoint final y CSV de log.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    # DataLoader
    dl = make_dataloader(
        root=root,
        parts=parts,
        batch_size=batch_size,
        samples_per_video=samples_per_video,
        out_size=image_size,
        to_gray=(channels == 1),
        max_videos=max_videos,
        seed=seed,
    )

    # Modelo
    model = DU.TinyUNet(in_ch=channels, base=32, t_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Schedule
    sched = DU.build_schedule(T=T, device=device)

    # Logging
    csv_path = outdir / "training_log.csv"
    f_csv = open(csv_path, "w", newline="")
    writer = csv.writer(f_csv); writer.writerow(["step","epoch","loss_simple","ELBO_est"])

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    global_step = 0
    model.train()
    for epoch in range(1, epochs + 1):
        it = 0
        for x0 in dl:
            it += 1
            if it > steps_per_epoch:
                break
            global_step += 1

            x0 = x0.to(device, non_blocking=True)  # [B,C,H,W]
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                loss_simple, elbo_est = DU.train_step(model, x0, sched, opt)

            # (nota: train_step ya hace backward+step)
            writer.writerow([global_step, epoch, float(loss_simple), float(elbo_est)])

        ckpt_path = outdir / f"eps_unet_epoch{epoch}.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_path)
        print(f"[Epoch {epoch}] loss={float(loss_simple):.4f} | ELBO_est={float(elbo_est):.4f} | saved: {ckpt_path}")

    f_csv.close()
    final_ckpt = outdir / "eps_unet_final.pt"
    torch.save({"model": model.state_dict(), "epoch": epochs}, final_ckpt)
    print(f"Log guardado en: {csv_path}")
    print(f"Checkpoint final: {final_ckpt}")
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
            x = default_preprocess(frame, out_size=image_size, to_gray=(channels == 1), norm="tanh")
            x = x.unsqueeze(0).to(device)  # [1,C,H,W]
            e = DU.estimate_elbo_mc(model, x, sched).mean().item()
            elbos.append(e)
        cap.release()
        if elbos:
            results.append((str(vpath), float(np.mean(elbos))))
    return results
