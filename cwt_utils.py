# cwt_module.py
from __future__ import annotations

import math, os, cv2, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Iterable, Tuple, List, Optional, Dict

__all__ = [
    "CWT1D",
    "load_video_frames",
    "to_grayscale",
    "detrend_per_pixel",
    "cwt_over_roi",
    "cwt_per_pixel_downsample",
    "cwt_per_pixel_dense",
    "plot_roi_spectrogram",
    "plot_chirp_ridge",
    "plot_band_maps",
    "analyze_video_with_cwt",
    "BandMLP",
    "CWTtoTokens",
]

# -----------------------------------------------------------------------------
# Config global de device (puedes forzarlo desde fuera si quieres)
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _ensure_odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1

# -----------------------------------------------------------------------------
# CWT Morlet COMPLEJA corregida
# -----------------------------------------------------------------------------
class CWT1D(nn.Module):
    """
    CWT 1D con Morlet COMPLEJA corregida (ideal para video deepfakes).
    Defaults pensados para ~30 FPS, 32–48 frames, ROI facial.

    Salida: magnitud |W| [B,S,T] (y opcionalmente fase).
    """
    def __init__(
        self,
        fs: float = 30.0,
        kernel_size: int = 161,
        w0: float = 6.5,
        scales=None,
        freqs_hz=None,
        fmin: float = 0.5,
        fmax: float = 12.0,
        n_bands: int = 16,
        norm_mode: str = "l2",
        pad_mode: Optional[str] = "reflect",  # 'reflect'/'replicate'/None
        return_phase: bool = False,
    ):
        super().__init__()
        self.fs = float(fs) if fs is not None else None
        self.kernel_size = _ensure_odd(int(kernel_size))
        self.w0 = float(w0)
        self.return_phase = bool(return_phase)
        self.norm_mode = norm_mode
        self.pad_mode = pad_mode

        if (scales is None) and (freqs_hz is None):
            assert self.fs is not None, "Debes indicar fs para bandas por frecuencia."
            freqs_hz = torch.logspace(
                math.log10(fmin), math.log10(fmax), steps=int(n_bands)
            ).tolist()

        if scales is None:
            assert self.fs is not None, "Para freqs_hz necesitas fs."
            scales = [self.w0 / (2.0 * math.pi * float(f)) for f in freqs_hz]

        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32), persistent=False)

        self.corr_const = math.exp(-0.5 * (self.w0 ** 2))  # Morlet corregida (sin DC)
        self.pi_norm = math.pi ** (-0.25)

        self._built = False
        self.register_buffer("Kr", None, persistent=False)
        self.register_buffer("Ki", None, persistent=False)

    def _build_kernels(self, device, dtype, dt: float):
        """
        Construye los núcleos Morlet (CWT) evitando tensores complejos en CUDA.
        Guarda:
        - self.Kr: [S, 1, L] parte real
        - self.Ki: [S, 1, L] parte imaginaria
        """
        dt = float(dt)
        L = self.kernel_size

        # Eje temporal (real)
        t = torch.linspace(
            -L // 2,
            L // 2,
            steps=L,
            device=device,
            dtype=dtype
        ) * dt

        # Escalas en el device correcto
        S = self.scales.to(device=device, dtype=dtype)

        # Constantes reales
        pi_norm = torch.tensor(float(self.pi_norm), device=device, dtype=dtype)
        corr_const = torch.tensor(float(self.corr_const), device=device, dtype=dtype)
        w0 = torch.tensor(float(self.w0), device=device, dtype=dtype)

        Kr = []
        Ki = []

        for s in S:
            # t escalado
            tau = t / s

            # envolvente gaussiana
            gauss = torch.exp(-0.5 * tau * tau)  # real

            # fase portadora
            theta = w0 * tau                      # real
            carrier_re = torch.cos(theta)         # Re{e^{jθ}}
            carrier_im = torch.sin(theta)         # Im{e^{jθ}}

            # amplitud común real
            amp = (pi_norm / torch.sqrt(s)) * gauss  # real

            # Morlet corregida en DC:
            # psi = amp * (carrier - corr_const)
            # => Re = amp * (carrier_re - corr_const)
            #    Im = amp * carrier_im
            psi_re = amp * (carrier_re - corr_const)
            psi_im = amp * carrier_im

            # correlación: kernel invertido en el tiempo
            psi_re = psi_re.flip(0)
            psi_im = psi_im.flip(0)

            Kr.append(psi_re)
            Ki.append(psi_im)

        # [S, 1, L] para usar como filtros conv1d
        self.Kr = torch.stack(Kr, dim=0).unsqueeze(1)
        self.Ki = torch.stack(Ki, dim=0).unsqueeze(1)

        self._built = True


    def forward(self, x: torch.Tensor, dt: float = None):
        assert x.dim() == 3 and x.size(1) == 1, "Se espera [B,1,T]."
        device, dtype = x.device, x.dtype
        if dt is None:
            dt = 1.0 / self.fs if self.fs is not None else 1.0
        dt = float(dt)

        if (not self._built) or (self.Kr is None) or (self.Kr.device != device) or (self.Kr.dtype != dtype):
            self._build_kernels(device, dtype, dt)

        pad = self.kernel_size // 2
        T = x.size(-1)

        # reflect/replicate requieren pad < T; si no, fallback a padding interno de conv1d
        can_explicit_pad = (self.pad_mode is not None) and (pad < T)

        if can_explicit_pad:
            x_pad = F.pad(x, (pad, pad), mode=self.pad_mode)
            conv_pad = 0
        else:
            x_pad = x
            conv_pad = pad  # zero-padding interno

        yr = F.conv1d(x_pad, self.Kr, padding=conv_pad)
        yi = F.conv1d(x_pad, self.Ki, padding=conv_pad)
        mag = torch.sqrt(yr**2 + yi**2 + 1e-12)
        if self.return_phase:
            phase = torch.atan2(yi, yr)
            return mag, phase
        return mag

# -----------------------------------------------------------------------------
# I/O & preprocesamiento
# -----------------------------------------------------------------------------
def load_video_frames(path, target_fps=30.0, max_frames=None, resize=(192,192), center_crop_ratio=0.9):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir el video: {path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    # ratio para saltar frames y aproximar target_fps
    step = max(1, int(round(src_fps / target_fps)))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % step == 0:
            # BGR->RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            # center crop cuadrado
            side = int(min(h, w) * center_crop_ratio)
            y0 = (h - side) // 2
            x0 = (w - side) // 2
            crop = frame[y0:y0+side, x0:x0+side]
            # resize
            if resize is not None:
                crop = cv2.resize(crop, resize, interpolation=cv2.INTER_AREA)
            frames.append(crop)
        idx += 1
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    frames = np.stack(frames, axis=0)  # [T,H,W,3]
    return frames, target_fps

def to_grayscale(frames):  # [T,H,W,3] -> [T,1,H,W]
    r, g, b = frames[...,0], frames[...,1], frames[...,2]
    y = 0.2989*r + 0.5870*g + 0.1140*b
    y = y.astype(np.float32) / 255.0
    return y[:, None, ...]  # [T,1,H,W]

def detrend_per_pixel(x):  # [T,1,H,W]
    mu = x.mean(axis=0, keepdims=True)
    return (x - mu).astype(np.float32)

# -----------------------------------------------------------------------------
# CWT ROI y Pixelwise
# -----------------------------------------------------------------------------
@torch.no_grad()
def cwt_over_roi(x_tchw, fps=30.0, cwt_cfg=None):
    """
    x_tchw: [T,1,H,W] float32, ya detrendido
    Devuelve: W_roi [S,T], freqs (Hz)
    """
    assert x_tchw.ndim == 4 and x_tchw.shape[1] == 1
    # serie 1D: promedio espacial (ROI)
    s = x_tchw.mean(axis=(2,3), keepdims=False)  # [T,1]
    xt = torch.from_numpy(s.T).unsqueeze(0).to(DEVICE)  # [B=1,1,T]
    cwt = CWT1D(fs=fps, **(cwt_cfg or {})).to(DEVICE)
    W = cwt(xt).detach().cpu().numpy()[0]  # [S,T]
    # frecuencias a partir de escalas
    freqs = (cwt.w0 / (2*np.pi*cwt.scales.detach().cpu().numpy()))
    return W, freqs

@torch.no_grad()
def cwt_per_pixel_downsample(x_tchw, fps=30.0, grid=(12,12), cwt_cfg=None):
    """
    CWT por píxel en grilla reducida (rápido).
    x_tchw: [T,1,H,W] float32 detrend
    grid: (Gh,Gw) celdas uniformes; usa el centro de cada celda.
    Devuelve: W_maps [S,T,Gh,Gw], freqs
    """
    assert x_tchw.ndim == 4 and x_tchw.shape[1] == 1
    T, _, H, W = x_tchw.shape
    Gh, Gw = grid
    ys = np.linspace(0, H-1, Gh, dtype=int)
    xs = np.linspace(0, W-1, Gw, dtype=int)

    series = []
    for y in ys:
        for x in xs:
            series.append(x_tchw[:, 0, y, x])  # [T]
    series = np.stack(series, axis=0).astype(np.float32)  # [Gh*Gw, T]

    xt = torch.from_numpy(series).unsqueeze(1).to(DEVICE)  # [Gh*Gw,1,T]
    cwt = CWT1D(fs=fps, **(cwt_cfg or {})).to(DEVICE)
    Wb = cwt(xt).detach()  # [P,S,T]
    P, S, T = Wb.shape
    W = Wb.view(Gh, Gw, S, T).permute(2,3,0,1).contiguous()  # [S,T,Gh,Gw]
    freqs = (cwt.w0 / (2*np.pi*cwt.scales.detach().cpu().numpy()))
    return W.cpu().numpy(), freqs

@torch.no_grad()
def cwt_per_pixel_dense(x_tchw, fps=30.0, batch_size: int = 1024, cwt_cfg=None):
    """
    NUEVO: CWT pixelwise denso (cada píxel), con batching.
    x_tchw: [T,1,H,W] float32 detrend
    Devuelve: W_full [S,T,H,W], freqs
    Notas: usar con cuidado; ajusta batch_size según tu VRAM.
    """
    assert x_tchw.ndim == 4 and x_tchw.shape[1] == 1
    T, _, H, W = x_tchw.shape
    P = H * W

    series = x_tchw[:,0].reshape(T, -1).T.astype(np.float32)  # [P,T]
    xt = torch.from_numpy(series).unsqueeze(1).to(DEVICE)     # [P,1,T]

    cwt = CWT1D(fs=fps, **(cwt_cfg or {})).to(DEVICE)
    S = len(cwt.scales)

    Ws = []
    for i in range(0, P, batch_size):
        chunk = xt[i:i+batch_size]                # [p,1,T]
        Wi = cwt(chunk)                           # [p,S,T]
        Ws.append(Wi.detach().cpu())
    W_all = torch.cat(Ws, dim=0).numpy()          # [P,S,T]
    W_full = W_all.reshape(H, W, S, T).transpose(2,3,0,1)  # [S,T,H,W]
    freqs = (cwt.w0 / (2*np.pi*cwt.scales.detach().cpu().numpy()))
    return W_full, freqs

# -----------------------------------------------------------------------------
# Visualización
# -----------------------------------------------------------------------------
def plot_roi_spectrogram(W_roi, freqs, title="CWT ROI (|W|)"):
    plt.figure()
    plt.imshow(W_roi, aspect='auto', origin='lower')
    yt = np.linspace(0, len(freqs)-1, 6, dtype=int)
    plt.yticks(ticks=yt, labels=[f"{freqs[i]:.1f} Hz" for i in yt])
    plt.xlabel("Tiempo (frames)")
    plt.ylabel("Frecuencia (Hz) ~")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_chirp_ridge(W_roi, freqs, title="CWT ROI (cresta)"):
    # cresta = argmax por tiempo
    ridge_idx = np.argmax(W_roi, axis=0)  # [T]
    plt.figure()
    plt.imshow(W_roi, aspect='auto', origin='lower')
    plt.plot(np.arange(W_roi.shape[1]), ridge_idx, 'w-', linewidth=1.5)
    yt = np.linspace(0, len(freqs)-1, 6, dtype=int)
    plt.yticks(ticks=yt, labels=[f"{freqs[i]:.1f} Hz" for i in yt])
    plt.xlabel("Tiempo (frames)")
    plt.ylabel("Frecuencia (Hz) ~")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_band_maps(W_maps, freqs, band_indices=(2,5,8)):
    # W_maps: [S,T,Gh,Gw]  -> promediamos en T para ver "energía por banda"
    E = W_maps.mean(axis=1)  # [S,Gh,Gw]
    n = len(band_indices)
    plt.figure(figsize=(4*n, 4))
    for k, bi in enumerate(band_indices, 1):
        plt.subplot(1, n, k)
        plt.imshow(E[bi], origin='lower')
        plt.title(f"Banda {bi} ~ {freqs[bi]:.1f} Hz")
        plt.axis('off')
    plt.suptitle("Energía espacial por banda (promedio en el tiempo)")
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Pipeline de análisis (conmutador de pixelwise)
# -----------------------------------------------------------------------------
def analyze_video_with_cwt(video_path,
                           target_fps=30.0,
                           max_frames=48,
                           resize=(192,192),
                           grid=(12,12),
                           cwt_cfg=None,
                           pixelwise_mode: str = "grid",   # "grid" | "dense" | "none"
                           dense_batch_size: int = 1024,
                           visualize=True):
    """
    pixelwise_mode:
      - "none": sólo ROI
      - "grid": CWT por grilla (rápido)
      - "dense": CWT por todos los píxeles (cuidado con VRAM/tiempo)
    """
    frames_rgb, fps_eff = load_video_frames(video_path, target_fps=target_fps,
                                            max_frames=max_frames, resize=resize)
    x = to_grayscale(frames_rgb)                 # [T,1,H,W] float32
    x = detrend_per_pixel(x)                     # quita media temporal

    # CWT ROI (rápido)
    W_roi, freqs = cwt_over_roi(x, fps=fps_eff, cwt_cfg=cwt_cfg)

    # CWT pixelwise según modo
    W_maps = None
    if pixelwise_mode == "grid":
        W_maps, freqs2 = cwt_per_pixel_downsample(x, fps=fps_eff, grid=grid, cwt_cfg=cwt_cfg)
    elif pixelwise_mode == "dense":
        W_maps, freqs2 = cwt_per_pixel_dense(x, fps=fps_eff, batch_size=dense_batch_size, cwt_cfg=cwt_cfg)
    elif pixelwise_mode == "none":
        pass
    else:
        raise ValueError("pixelwise_mode debe ser 'grid', 'dense' o 'none'.")

    if visualize:
        plot_roi_spectrogram(W_roi, freqs, title="CWT ROI (|W|)")
        plot_chirp_ridge(W_roi, freqs, title="CWT ROI (cresta)")
        if W_maps is not None and pixelwise_mode == "grid":
            sel = np.linspace(1, len(freqs)-2, num=min(3, len(freqs)-2), dtype=int).tolist()
            plot_band_maps(W_maps, freqs, band_indices=sel)

    # Features sencillas (para una red):
    band_mean = W_roi.mean(axis=1)        # [S]
    band_std  = W_roi.std(axis=1)         # [S]
    features = np.concatenate([band_mean, band_std], axis=0).astype(np.float32)  # [2S]

    return {
        "fps": fps_eff,
        "frames": frames_rgb.shape[0],
        "freqs": freqs,
        "W_roi": W_roi,       # [S,T]
        "W_maps": W_maps,     # [S,T,Gh,Gw] si grid, o [S,T,H,W] si dense, o None
        "features": features  # [2S]
    }

# -----------------------------------------------------------------------------
# MLP demo y Tokens para TimeSformer
# -----------------------------------------------------------------------------
class BandMLP(nn.Module):
    def __init__(self, in_dim, hidden=128, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class CWTtoTokens(nn.Module):
    """
    Reduce bandas S y hace patchify para producir tokens alineados con ViT/TimeSformer.
    Entrada:  cwt_feats [B,S,T,H,W]
    Salida:   tokens    [B,T,Np,D]
    """
    def __init__(self, S, patch=1, reduce_channels=64, embed_dim=256):
        super().__init__()
        self.patch = patch
        self.reduce = nn.Conv3d(S, reduce_channels, kernel_size=1)  # S -> Cr
        self.embed_dim = embed_dim
        self.proj = None
        self.reduce_channels = reduce_channels

    def _build_proj(self, H, W, device, dtype):
        ph = pw = self.patch
        in_dim = self.reduce_channels * ph * pw
        self.proj = nn.Linear(in_dim, self.embed_dim, bias=True).to(device=device, dtype=dtype)

    def forward(self, cwt_feats):  # [B,S,T,H,W]
        assert cwt_feats.dim() == 5, "cwt_feats debe ser [B,S,T,H,W]"
        device = cwt_feats.device
        dtype  = cwt_feats.dtype

        B,S,T,H,W = cwt_feats.shape

        z = self.reduce(cwt_feats)          # [B,Cr,T,H,W]
        z = z.permute(0,2,1,3,4)            # [B,T,Cr,H,W]

        if (self.proj is None) or (self.proj.weight.device != device) or (self.proj.weight.dtype != dtype):
            self._build_proj(H, W, device, dtype)

        ph = pw = self.patch
        z = z.unfold(3, ph, ph).unfold(4, pw, pw)   # [B,T,Cr,H/ph,W/pw,ph,pw]
        B,T,Cr,Hn,Wn,ph,pw = z.shape
        z = z.contiguous().view(B, T, Hn*Wn, Cr*ph*pw)  # [B,T,Np,Cr*ph*pw]
        z = z.to(dtype=self.proj.weight.dtype)
        tokens = self.proj(z)                        # [B,T,Np,D]
        return tokens
