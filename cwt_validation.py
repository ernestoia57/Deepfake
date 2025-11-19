# CWT Validation (Morlet, 1/sqrt(s)) - Minimal Script
# Ejecuta:  python cwt_validation.py
# Figuras:  ./cwt_sine_our.png, ./cwt_sine_pywt.png (si hay pywt), ./cwt_chirp_ridge.png

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(".")  # cambia a tu carpeta de artefactos si quieres

def _ensure_odd(k): return k if k % 2 == 1 else k + 1

class CWT1D(nn.Module):
    """
    CWT 1D con Morlet compleja corregida y normalización 1/sqrt(s).
    Correlación implementada como conv1d con kernel invertido (flip).
    Normalización L2 por escala para estabilidad numérica.
    """
    def __init__(self, fs=None, kernel_size=129, w0=6.0, scales=None, freqs_hz=None, return_phase=False):
        super().__init__()
        self.fs = fs
        self.kernel_size = _ensure_odd(kernel_size)
        self.w0 = float(w0)
        self.return_phase = return_phase

        if (scales is None) and (freqs_hz is None):
            if fs is None:
                scales = [0.02, 0.03, 0.05, 0.08, 0.12, 0.20]
            else:
                freqs_hz = torch.logspace(math.log10(0.5), math.log10(32.0), steps=6).tolist()
        if scales is None:
            assert fs is not None, "Para freqs_hz debes indicar fs."
            scales = [self.w0 / (2 * math.pi * float(f)) for f in freqs_hz]

        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32), persistent=False)
        self.corr_const = math.exp(-0.5 * (self.w0 ** 2))    # término de corrección Morlet
        self.pi_norm = math.pi ** (-0.25)
        self._built = False
        self.register_buffer("Kr", None, persistent=False)
        self.register_buffer("Ki", None, persistent=False)

    def _build_kernels(self, device, dtype, dt):
        S = self.scales.to(device=device, dtype=torch.float32)
        K = self.kernel_size
        half = K // 2
        t = torch.arange(-half, half + 1, device=device, dtype=torch.float32) * dt  # [K]

        kernels_r, kernels_i = [], []
        for s in S:
            gauss = torch.exp(-0.5 * (t / s) ** 2)
            carrier = torch.exp(1j * (self.w0 * (t / s)))
            psi = (self.pi_norm / torch.sqrt(s)) * (carrier - self.corr_const) * gauss  # complejo
            psi = psi.flip(0)  # correlación = conv con kernel invertido
            kernels_r.append(psi.real.view(1, 1, K))
            kernels_i.append(psi.imag.view(1, 1, K))
        Kr = torch.cat(kernels_r, dim=0)  # [S,1,K]
        Ki = torch.cat(kernels_i, dim=0)  # [S,1,K]

        # Normalización L2 por escala (sobre dims 1 y K)
        energy = torch.sqrt((Kr**2 + Ki**2).sum(dim=(1, 2), keepdim=True) + 1e-12)  # [S,1,1]
        Kr = Kr / energy
        Ki = Ki / energy

        self.Kr = Kr.to(device=device, dtype=dtype)
        self.Ki = Ki.to(device=device, dtype=dtype)
        self._built = True

    def forward(self, x, dt=None):
        """
        x: [B,1,T]  →  salida: magnitud [B,S,T]  (y opcionalmente fase)
        """
        assert x.dim() == 3 and x.size(1) == 1, "Se espera entrada [B,1,T]."
        device, dtype = x.device, x.dtype
        if dt is None:
            dt = 1.0 / self.fs if self.fs is not None else 1.0
        dt = float(dt)

        if (not self._built) or (self.Kr is None) or (self.Kr.device != device) or (self.Kr.dtype != dtype):
            self._build_kernels(device, dtype, dt)

        yr = F.conv1d(x, self.Kr, padding=self.kernel_size // 2)  # [B,S,T]
        yi = F.conv1d(x, self.Ki, padding=self.kernel_size // 2)
        mag = torch.sqrt(yr**2 + yi**2 + 1e-12)

        if self.return_phase:
            phase = torch.atan2(yi, yr)
            return mag, phase
        return mag

# ----- helpers -----
def make_time(fs, T):
    n = int(fs * T)
    return np.arange(n) / fs

def best_alpha(A, B):
    num = np.vdot(A.ravel(), B.ravel()).real
    den = np.vdot(A.ravel(), A.ravel()).real + 1e-12
    return float(num / den)

def interior_slice(n, border):
    b = int(border)
    if 2 * b >= n: return slice(0, n)
    return slice(b, n - b)

# ----- pruebas -----
try:
    import pywt
    HAS_PYWT = True
except Exception:
    HAS_PYWT = False

# 1) Seno puro: localización en frecuencia y heatmaps
fs, T, f0 = 30.0, 8.0, 6.0
n_bands, fmin, fmax, w0 = 12, 0.5, 12.0, 6.0
border = 16

t = make_time(fs, T)
x = np.sin(2 * np.pi * f0 * t).astype(np.float32)
xt = torch.from_numpy(x)[None, None, :]

freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_bands).tolist()
cwt_torch = CWT1D(fs=fs, kernel_size=129, w0=w0, freqs_hz=freqs, return_phase=False)
W_t = cwt_torch(xt).detach().cpu().numpy()[0]  # [S,T]

tau = len(t) // 2
s_idx = int(np.argmax(W_t[:, tau]))
f_hat = freqs[s_idx]
print({"f0": f0, "f_hat": f_hat})

plt.figure()
plt.imshow(W_t, aspect='auto', origin='lower')
plt.title(f"CWT magnitude (our impl) — sine {f0} Hz")
plt.xlabel("Time index"); plt.ylabel("Scale index")
plt.tight_layout()
plt.savefig(OUT / "cwt_sine_our.png", dpi=150); plt.close()

if HAS_PYWT:
    scales = (w0 / (2 * np.pi * np.array(freqs)))
    widths = scales * fs
    W_py, _ = pywt.cwt(x, widths, 'morl', sampling_period=1.0 / fs)  # [S,T]
    S, Tn = W_py.shape
    sl = (slice(0, S), interior_slice(Tn, border))
    A = W_t[sl]; B = np.abs(W_py[sl])
    alpha = best_alpha(A, B)
    mse = float(np.mean((alpha * A - B) ** 2))
    corr = float(np.corrcoef(A.ravel(), B.ravel())[0, 1])
    print({"pywt_mse": mse, "pywt_corr": corr, "alpha": alpha})

    plt.figure()
    plt.imshow(np.abs(W_py), aspect='auto', origin='lower')
    plt.title("CWT magnitude (PyWavelets morl)")
    plt.xlabel("Time index"); plt.ylabel("Scale index")
    plt.tight_layout()
    plt.savefig(OUT / "cwt_sine_pywt.png", dpi=150); plt.close()

# 2) Chirp lineal: cresta y MAE
fs, T = 30.0, 8.0
f1, f2 = 1.0, 12.0
n_bands, w0 = 16, 6.0
border = 16

t = make_time(fs, T)
k = (f2 - f1) / T
phase = 2 * np.pi * (f1 * t + 0.5 * k * t**2)
x = np.sin(phase).astype(np.float32)
xt = torch.from_numpy(x)[None, None, :]

freqs = np.logspace(np.log10(f1), np.log10(f2), n_bands).tolist()
cwt_torch = CWT1D(fs=fs, kernel_size=129, w0=w0, freqs_hz=freqs, return_phase=False)
W = cwt_torch(xt).detach().cpu().numpy()[0]  # [S,T]

ridge_idx = np.argmax(W, axis=0)
f_hat_t = np.array([freqs[i] for i in ridge_idx])
f_true = f1 + k * t

sl = interior_slice(len(t), border)
mae = float(np.mean(np.abs(f_hat_t[sl] - f_true[sl])))
print({"chirp_mae_hz": mae})

plt.figure()
plt.imshow(W, aspect='auto', origin='lower')
plt.plot(np.arange(len(t)), ridge_idx, linewidth=1.0)
plt.title(f"CWT (our impl) — chirp ridge, MAE≈{mae:.3f} Hz (interior)")
plt.xlabel("Time index"); plt.ylabel("Scale index")
plt.tight_layout()
plt.savefig(OUT / "cwt_chirp_ridge.png", dpi=150); plt.close()
