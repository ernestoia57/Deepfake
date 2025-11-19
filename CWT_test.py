import numpy as np
import torch
import pywt
from cwt import CWT1D

# ---------- utilidades ----------
def make_time(fs, T):  # segundos de señal
    t = np.arange(int(fs*T)) / fs
    return t

def best_alpha(A, B):
    # escala óptima para alinear magnitudes (least squares)
    # minimiza ||αA - B||_2  -> α = (A·B)/(A·A)
    num = np.vdot(A.ravel(), B.ravel()).real
    den = np.vdot(A.ravel(), A.ravel()).real + 1e-12
    return float(num/den)

def interior_slice(n, border):
    return slice(border, n - border)

# ---------- prueba 1: seno puro ----------
def test_sine_match(fs=30.0, T=8.0, f0=6.0, n_bands=12, fmin=0.5, fmax=12.0, w0=6.0, border=16):
    t = make_time(fs, T)
    x = np.sin(2*np.pi*f0*t).astype(np.float32)
    xt = torch.from_numpy(x)[None, None, :]  # [B=1,1,T]

    # tu CWT
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_bands).tolist()
    cwt_torch = CWT1D(fs=fs, kernel_size=129, w0=w0, freqs_hz=freqs, return_phase=False)
    W_t = cwt_torch(xt).detach().cpu().numpy()[0]  # [S,T]

    # pywt CWT (usa "widths" ~ escalas; sampling_period = 1/fs)
    # relación típica: s = w0/(2π f)  -> widths = s / dt  = s * fs
    scales = (w0 / (2*np.pi*np.array(freqs)))  # en segundos
    widths = scales * fs
    W_py, py_scales = pywt.cwt(x, widths, 'morl', sampling_period=1.0/fs)  # [S,T]

    # re-escala para comparar
    S, Tn = W_py.shape
    sl = (slice(0, S), interior_slice(Tn, border))
    A = W_t[sl]
    B = np.abs(W_py[sl])  # magnitud
    α = best_alpha(A, B)
    mse = np.mean((α*A - B)**2)
    corr = np.corrcoef(A.ravel(), B.ravel())[0,1]

    # localización de la banda dominante cerca del centro temporal
    tau = Tn//2
    s_idx = int(np.argmax(W_t[:, tau]))
    f_hat = freqs[s_idx]
    return dict(mse=mse, corr=float(corr), f_hat=f_hat, f0=f0, alpha=α)

# ---------- prueba 2: chirp lineal ----------
def test_chirp(fs=30.0, T=8.0, f1=1.0, f2=12.0, n_bands=16, w0=6.0, border=16):
    t = make_time(fs, T)
    k = (f2 - f1) / T
    phase = 2*np.pi*(f1*t + 0.5*k*t**2)
    x = np.sin(phase).astype(np.float32)
    xt = torch.from_numpy(x)[None, None, :]

    freqs = np.logspace(np.log10(f1), np.log10(f2), n_bands).tolist()
    cwt_torch = CWT1D(fs=fs, kernel_size=129, w0=w0, freqs_hz=freqs, return_phase=False)
    W = cwt_torch(xt).detach().cpu().numpy()[0]  # [S,T]

    # cresta: índice de escala con mayor magnitud por tiempo
    ridge_idx = np.argmax(W, axis=0)
    f_hat_t = np.array([freqs[i] for i in ridge_idx])

    # verdad de tierra: f(t) = f1 + k t
    f_true = f1 + k * t
    # error medio (interior temporal)
    sl = interior_slice(len(t), border)
    mae = float(np.mean(np.abs(f_hat_t[sl] - f_true[sl])))
    return dict(mae_hz=mae, f_hat_t=f_hat_t, f_true=f_true)
