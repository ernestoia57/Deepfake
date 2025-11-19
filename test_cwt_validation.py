import math
import numpy as np
import torch
import pytest
#from cwt import CWT1D

def _ensure_odd(k): return k if k % 2 == 1 else k + 1

class CWT1D(torch.nn.Module):
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
        self.corr_const = math.exp(-0.5 * (self.w0 ** 2))
        self.pi_norm = math.pi ** (-0.25)
        self._built = False
        self.register_buffer("Kr", None, persistent=False)
        self.register_buffer("Ki", None, persistent=False)

    def _build_kernels(self, device, dtype, dt):
        S = self.scales.to(device=device, dtype=torch.float32)
        K = self.kernel_size
        half = K // 2
        t = torch.arange(-half, half + 1, device=device, dtype=torch.float32) * dt
        kernels_r, kernels_i = [], []
        for s in S:
            gauss = torch.exp(-0.5 * (t / s) ** 2)
            carrier = torch.exp(1j * (self.w0 * (t / s)))
            psi = (self.pi_norm / torch.sqrt(s)) * (carrier - self.corr_const) * gauss
            psi = psi.flip(0)
            kernels_r.append(psi.real.view(1, 1, K))
            kernels_i.append(psi.imag.view(1, 1, K))
        Kr = torch.cat(kernels_r, dim=0)  # [S,1,K]
        Ki = torch.cat(kernels_i, dim=0)  # [S,1,K]
        energy = torch.sqrt((Kr**2 + Ki**2).sum(dim=(1, 2), keepdim=True) + 1e-12)  # [S,1,1]
        Kr = Kr / energy
        Ki = Ki / energy
        self.Kr = Kr.to(device=device, dtype=dtype)
        self.Ki = Ki.to(device=device, dtype=dtype)
        self._built = True

    def forward(self, x, dt=None):
        assert x.dim() == 3 and x.size(1) == 1, "Se espera [B,1,T]."
        device, dtype = x.device, x.dtype
        if dt is None:
            dt = 1.0 / self.fs if self.fs is not None else 1.0
        dt = float(dt)
        if (not self._built) or (self.Kr is None) or (self.Kr.device != device) or (self.Kr.dtype != dtype):
            self._build_kernels(device, dtype, dt)
        yr = torch.nn.functional.conv1d(x, self.Kr, padding=self.kernel_size // 2)
        yi = torch.nn.functional.conv1d(x, self.Ki, padding=self.kernel_size // 2)
        mag = torch.sqrt(yr**2 + yi**2 + 1e-12)
        if self.return_phase:
            phase = torch.atan2(yi, yr)
            return mag, phase
        return mag

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

@pytest.mark.parametrize("f0", [1.0, 3.0, 6.0, 10.0])
def test_sine_localization(f0):
    fs, T = 30.0, 8.0
    n_bands, fmin, fmax, w0 = 12, 0.5, 12.0, 6.0
    t = make_time(fs, T)
    x = np.sin(2 * np.pi * f0 * t).astype(np.float32)
    xt = torch.from_numpy(x)[None, None, :]
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_bands).tolist()
    W = CWT1D(fs=fs, kernel_size=129, w0=w0, freqs_hz=freqs)(xt).detach().cpu().numpy()[0]
    tau = len(t) // 2
    s_idx = int(np.argmax(W[:, tau]))
    f_hat = freqs[s_idx]
    # tolerancia: a una banda de distancia en una rejilla logarítmica
    assert s_idx in range(0, n_bands)
    assert abs(np.log(f_hat) - np.log(f0)) <= (np.log(fmax) - np.log(fmin)) / n_bands + 1e-6

def test_pywt_consistency():
    try:
        import pywt
    except Exception:
        pytest.skip("PyWavelets no disponible.")
    fs, T, f0 = 30.0, 8.0, 6.0
    n_bands, fmin, fmax, w0, border = 12, 0.5, 12.0, 6.0, 16
    t = make_time(fs, T)
    x = np.sin(2 * np.pi * f0 * t).astype(np.float32)
    xt = torch.from_numpy(x)[None, None, :]
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_bands).tolist()
    W_t = CWT1D(fs=fs, kernel_size=129, w0=w0, freqs_hz=freqs)(xt).detach().cpu().numpy()[0]
    scales = (w0 / (2 * np.pi * np.array(freqs)))
    widths = scales * fs
    W_py, _ = pywt.cwt(x, widths, 'morl', sampling_period=1.0 / fs)
    S, Tn = W_py.shape
    sl = (slice(0, S), interior_slice(Tn, border))
    A = W_t[sl]; B = np.abs(W_py[sl])
    alpha = best_alpha(A, B)
    mse = float(np.mean((alpha * A - B) ** 2))
    corr = float(np.corrcoef(A.ravel(), B.ravel())[0, 1])
    assert corr >= 0.95
    assert mse <= 0.05

def test_chirp_ridge_mae():
    fs, T = 30.0, 8.0
    f1, f2 = 1.0, 12.0
    n_bands, w0, border = 16, 6.0, 16
    t = make_time(fs, T)
    k = (f2 - f1) / T
    phase = 2 * np.pi * (f1 * t + 0.5 * k * t**2)
    x = np.sin(phase).astype(np.float32)
    xt = torch.from_numpy(x)[None, None, :]
    freqs = np.logspace(np.log10(f1), np.log10(f2), n_bands).tolist()
    W = CWT1D(fs=fs, kernel_size=129, w0=w0, freqs_hz=freqs)(xt).detach().cpu().numpy()[0]
    ridge_idx = np.argmax(W, axis=0)
    f_hat_t = np.array([freqs[i] for i in ridge_idx])
    f_true = f1 + k * t
    sl = interior_slice(len(t), border)
    mae = float(np.mean(np.abs(f_hat_t[sl] - f_true[sl])))
    assert mae <= 1.0
