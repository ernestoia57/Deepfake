"""Video frequency features. Input RGB uint8 [T,H,W,3], output float32 [bands,H,W].

Temporal FFT is along AXIS 0 (time), NOT frame-wise 2D spatial FFT.
CWT is exploratory, not a demonstrated improvement. Both use exactly the same clip.
"""
import math
import cv2
import numpy as np
from scipy.signal import fftconvolve


def residual_video(frames, mode="median", kernel=5):
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Expected RGB video [T,H,W,3]")
    if frames.dtype != np.uint8:
        raise TypeError("Input frames must be uint8 RGB before frequency preprocessing")
    if kernel < 3 or kernel % 2 != 1:
        raise ValueError("Median kernel size must be odd and >= 3")
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames])
    if mode == "median":
        smooth = np.stack([cv2.medianBlur(f, kernel) for f in gray])
        residual = gray.astype(np.float32) - smooth.astype(np.float32)
    elif mode == "none":
        residual = gray.astype(np.float32)
    elif mode == "mean":
        smooth = np.stack([cv2.blur(f, (kernel, kernel)) for f in gray])
        residual = gray.astype(np.float32) - smooth.astype(np.float32)
    else:
        raise ValueError("mode must be median, mean or none")
    return residual / 255.0


def temporal_fft(frames, filter_mode="median", kernel=5):
    """Return T//2 channels for even T: discard DC, retain Nyquist."""
    residual = residual_video(frames, filter_mode, kernel)
    T = len(residual)
    if T < 8 or T % 2:
        raise ValueError("FFT requires even T >= 8")
    # Demean removes temporal DC; rfft computes ONLY along the time axis.
    residual = residual - residual.mean(axis=0, keepdims=True)
    spectrum = np.fft.rfft(residual, axis=0)
    magnitude = np.abs(spectrum[1:]) / math.sqrt(T)
    return np.log1p(magnitude).astype(np.float32)


def temporal_cwt(frames, fps, bands=8, fmin=3.0, fmax=10.0,
                 filter_mode="median", kernel=5, w0=6.0):
    """Exploratory Morlet-CWT magnitude maps pooled over valid center times.

    Finite clips have a cone-of-influence problem; reject bands requiring
    a wavelet half-support longer than half the clip. Output [bands,H,W].
    """
    if not (np.isfinite(fps) and fps > 0):
        raise ValueError("Positive fps required")
    if fmin <= 0 or fmax <= fmin or fmax >= fps / 2:
        raise ValueError("Frequency range must be inside (0, Nyquist)")
    x = residual_video(frames, filter_mode, kernel)
    x = x - x.mean(axis=0, keepdims=True)
    T = len(x)
    freqs = np.geomspace(fmin, fmax, bands)
    maps = []
    for f in freqs:
        scale_samples = w0 * fps / (2 * np.pi * f)
        half_support = int(math.ceil(3 * scale_samples))
        # no pretend valid center coefficients where wavelet meets the clip edge
        if 2 * half_support >= T:
            raise ValueError(
                f"CWT band {f:.2f} Hz cannot be estimated from T={T}, fps={fps:.2f}; "
                "increase clip length or fmin"
            )
        tau = np.arange(-half_support, half_support + 1, dtype=np.float64)
        z = tau / scale_samples
        wavelet = np.exp(-0.5 * z*z) * (
            np.exp(1j*w0*z) - np.exp(-0.5*w0*w0)
        ) / np.sqrt(scale_samples)
        wavelet = wavelet / np.sqrt(np.sum(np.abs(wavelet)**2))
        coef = fftconvolve(x, np.conj(wavelet[::-1, None, None]), mode="same", axes=0)
        valid = np.abs(coef[half_support:T-half_support])
        maps.append(np.log1p(valid.mean(axis=0)).astype(np.float32))
    return np.stack(maps)
