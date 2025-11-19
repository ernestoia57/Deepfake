import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _ensure_odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1

class CWT1D(nn.Module):
    """
    Continuous Wavelet Transform (CWT) 1D con wavelet Morlet COMPLEJA corregida.

      ψ_s,τ(t) = (π^{-1/4}/√s) * [exp(j ω0 (t-τ)/s) - exp(-ω0^2/2)] * exp(-(t-τ)^2 / (2 s^2))

    - Correlación implementada vía conv1d con kernel invertido (flip).
    - Normalización por escala configurable (L2 por defecto) además del factor teórico 1/√s.
    - Padding 'reflect' por defecto para reducir el cono de influencia en bordes.
    - Presets por defecto optimizados para detección de deepfakes (30 FPS aprox.).

    Parámetros
    ----------
    fs : float
        Frecuencia de muestreo (Hz). Relevante si defines bandas por frecuencia (fmin, fmax, n_bands).
    kernel_size : int
        Longitud del kernel (impar). Por defecto 161 (buena resolución en frecuencia).
    w0 : float
        Frecuencia central adimensional de Morlet. Por defecto 6.5 (ligeramente >6 para mejor resolución en f).
    scales : list[float], opcional
        Escalas (segundos). Si se suministra, ignora fmin/fmax/n_bands.
    freqs_hz : list[float], opcional
        Frecuencias centrales (Hz). Si se suministra, ignora fmin/fmax/n_bands.
    fmin, fmax : float
        Rango de frecuencias a cubrir si no pasas escalas/freqs (por defecto 0.5–12 Hz).
    n_bands : int
        Número de bandas log-espaciadas en [fmin, fmax] (por defecto 16).
    norm_mode : {"l2","none"}
        Normalización adicional por banda (además de 1/√s). "l2" por defecto.
    pad_mode : {"reflect","constant","replicate", None}
        Estrategia de padding antes de conv (None = padding interno de conv). Por defecto "reflect".
    return_phase : bool
        Si True, devuelve (magnitud, fase). Por defecto False (solo magnitud).

    Entrada
    -------
    x : [B, 1, T]

    Salida
    ------
    |W| : [B, S, T]   (y opcionalmente phase [B, S, T])
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
        pad_mode: str = "reflect",
        return_phase: bool = False,
    ):
        super().__init__()
        self.fs = float(fs) if fs is not None else None
        self.kernel_size = _ensure_odd(int(kernel_size))
        self.w0 = float(w0)
        self.return_phase = bool(return_phase)
        self.norm_mode = norm_mode
        self.pad_mode = pad_mode

        # --- definir escalas ---
        if (scales is None) and (freqs_hz is None):
            # bandas log en [fmin, fmax]
            assert self.fs is not None, "Debes especificar fs para generar bandas por frecuencia."
            # usar torch.logspace para reproducibilidad en GPU si fuese necesario
            freqs_hz = torch.logspace(
                math.log10(fmin), math.log10(fmax), steps=int(n_bands)
            ).tolist()

        if scales is None:
            assert self.fs is not None, "Para freqs_hz necesitas fs."
            # Relación Morlet compleja: f_c ≈ w0 / (2π s)  →  s = w0 / (2π f_c)
            scales = [self.w0 / (2.0 * math.pi * float(f)) for f in freqs_hz]

        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32), persistent=False)

        # Constantes Morlet
        self.corr_const = math.exp(-0.5 * (self.w0 ** 2))  # término de corrección para media≈0
        self.pi_norm = math.pi ** (-0.25)

        # kernels (se construyen lazily en el primer forward y al cambiar device/dtype)
        self._built = False
        self.register_buffer("Kr", None, persistent=False)
        self.register_buffer("Ki", None, persistent=False)

    # ----------------------
    # construcción de kernels
    # ----------------------
    def _build_kernels(self, device, dtype, dt: float):
        S = self.scales.to(device=device, dtype=torch.float32)
        K = self.kernel_size
        half = K // 2
        # eje temporal (segundos) centrado
        t = torch.arange(-half, half + 1, device=device, dtype=torch.float32) * dt  # [K]

        kernels_r = []
        kernels_i = []
        for s in S:
            # Morlet compleja corregida
            gauss = torch.exp(-0.5 * (t / s) ** 2)                            # [K]
            carrier = torch.exp(1j * (self.w0 * (t / s)))                     # [K] complejo
            psi = (self.pi_norm / torch.sqrt(s)) * (carrier - self.corr_const) * gauss  # [K] complejo
            # correlación -> kernel invertido
            psi = psi.flip(0)
            kernels_r.append(psi.real.view(1, 1, K))
            kernels_i.append(psi.imag.view(1, 1, K))

        Kr = torch.cat(kernels_r, dim=0)  # [S,1,K]
        Ki = torch.cat(kernels_i, dim=0)  # [S,1,K]

        # Normalización adicional por escala (además del 1/√s ya incluido):
        if self.norm_mode == "l2":
            # norma L2 del kernel complejo discreto por banda
            energy = torch.sqrt((Kr**2 + Ki**2).sum(dim=(1, 2), keepdim=True) + 1e-12)  # [S,1,1]
            Kr = Kr / energy
            Ki = Ki / energy
        elif self.norm_mode == "none":
            pass
        else:
            raise ValueError(f"norm_mode desconocido: {self.norm_mode}")

        self.Kr = Kr.to(device=device, dtype=dtype)  # [S,1,K]
        self.Ki = Ki.to(device=device, dtype=dtype)
        self._built = True

    # -----------
    # forward CWT
    # -----------
    def forward(self, x: torch.Tensor, dt: float = None):
        """
        x: [B, 1, T]
        dt: periodo de muestreo (s). Si None y fs está definido, usa dt=1/fs.
        """

        assert x.dim() == 3 and x.size(1) == 1, "Se espera [B,1,T]."
        device, dtype = x.device, x.dtype
        if dt is None:
            dt = 1.0 / self.fs if self.fs is not None else 1.0
        dt = float(dt)

        if (not self._built) or (self.Kr is None) or (self.Kr.device != device) or (self.Kr.dtype != dtype):
            self._build_kernels(device, dtype, dt)

        pad = self.kernel_size // 2
        T = x.size(-1)

        # reflect/replicate requieren pad < T; si no, hacemos fallback a padding interno de conv1d
        can_explicit_pad = (self.pad_mode is not None) and (pad < T)

        if can_explicit_pad:
            x_pad = F.pad(x, (pad, pad), mode=self.pad_mode)
            conv_pad = 0
        else:
            x_pad = x
            conv_pad = pad  # zero-padding interno de conv1d (si self.pad_mode es None o pad>=T)

        yr = F.conv1d(x_pad, self.Kr, padding=conv_pad)
        yi = F.conv1d(x_pad, self.Ki, padding=conv_pad)
        mag = torch.sqrt(yr**2 + yi**2 + 1e-12)
        if self.return_phase:
            phase = torch.atan2(yi, yr)
            return mag, phase
        return mag

