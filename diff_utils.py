# diff_utils.py
from __future__ import annotations

import math, csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

__all__ = [
    # schedules
    "betas_for_alpha_bar", "build_schedule",
    # forward diffusion helpers
    "sample_xt", "ddpm_mean_theta",
    # ELBO / VLB terms
    "normal_kl", "discretized_gaussian_nll", "compute_vlb_terms",
    "elbo_per_timestep", "accumulate_per_timestep", "estimate_elbo_mc",
    # training utils
    "train_step", "print_summary", "plot_elbo_curve", "write_training_row",
    # tiny reference model & dataset (opcionales)
    "SinusoidalTimeEmbedding", "ResBlock", "TinyUNet",
    "ToyDataset",
]

# ------------------------------
# Schedule (cosine) y derivados
# ------------------------------

def betas_for_alpha_bar(T: int, max_beta: float = 0.999) -> torch.Tensor:
    """
    Cosine schedule de Nichol & Dhariwal: produce betas[t] size [T].
    """
    def alpha_bar(t: float) -> float:
        s = 0.008
        return np.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2
    betas = []
    for i in range(T):
        t1 = i / T
        t2 = (i + 1) / T
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def build_schedule(T: int, device: torch.device,
                   beta_start_end: Optional[Tuple[float, float]] = None) -> Dict[str, torch.Tensor]:
    """
    Construye el diccionario de schedule con campos útiles.
    Por default usa cosine (como en tu script). Si pasas beta_start_end, usa lineal.
    """
    if beta_start_end is None:
        betas = betas_for_alpha_bar(T).to(device)
    else:
        b0, b1 = beta_start_end
        betas = torch.linspace(b0, b1, T, device=device)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    return dict(
        betas=betas,
        alphas=alphas,
        abar=abar,
        sqrt_alphas=torch.sqrt(alphas),
        sqrt_abar=torch.sqrt(abar),
        sqrt_one_minus_abar=torch.sqrt(1.0 - abar),
        log_betas=torch.log(betas.clamp_min(1e-12)),
    )

# ---------------------------------
# Forward diffusion & mean DDPM
# ---------------------------------

def sample_xt(x0: torch.Tensor, t: torch.Tensor, sched: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x_t = sqrt(abar_t)*x0 + sqrt(1-abar_t)*eps, eps~N(0,I)
    Devuelve (x_t, eps) para poder usar eps_true en la recon.
    """
    B = x0.size(0)
    abar_t = sched["abar"][t - 1].view(B, 1, 1, 1)
    x_eps = torch.randn_like(x0)
    xt = torch.sqrt(abar_t) * x0 + torch.sqrt(1.0 - abar_t) * x_eps
    return xt, x_eps


def ddpm_mean_theta(eps_hat: torch.Tensor, xt: torch.Tensor, t: torch.Tensor,
                    sched: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Media del modelo p_theta(x_{t-1}|x_t) para DDPM con var fija=beta_t:
      mu_theta = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-abar_t) * eps_hat)
    """
    B = xt.size(0)
    alpha_t = sched["alphas"][t - 1].view(B, 1, 1, 1)
    beta_t = (1.0 - alpha_t)
    abar_t = sched["abar"][t - 1].view(B, 1, 1, 1)
    mu = (1.0 / torch.sqrt(alpha_t)) * (xt - (beta_t / torch.sqrt(1.0 - abar_t)) * eps_hat)
    return mu

# ---------------------------------
# Términos ELBO (DDPM eps-pred)
# ---------------------------------

def normal_kl(mean1: torch.Tensor, logvar1: torch.Tensor,
              mean2: torch.Tensor, logvar2: torch.Tensor) -> torch.Tensor:
    """
    KL(N1||N2) por elemento. Devuelve same-shape tensor (no reduce).
    """
    var1 = logvar1.exp()
    var2 = logvar2.exp()
    return 0.5 * (logvar2 - logvar1 + (var1 + (mean1 - mean2) ** 2) / var2 - 1.0)


def discretized_gaussian_nll(x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    NLL ~ 0.5 * ((x-mean)^2 / var + logvar) (ignorando constantes)
    Devuelve same-shape tensor (no reduce).
    """
    inv_var = torch.exp(-logvar)
    return 0.5 * ((x - mean) ** 2 * inv_var + logvar)


@torch.no_grad()
def compute_vlb_terms(
    x0: torch.Tensor,
    eps_hat: torch.Tensor,
    xt: torch.Tensor,
    t: torch.Tensor,                        # enteros en [1..T]
    sched: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Devuelve (L_T, KL_t, L0) por muestra, shape [B] cada uno.
    - L_T: término de prior al final de la cadena (aprox, como en tu script).
    - KL_t: para t>1, KL(q(x_{t-1}|x_t,x0) || p_theta)
    - L0: para t=1, -log p(x0|x1)
    """
    device = x0.device
    B = x0.size(0)

    betas = sched["betas"]
    alphas = sched["alphas"]
    abar = sched["abar"]

    beta_t = betas[t - 1].view(B, 1, 1, 1)
    alpha_t = alphas[t - 1].view(B, 1, 1, 1)
    abar_t  = abar[t - 1].view(B, 1, 1, 1)

    # abar_{t-1} con guardia en t=1
    abar_tm1_flat = torch.where(t > 1, abar[t - 2], abar[t - 1])
    abar_tm1 = abar_tm1_flat.view(B, 1, 1, 1)

    # Posterior q(x_{t-1}|x_t,x0): media y varianza cerradas (DDPM)
    mean_tilde = (torch.sqrt(abar_tm1) * beta_t / (1.0 - abar_t)) * x0 + \
                 (torch.sqrt(alpha_t) * (1.0 - abar_tm1) / (1.0 - abar_t)) * xt
    var_tilde = (1.0 - abar_tm1) / (1.0 - abar_t) * beta_t
    logvar_tilde = torch.log(var_tilde + 1e-12)

    # Modelo con var fija=var_tilde (mismas varianzas para KL):
    mean_theta = ddpm_mean_theta(eps_hat, xt, t, sched)

    # L_T (prior al final) — misma forma que tu script (up to const)
    abar_T = abar[-1]
    LT = 0.5 * (
        (abar_T / (1 - abar_T)) * (x0 ** 2) + torch.log(1 - abar_T) * torch.ones_like(x0)
    )
    LT = LT.flatten(1).mean(dim=1)  # [B]

    # Flags t==1 o >1
    is_t1 = (t == 1).float().view(B, 1, 1, 1)
    is_gt1 = 1.0 - is_t1

    # KL para t>1 (var iguales -> KL = (mu diff)^2/(2 var))
    KL_t = normal_kl(mean_tilde, logvar_tilde, mean_theta, logvar_tilde)
    KL_t = KL_t.flatten(1).mean(dim=1) * is_gt1.view(B)

    # L0 para t=1 : NLL con var≈beta_1
    x0_hat = (xt - torch.sqrt(1.0 - abar_t) * eps_hat) / torch.sqrt(abar_t)
    logvar0 = torch.log(betas[0].clamp_min(1e-12)).view(1, 1, 1, 1)
    L0 = discretized_gaussian_nll(x0, x0_hat, logvar0).flatten(1).mean(dim=1) * is_t1.view(B)

    return LT, KL_t, L0


# ---------------------------------
# ELBO por paso y acumulación
# ---------------------------------

@dataclass
class StepTerms:
    t: int
    LT: torch.Tensor     # [B]
    KL_t: torch.Tensor   # [B]
    L0: torch.Tensor     # [B]
    total: torch.Tensor  # [B] = LT + KL_t + L0

@dataclass
class AccumulatedELBO:
    LT_sum: torch.Tensor   # [B]
    KL_sum: torch.Tensor   # [B]
    L0_sum: torch.Tensor   # [B]
    elbo_sum: torch.Tensor # [B] (↓ mejor)
    per_step: List[StepTerms]

@torch.no_grad()
def elbo_per_timestep(
    model_eps: nn.Module,
    x0: torch.Tensor,
    T: int,
    sched: Dict[str, torch.Tensor],
    t_order: str = "desc",
) -> List[StepTerms]:
    """
    Calcula VLB términos para TODOS los t (1..T) sobre el mismo batch x0.
    Útil para curvas por paso. Usa una muestra fresh de eps para cada t.
    """
    device = x0.device
    steps = list(range(1, T + 1))
    if t_order == "desc":
        steps = steps[::-1]

    out: List[StepTerms] = []
    for t_int in steps:
        t = torch.full((x0.size(0),), t_int, device=device, dtype=torch.long)
        xt, eps_true = sample_xt(x0, t, sched)
        eps_hat = model_eps(xt, t)
        LT, KL_t, L0 = compute_vlb_terms(x0, eps_hat, xt, t, sched)
        total = LT + KL_t + L0
        out.append(StepTerms(t=t_int, LT=LT, KL_t=KL_t, L0=L0, total=total))
    return out

@torch.no_grad()
def accumulate_per_timestep(steps: List[StepTerms]) -> AccumulatedELBO:
    LT = torch.stack([s.LT for s in steps], dim=0)     # [T,B]
    KL = torch.stack([s.KL_t for s in steps], dim=0)   # [T,B]
    L0 = torch.stack([s.L0 for s in steps], dim=0)     # [T,B]
    LT_sum = LT.sum(dim=0)
    KL_sum = KL.sum(dim=0)
    L0_sum = L0.sum(dim=0)
    elbo_sum = LT_sum + KL_sum + L0_sum
    return AccumulatedELBO(LT_sum, KL_sum, L0_sum, elbo_sum, steps)

@torch.no_grad()
def estimate_elbo_mc(
    model_eps: nn.Module,
    x0: torch.Tensor,
    sched: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Estimador MC por muestra como en tu training loop:
      ELBO_est = L_T + KL_t (t>1) ó L0 (t=1), con t ~ U{1..T}
    Devuelve [B].
    """
    B = x0.size(0)
    T = sched["betas"].numel()
    t = torch.randint(1, T + 1, (B,), device=x0.device)
    xt, _ = sample_xt(x0, t, sched)
    eps_hat = model_eps(xt, t)
    LT, KL_t, L0 = compute_vlb_terms(x0, eps_hat, xt, t, sched)
    return LT + KL_t + L0  # [B]

# ---------------------------------
# Training helpers
# ---------------------------------

def train_step(model_eps: nn.Module, x0: torch.Tensor, sched: Dict[str, torch.Tensor],
               optimizer: torch.optim.Optimizer) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Un paso de entrenamiento DDPM (loss simple = MSE de ε) + ELBO_est para log.
    Devuelve (loss_simple, elbo_est_mean).
    """
    device = x0.device
    B = x0.size(0)
    T = sched["betas"].numel()

    t = torch.randint(1, T + 1, (B,), device=device)
    xt, eps_true = sample_xt(x0, t, sched)
    eps_hat = model_eps(xt, t)

    loss = F.mse_loss(eps_hat, eps_true)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        elbo_est = estimate_elbo_mc(model_eps, x0, sched).mean()

    return loss.detach(), elbo_est.detach()


def print_summary(acc: AccumulatedELBO, title: str = "ELBO summary"):
    def stats(x):
        return dict(mean=float(x.mean()), std=float(x.std()), min=float(x.min()), max=float(x.max()))
    print(f"== {title} ==")
    print("ELBO_sum (↓ mejor):", stats(acc.elbo_sum))
    print("LT_sum:", stats(acc.LT_sum))
    print("KL_sum:", stats(acc.KL_sum))
    print("L0_sum:", stats(acc.L0_sum))


def plot_elbo_curve(values: List[float], title="Evolución ELBO (↓ mejor)", path: Optional[str] = None):
    plt.figure()
    plt.plot(values)
    plt.xlabel("Paso de entrenamiento")
    plt.ylabel("ELBO (estimado)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
    plt.show()


def write_training_row(writer: csv.writer, step: int, epoch: int, loss_simple: float, elbo_est: float):
    writer.writerow([step, epoch, float(loss_simple), float(elbo_est)])

# ---------------------------------
# Modelo y dataset de referencia
# ---------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(10000.0), half, device=device))
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, t_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c_in); self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, c_out); self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.time = nn.Linear(t_dim, c_out)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(F.silu(h))))
        return h + self.skip(x)

class TinyUNet(nn.Module):
    """
    Mínima ε-UNet 64x64 para pruebas (predice epsilon).
    """
    def __init__(self, in_ch=1, base=32, t_dim=128):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(t_dim),
            nn.Linear(t_dim, t_dim), nn.SiLU(),
            nn.Linear(t_dim, t_dim)
        )
        # Down
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)
        self.rb1 = ResBlock(base, base, t_dim)
        self.down1 = nn.Conv2d(base, base*2, 4, 2, 1)
        self.rb2 = ResBlock(base*2, base*2, t_dim)
        self.down2 = nn.Conv2d(base*2, base*4, 4, 2, 1)
        self.rb3 = ResBlock(base*4, base*4, t_dim)
        # Up
        self.up1 = nn.ConvTranspose2d(base*4, base*2, 4, 2, 1)
        self.rb4 = ResBlock(base*4, base*2, t_dim)
        self.up2 = nn.ConvTranspose2d(base*2, base, 4, 2, 1)
        self.rb5 = ResBlock(base*2, base, t_dim)
        self.out = nn.Conv2d(base, in_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_emb(t)
        h0 = self.in_conv(x)
        h1 = self.rb1(h0, t_emb)
        d1 = self.down1(h1)
        h2 = self.rb2(d1, t_emb)
        d2 = self.down2(h2)
        h3 = self.rb3(d2, t_emb)
        u1 = self.up1(h3)
        h4 = self.rb4(torch.cat([u1, h2], dim=1), t_emb)
        u2 = self.up2(h4)
        h5 = self.rb5(torch.cat([u2, h1], dim=1), t_emb)
        return self.out(F.silu(h5))

class ToyDataset(torch.utils.data.Dataset):
    """
    Dataset sintético rápido para smoke tests (1 canal, 64x64).
    """
    def __init__(self, n=8192, size=64, seed=0):
        rng = np.random.RandomState(seed)
        imgs = []
        for _ in range(n):
            x = rng.randn(1, size, size).astype(np.float32) * 0.2
            cx, cy = rng.randint(16, size-16), rng.randint(16, size-16)
            rr, cc = np.ogrid[:size, :size]
            # círculo suave aproximado
            mask = (rr - cx) ** 2 + (cc - cy) ** 2 <= rng.randint(50, 200)
            x[0, mask] += rng.uniform(0.7, 1.3)
            imgs.append(np.clip(x, -1, 1))
        self.x = torch.from_numpy(np.stack(imgs))
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]
