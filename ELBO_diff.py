import math, os, csv, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# -------- Utils: beta schedule (cosine) ----------
def betas_for_alpha_bar(T, max_beta=0.999):
    def alpha_bar(t):
        # cosine schedule (Nichol & Dhariwal)
        s = 0.008
        return np.cos(( (t/T + s)/(1+s) ) * math.pi/2)**2
    betas=[]
    for i in range(T):
        t1 = i/T
        t2 = (i+1)/T
        betas.append( min(1 - alpha_bar(t2)/alpha_bar(t1), max_beta) )
    return torch.tensor(betas, dtype=torch.float32)

# --------- Tiny ε-UNet 64x64 (mínimo) ----------
class SinusoidalTimeEmbedding(nn.Module):
    def _init_(self, dim):
        super()._init_()
        self.dim = dim
    def forward(self, t):  # t: (B,)
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(10000.0), half, device=device)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:  # padding si impar
            emb = F.pad(emb, (0,1))
        return emb

class ResBlock(nn.Module):
    def _init_(self, c_in, c_out, t_dim):
        super()._init_()
        self.norm1 = nn.GroupNorm(8, c_in); self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, c_out); self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.time = nn.Linear(t_dim, c_out)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in!=c_out else nn.Identity()
    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(F.silu(h))))
        return h + self.skip(x)

class TinyUNet(nn.Module):
    def _init_(self, in_ch=1, base=32, t_dim=128):
        super()._init_()
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
    def forward(self, x, t):  # t: (B,) integers
        t_emb = self.time_emb(t.float())
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

# ------ Dataset sintético placeholder (reemplaza por tu DataLoader) ------
class ToyDataset(Dataset):
    def _init_(self, n=8192, size=64):
        rng = np.random.RandomState(0)
        imgs = []
        for _ in range(n):
            x = rng.randn(1, size, size).astype(np.float32)*0.2
            # añade un circulito suave para que haya estructura
            cx, cy = rng.randint(16, size-16), rng.randint(16, size-16)
            rr, cc = np.ogrid[:size, :size]
            mask = (rr-cx)*2 + (cc-cy)*2 <= rng.randint(20,100)
            x[0, mask] += rng.uniform(0.7, 1.3)
            imgs.append(np.clip(x, -1, 1))
        self.x = torch.from_numpy(np.stack(imgs))
    def _len_(self): return len(self.x)
    def _getitem_(self, i): return self.x[i]

# ------------- ELBO helpers ----------------
def normal_kl(mean1, logvar1, mean2, logvar2):
    # KL(N1||N2) por elemento
    return 0.5 * (logvar2 - logvar1 + (torch.exp(logvar1) + (mean1-mean2)**2)/torch.exp(logvar2) - 1)

def discretized_gaussian_nll(x, mean, logvar):
    # NLL ~ 0.5*[(x-mean)^2/var + logvar] + const (const ignorado para monitoreo)
    return 0.5*((x-mean)**2*torch.exp(-logvar) + logvar)

def compute_vlb_terms(x0, eps_hat, xt, t, betas, alphas, abar):
    """
    Devuelve L_T, KL_t (si t>1) ó L0 (si t=1) por muestra.
    """
    B = x0.size(0)
    device = x0.device
    # reshape params
    beta_t     = betas[t-1].view(B, 1, 1, 1)
    alpha_t    = alphas[t-1].view(B, 1, 1, 1)
    abar_t     = abar[t-1].view(B, 1, 1, 1)
    abar_t_1   = torch.where(t>1, abar[t-2], abar[t-1]).view(B,1,1,1)  # safe for t=1
    # Posterior q(x_{t-1}|x_t,x0): mean_tilde, var_tilde
    mean_tilde = ( torch.sqrt(abar_t_1)*beta_t/(1. - abar_t) )*x0 + \
                 ( torch.sqrt(alpha_t)*(1.-abar_t_1)/(1.-abar_t) )*xt
    var_tilde  = (1.-abar_t_1)/(1.-abar_t) * beta_t
    logvar_tilde = torch.log(var_tilde + 1e-12)
    # Modelo p_theta(x_{t-1}|x_t): mean de DDPM con var = var_tilde (fija)
    mean_theta = (1./torch.sqrt(alpha_t))*(xt - (beta_t/torch.sqrt(1.-abar_t))*eps_hat)

    # L_T
    abar_T = abar[-1]
    LT = 0.5*(
        (abar_T/(1-abar_T))* (x0**2)
        + torch.log(1-abar_T)*torch.ones_like(x0)
    )  # up to constant
    LT = LT.flatten(1).mean(dim=1)  # por muestra

    # Selección por t
    is_t1 = (t==1).float().view(B,1,1,1)
    is_gt1= 1.-is_t1

    # KL para t>1
    KL_t = normal_kl(mean_tilde, logvar_tilde, mean_theta, logvar_tilde)  # var iguales -> KL = (mu diff)^2/(2var)
    KL_t = KL_t.flatten(1).mean(dim=1) * is_gt1.view(B)  # por muestra

    # L0 para t=1 : -log p(x0 | x1) ~ NLL con var ~ beta_1
    x0_hat = (xt - torch.sqrt(1.-abar_t)*eps_hat) / torch.sqrt(abar_t)
    logvar0 = torch.log(betas[0]).view(1,1,1,1)  # usar beta_1
    L0 = discretized_gaussian_nll(x0, x0_hat, logvar0).flatten(1).mean(dim=1) * is_t1.view(B)

    return LT, KL_t, L0

# ------------- Entrenamiento ---------------
def main():
    outdir = Path("./")
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparams
    image_size = 64
    in_ch = 1
    T = 300  # pasos de difusión (demo)
    batch_size = 64
    epochs = 3
    steps_per_epoch = 200  # recorta por velocidad de demo
    lr = 2e-4

    # Betas y derivados
    betas = betas_for_alpha_bar(T).to(device)
    alphas = (1.0 - betas)
    abar = torch.cumprod(alphas, dim=0)

    # Modelo
    model = TinyUNet(in_ch=in_ch, base=32, t_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # Data
    ds = ToyDataset(n=steps_per_epoch*batch_size, size=image_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # Log
    log_csv = open("training_log.csv", "w", newline="")
    writer = csv.writer(log_csv); writer.writerow(["step","epoch","L_simple","ELBO_est"])

    global_step = 0
    elbo_curve = []

    model.train()
    for epoch in range(1, epochs+1):
        it = 0
        for x0 in dl:
            it += 1
            if it>steps_per_epoch: break
            global_step += 1
            x0 = x0.to(device)

            # sample t, noise
            B = x0.size(0)
            t = torch.randint(1, T+1, (B,), device=device)  # 1..T
            eps = torch.randn_like(x0)
            abar_t = abar[t-1].view(B, 1, 1, 1)
            xt = torch.sqrt(abar_t)*x0 + torch.sqrt(1. - abar_t)*eps

            # pred noise
            eps_hat = model(xt, t)

            # loss simple
            loss = F.mse_loss(eps_hat, eps)

            # grad
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            # --- ELBO (estimado MC) para logging ---
            with torch.no_grad():
                LT, KL_t, L0 = compute_vlb_terms(x0, eps_hat, xt, t, betas, alphas, abar)
                # estimador: L_T + KL_t (si t>1) ó L0 (si t=1)
                elbo_est = LT + KL_t + L0
                elbo_mean = elbo_est.mean().item()
                elbo_curve.append(elbo_mean)

            if global_step % 20 == 0:
                writer.writerow([global_step, epoch, float(loss.item()), elbo_mean])

        print(f"[Epoch {epoch}] L_simple={loss.item():.4f} | ELBO_est={elbo_mean:.4f}")

    log_csv.close()

    # --------- Plot ELBO ----------
    plt.figure()
    plt.plot(elbo_curve)
    plt.xlabel("Paso de entrenamiento")
    plt.ylabel("ELBO (estimado, ↓ mejor)")
    plt.title("Evolución del Lower Bound (DDPM demo)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("diffusion_elbo.png", dpi=150)
    print("Guardado: diffusion_elbo.png y training_log.csv")

if _name_ == "_main_":
    main()