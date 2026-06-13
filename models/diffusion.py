import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half   = self.dim // 2
        freqs  = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sinusoidal = SinusoidalEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        return self.mlp(self.sinusoidal(t))


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1  = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1  = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.norm2  = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip   = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act    = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.norm  = nn.GroupNorm(min(8, dim), dim)
        self.qkv   = nn.Conv2d(dim, dim * 3, 1)
        self.proj  = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = qkv.unbind(1)
        scale   = (C // self.heads) ** -0.5
        attn    = torch.softmax((q * scale) @ k.transpose(-2, -1), dim=-1)
        out     = (attn @ v).reshape(B, C, H, W)
        return x + self.proj(out)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, use_attn=False):
        super().__init__()
        self.res1    = ResBlock(in_ch,  out_ch, time_dim)
        self.res2    = ResBlock(out_ch, out_ch, time_dim)
        self.attn    = AttentionBlock(out_ch) if use_attn else nn.Identity()
        self.down    = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.res2(x, t)
        x = self.attn(x) if not isinstance(self.attn, nn.Identity) else self.attn(x)
        return self.down(x), x  # (downsampled, skip)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_dim, use_attn=False):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res1 = ResBlock(in_ch + skip_ch, out_ch, time_dim)
        self.res2 = ResBlock(out_ch, out_ch, time_dim)
        self.attn = AttentionBlock(out_ch) if use_attn else nn.Identity()

    def forward(self, x, skip, t):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t)
        x = self.res2(x, t)
        return self.attn(x) if not isinstance(self.attn, nn.Identity) else self.attn(x)


# ---------------------------------------------------------------------------
# Diffusion UNet
# ---------------------------------------------------------------------------

class DiffusionUNet(nn.Module):
    """
    Conditional UNet for DDPM.
    in_ch = 2: (noisy_ct | condition) concatenated on channel dim.
    """
    def __init__(self, in_ch=2, base_ch=64):
        super().__init__()
        ch   = base_ch
        tdim = ch * 4

        self.time_embed = TimeEmbedding(tdim)
        self.init_conv  = nn.Conv2d(in_ch, ch, 3, padding=1)

        # Encoder: ch -> 2ch -> 4ch -> 8ch
        self.down1 = DownBlock(ch,    ch*2, tdim, use_attn=False)
        self.down2 = DownBlock(ch*2,  ch*4, tdim, use_attn=False)
        self.down3 = DownBlock(ch*4,  ch*8, tdim, use_attn=True)

        # Bottleneck
        self.mid1 = ResBlock(ch*8, ch*8, tdim)
        self.mid_attn = AttentionBlock(ch*8)
        self.mid2 = ResBlock(ch*8, ch*8, tdim)

        # Decoder
        self.up3 = UpBlock(ch*8, ch*8, ch*4, tdim, use_attn=True)
        self.up2 = UpBlock(ch*4, ch*4, ch*2, tdim, use_attn=False)
        self.up1 = UpBlock(ch*2, ch*2, ch,   tdim, use_attn=False)

        self.out_norm = nn.GroupNorm(min(8, ch), ch)
        self.out_conv = nn.Conv2d(ch, 1, 1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        x = self.init_conv(x)

        x, s1 = self.down1(x, t_emb)
        x, s2 = self.down2(x, t_emb)
        x, s3 = self.down3(x, t_emb)

        x = self.mid1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb)

        x = self.up3(x, s3, t_emb)
        x = self.up2(x, s2, t_emb)
        x = self.up1(x, s1, t_emb)

        return self.out_conv(F.silu(self.out_norm(x)))


# ---------------------------------------------------------------------------
# Gaussian Diffusion (DDPM + DDIM)
# ---------------------------------------------------------------------------

class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_schedule="cosine"):
        self.T = timesteps
        betas  = self._make_betas(timesteps, beta_schedule)
        self.register(betas)

    def _make_betas(self, T, schedule):
        if schedule == "linear":
            return torch.linspace(1e-4, 0.02, T)
        # cosine schedule (Nichol & Dhariwal 2021)
        steps = T + 1
        t     = torch.linspace(0, T, steps) / T
        alpha_bar = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(0, 0.999)

    def register(self, betas):
        self.betas            = betas
        alphas                = 1.0 - betas
        self.alphas_cumprod   = torch.cumprod(alphas, dim=0)
        self.sqrt_acp         = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus   = (1.0 - self.alphas_cumprod).sqrt()

    def _to(self, device):
        if self.betas.device != device:
            self.betas          = self.betas.to(device)
            self.alphas_cumprod = self.alphas_cumprod.to(device)
            self.sqrt_acp       = self.sqrt_acp.to(device)
            self.sqrt_one_minus = self.sqrt_one_minus.to(device)

    def q_sample(self, x0, t, noise=None):
        """Forward: add noise to x0 at timestep t."""
        self._to(x0.device)
        if noise is None:
            noise = torch.randn_like(x0)
        acp  = self.sqrt_acp[t][:, None, None, None]
        nacp = self.sqrt_one_minus[t][:, None, None, None]
        return acp * x0 + nacp * noise, noise

    def p_losses(self, model, x0, condition, t):
        """Training loss: predict noise given noisy x and condition."""
        xt, noise = self.q_sample(x0, t)
        model_in  = torch.cat([xt, condition], dim=1)
        pred      = model(model_in, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def ddim_sample(self, model, condition, n_steps=50, eta=0.0):
        """
        Fast DDIM inference.
        condition: (B, 1, H, W) tensor (MR or CBCT), already on correct device.
        Returns: (B, 1, H, W) synthetic CT.
        """
        device = condition.device
        self._to(device)
        B, _, H, W = condition.shape

        # Evenly-spaced timestep subsequence
        step_size = self.T // n_steps
        timesteps = list(range(0, self.T, step_size))[::-1]

        x = torch.randn(B, 1, H, W, device=device)

        for i, t_val in enumerate(timesteps):
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            model_in = torch.cat([x, condition], dim=1)
            eps      = model(model_in, t_batch)

            acp  = self.alphas_cumprod[t_val]
            x0_pred = (x - self.sqrt_one_minus[t_val] * eps) / self.sqrt_acp[t_val]
            x0_pred = x0_pred.clamp(-4, 4)

            if i < len(timesteps) - 1:
                t_prev  = timesteps[i + 1]
                acp_prev = self.alphas_cumprod[t_prev]
                sigma   = eta * ((1 - acp_prev) / (1 - acp) * (1 - acp / acp_prev)).sqrt()
                x = acp_prev.sqrt() * x0_pred + (1 - acp_prev - sigma**2).sqrt() * eps
                if eta > 0:
                    x = x + sigma * torch.randn_like(x)
            else:
                x = x0_pred

        return x
