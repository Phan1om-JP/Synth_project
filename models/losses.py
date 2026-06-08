import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_l1(pred, target, mask):
    loss = torch.abs(pred - target) * mask
    return loss.sum() / (mask.sum() + 1e-8)


def gan_generator_loss(disc_fake):
    return F.mse_loss(disc_fake, torch.ones_like(disc_fake))


def gan_discriminator_loss(disc_real, disc_fake):
    loss_real = F.mse_loss(disc_real, torch.ones_like(disc_real))
    loss_fake = F.mse_loss(disc_fake, torch.zeros_like(disc_fake))
    return (loss_real + loss_fake) * 0.5


# ---------------------------------------------------------------------------
# Hybrid loss components
# ---------------------------------------------------------------------------

def _gaussian_window_2d(size: int, sigma: float, device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g /= g.sum()
    return g.outer(g)


def ssim_loss(pred: torch.Tensor, target: torch.Tensor,
              mask: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """1 - SSIM on masked regions. Input is z-scored CT (training space)."""
    B, C, H, W = pred.shape
    win = _gaussian_window_2d(window_size, sigma=1.5, device=pred.device)
    win = win.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
    pad = window_size // 2

    mu_p  = F.conv2d(pred,            win, padding=pad, groups=C)
    mu_t  = F.conv2d(target,          win, padding=pad, groups=C)
    sig_p = F.conv2d(pred   * pred,   win, padding=pad, groups=C) - mu_p * mu_p
    sig_t = F.conv2d(target * target, win, padding=pad, groups=C) - mu_t * mu_t
    sig_pt= F.conv2d(pred   * target, win, padding=pad, groups=C) - mu_p * mu_t

    # C1/C2 stabilise the denominator; small values work for z-score space
    C1, C2   = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)) / (
                (mu_p ** 2 + mu_t ** 2 + C1) * (sig_p + sig_t + C2))

    return (1.0 - ssim_map.clamp(-1.0, 1.0))[mask.bool()].mean()


def gradient_loss(pred: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
    """Sobel edge-difference loss — sharpens bone/soft-tissue boundaries."""
    sobel_x = torch.tensor(
        [[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]],
        device=pred.device,
    ).expand(pred.shape[1], 1, -1, -1)
    sobel_y = torch.tensor(
        [[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]],
        device=pred.device,
    ).expand(pred.shape[1], 1, -1, -1)

    C = pred.shape[1]
    gp_x = F.conv2d(pred,   sobel_x, padding=1, groups=C)
    gp_y = F.conv2d(pred,   sobel_y, padding=1, groups=C)
    gt_x = F.conv2d(target, sobel_x, padding=1, groups=C)
    gt_y = F.conv2d(target, sobel_y, padding=1, groups=C)

    return (masked_l1(gp_x, gt_x, mask) + masked_l1(gp_y, gt_y, mask)) * 0.5


def frequency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Log-FFT magnitude loss — captures high-frequency bone texture."""
    pred_mag = torch.abs(torch.fft.rfft2(pred))
    targ_mag = torch.abs(torch.fft.rfft2(target))
    return F.l1_loss(torch.log(pred_mag + 1.0), torch.log(targ_mag + 1.0))


def residual_skip(pred: torch.Tensor, inp: torch.Tensor, stats: dict) -> torch.Tensor:
    """Re-express CBCT input in CT z-score space and add to model output.

    The model learns only the correction Δ (CBCT→CT residual). For multi-channel
    inputs (2.5D), the centre channel is used as the residual carrier.

    Only call this when task == "task2" (stats must contain cbct_global_* keys).
    """
    n_ch = inp.shape[1]
    inp_center = inp[:, n_ch // 2 : n_ch // 2 + 1, ...]
    cbct_hu = inp_center * stats["cbct_global_std"] + stats["cbct_global_mean"]
    inp_ct  = (cbct_hu - stats["ct_global_mean"]) / (stats["ct_global_std"] + 1e-8)
    return pred + inp_ct


class HybridLoss(nn.Module):
    """
    Uncertainty-weighted hybrid loss: L1 + SSIM + gradient + frequency.

    Weights are learned automatically via homoscedastic uncertainty
    (Kendall et al., CVPR 2018):
        L_total = Σᵢ  exp(−log_varᵢ) · Lᵢ  +  log_varᵢ

    Larger log_varᵢ → smaller weight on that term. log_vars are clamped
    to [−4, 4] so weights stay in [0.018, 54.6] (prevents collapse).

    Usage:
        loss_fn = HybridLoss().to(device)
        opt = Adam(list(model.params()) + list(loss_fn.params()), lr=lr)
        loss, info = loss_fn(pred, target, mask)
    """

    def __init__(self):
        super().__init__()
        # All four losses start equally weighted (log_var=0 → weight=1)
        self.log_vars = nn.Parameter(torch.zeros(4))

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor):
        l1   = masked_l1(pred, target, mask)
        s    = ssim_loss(pred, target, mask)
        grad = gradient_loss(pred, target, mask)
        freq = frequency_loss(pred, target)

        losses  = torch.stack([l1, s, grad, freq])
        lv      = self.log_vars.clamp(-4.0, 4.0)
        weights = torch.exp(-lv)
        total   = (weights * losses + lv).sum()

        return total, {
            "l1"    : l1.item(),
            "ssim"  : s.item(),
            "grad"  : grad.item(),
            "freq"  : freq.item(),
            "w_l1"  : weights[0].item(),
            "w_ssim": weights[1].item(),
            "w_grad": weights[2].item(),
            "w_freq": weights[3].item(),
        }
