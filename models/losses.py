import torch
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