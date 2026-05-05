import torch
import torch.nn as nn

from models.unet import UNet2D, UNet3D


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_ch=2):
        super().__init__()

        def block(ic, oc, stride=2, norm=True):
            layers = [nn.Conv2d(ic, oc, 4, stride=stride, padding=1, bias=not norm)]
            if norm:
                layers.append(nn.InstanceNorm2d(oc))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_ch, 64,  stride=2, norm=False),
            *block(64,    128, stride=2),
            *block(128,   256, stride=2),
            *block(256,   512, stride=1),
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
        )

    def forward(self, mr, ct):
        return self.model(torch.cat([mr, ct], dim=1))


def build_model(cfg):
    mode      = cfg["data"]["spatial_mode"]
    in_ch     = cfg["model"]["in_ch"]
    out_ch    = cfg["model"]["out_ch"]
    base_ch   = cfg["model"]["base_ch"]
    device    = cfg["device"]
    multi_gpu = cfg["training"]["multi_gpu"]

    if mode in ("2D", "2.5D"):
        n_in      = cfg["data"]["n_adjacent"] if mode == "2.5D" else in_ch
        generator = UNet2D(n_in, out_ch, base_ch)
    elif mode == "3D":
        generator = UNet3D(in_ch, out_ch, base_ch)
    else:
        raise ValueError(f"Unknown spatial_mode: {mode}")

    if multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel.")
        generator = nn.DataParallel(generator)

    generator = generator.to(device)
    n_params  = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f"Generator params: {n_params:,}")

    discriminator = None
    if cfg["training"]["loss_type"] == "l1_gan":
        discriminator = PatchGANDiscriminator(in_ch=2)
        if multi_gpu and torch.cuda.device_count() > 1:
            discriminator = nn.DataParallel(discriminator)
        discriminator = discriminator.to(device)
        n_params_d = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
        print(f"Discriminator params: {n_params_d:,}")

    return generator, discriminator
