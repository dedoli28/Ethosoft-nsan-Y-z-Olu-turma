"""
FusionFaceGAN — Orijinal Model Mimarisi
Bu dosya Generator ve yardımcı modüllerin tanımlarını içerir.
Arayüz (app.py) bu dosyayı kullanır.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveChannelAttention(nn.Module):
    """Adaptive Channel Attention — Orijinal, gamma warm-up."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False), nn.ReLU(True),
            nn.Linear(mid, channels, bias=False), nn.Hardsigmoid(True))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C = x.size(0), x.size(1)
        w = self.fc(self.pool(x).view(B, C)).view(B, C, 1, 1)
        return x * (1 + self.gamma * (w - 1))


class PRU(nn.Module):
    """Progressive Residual Upsample — Orijinal."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.skip = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                   nn.BatchNorm2d(out_ch)) if in_ch != out_ch else nn.Identity()
        self.aca = AdaptiveChannelAttention(out_ch)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.aca(F.relu(self.conv(x) + self.skip(x), True))


class FusionGenerator(nn.Module):
    """
    Multi-Scale Fusion Generator — Orijinal mimari.

    Akış: z(128) → 4x4 → PRU→8x8 → PRU→16x16 → PRU→32x32 → PRU→64x64
    Her seviyeden RGB çıktı, öğrenilebilir fusion ağırlıklarıyla birleştirilir.
    """
    def __init__(self, latent_dim=128, g_base_ch=64, num_channels=3):
        super().__init__()
        ch = g_base_ch
        nz = latent_dim
        nc = num_channels

        self.proj = nn.Sequential(
            nn.ConvTranspose2d(nz, ch*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ch*8), nn.ReLU(True))

        self.up1 = PRU(ch*8, ch*4)
        self.up2 = PRU(ch*4, ch*2)
        self.up3 = PRU(ch*2, ch)
        self.up4 = PRU(ch, ch)

        self.toRGB_8  = nn.Conv2d(ch*4, nc, 1)
        self.toRGB_16 = nn.Conv2d(ch*2, nc, 1)
        self.toRGB_32 = nn.Conv2d(ch, nc, 1)
        self.toRGB_64 = nn.Conv2d(ch, nc, 1)

        self.fusion_weights = nn.Parameter(torch.tensor([0.05, 0.1, 0.2, 1.0]))

    def forward(self, z):
        x = self.proj(z)
        x = self.up1(x); rgb8 = self.toRGB_8(x)
        x = self.up2(x); rgb16 = self.toRGB_16(x)
        x = self.up3(x); rgb32 = self.toRGB_32(x)
        x = self.up4(x); rgb64 = self.toRGB_64(x)

        w = F.softmax(self.fusion_weights, dim=0)
        fused = (w[0]*F.interpolate(rgb8, 64, mode='bilinear', align_corners=False) +
                 w[1]*F.interpolate(rgb16, 64, mode='bilinear', align_corners=False) +
                 w[2]*F.interpolate(rgb32, 64, mode='bilinear', align_corners=False) +
                 w[3]*rgb64)
        return torch.tanh(fused)
