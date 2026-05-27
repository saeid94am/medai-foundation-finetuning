import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Standard UNet (Ronneberger et al. 2015) — from-scratch segmentation baseline.

    Used to establish the lower bound: how well can a task-specific model
    trained from scratch do compared to LoRA-fine-tuned MedSAM?

    Args:
        in_channels: 1 for grayscale ultrasound.
        out_channels: 1 for binary lesion mask.
        features: Channel widths at each encoder resolution.
        bilinear: Use bilinear upsampling instead of transposed convolutions.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list[int] | None = None,
        bilinear: bool = False,
    ) -> None:
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoder.append(_DoubleConv(ch, f))
            ch = f

        self.bottleneck = _DoubleConv(features[-1], features[-1] * 2)

        self.decoder_up = nn.ModuleList()
        self.decoder_conv = nn.ModuleList()
        for f in reversed(features):
            self.decoder_up.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.decoder_conv.append(_DoubleConv(f * 2, f))

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, bbox: torch.Tensor | None = None) -> torch.Tensor:
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)  # RGB → grayscale for 1-channel UNet
        skip_connections = []
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, conv, skip in zip(
            self.decoder_up, self.decoder_conv, reversed(skip_connections), strict=True
        ):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = conv(x)

        return self.final(x)
