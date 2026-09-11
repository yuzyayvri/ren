"""Compact multiscale HoVer-style encoder-decoder for 256px PanNuke patches."""

from __future__ import annotations

import torch
from torch import nn

IMPLEMENTATION_ID = "ren-hover-fast-v3-optional-type-decoder"
INITIALIZATION_PROVENANCE = (
    "torch random initialization; no pretrained checkpoint; PanNuke fold exposure: none"
)


def _groups(channels: int) -> int:
    return min(8, channels)


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.ReLU(inplace=True),
        )


class HoVerFast(nn.Module):
    """Three-scale shared trunk with independent NP, HV, and type heads."""

    def __init__(
        self,
        width: int = 24,
        detach_type_features: bool = False,
        separate_type_decoder: bool = False,
    ):
        super().__init__()
        self.detach_type_features = detach_type_features
        self.separate_type_decoder = separate_type_decoder
        self.pool = nn.MaxPool2d(2)
        self.encoder1 = ConvBlock(3, width)
        self.encoder2 = ConvBlock(width, width * 2)
        self.encoder3 = ConvBlock(width * 2, width * 4)
        self.bottleneck = ConvBlock(width * 4, width * 8)
        self.up3 = nn.ConvTranspose2d(width * 8, width * 4, 2, 2)
        self.decoder3 = ConvBlock(width * 8, width * 4)
        self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 2, 2)
        self.decoder2 = ConvBlock(width * 4, width * 2)
        self.up1 = nn.ConvTranspose2d(width * 2, width, 2, 2)
        self.decoder1 = ConvBlock(width * 2, width)
        self.np_head = nn.Conv2d(width, 1, 1)
        self.hv_head = nn.Conv2d(width, 2, 1)
        self.type_head = nn.Conv2d(width, 6, 1)
        if separate_type_decoder:
            self.type_up3 = nn.ConvTranspose2d(width * 8, width * 4, 2, 2)
            self.type_decoder3 = ConvBlock(width * 8, width * 4)
            self.type_up2 = nn.ConvTranspose2d(width * 4, width * 2, 2, 2)
            self.type_decoder2 = ConvBlock(width * 4, width * 2)
            self.type_up1 = nn.ConvTranspose2d(width * 2, width, 2, 2)
            self.type_decoder1 = ConvBlock(width * 2, width)

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        bottleneck = self.bottleneck(self.pool(encoder3))
        decoder3 = self.decoder3(torch.cat((encoder3, self.up3(bottleneck)), dim=1))
        decoder2 = self.decoder2(torch.cat((encoder2, self.up2(decoder3)), dim=1))
        decoder1 = self.decoder1(torch.cat((encoder1, self.up1(decoder2)), dim=1))
        if self.separate_type_decoder:
            type_encoder1 = encoder1.detach() if self.detach_type_features else encoder1
            type_encoder2 = encoder2.detach() if self.detach_type_features else encoder2
            type_encoder3 = encoder3.detach() if self.detach_type_features else encoder3
            type_bottleneck = bottleneck.detach() if self.detach_type_features else bottleneck
            type_decoder3 = self.type_decoder3(
                torch.cat((type_encoder3, self.type_up3(type_bottleneck)), dim=1)
            )
            type_decoder2 = self.type_decoder2(
                torch.cat((type_encoder2, self.type_up2(type_decoder3)), dim=1)
            )
            type_features = self.type_decoder1(
                torch.cat((type_encoder1, self.type_up1(type_decoder2)), dim=1)
            )
        else:
            type_features = decoder1.detach() if self.detach_type_features else decoder1
        return (
            self.np_head(decoder1),
            self.hv_head(decoder1),
            self.type_head(type_features),
        )
