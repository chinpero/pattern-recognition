import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class HeatmapKeypointModel(nn.Module):
    """U-Net style model with ResNet18 encoder for keypoint heatmap regression."""

    def __init__(self, num_keypoints: int = 10, pretrained: bool = True):
        super().__init__()
        self.num_keypoints = num_keypoints
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )

        self.encoder0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.encoder1 = backbone.layer1  # 64ch, /4
        self.encoder2 = backbone.layer2  # 128ch, /8
        self.encoder3 = backbone.layer3  # 256ch, /16
        self.encoder4 = backbone.layer4  # 512ch, /32

        self.bridge = ConvBlock(512, 512)

        self.decoder3 = DecoderBlock(512, 256, 256)
        self.decoder2 = DecoderBlock(256, 128, 128)
        self.decoder1 = DecoderBlock(128, 64, 64)

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_keypoints, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.encoder0(x)        # (B, 64, H/2, W/2)
        e1 = self.pool(e0)            # (B, 64, H/4, W/4)
        s1 = self.encoder1(e1)        # (B, 64, H/4, W/4)
        s2 = self.encoder2(s1)        # (B, 128, H/8, W/8)
        s3 = self.encoder3(s2)        # (B, 256, H/16, W/16)
        s4 = self.encoder4(s3)        # (B, 512, H/32, W/32)

        b = self.bridge(s4)           # (B, 512, H/32, W/32)

        d3 = self.decoder3(b, s3)     # (B, 256, H/16, W/16)
        d2 = self.decoder2(d3, s2)    # (B, 128, H/8, W/8)
        d1 = self.decoder1(d2, s1)    # (B, 64, H/4, W/4)

        heatmaps = self.final_conv(d1)  # (B, 10, H/4, W/4)
        return heatmaps
