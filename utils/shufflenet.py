from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Channel shuffle operation
    """
    batch_size, channels, height, width = x.size()
    channels_per_group = channels // groups  # Calculate how many channels in each group

    # Split channels by groups
    x = x.view(batch_size, groups, channels_per_group, height, width)

    # Swap axes to mix groups
    x = x.transpose(1, 2).contiguous()

    # Back to normal tensor shape.
    x = x.view(batch_size, channels, height, width)

    return x


class ShuffleNetUnit(nn.Module):
    """
    ShuffleNet Building Block
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int,
        stride: int = 1,
        is_stage2: bool = False,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.groups = groups

        # For stride=2, the output channels are split: out_channels and in_channels
        if stride == 2:
            main_out_channels = out_channels - in_channels
        # For stride=1 residual add requires same channel size.
        else:
            main_out_channels = out_channels

        # Bottleneck channels are usually 1/4 of main branch output
        bottleneck_channels = main_out_channels // 4
        # Make bottleneck divisible by groups for group convolution.
        bottleneck_channels = max(((bottleneck_channels + groups - 1) // groups) * groups, groups)

        # First group convolution 1x1
        first_groups = 1 if is_stage2 else groups

        # 1x1 group convolution reduces channels
        self.gconv1 = nn.Conv2d(
            in_channels, bottleneck_channels,
            kernel_size=1, groups=first_groups, bias=False
        )
        self.bn1 = nn.BatchNorm2d(bottleneck_channels) # Normalize after conv.

        # Depthwise convolution (one filter per channel)
        self.dwconv = nn.Conv2d(
            bottleneck_channels, bottleneck_channels,
            kernel_size=3, stride=stride, padding=1,
            groups=bottleneck_channels, bias=False
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        # Second pointwise group convolution to project channels
        self.gconv2 = nn.Conv2d(
            bottleneck_channels, main_out_channels,
            kernel_size=1, groups=groups, bias=False
        )
        self.bn3 = nn.BatchNorm2d(main_out_channels)

        # Non-linearity
        self.relu = nn.ReLU(inplace=True)

        # Shortcut when stride is 2 using average pooling
        if stride == 2:
            self.shortcut = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.gconv1(x)  # 1x1 group conv
        out = self.bn1(out)  # BatchNormalization
        out = self.relu(out) # ReLU

        # Channel shuffle
        out = channel_shuffle(out, self.groups)

        # 3x3 depthwise conv -> BN
        out = self.dwconv(out)  # 3x3 depth wise convolution
        out = self.bn2(out) # Batch Normalization

        # 1x1 group conv -> BN
        out = self.gconv2(out)  # 1x1 group convolution
        out = self.bn3(out)  # BatchNorm

        if self.stride == 2:
            residual = self.shortcut(residual)  # Downsample shortcut
            out = torch.cat([out, residual], dim=1)  # Concatenate channels
        else:
            # Element-wise addition
            out = out + residual

        out = self.relu(out)
        return out


class ShuffleNet(nn.Module):
    """
    Full ShuffleNet network
    """

    # Channel configurations
    STAGE_CHANNELS = {
        1: [144, 288, 576],
        2: [200, 400, 800],
        3: [240, 480, 960],
        4: [272, 544, 1088],
        8: [384, 768, 1536],
    }

    # Number of repeat units per stage2, stage3, stage4
    STAGE_REPEATS = [4, 8, 4]

    def __init__(
        self,
        groups: int = 3,
        num_classes: int = 1000,
        scale: float = 1.0,
    ) -> None:
        super().__init__()

        # Validate group setting
        if groups not in self.STAGE_CHANNELS:
            raise ValueError(
                f"groups must be one of {list(self.STAGE_CHANNELS.keys())}, got {groups}"
            )

        self.groups = groups
        self.num_classes = num_classes

        # Compute scaled channel counts.
        # Keep channel counts divisible by groups
        base_channels = self.STAGE_CHANNELS[groups]
        self.stage_out_channels = [
            self._round_to_groups(int(c * scale), groups) for c in base_channels
        ]

        # First convolotion output channels
        self.conv1_out = max(int(24 * scale), 1) if scale != 1.0 else 24
        # Downsample
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.conv1_out, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.conv1_out),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build stages
        self.stage2 = self._make_stage(
            in_channels=self.conv1_out,
            out_channels=self.stage_out_channels[0],
            repeat=self.STAGE_REPEATS[0],
            groups=groups,
            is_stage2=True,
        )
        self.stage3 = self._make_stage(
            in_channels=self.stage_out_channels[0],
            out_channels=self.stage_out_channels[1],
            repeat=self.STAGE_REPEATS[1],
            groups=groups,
        )
        self.stage4 = self._make_stage(
            in_channels=self.stage_out_channels[1],
            out_channels=self.stage_out_channels[2],
            repeat=self.STAGE_REPEATS[2],
            groups=groups,
        )

        # Classifier head
        if num_classes > 0:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(self.stage_out_channels[2], num_classes)
        else:
            self.global_pool = None
            self.fc = None

        # Initialize weights
        self._initialize_weights()

    @staticmethod
    def _round_to_groups(channels: int, groups: int) -> int:
        # Round up to nearest multiple of groups
        return max(((channels + groups - 1) // groups) * groups, groups)

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        repeat: int,
        groups: int,
        is_stage2: bool = False,
    ) -> nn.Sequential:
        # Container for stage units
        layers: List[nn.Module] = []

        # First unit downsamples feature map
        layers.append(
            ShuffleNetUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                groups=groups,
                stride=2,
                is_stage2=is_stage2,
            )
        )

        # Remaining units keep spatial size
        for _ in range(repeat - 1):
            layers.append(
                ShuffleNetUnit(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    groups=groups,
                    stride=1,
                )
            )
        # Put units into one sequential stage
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        # Iterate over all submodules
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Good initialization for ReLU conv
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)  # Start BatchNorm at 1
                nn.init.zeros_(m.bias)  # Start BatchNorm bias at 0
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01) # small random init
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features
        """
        x = self.conv1(x)
        x = self.maxpool(x)

        l1 = self.stage2(x)   # stage2 output - Low level
        l2 = self.stage3(l1)  # stage3 output - Mid Level
        l3 = self.stage4(l2)  # stage4 output - High Level

        # Multi- scale output
        return {"l1": l1, "l2": l2, "l3": l3}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass with classifier head."""
        feats = self.forward_features(x)  # Extract multi-scale backbone features

        if self.global_pool is not None and self.fc is not None:
            out = self.global_pool(feats["l3"])
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out
        else:
            return feats["l3"]


def shufflenet_g1(scale: float = 1.0, num_classes: int = 1000) -> ShuffleNet:
    """ShuffleNet with g=1"""
    return ShuffleNet(groups=1, num_classes=num_classes, scale=scale)


def shufflenet_g3(scale: float = 1.0, num_classes: int = 1000) -> ShuffleNet:
    """ShuffleNet with g=3"""
    return ShuffleNet(groups=3, num_classes=num_classes, scale=scale)


def shufflenet_g8(scale: float = 1.0, num_classes: int = 1000) -> ShuffleNet:
    """ShuffleNet with g=8"""
    return ShuffleNet(groups=8, num_classes=num_classes, scale=scale)
