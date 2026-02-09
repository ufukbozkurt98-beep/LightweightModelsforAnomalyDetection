"""
ShuffleNet V1 implementation for multi-scale feature extraction.

Based on the paper:
  "ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices"
  Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun (2017)
  https://arxiv.org/abs/1707.01083

Key components:
  - Channel Shuffle: redistributes channels across groups after group convolution
  - ShuffleNet Unit: group conv 1x1 -> channel shuffle -> depthwise conv 3x3 -> group conv 1x1
  - Three stages with configurable output channels and number of groups

This implementation supports multi-scale feature extraction (returning features from
stage2, stage3, stage4) for use as a backbone in anomaly detection methods.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Channel shuffle operation from ShuffleNet V1 (Section 3.1, Figure 1).

    After a group convolution, channels within each group are independent.
    Channel shuffle redistributes them so that the next group convolution
    can mix information across groups.

    Steps:
      1. Reshape (B, C, H, W) -> (B, groups, C // groups, H, W)
      2. Transpose groups and channels_per_group dimensions
      3. Flatten back to (B, C, H, W)
    """
    batch_size, channels, height, width = x.size()
    channels_per_group = channels // groups

    # reshape: (B, C, H, W) -> (B, g, C/g, H, W)
    x = x.view(batch_size, groups, channels_per_group, height, width)

    # transpose: swap group and channel dimensions -> (B, C/g, g, H, W)
    x = x.transpose(1, 2).contiguous()

    # flatten: (B, C, H, W)
    x = x.view(batch_size, channels, height, width)

    return x


class ShuffleNetUnit(nn.Module):
    """
    ShuffleNet Unit (Section 3.2, Figure 2).

    Two variants:
      - stride=1 (Figure 2b): residual connection with element-wise addition
      - stride=2 (Figure 2c): residual connection with concatenation + avg pool on shortcut

    Architecture:
      1x1 Group Conv -> BN -> ReLU ->
      Channel Shuffle ->
      3x3 Depthwise Conv (stride) -> BN ->
      1x1 Group Conv -> BN ->
      Add/Cat with shortcut -> ReLU
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

        # For stride=2, the output channels are split: out_channels - in_channels
        # come from the main branch, and in_channels come from the avg pool shortcut
        if stride == 2:
            main_out_channels = out_channels - in_channels
        else:
            main_out_channels = out_channels

        # Bottleneck channels (1/4 of output as per the paper, Table 1)
        # Must be divisible by groups for the second group conv
        bottleneck_channels = main_out_channels // 4
        # Round up to nearest multiple of groups to ensure divisibility
        bottleneck_channels = max(((bottleneck_channels + groups - 1) // groups) * groups, groups)

        # First group conv 1x1
        # Note: for stage 2 first block, input comes from a regular conv (not group conv),
        # so we don't use groups on the first 1x1 conv (Section 3.3 in the paper)
        first_groups = 1 if is_stage2 else groups

        self.gconv1 = nn.Conv2d(
            in_channels, bottleneck_channels,
            kernel_size=1, groups=first_groups, bias=False
        )
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        # 3x3 depthwise convolution
        self.dwconv = nn.Conv2d(
            bottleneck_channels, bottleneck_channels,
            kernel_size=3, stride=stride, padding=1,
            groups=bottleneck_channels, bias=False
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        # Second group conv 1x1
        self.gconv2 = nn.Conv2d(
            bottleneck_channels, main_out_channels,
            kernel_size=1, groups=groups, bias=False
        )
        self.bn3 = nn.BatchNorm2d(main_out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Shortcut for stride=2: average pooling
        if stride == 2:
            self.shortcut = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # 1x1 group conv -> BN -> ReLU
        out = self.gconv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Channel shuffle
        out = channel_shuffle(out, self.groups)

        # 3x3 depthwise conv -> BN
        out = self.dwconv(out)
        out = self.bn2(out)

        # 1x1 group conv -> BN
        out = self.gconv2(out)
        out = self.bn3(out)

        # Shortcut connection
        if self.stride == 2:
            residual = self.shortcut(residual)
            # Concatenate along channel dimension (Figure 2c)
            out = torch.cat([out, residual], dim=1)
        else:
            # Element-wise addition (Figure 2b)
            out = out + residual

        out = self.relu(out)
        return out


class ShuffleNet(nn.Module):
    """
    Full ShuffleNet V1 network (Section 3.3, Table 1).

    Architecture:
      Conv1 (3x3, stride=2) -> MaxPool (3x3, stride=2) ->
      Stage2 (repeat units) -> Stage3 (repeat units) -> Stage4 (repeat units) ->
      GlobalAvgPool -> FC

    The number of output channels per stage depends on the number of groups (g).
    From Table 1 in the paper:

    | Stage     | g=1  | g=2  | g=3  | g=4  | g=8  |
    |-----------|------|------|------|------|------|
    | Stage 2   | 144  | 200  | 240  | 272  | 384  |
    | Stage 3   | 288  | 400  | 480  | 544  | 768  |
    | Stage 4   | 576  | 800  | 960  | 1088 | 1536 |

    Each stage has [4, 8, 4] repeat units respectively (first unit has stride=2).

    This implementation supports a `scale` multiplier to adjust channel width,
    and exposes multi-scale features for use as a backbone.
    """

    # Channel configurations from Table 1 (indexed by number of groups)
    STAGE_CHANNELS = {
        1: [144, 288, 576],
        2: [200, 400, 800],
        3: [240, 480, 960],
        4: [272, 544, 1088],
        8: [384, 768, 1536],
    }

    # Number of repeat units per stage [stage2, stage3, stage4]
    STAGE_REPEATS = [4, 8, 4]

    def __init__(
        self,
        groups: int = 3,
        num_classes: int = 1000,
        scale: float = 1.0,
    ) -> None:
        """
        Args:
            groups: Number of groups for group convolutions (1, 2, 3, 4, or 8).
            num_classes: Number of output classes. Set to 0 to disable the classifier head.
            scale: Width multiplier for channel counts (e.g., 0.5x, 1.0x, 1.5x, 2.0x).
        """
        super().__init__()

        if groups not in self.STAGE_CHANNELS:
            raise ValueError(
                f"groups must be one of {list(self.STAGE_CHANNELS.keys())}, got {groups}"
            )

        self.groups = groups
        self.num_classes = num_classes

        # Compute scaled channel counts.
        # Channels must be divisible by groups for group convolution,
        # so we round each scaled value to the nearest multiple of groups.
        base_channels = self.STAGE_CHANNELS[groups]
        self.stage_out_channels = [
            self._round_to_groups(int(c * scale), groups) for c in base_channels
        ]

        # Initial convolution: 3x3, stride=2, 24 output channels (from the paper)
        self.conv1_out = max(int(24 * scale), 1) if scale != 1.0 else 24
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

        # Classifier head (optional)
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
        """Round channel count up to the nearest multiple of groups."""
        return max(((channels + groups - 1) // groups) * groups, groups)

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        repeat: int,
        groups: int,
        is_stage2: bool = False,
    ) -> nn.Sequential:
        """Build one ShuffleNet stage with the given number of repeat units."""
        layers: List[nn.Module] = []

        # First unit has stride=2 (spatial downsampling)
        layers.append(
            ShuffleNetUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                groups=groups,
                stride=2,
                is_stage2=is_stage2,
            )
        )

        # Remaining units have stride=1
        for _ in range(repeat - 1):
            layers.append(
                ShuffleNetUnit(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    groups=groups,
                    stride=1,
                )
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """Kaiming initialization for conv layers, constant for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features (for use as a backbone).

        Returns:
            dict with keys "l1", "l2", "l3" mapping to feature tensors from
            stage2, stage3, stage4 respectively.
        """
        x = self.conv1(x)
        x = self.maxpool(x)

        l1 = self.stage2(x)   # stage2 output
        l2 = self.stage3(l1)  # stage3 output
        l3 = self.stage4(l2)  # stage4 output

        return {"l1": l1, "l2": l2, "l3": l3}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass with classifier head."""
        feats = self.forward_features(x)

        if self.global_pool is not None and self.fc is not None:
            out = self.global_pool(feats["l3"])
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out
        else:
            return feats["l3"]


# ---------------------------------------------------------------------------
# Factory functions for common ShuffleNet V1 configurations
# ---------------------------------------------------------------------------

def shufflenet_g1(scale: float = 1.0, num_classes: int = 1000) -> ShuffleNet:
    """ShuffleNet with g=1 (no group conv in first 1x1)."""
    return ShuffleNet(groups=1, num_classes=num_classes, scale=scale)


def shufflenet_g2(scale: float = 1.0, num_classes: int = 1000) -> ShuffleNet:
    """ShuffleNet with g=2."""
    return ShuffleNet(groups=2, num_classes=num_classes, scale=scale)


def shufflenet_g3(scale: float = 1.0, num_classes: int = 1000) -> ShuffleNet:
    """ShuffleNet with g=3 (default configuration from the paper)."""
    return ShuffleNet(groups=3, num_classes=num_classes, scale=scale)


def shufflenet_g4(scale: float = 1.0, num_classes: int = 1000) -> ShuffleNet:
    """ShuffleNet with g=4."""
    return ShuffleNet(groups=4, num_classes=num_classes, scale=scale)


def shufflenet_g8(scale: float = 1.0, num_classes: int = 1000) -> ShuffleNet:
    """ShuffleNet with g=8."""
    return ShuffleNet(groups=8, num_classes=num_classes, scale=scale)
