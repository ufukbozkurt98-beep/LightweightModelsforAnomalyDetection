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


def _map_megvii_state_dict(src_state: dict, model: ShuffleNet) -> dict:
    """
    Map state_dict keys from the official Megvii ShuffleNet-Series checkpoint
    to our implementation.

    Megvii (megvii-model/ShuffleNet-Series) uses:
      - first_conv.{i}                    -> conv1.{i}
      - features.{flat_idx}.branch_main_1.0  -> stageN.{local}.gconv1  (1x1 group conv)
      - features.{flat_idx}.branch_main_1.1  -> stageN.{local}.bn1
      - features.{flat_idx}.branch_main_1.3  -> stageN.{local}.dwconv  (3x3 depthwise)
      - features.{flat_idx}.branch_main_1.4  -> stageN.{local}.bn2
      - features.{flat_idx}.branch_main_2.0  -> stageN.{local}.gconv2  (1x1 group conv)
      - features.{flat_idx}.branch_main_2.1  -> stageN.{local}.bn3
      - classifier.*                         -> skipped (no classifier needed)

    Megvii stores all blocks in one flat 'features' Sequential [0..15].
    We split them into stage2 [0..3], stage3 [0..7], stage4 [0..3]
    using STAGE_REPEATS = [4, 8, 4].
    """
    import re

    mapped = {}
    dst_keys = set(model.state_dict().keys())

    # Build flat-index to (stage_name, local_index) mapping
    # stage_repeats = [4, 8, 4] -> flat 0-3 = stage2, 4-11 = stage3, 12-15 = stage4
    stage_repeats = model.STAGE_REPEATS
    flat_to_stage = {}
    flat_idx = 0
    for stage_offset, repeats in enumerate(stage_repeats):
        stage_name = f"stage{stage_offset + 2}"
        for local_idx in range(repeats):
            flat_to_stage[flat_idx] = (stage_name, local_idx)
            flat_idx += 1

    # Layer index mapping within each block:
    # branch_main_1: [0]=gconv1, [1]=bn1, [2]=ReLU(no params), [3]=dwconv, [4]=bn2
    # branch_main_2: [0]=gconv2, [1]=bn3
    block_layer_map = {
        "branch_main_1.0": "gconv1",
        "branch_main_1.1": "bn1",
        "branch_main_1.3": "dwconv",
        "branch_main_1.4": "bn2",
        "branch_main_2.0": "gconv2",
        "branch_main_2.1": "bn3",
    }

    for src_key, value in src_state.items():
        # Skip classifier
        if src_key.startswith("classifier."):
            continue

        # first_conv.{i} -> conv1.{i}
        if src_key.startswith("first_conv."):
            dst_key = src_key.replace("first_conv.", "conv1.", 1)
            if dst_key in dst_keys:
                mapped[dst_key] = value
            continue

        # features.{flat_idx}.{block_layer}.{param}
        m = re.match(r"features\.(\d+)\.(.*)", src_key)
        if m:
            fidx = int(m.group(1))
            rest = m.group(2)

            if fidx not in flat_to_stage:
                continue

            stage_name, local_idx = flat_to_stage[fidx]

            # Map branch_main_X.Y to our layer name
            for megvii_prefix, our_name in block_layer_map.items():
                if rest.startswith(megvii_prefix):
                    param_suffix = rest[len(megvii_prefix):]  # e.g. ".weight"
                    dst_key = f"{stage_name}.{local_idx}.{our_name}{param_suffix}"
                    if dst_key in dst_keys:
                        mapped[dst_key] = value
                    break
            continue

        # Any other key that directly matches (unlikely but safe)
        if src_key in dst_keys:
            mapped[src_key] = value

    return mapped


def _map_jaxony_state_dict(src_state: dict, model: ShuffleNet) -> dict:
    """
    Map state_dict keys from jaxony/ShuffleNet (or ericsun99/ShuffleNet-1g8)
    checkpoint format to our implementation.

    These third-party implementations use:
      - conv1 (plain Conv2d, no BN)
      - stageN.ShuffleUnit_StageN_I.g_conv_1x1_compress/depthwise_conv3x3/...
      - Conv layers have bias=True (ours have bias=False, so biases are skipped)
    """
    import re

    mapped = {}
    dst_keys = set(model.state_dict().keys())

    for src_key, value in src_state.items():
        # Skip FC classifier
        if src_key.startswith("fc."):
            continue

        # conv1: external has plain Conv2d, ours wraps in Sequential
        if src_key == "conv1.weight":
            dst_key = "conv1.0.weight"
            if dst_key in dst_keys:
                mapped[dst_key] = value
            continue
        if src_key == "conv1.bias":
            continue

        dst_key = src_key

        m = re.match(
            r"(stage\d+)\.ShuffleUnit_Stage\d+_(\d+)\.(.*)", src_key
        )
        if m:
            stage, idx, rest = m.group(1), m.group(2), m.group(3)
            rest = rest.replace("g_conv_1x1_compress.conv1x1.", "gconv1.")
            rest = rest.replace("g_conv_1x1_compress.batch_norm.", "bn1.")
            rest = rest.replace("depthwise_conv3x3.", "dwconv.")
            rest = rest.replace("bn_after_depthwise.", "bn2.")
            rest = rest.replace("g_conv_1x1_expand.conv1x1.", "gconv2.")
            rest = rest.replace("g_conv_1x1_expand.batch_norm.", "bn3.")
            dst_key = f"{stage}.{idx}.{rest}"

        # Skip bias tensors from conv layers (ours uses bias=False)
        if dst_key.endswith(".bias") and dst_key not in dst_keys:
            continue

        if dst_key in dst_keys:
            mapped[dst_key] = value

    return mapped


def _detect_checkpoint_format(src_state: dict) -> str:
    """Auto-detect which checkpoint format the state_dict uses."""
    keys = set(src_state.keys())
    if any(k.startswith("first_conv.") for k in keys):
        return "megvii"
    if any("ShuffleUnit_Stage" in k for k in keys):
        return "jaxony"
    if any(k.startswith("features.") for k in keys):
        return "megvii"
    return "unknown"


def load_pretrained_shufflenet(model: ShuffleNet, checkpoint_path: str) -> None:
    """
    Load pretrained weights into our ShuffleNet model.
    Auto-detects the checkpoint format (official Megvii or third-party jaxony/ericsun99).

    Supported checkpoints:
      - Official Megvii (megvii-model/ShuffleNet-Series), MIT license
        Download: https://1drv.ms/f/s!AgaP37NGYuEXhRfQxHRseR7eSxXo
      - jaxony/ShuffleNet g=3 (third-party, MIT license)
      - ericsun99/ShuffleNet-1g8-Pytorch g=8 (third-party, BSD-2-Clause)
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Checkpoints may wrap the state_dict under a 'state_dict' key
    if "state_dict" in checkpoint:
        src_state = checkpoint["state_dict"]
    else:
        src_state = checkpoint

    # Auto-detect format and apply the right mapping
    fmt = _detect_checkpoint_format(src_state)
    if fmt == "megvii":
        mapped = _map_megvii_state_dict(src_state, model)
    elif fmt == "jaxony":
        mapped = _map_jaxony_state_dict(src_state, model)
    else:
        raise ValueError(
            f"Could not detect checkpoint format. "
            f"Expected keys starting with 'first_conv.' (Megvii) "
            f"or containing 'ShuffleUnit_Stage' (jaxony). "
            f"Got keys: {list(src_state.keys())[:5]}..."
        )

    # Load with strict=False: any unmatched keys stay at default init
    missing, unexpected = model.load_state_dict(mapped, strict=False)

    loaded_count = len(mapped)
    total_count = len(model.state_dict())
    print(f"  [{fmt} format] Loaded {loaded_count}/{total_count} pretrained parameters")
    if missing:
        print(f"  Not loaded (using default init): {missing}")


def shufflenet_g1(scale: float = 1.0, num_classes: int = 1000) -> ShuffleNet:
    """ShuffleNet with g=1"""
    return ShuffleNet(groups=1, num_classes=num_classes, scale=scale)


def shufflenet_g3(scale: float = 1.0, num_classes: int = 1000) -> ShuffleNet:
    """ShuffleNet with g=3"""
    return ShuffleNet(groups=3, num_classes=num_classes, scale=scale)


def shufflenet_g8(scale: float = 1.0, num_classes: int = 1000) -> ShuffleNet:
    """ShuffleNet with g=8"""
    return ShuffleNet(groups=8, num_classes=num_classes, scale=scale)
