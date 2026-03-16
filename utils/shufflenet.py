"""
ShuffleNet V1 implementation.

Based on the official Megvii code (MIT license):
  https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV1

Original paper:
  Zhang et al., "ShuffleNet: An Extremely Efficient Convolutional Neural
  Network for Mobile Devices", CVPR 2018.

Modifications from the original:
  - Merged blocks.py and network.py into a single file
  - Added forward_features() for multi-scale feature extraction
  - Added num_classes=0 mode to disable classifier head
  - Added pretrained weight loading with auto-format detection
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building block  (from blocks.py — kept identical to original)
# ---------------------------------------------------------------------------

class ShuffleV1Block(nn.Module):
    def __init__(self, inp, oup, *, group, first_group, mid_channels, ksize, stride):
        super(ShuffleV1Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp
        self.group = group

        if stride == 2:
            outputs = oup - inp
        else:
            outputs = oup

        branch_main_1 = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, groups=1 if first_group else group, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
        ]
        branch_main_2 = [
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, groups=group, bias=False),
            nn.BatchNorm2d(outputs),
        ]
        self.branch_main_1 = nn.Sequential(*branch_main_1)
        self.branch_main_2 = nn.Sequential(*branch_main_2)

        if stride == 2:
            self.branch_proj = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, old_x):
        x = old_x
        x_proj = old_x
        x = self.branch_main_1(x)
        if self.group > 1:
            x = self.channel_shuffle(x)
        x = self.branch_main_2(x)
        if self.stride == 1:
            return F.relu(x + x_proj)
        elif self.stride == 2:
            return torch.cat((self.branch_proj(x_proj), F.relu(x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group

        x = x.reshape(batchsize, group_channels, self.group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x


# ---------------------------------------------------------------------------
# Full network  (from network.py — added forward_features + num_classes=0)
# ---------------------------------------------------------------------------

class ShuffleNetV1(nn.Module):
    def __init__(self, input_size=224, n_class=1000, model_size='1.0x', group=None):
        super(ShuffleNetV1, self).__init__()

        assert group is not None

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if group == 3:
            if model_size == '0.5x':
                self.stage_out_channels = [-1, 12, 120, 240, 480]
            elif model_size == '1.0x':
                self.stage_out_channels = [-1, 24, 240, 480, 960]
            elif model_size == '1.5x':
                self.stage_out_channels = [-1, 24, 360, 720, 1440]
            elif model_size == '2.0x':
                self.stage_out_channels = [-1, 48, 480, 960, 1920]
            else:
                raise NotImplementedError
        elif group == 8:
            if model_size == '0.5x':
                self.stage_out_channels = [-1, 16, 192, 384, 768]
            elif model_size == '1.0x':
                self.stage_out_channels = [-1, 24, 384, 768, 1536]
            elif model_size == '1.5x':
                self.stage_out_channels = [-1, 24, 576, 1152, 2304]
            elif model_size == '2.0x':
                self.stage_out_channels = [-1, 48, 768, 1536, 3072]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError(f"group={group} not supported, use 3 or 8")

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            for i in range(numrepeat):
                stride = 2 if i == 0 else 1
                first_group = idxstage == 0 and i == 0
                self.features.append(ShuffleV1Block(input_channel, output_channel,
                                                    group=group, first_group=first_group,
                                                    mid_channels=output_channel // 4, ksize=3, stride=stride))
                input_channel = output_channel

        self.features = nn.Sequential(*self.features)

        # Classifier head (disabled when n_class=0 for feature extraction)
        if n_class > 0:
            self.globalpool = nn.AvgPool2d(7)
            self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class, bias=False))
        else:
            self.globalpool = None
            self.classifier = None

        self._initialize_weights()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.features(x)

        if self.globalpool is not None and self.classifier is not None:
            x = self.globalpool(x)
            x = x.contiguous().view(-1, self.stage_out_channels[-1])
            x = self.classifier(x)
        return x

    def forward_features(self, x) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features from stage2, stage3, stage4.
        Returns {"l1": stage2_out, "l2": stage3_out, "l3": stage4_out}.
        """
        x = self.first_conv(x)
        x = self.maxpool(x)

        # Run through each block, capturing output at stage boundaries
        # stage_repeats = [4, 8, 4] -> blocks 0-3 = stage2, 4-11 = stage3, 12-15 = stage4
        stage_ends = []
        cumulative = 0
        for r in self.stage_repeats:
            cumulative += r
            stage_ends.append(cumulative - 1)  # [3, 11, 15]

        features = {}
        level = 1
        for i, block in enumerate(self.features):
            x = block(x)
            if i in stage_ends:
                features[f"l{level}"] = x
                level += 1

        return features

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# Pretrained weight loading
# ---------------------------------------------------------------------------

def load_pretrained_shufflenet(model: ShuffleNetV1, checkpoint_path: str) -> None:
    """
    Load pretrained weights from official Megvii checkpoint.
    Weights use the same layer names (first_conv, features, branch_main_1/2),
    so no key mapping is needed — just load directly.

    Download from: https://1drv.ms/f/s!AgaP37NGYuEXhRfQxHRseR7eSxXo
    Source: https://github.com/megvii-model/ShuffleNet-Series (MIT license)
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Checkpoints may wrap the state_dict under a 'state_dict' key
    if "state_dict" in checkpoint:
        src_state = checkpoint["state_dict"]
    else:
        src_state = checkpoint

    # Strip 'module.' prefix if saved with nn.DataParallel
    src_state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
                 for k, v in src_state.items()}

    # Filter out classifier weights (not needed for feature extraction)
    dst_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in src_state.items() if k in dst_keys}

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    loaded_count = len(filtered)
    total_count = len(dst_keys)
    print(f"  Loaded {loaded_count}/{total_count} pretrained parameters")
    if missing:
        print(f"  Not loaded (using default init): {missing}")
