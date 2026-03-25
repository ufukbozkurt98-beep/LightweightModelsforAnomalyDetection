"""
FastFlow core algorithm — adapted from anomalib (OpenVINO Toolkit).

Source: https://github.com/openvinotoolkit/anomalib
Files:  src/anomalib/models/image/fastflow/torch_model.py
        src/anomalib/models/image/fastflow/loss.py
        src/anomalib/models/image/fastflow/anomaly_map.py

Original Code Copyright (c) 2022 @gathierry (https://github.com/gathierry/FastFlow/)
Modified    Copyright (C) 2022-2025 Intel Corporation
SPDX-License-Identifier: Apache-2.0

Default anomalib config (examples/configs/model/fastflow.yaml):
    model:  backbone=resnet18, flow_steps=8, conv3x3_only=false, hidden_ratio=1.0
    train:  Adam(lr=0.001, weight_decay=1e-5), max_epochs=500
            EarlyStopping(patience=3, monitor=pixel_AUROC, mode=max)
            gradient_clip_val=0 (disabled)

Only change: AllInOneBlock imported from FrEIA.modules instead of
anomalib.models.components.flow (anomalib vendored an identical copy).
"""

from collections.abc import Callable

import torch
from FrEIA.framework import SequenceINN
from FrEIA.modules import AllInOneBlock
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# subnet_conv_func  (from torch_model.py — exact copy)
# ---------------------------------------------------------------------------
def subnet_conv_func(kernel_size: int, hidden_ratio: float) -> Callable:
    """Subnet Convolutional Function.

    Callable class or function ``f``, called as ``f(channels_in, channels_out)`` and
        should return a torch.nn.Module.
        Predicts coupling coefficients :math:`s, t`.

    Args:
        kernel_size (int): Kernel Size
        hidden_ratio (float): Hidden ratio to compute number of hidden channels.

    Returns:
        Callable: Sequential for the subnet constructor.
    """

    def subnet_conv(in_channels: int, out_channels: int) -> nn.Sequential:
        hidden_channels = int(in_channels * hidden_ratio)
        # NOTE: setting padding="same" in nn.Conv2d breaks the onnx export so manual padding required.
        # TODO(ashwinvaidya17): Use padding="same" in nn.Conv2d once PyTorch v2.1 is released
        # CVS-122671
        padding_dims = (kernel_size // 2 - ((1 + kernel_size) % 2), kernel_size // 2)
        padding = (*padding_dims, *padding_dims)
        return nn.Sequential(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_channels, hidden_channels, kernel_size),
            nn.ReLU(),
            nn.ZeroPad2d(padding),
            nn.Conv2d(hidden_channels, out_channels, kernel_size),
        )

    return subnet_conv


# ---------------------------------------------------------------------------
# create_fast_flow_block  (from torch_model.py — exact copy)
# ---------------------------------------------------------------------------
def create_fast_flow_block(
    input_dimensions: list[int],
    conv3x3_only: bool,
    hidden_ratio: float,
    flow_steps: int,
    clamp: float = 2.0,
) -> SequenceINN:
    """Create NF Fast Flow Block.

    This is to create Normalizing Flow (NF) Fast Flow model block based on
    Figure 2 and Section 3.3 in the paper.

    Args:
        input_dimensions (list[int]): Input dimensions (Channel, Height, Width)
        conv3x3_only (bool): Boolean whether to use conv3x3 only or conv3x3 and conv1x1.
        hidden_ratio (float): Ratio for the hidden layer channels.
        flow_steps (int): Flow steps.
        clamp (float, optional): Clamp.
            Defaults to ``2.0``.

    Returns:
        SequenceINN: FastFlow Block.
    """
    nodes = SequenceINN(*input_dimensions)
    for i in range(flow_steps):
        kernel_size = 1 if i % 2 == 1 and not conv3x3_only else 3
        nodes.append(
            AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes


# ---------------------------------------------------------------------------
# FastflowLoss  (from loss.py — exact copy)
# ---------------------------------------------------------------------------
class FastflowLoss(nn.Module):
    """FastFlow Loss Module."""

    @staticmethod
    def forward(hidden_variables: list[torch.Tensor], jacobians: list[torch.Tensor]) -> torch.Tensor:
        """Calculate the FastFlow loss."""
        loss = torch.tensor(0.0, device=hidden_variables[0].device)
        for hidden_variable, jacobian in zip(hidden_variables, jacobians, strict=True):
            loss += torch.mean(0.5 * torch.sum(hidden_variable**2, dim=(1, 2, 3)) - jacobian)
        return loss


# ---------------------------------------------------------------------------
# AnomalyMapGenerator  (from anomaly_map.py — exact copy)
# ---------------------------------------------------------------------------
class AnomalyMapGenerator(nn.Module):
    """Generate anomaly heatmaps from FastFlow hidden variables."""

    def __init__(self, input_size: tuple) -> None:
        super().__init__()
        self.input_size = input_size if isinstance(input_size, tuple) else tuple(input_size)

    def forward(self, hidden_variables: list[torch.Tensor]) -> torch.Tensor:
        """Generate anomaly heatmap from hidden variables."""
        flow_maps: list[torch.Tensor] = []
        for hidden_variable in hidden_variables:
            log_prob = -torch.mean(hidden_variable**2, dim=1, keepdim=True) * 0.5
            prob = torch.exp(log_prob)
            flow_map = F.interpolate(
                input=-prob,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )
            flow_maps.append(flow_map)
        flow_maps = torch.stack(flow_maps, dim=-1)
        return torch.mean(flow_maps, dim=-1)
