import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import FrEIA.framework as Ff
import FrEIA.modules as Fm

from utils.data_adapters import TupleLoader


# ---------------------------------------------------------------------------
# Subnet constructor for 2D normalizing flow coupling blocks.
#
# Taken from anomalib  torch_model.py  subnet_conv_func()
# Variable names (kernel_size, hidden_ratio, in_channels, out_channels,
# hidden_channels) are kept identical to anomalib.
# ---------------------------------------------------------------------------

def subnet_conv_func(kernel_size, hidden_ratio):
    """
    Return a callable subnet_conv(in_channels, out_channels) -> nn.Sequential.
    Used inside FrEIA AllInOneBlock as the subnet_constructor argument.

    Reference: anomalib  torch_model.py  subnet_conv_func()
    """

    def subnet_conv(in_channels, out_channels):
        hidden_channels = int(in_channels * hidden_ratio)
        padding = kernel_size // 2
        last_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size, padding=padding)
        # Zero-init: coupling layers start as identity transforms,
        # stabilising early training (standard practice in RealNVP / FastFlow).
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            last_conv,
        )

    return subnet_conv


# ---------------------------------------------------------------------------
# Build one normalizing-flow block (SequenceINN) for a single feature scale.
#
# Taken from anomalib  torch_model.py  create_fast_flow_block()
# Variable names (input_dimensions, conv3x3_only, hidden_ratio, flow_steps,
# clamp, nodes, kernel_size) are kept identical to anomalib.
# The only change: uses FrEIA.modules.AllInOneBlock (Fm.AllInOneBlock)
# instead of anomalib's own AllInOneBlock wrapper.
# ---------------------------------------------------------------------------

def create_fast_flow_block(input_dimensions, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
    """
    Create one NF FastFlow block for features of shape input_dimensions=[C, H, W].

    Alternating kernel sizes (3x3 / 1x1) follow Figure 2 and Section 3.3 of the
    FastFlow paper.  When conv3x3_only=True every step uses 3x3.

    Reference: anomalib  torch_model.py  create_fast_flow_block()
    """
    nodes = Ff.SequenceINN(*input_dimensions)
    for i in range(flow_steps):
        # Alternating kernel: even steps -> 3x3, odd steps -> 1x1
        # (unless conv3x3_only is True, then always 3x3)
        kernel_size = 1 if i % 2 == 1 and not conv3x3_only else 3
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes


# ---------------------------------------------------------------------------
# FastFlow loss function.
#
# Taken from anomalib  loss.py  FastflowLoss.forward()
# Variable names (hidden_variables, jacobians, hidden_variable, jacobian)
# are kept identical to anomalib.
# ---------------------------------------------------------------------------

class FastflowLoss(nn.Module):
    """
    Negative log-likelihood loss for FastFlow.
    loss = mean( 0.5 * sum(z^2, dim=(1,2,3)) - log_det_jacobian )

    Reference: anomalib  loss.py  FastflowLoss
    """

    @staticmethod
    def forward(hidden_variables, jacobians):
        loss = torch.tensor(0.0, device=hidden_variables[0].device)
        for hidden_variable, jacobian in zip(hidden_variables, jacobians):
            loss += torch.mean(
                0.5 * torch.sum(hidden_variable ** 2, dim=(1, 2, 3)) - jacobian
            )
        return loss


# ---------------------------------------------------------------------------
# Anomaly map generator.
#
# Taken from anomalib  anomaly_map.py  AnomalyMapGenerator.forward()
# Variable names (hidden_variables, hidden_variable, log_prob, prob,
# flow_map, flow_maps) are kept identical to anomalib.
# ---------------------------------------------------------------------------

class AnomalyMapGenerator(nn.Module):
    """
    Convert hidden variables from NF blocks into a single anomaly heatmap.

    For each hidden variable tensor:
      1. log_prob  = -0.5 * mean(z^2, dim=1, keepdim=True)
      2. prob      = exp(log_prob)
      3. flow_map  = interpolate(-prob, to input_size)
    Stack all flow_maps and average to get final anomaly map.
    Finally apply Gaussian smoothing to reduce noise (improves pixel AUROC & PRO).

    Reference: anomalib  anomaly_map.py  AnomalyMapGenerator
    """

    def __init__(self, input_size, sigma=1.5):
        super().__init__()
        self.input_size = tuple(input_size) if not isinstance(input_size, tuple) else input_size
        self.register_buffer("_gauss_kernel", self._make_gaussian_kernel(sigma))

    @staticmethod
    def _make_gaussian_kernel(sigma, channels=1):
        """Create a fixed 2-D Gaussian kernel for smoothing anomaly maps."""
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1  # covers ~4 sigma each side
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
        gauss_2d = gauss_2d / gauss_2d.sum()
        # shape: (out_channels, in_channels/groups, kH, kW)
        return gauss_2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

    def forward(self, hidden_variables):
        flow_maps = []
        for hidden_variable in hidden_variables:
            log_prob = -torch.mean(hidden_variable ** 2, dim=1, keepdim=True) * 0.5
            prob = torch.exp(log_prob)
            flow_map = F.interpolate(
                input=-prob,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )
            flow_maps.append(flow_map)
        flow_maps = torch.stack(flow_maps, dim=-1)
        anomaly_map = torch.mean(flow_maps, dim=-1)

        # Gaussian smoothing — reduces pixel-level noise
        pad = self._gauss_kernel.shape[-1] // 2
        anomaly_map = F.pad(anomaly_map, (pad, pad, pad, pad), mode="reflect")
        anomaly_map = F.conv2d(anomaly_map, self._gauss_kernel, groups=1)
        return anomaly_map


# ---------------------------------------------------------------------------
# Main FastFlow method class.
#
# This class follows the same fit() / predict() interface as CFlowMethod
# (methods/cflow_method.py) so that it plugs into the existing project
# pipeline (main.py, train_and_test pattern).
#
# The model architecture inside (frozen backbone + LayerNorm + 2D NF blocks)
# follows anomalib's FastflowModel.__init__() and forward().
# The training loop, LR schedule, and predict logic are written to match
# the CFlowMethod conventions in this project.
# ---------------------------------------------------------------------------

class FastFlowMethod:
    """
    Main class for FastFlow anomaly detection.

    Uses this project's MultiScaleFeatureExtractor as the frozen backbone,
    and builds one 2D normalizing flow per feature scale (l1, l2, l3).
    """

    def __init__(
            self,
            extractor,
            device="cuda",
            input_size=256,
            flow_steps=8,
            conv3x3_only=False,
            hidden_ratio=1.0,
            clamp=2.0,
            lr=1e-3,
            meta_epochs=50,
            weight_decay=1e-5,
            verbose=True,
    ):
        # ---- Freeze the backbone (same pattern as CFlowMethod.__init__) ----
        self.extractor = extractor.to(device).eval()
        for p in self.extractor.parameters():
            p.requires_grad = False

        self.device = torch.device(device)
        self.input_size = (input_size, input_size)
        self.flow_steps = flow_steps
        self.conv3x3_only = conv3x3_only
        self.hidden_ratio = hidden_ratio
        self.clamp = clamp
        self.lr = lr
        self.meta_epochs = meta_epochs
        self.weight_decay = weight_decay
        self.verbose = verbose

        # Will be built on first fit() call (same pattern as CFlowMethod._build)
        self.norms = None
        self.fast_flow_blocks = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = FastflowLoss()
        self.anomaly_map_generator = AnomalyMapGenerator(self.input_size)

    # ------------------------------------------------------------------
    # _build: discover feature shapes and create NF blocks + LayerNorms.
    #
    # LayerNorm per scale follows anomalib's FastflowModel.__init__() for
    # CNN (resnet) backbones.  Variable names (channels, scales, norms,
    # fast_flow_blocks) mirror anomalib where applicable.
    # ------------------------------------------------------------------

    def _build(self, train_loader):
        """Run one forward pass to discover feature map shapes, then build NF blocks."""
        image, _, _ = next(iter(TupleLoader(train_loader)))
        image = image.to(self.device)

        with torch.no_grad():
            feats = self.extractor(image)  # {"l1": (B,C1,H1,W1), "l2": ..., "l3": ...}

        # Collect channel counts and spatial sizes for each scale
        channels = []
        scales = []
        for key in ["l1", "l2", "l3"]:
            f = feats[key]
            _, C, H, W = f.shape
            channels.append(C)
            scales.append((H, W))

        # LayerNorm per feature scale (from anomalib FastflowModel.__init__ for CNN backbones)
        self.norms = nn.ModuleList()
        for i, key in enumerate(["l1", "l2", "l3"]):
            C = channels[i]
            H, W = scales[i]
            self.norms.append(
                nn.LayerNorm([C, H, W], elementwise_affine=True)
            )
        self.norms = self.norms.to(self.device)

        # One 2D normalizing flow block per feature scale
        # (from anomalib FastflowModel.__init__)
        self.fast_flow_blocks = nn.ModuleList()
        for i in range(len(channels)):
            C = channels[i]
            H, W = scales[i]
            self.fast_flow_blocks.append(
                create_fast_flow_block(
                    input_dimensions=[C, H, W],
                    conv3x3_only=self.conv3x3_only,
                    hidden_ratio=self.hidden_ratio,
                    flow_steps=self.flow_steps,
                    clamp=self.clamp,
                )
            )
        self.fast_flow_blocks = self.fast_flow_blocks.to(self.device)

        # Optimizer updates only NF block and LayerNorm parameters (backbone is frozen)
        params = list(self.fast_flow_blocks.parameters()) + list(self.norms.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        # Smooth cosine decay over the full training run (no restarts).
        # CosineAnnealingLR decays LR from initial lr to eta_min once,
        # which avoids the LR jumps that WarmRestarts can cause.
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.meta_epochs, eta_min=self.lr * 1e-3
        )

        # Print verification (same pattern as CFlowMethod._build)
        print("\n" + "=" * 60)
        print("FastFlow Build Verification:")
        print("=" * 60)
        for i, key in enumerate(["l1", "l2", "l3"]):
            C = channels[i]
            H, W = scales[i]
            print(f"  {key} -> C={C}, H={H}, W={W}  |  NF input_dimensions=[{C}, {H}, {W}]")
        total_params = sum(p.numel() for p in params)
        print(f"  Total trainable parameters: {total_params:,}")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # fit: training loop.
    #
    # The forward pass through NF blocks follows anomalib FastflowModel.forward()
    # (training branch): features -> norms -> fast_flow_blocks -> (hidden_variables, log_jacobians).
    # Loss computation uses FastflowLoss (anomalib loss.py).
    #
    # The outer epoch loop and logging follow CFlowMethod.fit() conventions.
    # ------------------------------------------------------------------

    def fit(self, train_loader):
        """Train FastFlow on normal images."""
        if self.fast_flow_blocks is None:
            self._build(train_loader)

        train_loader_t = TupleLoader(train_loader)
        self.norms.train()
        self.fast_flow_blocks.train()

        for epoch in range(self.meta_epochs):
            epoch_loss = 0.0
            batch_count = 0

            for image, _, _ in train_loader_t:
                image = image.to(self.device)

                # Extract multi-scale features from frozen backbone
                with torch.no_grad():
                    feats = self.extractor(image)  # {"l1", "l2", "l3"}

                # Forward pass through NF blocks
                # (follows anomalib FastflowModel.forward() training branch)
                features = [feats["l1"], feats["l2"], feats["l3"]]

                hidden_variables = []
                log_jacobians = []
                for i, feature in enumerate(features):
                    feature = self.norms[i](feature)
                    hidden_variable, log_jacobian = self.fast_flow_blocks[i](feature)
                    hidden_variables.append(hidden_variable)
                    log_jacobians.append(log_jacobian)

                # Loss (anomalib FastflowLoss)
                loss = self.criterion(hidden_variables, log_jacobians)

                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping — prevents rare gradient spikes from
                # corrupting the NF weights (important for lightweight backbones
                # whose thin feature channels make training less stable).
                torch.nn.utils.clip_grad_norm_(
                    list(self.fast_flow_blocks.parameters()) + list(self.norms.parameters()),
                    max_norm=1.0,
                )
                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            self.scheduler.step()

            if self.verbose:
                mean_loss = epoch_loss / max(batch_count, 1)
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    "Epoch: {:d} \t train loss: {:.4f}, lr={:.6f}".format(
                        epoch, mean_loss, current_lr
                    )
                )

        # Switch to eval after training (same as CFlowMethod.fit)
        self.norms.eval()
        self.fast_flow_blocks.eval()

    # ------------------------------------------------------------------
    # predict: inference / anomaly map generation.
    #
    # The forward pass and anomaly map generation follow
    # anomalib FastflowModel.forward() (eval branch) and
    # AnomalyMapGenerator.forward().
    #
    # The return format (scores, maps) matches CFlowMethod.predict().
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, test_loader):
        """Compute anomaly scores and maps for the test set."""
        if self.fast_flow_blocks is None:
            raise RuntimeError("Call fit() first.")

        self.norms.eval()
        self.fast_flow_blocks.eval()

        test_loader_t = TupleLoader(test_loader)

        all_scores = []
        all_maps = []

        for image, _, _ in test_loader_t:
            image = image.to(self.device)

            # Extract features
            feats = self.extractor(image)
            features = [feats["l1"], feats["l2"], feats["l3"]]

            # Forward through NF blocks (eval branch of anomalib FastflowModel.forward)
            hidden_variables = []
            for i, feature in enumerate(features):
                feature = self.norms[i](feature)
                hidden_variable, _log_jacobian = self.fast_flow_blocks[i](feature)
                hidden_variables.append(hidden_variable)

            # Generate anomaly map (anomalib AnomalyMapGenerator.forward)
            anomaly_map = self.anomaly_map_generator(hidden_variables)  # (B, 1, H, W)

            # Image-level score: max value in the anomaly map
            # (same as anomalib: pred_score = torch.amax(anomaly_map, dim=(-2, -1)))
            pred_score = torch.amax(anomaly_map, dim=(-2, -1))  # (B, 1)
            pred_score = pred_score.squeeze(1)  # (B,)

            all_scores.append(pred_score.cpu())
            all_maps.append(anomaly_map.cpu())

        scores = torch.cat(all_scores, dim=0)   # (N,)
        maps = torch.cat(all_maps, dim=0)        # (N, 1, H, W)

        return scores, maps
