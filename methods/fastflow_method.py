"""
FastFlow method — backbone-agnostic wrapper around the anomalib core.

The core algorithm (subnet_conv_func, create_fast_flow_block, FastflowLoss,
AnomalyMapGenerator) lives in fastflow_core.py and is the exact anomalib code.

This file adds:
  - Backbone-agnostic integration via MultiScaleFeatureExtractor
  - fit() / predict() / measure_fps() training loop
  - Enhancement toggles (all OFF by default = vanilla anomalib):
      * zero_init      — zero-initialize last conv in coupling subnets
      * gauss_sigma    — Gaussian smoothing on anomaly map
      * use_scheduler  — cosine annealing LR schedule
      * channel_cap    — 1x1 conv channel reduction for wide backbones
"""

import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data_adapters import FlowTupleLoader

# ── anomalib-exact core (fastflow_core.py) ────────────────────────────
from methods.fastflow_core import (
    subnet_conv_func as _anomalib_subnet_conv_func,
    create_fast_flow_block as _anomalib_create_fast_flow_block,
    FastflowLoss,
    AnomalyMapGenerator as _AnomalyMapGenerator,
)


# ======================================================================
# Enhancement wrappers — these extend the anomalib core with toggleable
# features. When enhancements are OFF, the exact anomalib code runs.
# ======================================================================

def subnet_conv_func(kernel_size, hidden_ratio, zero_init=False):
    """
    Wrapper around anomalib's subnet_conv_func.
    When zero_init=False (default), returns the exact anomalib subnet.
    When zero_init=True, additionally zero-initializes the last conv layer.
    """
    if not zero_init:
        # Exact anomalib code path — no modifications
        return _anomalib_subnet_conv_func(kernel_size, hidden_ratio)

    # Enhancement: zero-init last conv so the flow starts as identity
    def subnet_conv(in_channels, out_channels):
        hidden_channels = int(in_channels * hidden_ratio)
        padding_dims = (kernel_size // 2 - ((1 + kernel_size) % 2), kernel_size // 2)
        padding = (*padding_dims, *padding_dims)
        last_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size)
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)
        return nn.Sequential(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_channels, hidden_channels, kernel_size),
            nn.ReLU(),
            nn.ZeroPad2d(padding),
            last_conv,
        )
    return subnet_conv


def create_fast_flow_block(input_dimensions, conv3x3_only, hidden_ratio,
                           flow_steps, clamp=2.0, zero_init=False):
    """
    Wrapper around anomalib's create_fast_flow_block.
    When zero_init=False (default), calls the exact anomalib function.
    When zero_init=True, uses our enhanced subnet_conv_func.
    """
    if not zero_init:
        # Exact anomalib code path
        return _anomalib_create_fast_flow_block(
            input_dimensions, conv3x3_only, hidden_ratio, flow_steps, clamp
        )

    # Enhancement path: build with zero-init subnets
    from FrEIA.framework import SequenceINN
    from FrEIA.modules import AllInOneBlock
    nodes = SequenceINN(*input_dimensions)
    for i in range(flow_steps):
        kernel_size = 1 if i % 2 == 1 and not conv3x3_only else 3
        nodes.append(
            AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio, zero_init=True),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes


class AnomalyMapGenerator(nn.Module):
    """
    Anomaly map generator with optional Gaussian smoothing enhancement.
    When sigma=0.0 (default), delegates entirely to the anomalib core.
    When sigma>0.0, applies Gaussian post-processing smoothing.
    """

    def __init__(self, input_size, sigma=0.0):
        super().__init__()
        self.sigma = sigma
        # The core anomalib generator (always used)
        self._core = _AnomalyMapGenerator(input_size)
        # Enhancement: Gaussian kernel for smoothing
        if sigma > 0:
            self.register_buffer("_gauss_kernel", self._make_gaussian_kernel(sigma))

    @staticmethod
    def _make_gaussian_kernel(sigma, channels=1):
        """Builds fixed 2D Gaussian filter."""
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
        gauss_2d = gauss_2d / gauss_2d.sum()
        return gauss_2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

    def forward(self, hidden_variables):
        # Core anomalib anomaly map
        anomaly_map = self._core(hidden_variables)
        # Enhancement: Gaussian smoothing
        if self.sigma > 0:
            pad = self._gauss_kernel.shape[-1] // 2
            anomaly_map = F.pad(anomaly_map, (pad, pad, pad, pad), mode="reflect")
            anomaly_map = F.conv2d(anomaly_map, self._gauss_kernel, groups=1)
        return anomaly_map


# ======================================================================
# FastFlowMethod — backbone-agnostic training/inference wrapper
# ======================================================================

class FastFlowMethod:
    """
    Main method class with fit() / predict() / measure_fps().

    When all enhancement toggles are off (default), this runs the exact
    anomalib FastFlow algorithm. Enhancements are bound to parameters:
      zero_init=False       → anomalib (no zero-init)
      gauss_sigma=0.0       → anomalib (no smoothing)
      use_scheduler=False   → anomalib (no LR schedule)
      channel_cap=None      → anomalib (no channel reduction)
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
            # Enhancement toggles (default=off matches vanilla anomalib)
            zero_init=False,
            gauss_sigma=0.0,
            use_scheduler=False,
            channel_cap=None,
    ):
        # Moves extractor to device and sets eval mode and freeze it
        self.extractor = extractor.to(device).eval()
        for p in self.extractor.parameters():
            p.requires_grad = False

        # Set parameters
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

        # Enhancement toggles
        self.zero_init = zero_init
        self.use_scheduler = use_scheduler
        self.channel_cap = channel_cap

        # built them in fit()
        self.reducers = None
        self.norms = None
        self.fast_flow_blocks = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = FastflowLoss()
        self.anomaly_map_generator = AnomalyMapGenerator(self.input_size, sigma=gauss_sigma).to(self.device)

        # Best-epoch tracking (used when eval_fn is provided to fit())
        self._best_metric = -float("inf")
        self._best_epoch = -1
        self._best_reducers_state = None
        self._best_norms_state = None
        self._best_blocks_state = None

    def _build(self, train_loader):
        """Run one forward pass to discover feature map shapes, then build NF blocks."""
        # Takes one batch for feature shapes we need only image
        image, _, _ = next(iter(FlowTupleLoader(train_loader)))
        image = image.to(self.device)

        # Extract features without gradients (l1,l2,l3)
        with torch.no_grad():
            feats = self.extractor(image)

        # Collect channel counts and spatial sizes for each scale
        channels = []
        scales = []
        for key in ["l1", "l2", "l3"]:
            f = feats[key]
            _, C, H, W = f.shape
            channels.append(C)
            scales.append((H, W))

        # Channel reducers for wide-channel backbones (e.g. ShuffleNet 240/480/960)
        # Trainable 1x1 conv + BN that project to lower dimension before the NF
        # Enhancement: when channel_cap=None, all reducers are nn.Identity() (no effect)
        self.reducers = nn.ModuleList()
        for i in range(len(channels)):
            C = channels[i]
            if self.channel_cap is not None and C > self.channel_cap:
                self.reducers.append(nn.Sequential(
                    nn.Conv2d(C, self.channel_cap, 1, bias=False),
                    nn.BatchNorm2d(self.channel_cap),
                ))
                if self.verbose:
                    print(f"  Channel reduction: l{i+1} {C} -> {self.channel_cap}")
                channels[i] = self.channel_cap  # update for norm/NF creation
            else:
                self.reducers.append(nn.Identity())
        self.reducers = self.reducers.to(self.device)

        # LayerNorm per feature scale (anomalib-exact: elementwise_affine=True)
        self.norms = nn.ModuleList()
        for i, key in enumerate(["l1", "l2", "l3"]):
            C = channels[i]
            H, W = scales[i]
            self.norms.append(
                nn.LayerNorm([C, H, W], elementwise_affine=True)
            )
        self.norms = self.norms.to(self.device)

        # 2D normalizing flow block per feature scale
        # When zero_init=False, this calls the exact anomalib create_fast_flow_block
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
                    zero_init=self.zero_init,
                )
            )
        self.fast_flow_blocks = self.fast_flow_blocks.to(self.device)

        # Anomalib-exact: Adam(lr=0.001, weight_decay=1e-5)
        params = (list(self.fast_flow_blocks.parameters())
                  + list(self.norms.parameters())
                  + list(self.reducers.parameters()))
        self.optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        # Enhancement: cosine annealing LR schedule (not in anomalib)
        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=self.meta_epochs, eta_min=self.lr * 1e-3
            )

        # Prints verification info
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
    # Best-epoch helpers: save / restore model state on CPU
    # ------------------------------------------------------------------

    def _save_best_state(self):
        """Deep-copy current reducers + norms + flow block weights to CPU."""
        self._best_reducers_state = copy.deepcopy(self.reducers.state_dict())
        self._best_norms_state = copy.deepcopy(self.norms.state_dict())
        self._best_blocks_state = copy.deepcopy(self.fast_flow_blocks.state_dict())

    def _load_best_state(self):
        """Restore previously saved best weights."""
        self.reducers.load_state_dict(self._best_reducers_state)
        self.norms.load_state_dict(self._best_norms_state)
        self.fast_flow_blocks.load_state_dict(self._best_blocks_state)

    def fit(self, train_loader, eval_fn=None, eval_every=10, early_stopping_patience=0):
        """
        Train FastFlow on normal images.

        Args:
            train_loader: DataLoader with normal training images.
            eval_fn: Optional callable returning a scalar metric (higher=better).
                     Called every `eval_every` epochs; best model is restored at end.
            eval_every: How often (in epochs) to run eval_fn. Default 10.
            early_stopping_patience: Stop training if the monitored metric does not
                     improve for this many evaluation rounds. 0 = disabled.
                     Anomalib default config uses patience=3.
        """
        if self.fast_flow_blocks is None:
            self._build(train_loader)

        # Reset best-epoch tracking for this training run
        self._best_metric = -float("inf")
        self._best_epoch = -1
        self._best_norms_state = None
        self._best_blocks_state = None
        no_improve_count = 0  # for early stopping

        # Wrap loader and set modules to train mode.
        train_loader_t = FlowTupleLoader(train_loader)
        self.reducers.train()
        self.norms.train()
        self.fast_flow_blocks.train()

        for epoch in range(self.meta_epochs):
            epoch_loss = 0.0
            batch_count = 0

            # Iterate batches
            for image, _, _ in train_loader_t:
                image = image.to(self.device)

                # feature extraction from frozen backbone
                with torch.no_grad():
                    feats = self.extractor(image)

                # Collects three scales
                features = [feats["l1"], feats["l2"], feats["l3"]]

                # For each scale: reduce channels (if needed), normalize, flow forward.
                hidden_variables = []
                log_jacobians = []
                for i, feature in enumerate(features):
                    feature = self.reducers[i](feature)
                    feature = self.norms[i](feature)
                    hidden_variable, log_jacobian = self.fast_flow_blocks[i](feature)
                    hidden_variables.append(hidden_variable)
                    log_jacobians.append(log_jacobian)

                # compute loss (anomalib-exact FastflowLoss)
                loss = self.criterion(hidden_variables, log_jacobians)

                # Backprop (anomalib-exact: no gradient clipping)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            # Enhancement: cosine annealing scheduler step
            if self.scheduler is not None:
                self.scheduler.step(epoch)

            # print mean loss and lr
            if self.verbose:
                mean_loss = epoch_loss / max(batch_count, 1)
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    "Epoch: {:d} \t train loss: {:.4f}, lr={:.6f}".format(
                        epoch, mean_loss, current_lr
                    )
                )

            # Periodic evaluation for best-epoch selection + early stopping
            if eval_fn is not None and (epoch + 1) % eval_every == 0:
                self.reducers.eval()
                self.norms.eval()
                self.fast_flow_blocks.eval()
                result = eval_fn()
                # Support both scalar and (combined, img, pix) tuple
                if isinstance(result, tuple):
                    metric, img_auc, pix_auc = result
                else:
                    metric, img_auc, pix_auc = result, None, None
                if self.verbose:
                    if img_auc is not None:
                        combined = (img_auc + pix_auc) / 2
                        detail = f"img={img_auc:.4f}  pix={pix_auc:.4f}  combined={combined:.4f}"
                    else:
                        detail = f"metric={metric:.4f}"
                    best_flag = "  ★ new best" if metric > self._best_metric else ""
                    print(f"  >> eval @ epoch {epoch}: {detail}{best_flag}")
                if metric > self._best_metric:
                    self._best_metric = metric
                    self._best_epoch = epoch
                    self._save_best_state()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                # Early stopping (anomalib default config: patience=3, monitor=pixel_AUROC)
                if early_stopping_patience > 0 and no_improve_count >= early_stopping_patience:
                    if self.verbose:
                        print(f"\n  Early stopping at epoch {epoch}: no improvement "
                              f"for {no_improve_count} eval rounds (patience={early_stopping_patience})")
                    break
                self.reducers.train()
                self.norms.train()
                self.fast_flow_blocks.train()

        # Restore best model if eval_fn was used and a best was found
        if eval_fn is not None and self._best_norms_state is not None:
            self._load_best_state()
            if self.verbose:
                print(f"\n  Restored best model from epoch {self._best_epoch} "
                      f"(combined={self._best_metric:.4f})")

        #  training is done
        self.reducers.eval()
        self.norms.eval()
        self.fast_flow_blocks.eval()


    @torch.no_grad()
    def predict(self, test_loader):
        """Compute anomaly scores and maps for the test."""
        if self.fast_flow_blocks is None:
            raise RuntimeError("Call fit() first.")

        self.reducers.eval()
        self.norms.eval()
        self.fast_flow_blocks.eval()

        test_loader_t = FlowTupleLoader(test_loader)

        all_scores = []
        all_maps = []

        for image, _, _ in test_loader_t:
            image = image.to(self.device)

            # Extract features
            feats = self.extractor(image)
            features = [feats["l1"], feats["l2"], feats["l3"]]

            # Forward through reducers + norms + NF blocks
            hidden_variables = []
            for i, feature in enumerate(features):
                feature = self.reducers[i](feature)
                feature = self.norms[i](feature)
                hidden_variable, _log_jacobian = self.fast_flow_blocks[i](feature)
                hidden_variables.append(hidden_variable)

            # Generate anomaly map (uses anomalib core, optionally with Gaussian smoothing)
            anomaly_map = self.anomaly_map_generator(hidden_variables)  # (B, 1, H, W)

            # Image-level score: max value in the anomaly map (anomalib-exact)
            pred_score = torch.amax(anomaly_map, dim=(-2, -1))
            pred_score = pred_score.squeeze(1)

            all_scores.append(pred_score.cpu())
            all_maps.append(anomaly_map.cpu())

        # Concatenates and returns all scores and maps
        scores = torch.cat(all_scores, dim=0)
        maps = torch.cat(all_maps, dim=0)

        return scores, maps

    @torch.no_grad()
    def measure_fps(self, test_loader):
        """
        Measure FPS with warm-up and CUDA sync.
        Matches FastFlow paper methodology: reports "additional inference time"
        (NF + anomaly map only, excluding backbone) as well as encoder-only
        and full pipeline FPS.
        """
        if self.fast_flow_blocks is None:
            raise RuntimeError("Call fit() first.")

        test_loader_t = FlowTupleLoader(test_loader)
        use_cuda = self.device.type == "cuda"
        A = len(test_loader.dataset)

        self.reducers.eval()
        self.norms.eval()
        self.fast_flow_blocks.eval()

        # 1) Warm-up: full pass
        for image, _, _ in test_loader_t:
            image = image.to(self.device)
            feats = self.extractor(image)
            features = [feats["l1"], feats["l2"], feats["l3"]]
            hidden_variables = []
            for i, feature in enumerate(features):
                feature = self.reducers[i](feature)
                feature = self.norms[i](feature)
                z, _ = self.fast_flow_blocks[i](feature)
                hidden_variables.append(z)
            _ = self.anomaly_map_generator(hidden_variables)

        # 2) Encoder-only timing (backbone)
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        for image, _, _ in test_loader_t:
            image = image.to(self.device)
            _ = self.extractor(image)
        if use_cuda:
            torch.cuda.synchronize()
        time_enc = time.time() - t0

        # 3) "Additional inference time" (NF + anomaly map only, no backbone)
        cached_features = []
        for image, _, _ in test_loader_t:
            image = image.to(self.device)
            feats = self.extractor(image)
            cached_features.append([feats["l1"], feats["l2"], feats["l3"]])

        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        for features in cached_features:
            hidden_variables = []
            for i, feature in enumerate(features):
                feature = self.reducers[i](feature)
                feature = self.norms[i](feature)
                z, _ = self.fast_flow_blocks[i](feature)
                hidden_variables.append(z)
            _ = self.anomaly_map_generator(hidden_variables)
        if use_cuda:
            torch.cuda.synchronize()
        time_additional = time.time() - t0

        # 4) Full pipeline timing (encoder + NF + anomaly map)
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        for image, _, _ in test_loader_t:
            image = image.to(self.device)
            feats = self.extractor(image)
            features = [feats["l1"], feats["l2"], feats["l3"]]
            hidden_variables = []
            for i, feature in enumerate(features):
                feature = self.reducers[i](feature)
                feature = self.norms[i](feature)
                z, _ = self.fast_flow_blocks[i](feature)
                hidden_variables.append(z)
            _ = self.anomaly_map_generator(hidden_variables)
        if use_cuda:
            torch.cuda.synchronize()
        time_all = time.time() - t0

        fps_enc = A / time_enc
        fps_additional = A / time_additional
        fps_all = A / time_all

        return {
            "fps_encoder": round(fps_enc, 2),
            "fps_additional": round(fps_additional, 2),
            "fps_all": round(fps_all, 2),
        }
