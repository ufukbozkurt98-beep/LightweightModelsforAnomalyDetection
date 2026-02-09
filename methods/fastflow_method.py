import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# to build normalizing flows (invertible networks).
import FrEIA.framework as Ff
import FrEIA.modules as Fm

from utils.data_adapters import TupleLoader


def subnet_conv_func(kernel_size, hidden_ratio):
    """
    This builds the small sub-network used inside coupling blocks
    It returns a function that builds a small nn.Sequential
    """

    def subnet_conv(in_channels, out_channels):
        """"
        Inner builder: creates the subnet given input/output channels
        """
        # Hidden channel count
        hidden_channels = int(in_channels * hidden_ratio)
        # Padding to keep spatial size for 3×3.
        padding = kernel_size // 2
        last_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size, padding=padding)

        # Zero init makes the coupling start near identity for stable training
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)

        # Subnet refers respectively Conv, BN, ReLU, Conv
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            last_conv,
        )

    return subnet_conv


def create_fast_flow_block(input_dimensions, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
    """
    Builds a flow for a feature map
    """
    # Creates an invertible sequence
    nodes = Ff.SequenceINN(*input_dimensions)
    for i in range(flow_steps):
        # Alternate kernel for even steps it is 3x3, for odd steps it is1x1
        # if conv3x3_only is True, then always 3x3
        kernel_size = 1 if i % 2 == 1 and not conv3x3_only else 3

        # Appends an invertible
        nodes.append(
            Fm.AllInOneBlock,
            # Use subnet function for the coupling subnet
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            # Clamping for stability
            affine_clamping=clamp,
            # Controls permutation behavior in the block
            permute_soft=False,
        )
    return nodes  # Returns the built invertible sequence


class FastflowLoss(nn.Module):
    """
    FastFlow loss function.
    """

    @staticmethod
    def forward(hidden_variables, jacobians):
        # Initializes loss on correct device
        loss = torch.tensor(0.0, device=hidden_variables[0].device)
        # Loop over scales
        for hidden_variable, jacobian in zip(hidden_variables, jacobians):
            # Standard flow negative log-likelihood term
            loss += torch.mean(
                0.5 * torch.sum(hidden_variable ** 2, dim=(1, 2, 3)) - jacobian
            )
        return loss


class AnomalyMapGenerator(nn.Module):
    """
    Anomaly map generator
    Generates pixel-level anomaly heatmap
    """

    def __init__(self, input_size, sigma=1.5):
        super().__init__()
        # check input size is a tuple
        self.input_size = tuple(input_size) if not isinstance(input_size, tuple) else input_size
        # Store Gaussian kernel as a buffer
        self.register_buffer("_gauss_kernel", self._make_gaussian_kernel(sigma))

    @staticmethod
    def _make_gaussian_kernel(sigma, channels=1):
        """Builds fixed 2D Gaussian filter"""
        # Kernel size
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        # Centered coordinate vector
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        # 1D Gaussian values
        gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        # Creates 2D kernel via outer product
        gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
        # Normalize to sum=1
        gauss_2d = gauss_2d / gauss_2d.sum()
        # Reshapes for conv2d weight format
        return gauss_2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

    def forward(self, hidden_variables):
        # Collect per-scale maps
        flow_maps = []
        # Iterate scales
        for hidden_variable in hidden_variables:
            # Computes log-prob proxy from latent energy
            log_prob = -torch.mean(hidden_variable ** 2, dim=1, keepdim=True) * 0.5
            # Exponentiates to get probability
            prob = torch.exp(log_prob)
            # Uses prob as anomaly score map and upsamples to input resolution
            flow_map = F.interpolate(
                input=-prob,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )
            flow_maps.append(flow_map)
        # Stack and average across scales
        flow_maps = torch.stack(flow_maps, dim=-1)
        anomaly_map = torch.mean(flow_maps, dim=-1)

        # Reflect padding before smoothing
        pad = self._gauss_kernel.shape[-1] // 2
        anomaly_map = F.pad(anomaly_map, (pad, pad, pad, pad), mode="reflect")
        # Applies Gaussian smoothing and returns map
        anomaly_map = F.conv2d(anomaly_map, self._gauss_kernel, groups=1)
        return anomaly_map

class FastFlowMethod:
    """
    Main method class with fit() / predict()
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

        # built them in fit()
        self.norms = None
        self.fast_flow_blocks = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = FastflowLoss()
        self.anomaly_map_generator = AnomalyMapGenerator(self.input_size).to(self.device)

    def _build(self, train_loader):
        """Run one forward pass to discover feature map shapes, then build NF blocks."""
        # Takes one batch for feature shapes we need only image
        image, _, _ = next(iter(TupleLoader(train_loader)))
        image = image.to(self.device)

        # Extract features without gradients (l1,l2,l2)
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

        # LayerNorm per feature scale for stability
        self.norms = nn.ModuleList()
        for i, key in enumerate(["l1", "l2", "l3"]):
            C = channels[i]
            H, W = scales[i]
            self.norms.append(
                nn.LayerNorm([C, H, W], elementwise_affine=True)
            )
        self.norms = self.norms.to(self.device)

        # 2D normalizing flow block per feature scale
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

        # Optimize NF block and LayerNorm parameters, backbone is still frozen
        params = list(self.fast_flow_blocks.parameters()) + list(self.norms.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        # Smooth cosine LR decay
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.meta_epochs, eta_min=self.lr * 1e-3
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

    def fit(self, train_loader):
        """Train FastFlow on normal images."""
        if self.fast_flow_blocks is None:
            self._build(train_loader)

        # Wrap loader and set modules to train mode.
        train_loader_t = TupleLoader(train_loader)
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

                # For each scale normalize, flow forward, collect z and jacobian.
                hidden_variables = []
                log_jacobians = []
                for i, feature in enumerate(features):
                    feature = self.norms[i](feature)
                    hidden_variable, log_jacobian = self.fast_flow_blocks[i](feature)
                    hidden_variables.append(hidden_variable)
                    log_jacobians.append(log_jacobian)

                # compute loss
                loss = self.criterion(hidden_variables, log_jacobians)

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    list(self.fast_flow_blocks.parameters()) + list(self.norms.parameters()),
                    max_norm=1.0,
                )
                # Update parameters
                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1
            # Step LR scheduler
            self.scheduler.step()

            # print mean loss and lr
            if self.verbose:
                mean_loss = epoch_loss / max(batch_count, 1)
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    "Epoch: {:d} \t train loss: {:.4f}, lr={:.6f}".format(
                        epoch, mean_loss, current_lr
                    )
                )

        #  training is done
        self.norms.eval()
        self.fast_flow_blocks.eval()



    @torch.no_grad()
    def predict(self, test_loader):
        """Compute anomaly scores and maps for the test"""
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

            # Forward through NF blocks
            hidden_variables = []
            for i, feature in enumerate(features):
                feature = self.norms[i](feature)
                hidden_variable, _log_jacobian = self.fast_flow_blocks[i](feature)
                hidden_variables.append(hidden_variable)

            # Generate anomaly map
            anomaly_map = self.anomaly_map_generator(hidden_variables)  # (B, 1, H, W)

            # Image-level score, max value in the anomaly map
            pred_score = torch.amax(anomaly_map, dim=(-2, -1))
            pred_score = pred_score.squeeze(1)

            all_scores.append(pred_score.cpu())
            all_maps.append(anomaly_map.cpu())

        # Concatenates and returns all scores and maps
        scores = torch.cat(all_scores, dim=0)
        maps = torch.cat(all_maps, dim=0)

        return scores, maps
