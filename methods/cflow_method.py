import math
import numpy as np
import torch
import torch.nn.functional as F
from methods.cflow_freia import load_decoder_arch, activation
from utils.cflow_utils import train_meta_epoch, test_meta_epoch
from utils.data_adapters import TupleLoader


class TimmActivationEncoder(torch.nn.Module):
    """
    In the original CFLOW, the encoder fills activation['layer0', 'layer1', 'layer2'] using forward hooks.
    In this project, the extractor returns features 'l1', 'l2', 'l3', and we map them to activation.
    """

    def __init__(self, extractor, pool_layers):
        super().__init__()
        self.extractor = extractor
        self.pool_layers = pool_layers  # ['layer0','layer1','layer2']

    @torch.no_grad()
    def forward(self, x):
        feats = self.extractor(x)  # dict: l1,l2,l3
        # Save features to the global activation dict (It is initialized in cflow_freia.py globally) using CFLOW layer
        # names.
        activation[self.pool_layers[0]] = feats["l1"].detach()
        activation[self.pool_layers[1]] = feats["l2"].detach()
        activation[self.pool_layers[2]] = feats["l3"].detach()
        return feats


class _C:
    """Instead of Argparse config object, a simple container with same attributes """
    pass


class CFlowMethod:
    """
    Main class of CFlow method
    """
    # Takes hyperparameters and sets them.
    def __init__(
            self,
            extractor,
            device="cuda",
            enc_arch="timm",
            dec_arch="freia-cflow",
            pool_layers=3,
            coupling_blocks=8,
            condition_vec=128,
            clamp_alpha=1.9,
            lr=2e-4,
            meta_epochs=25,
            sub_epochs=8,
            N=256,
            input_size=256,
            verbose=True,
            hide_tqdm_bar=False,
    ):
        # Freeze the extractor (eval mode). Only the decoders will learn.
        self.extractor = extractor.to(device).eval()
        for p in self.extractor.parameters():
            p.requires_grad = False

        # Simple container that gets the attributes
        c = _C()
        c.enc_arch = enc_arch
        c.dec_arch = dec_arch
        c.pool_layers = pool_layers
        c.coupling_blocks = coupling_blocks
        c.condition_vec = condition_vec
        c.clamp_alpha = clamp_alpha
        c.lr = lr
        c.meta_epochs = meta_epochs
        c.sub_epochs = sub_epochs
        c.verbose = verbose
        c.hide_tqdm_bar = hide_tqdm_bar
        c.crp_size = (input_size, input_size)

        # lr change over epochs instead of staying constant
        # Sama logic as original CFLOW structure
        # Define learning rate schedule
        c.lr_decay_epochs = [i * c.meta_epochs // 100 for i in [50, 75, 90]]
        # Decrease lr 10x
        c.lr_decay_rate = 0.1
        # Warm up first at 2 epochs. Increase lr slowly
        c.lr_warm_epochs = 2
        c.lr_warm = True
        # Decrease lr softly
        c.lr_cosine = True
        # Warm up
        if c.lr_warm:
            # Set initial lr before warm up
            c.lr_warmup_from = c.lr / 10.0
            if c.lr_cosine:
                # Minimum lr after cosine schedule
                eta_min = c.lr * (c.lr_decay_rate ** 3)
                # Calculating upper limit of warmup
                c.lr_warmup_to = eta_min + (c.lr - eta_min) * (
                        1 + math.cos(math.pi * c.lr_warm_epochs / c.meta_epochs)
                ) / 2
            else:
                c.lr_warmup_to = c.lr
        # Device information
        c.device = torch.device(device)

        # Set configuration ( the container)
        self.c = c
        # Set fiber batch for memory usage
        self.N = N
        # Set pool layer names
        self.pool_layers = ["layer" + str(i) for i in range(pool_layers)]
        # Call encoder wrapper for timm backbones (MobileNetV3 / EfficientNet-Lite / MobileViT).
        self.encoder = TimmActivationEncoder(self.extractor, self.pool_layers).to(device).eval()

        # At the beginning decoder and optimizer are None
        self.decoders = None
        self.optimizer = None

    # Initialize model for training
    def _build(self, train_loader):
        # After converting the train_loader to tuple, take the first batch
        # Take image others(label and mask) are not important for this method
        image, _, _ = next(iter(TupleLoader(train_loader)))
        image = image.to(self.c.device)

        # Run encoder once without training to fill the variables in the encoder
        with torch.no_grad():
            _ = self.encoder(image)

        # Get the channel size (C) of each saved feature map
        pool_dims = [activation[layer].size(1) for layer in self.pool_layers]

        # Create one decoder for each feature level
        self.decoders = [load_decoder_arch(self.c, dim_in).to(self.c.device) for dim_in in pool_dims]
        # Collect all decoder parameters for the optimizer
        params = []
        for d in self.decoders:
            params += list(d.parameters())
        # Optimizer updates only decoder weights
        self.optimizer = torch.optim.Adam(params, lr=self.c.lr)

    def fit(self, train_loader):
        # Build decoders and optimizer if not built yet
        if self.decoders is None:
            self._build(train_loader)

        # Ensure the loader yields (image, label, mask) tuples
        train_loader_t = TupleLoader(train_loader)

        # Train for the given number of meta-epochs
        for epoch in range(self.c.meta_epochs):
            print("Train meta epoch: {}".format(epoch))
            # Run one meta-epoch training loop
            train_meta_epoch(
                self.c,
                epoch,
                train_loader_t,
                self.encoder,
                self.decoders,
                self.optimizer,
                self.pool_layers,
                self.N,
            )
        # Switch decoders to evaluation mode after training
        self.decoders = [d.eval() for d in self.decoders]

    @torch.no_grad()
    def predict(self, test_loader):
        # Predict anomaly scores and anomaly maps for the test set

        if self.decoders is None:
            raise RuntimeError("Call fit() first.") # Need trained decoders

        # Ensure test loader yields (image, label, mask)
        test_loader_t = TupleLoader(test_loader)

        # Run one test epoch and get feature map sizes and distances
        height, width, _, test_dist, _, _ = test_meta_epoch(
            self.c, 0, test_loader_t, self.encoder, self.decoders, self.pool_layers, self.N
        )

        # Build a score map for each pooled layer
        test_map = [list() for _ in self.pool_layers]
        for l, _p in enumerate(self.pool_layers):
            # Convert distances to a tensor
            test_norm = torch.tensor(test_dist[l], dtype=torch.double)
            # Normalize (max 0)
            test_norm -= torch.max(test_norm)
            # Convert to positive scores
            test_prob = torch.exp(test_norm)
            test_mask = test_prob.reshape(-1, height[l], width[l])

            # Upsample to image crop size and convert to numpy
            test_map[l] = (
                F.interpolate(
                    test_mask.unsqueeze(1),
                    size=self.c.crp_size,
                    mode="bilinear",
                    align_corners=True,
                )
                .squeeze(1)
                .cpu()
                .numpy()
            )

        # Sum score maps from all layers
        score_map = np.zeros_like(test_map[0])
        for l, _p in enumerate(self.pool_layers):
            score_map += test_map[l]

        score_mask = score_map

        # Invert to get final anomaly map
        # higher = more anomalous
        super_mask = score_mask.max() - score_mask

        # Image-level score: max value in the anomaly map
        score_label = np.max(super_mask, axis=(1, 2))

        # Return torch tensors
        scores = torch.from_numpy(score_label.astype(np.float32))
        maps = torch.from_numpy(super_mask.astype(np.float32)).unsqueeze(1)  # (N,1,H,W)
        return scores, maps
