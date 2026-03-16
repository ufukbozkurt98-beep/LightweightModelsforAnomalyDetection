"""
CFlowMethod — wrapper class that connects our backbone system to the
original gudovskiy/cflow-ad training/testing logic.

The only non-original part is TimmActivationEncoder, which replaces
the original forward-hook mechanism with our MultiScaleFeatureExtractor.
All other logic (training loop, testing loop, score computation, LR
schedule, FPS measurement) is called from cflow_utils.py / cflow_freia.py
which are exact copies of the original code.
"""
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.cflow_freia import load_decoder_arch, activation
from utils.cflow_utils import train_meta_epoch, test_meta_epoch, test_meta_fps
from utils.data_adapters import TupleLoader


class TimmActivationEncoder(torch.nn.Module):
    """
    In the original CFLOW, the encoder fills activation['layer0', 'layer1', 'layer2'] using forward hooks.
    In this project, the extractor returns features 'l1', 'l2', 'l3', and we map them to activation.

    Optional channel reduction: for wide-channel backbones (e.g. ShuffleNet 240/480/960),
    fixed 1x1 conv projections reduce channels before the normalizing flow sees them.
    The reducers are non-trainable (random projection) since CFlow's training loop
    in cflow_utils.py detaches features, preventing gradient flow to the encoder.
    """

    def __init__(self, extractor, pool_layers, reducers=None):
        super().__init__()
        self.extractor = extractor
        self.pool_layers = pool_layers  # ['layer0','layer1','layer2']
        self.reducers = reducers  # nn.ModuleList of [Conv2d or None] or None

    @torch.no_grad()
    def forward(self, x):
        feats = self.extractor(x)  # dict: l1,l2,l3
        # Save features to the global activation dict using CFLOW layer names.
        for i, key in enumerate(["l1", "l2", "l3"]):
            f = feats[key].detach()
            if self.reducers is not None and self.reducers[i] is not None:
                f = self.reducers[i](f)
            activation[self.pool_layers[i]] = f
        return feats


class _C:
    """Instead of Argparse config object, a simple container with same attributes."""
    pass


class CFlowMethod:
    """
    Main class of CFlow method.
    """
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
            channel_cap=None,
    ):
        # Freeze the extractor (eval mode). Only the decoders will learn.
        self.extractor = extractor.to(device).eval()
        for p in self.extractor.parameters():
            p.requires_grad = False

        # Config container — mirrors original argparse config
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
        c.viz = False  # no visualization export (original default)
        c.pro = True   # compute AUPRO (original default)
        c.crp_size = (input_size, input_size)

        # LR schedule — same as original main.py
        c.lr_decay_epochs = [i * c.meta_epochs // 100 for i in [50, 75, 90]]
        c.lr_decay_rate = 0.1
        c.lr_warm_epochs = 2
        c.lr_warm = True
        c.lr_cosine = True
        if c.lr_warm:
            c.lr_warmup_from = c.lr / 10.0
            if c.lr_cosine:
                eta_min = c.lr * (c.lr_decay_rate ** 3)
                c.lr_warmup_to = eta_min + (c.lr - eta_min) * (
                        1 + math.cos(math.pi * c.lr_warm_epochs / c.meta_epochs)
                ) / 2
            else:
                c.lr_warmup_to = c.lr

        c.device = torch.device(device)

        self.c = c
        self.N = N
        self.pool_layers = ["layer" + str(i) for i in range(pool_layers)]

        # Build fixed channel reducers for wide-channel backbones (e.g. ShuffleNet)
        reducers = None
        if channel_cap is not None:
            fc = extractor.feature_channels
            reducer_list = []
            needs_reduction = False
            for key in ["l1", "l2", "l3"]:
                C = fc[key]
                if C > channel_cap:
                    r = nn.Conv2d(C, channel_cap, 1, bias=False)
                    nn.init.kaiming_normal_(r.weight)
                    for p in r.parameters():
                        p.requires_grad = False
                    reducer_list.append(r)
                    needs_reduction = True
                    if verbose:
                        print(f"  Channel reduction: {key} {C} -> {channel_cap}")
                else:
                    reducer_list.append(None)
            if needs_reduction:
                reducers = nn.ModuleList(reducer_list).to(device)
                reducers.eval()

        self.encoder = TimmActivationEncoder(self.extractor, self.pool_layers, reducers=reducers).to(device).eval()

        self.decoders = None
        self.optimizer = None

        # Best-epoch tracking
        self._best_metric = -float("inf")
        self._best_epoch = -1
        self._best_decoder_states = None

    def _build(self, train_loader):
        image, _, _ = next(iter(TupleLoader(train_loader)))
        image = image.to(self.c.device)

        with torch.no_grad():
            _ = self.encoder(image)

        pool_dims = [activation[layer].size(1) for layer in self.pool_layers]

        self.decoders = [load_decoder_arch(self.c, dim_in).to(self.c.device) for dim_in in pool_dims]
        # Collect all decoder parameters — same as original
        params = list(self.decoders[0].parameters())
        for l in range(1, len(self.decoders)):
            params += list(self.decoders[l].parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.c.lr)

        print("\n" + "=" * 60)
        print("CFLOW Feature Extraction Verification:")
        print("=" * 60)
        for i, layer in enumerate(self.pool_layers):
            feat = activation[layer]
            print(f"{layer} -> Shape: {feat.shape}, Channels: {pool_dims[i]}")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Best-epoch helpers
    # ------------------------------------------------------------------

    def _save_best_state(self):
        self._best_decoder_states = [copy.deepcopy(d.state_dict()) for d in self.decoders]

    def _load_best_state(self):
        for d, state in zip(self.decoders, self._best_decoder_states):
            d.load_state_dict(state)

    def fit(self, train_loader, eval_fn=None, eval_every=5):
        if self.decoders is None:
            self._build(train_loader)

        self._best_metric = -float("inf")
        self._best_epoch = -1
        self._best_decoder_states = None

        train_loader_t = TupleLoader(train_loader)

        for epoch in range(self.c.meta_epochs):
            print("Train meta epoch: {}".format(epoch))
            train_meta_epoch(
                self.c, epoch, train_loader_t,
                self.encoder, self.decoders, self.optimizer,
                self.pool_layers, self.N,
            )

            if eval_fn is not None and (epoch + 1) % eval_every == 0:
                self.decoders = [d.eval() for d in self.decoders]
                result = eval_fn()
                if isinstance(result, tuple):
                    metric, img_auc, pix_auc = result
                else:
                    metric, img_auc, pix_auc = result, None, None
                if self.c.verbose:
                    detail = f"metric={metric:.4f}"
                    if img_auc is not None:
                        detail += f"  img={img_auc:.4f}  pix={pix_auc:.4f}"
                    best_flag = "  ★ new best" if metric > self._best_metric else ""
                    best_info = f"  (best so far: epoch {self._best_epoch}, metric={self._best_metric:.4f})" if self._best_epoch >= 0 and metric <= self._best_metric else ""
                    print(f"  >> eval @ epoch {epoch}: {detail}{best_flag}{best_info}")
                if metric > self._best_metric:
                    self._best_metric = metric
                    self._best_epoch = epoch
                    self._save_best_state()
                self.decoders = [d.train() for d in self.decoders]

        if eval_fn is not None and self._best_decoder_states is not None:
            self._load_best_state()
            if self.c.verbose:
                print(f"\n  Restored best model from epoch {self._best_epoch} "
                      f"(metric={self._best_metric:.4f})")

        self.decoders = [d.eval() for d in self.decoders]

    @torch.no_grad()
    def measure_fps(self, test_loader):
        """
        Measure FPS using the exact same test_meta_fps from the original
        CFlow-AD paper (Table 6).
        """
        if self.decoders is None:
            raise RuntimeError("Call fit() first.")

        test_loader_t = TupleLoader(test_loader)

        fps_enc, fps_all = test_meta_fps(
            self.c, 0, test_loader_t, self.encoder,
            self.decoders, self.pool_layers, self.N
        )

        return {
            "fps_encoder": round(fps_enc, 2),
            "fps_all": round(fps_all, 2),
        }

    @torch.no_grad()
    def predict(self, test_loader):
        """
        Predict anomaly scores and maps.
        Score computation is exact copy from original train.py train() function.
        """
        if self.decoders is None:
            raise RuntimeError("Call fit() first.")

        test_loader_t = TupleLoader(test_loader)

        height, width, test_image_list, test_dist, gt_label_list, gt_mask_list = test_meta_epoch(
            self.c, 0, test_loader_t, self.encoder, self.decoders, self.pool_layers, self.N
        )

        # --- Score computation: exact copy from original train.py ---
        test_map = [list() for p in self.pool_layers]
        for l, p in enumerate(self.pool_layers):
            test_norm = torch.tensor(test_dist[l], dtype=torch.double)  # EHWx1
            test_norm -= torch.max(test_norm)  # normalize likelihoods to (-Inf:0]
            test_prob = torch.exp(test_norm)  # convert to probs in range [0:1]
            test_mask = test_prob.reshape(-1, height[l], width[l])
            test_mask = test_prob.reshape(-1, height[l], width[l])  # duplicate line in original
            # upsample
            test_map[l] = F.interpolate(test_mask.unsqueeze(1),
                size=self.c.crp_size, mode='bilinear', align_corners=True).squeeze().numpy()
        # score aggregation
        score_map = np.zeros_like(test_map[0])
        for l, p in enumerate(self.pool_layers):
            score_map += test_map[l]
        score_mask = score_map
        # invert probs to anomaly scores
        super_mask = score_mask.max() - score_mask
        # image-level score: max value in the anomaly map
        score_label = np.max(super_mask, axis=(1, 2))

        # Return torch tensors
        scores = torch.from_numpy(score_label.astype(np.float32))
        maps = torch.from_numpy(super_mask.astype(np.float32)).unsqueeze(1)  # (N,1,H,W)
        return scores, maps
