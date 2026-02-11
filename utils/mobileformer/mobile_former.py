"""Mobile-Former model with multi-scale feature extraction.

Ported from the official implementation:
  https://github.com/AAboys/MobileFormer

Paper: Mobile-Former: Bridging MobileNet and Transformer (CVPR 2022)
       https://arxiv.org/abs/2108.05895
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .dna_blocks import (
    DnaBlock,
    DnaBlock3,
    Local2Global,
    MergeClassifier,
    _make_divisible,
)

# ---------------------------------------------------------------------------
# Common kwargs shared by all variants
# ---------------------------------------------------------------------------
_COMMON_KWARGS = dict(
    cnn_drop_path_rate=0.1,
    dw_conv="dw",
    kernel_size=(3, 3),
    cnn_exp=(6, 4),
    cls_token_num=1,
    hyper_token_id=0,
    hyper_reduction_ratio=4,
    attn_num_heads=2,
    gbr_norm="post",
    mlp_token_exp=4,
    gbr_before_skip=False,
    gbr_drop=[0.0, 0.0],
    last_act="relu",
    remove_proj_local=True,
)

# ---------------------------------------------------------------------------
# Variant configurations
#   Each entry: (stem_chs, token_num, token_dim, num_features, block_args, extra_kwargs)
#   block_args rows: [block_cls_name, e1, channels, repeat, stride, e2]
# ---------------------------------------------------------------------------

_VARIANT_CONFIGS = {
    "mobileformer_508m": dict(
        stem_chs=24,
        token_num=6,
        token_dim=192,
        num_features=1920,
        block_args=[
            ["DnaBlock3", 2, 24, 1, 1, 0],
            ["DnaBlock3", 6, 40, 1, 2, 4],
            ["DnaBlock", 3, 40, 1, 1, 3],
            ["DnaBlock3", 6, 72, 1, 2, 4],
            ["DnaBlock", 3, 72, 1, 1, 3],
            ["DnaBlock3", 6, 128, 1, 2, 4],
            ["DnaBlock", 4, 128, 1, 1, 4],
            ["DnaBlock", 6, 176, 1, 1, 4],
            ["DnaBlock", 6, 176, 1, 1, 4],
            ["DnaBlock3", 6, 240, 1, 2, 4],
            ["DnaBlock", 6, 240, 1, 1, 4],
            ["DnaBlock", 6, 240, 1, 1, 4],
        ],
    ),
    "mobileformer_294m": dict(
        stem_chs=16,
        token_num=6,
        token_dim=192,
        num_features=1920,
        block_args=[
            ["DnaBlock3", 2, 16, 1, 1, 0],
            ["DnaBlock3", 6, 24, 1, 2, 4],
            ["DnaBlock", 4, 24, 1, 1, 4],
            ["DnaBlock3", 6, 48, 1, 2, 4],
            ["DnaBlock", 4, 48, 1, 1, 4],
            ["DnaBlock3", 6, 96, 1, 2, 4],
            ["DnaBlock", 4, 96, 1, 1, 4],
            ["DnaBlock", 6, 128, 1, 1, 4],
            ["DnaBlock", 6, 128, 1, 1, 4],
            ["DnaBlock3", 6, 192, 1, 2, 4],
            ["DnaBlock", 6, 192, 1, 1, 4],
            ["DnaBlock", 6, 192, 1, 1, 4],
        ],
    ),
    "mobileformer_214m": dict(
        stem_chs=12,
        token_num=6,
        token_dim=192,
        num_features=1600,
        block_args=[
            ["DnaBlock3", 2, 12, 1, 1, 0],
            ["DnaBlock3", 6, 20, 1, 2, 4],
            ["DnaBlock", 3, 20, 1, 1, 4],
            ["DnaBlock3", 6, 40, 1, 2, 4],
            ["DnaBlock", 4, 40, 1, 1, 4],
            ["DnaBlock3", 6, 80, 1, 2, 4],
            ["DnaBlock", 4, 80, 1, 1, 4],
            ["DnaBlock", 6, 112, 1, 1, 4],
            ["DnaBlock", 6, 112, 1, 1, 4],
            ["DnaBlock3", 6, 160, 1, 2, 4],
            ["DnaBlock", 6, 160, 1, 1, 4],
            ["DnaBlock", 6, 160, 1, 1, 4],
        ],
    ),
    "mobileformer_151m": dict(
        stem_chs=12,
        token_num=6,
        token_dim=192,
        num_features=1280,
        block_args=[
            ["DnaBlock3", 2, 12, 1, 1, 0],
            ["DnaBlock3", 6, 16, 1, 2, 4],
            ["DnaBlock", 3, 16, 1, 1, 3],
            ["DnaBlock3", 6, 32, 1, 2, 4],
            ["DnaBlock", 3, 32, 1, 1, 3],
            ["DnaBlock3", 6, 64, 1, 2, 4],
            ["DnaBlock", 4, 64, 1, 1, 4],
            ["DnaBlock", 6, 88, 1, 1, 4],
            ["DnaBlock", 6, 88, 1, 1, 4],
            ["DnaBlock3", 6, 128, 1, 2, 4],
            ["DnaBlock", 6, 128, 1, 1, 4],
            ["DnaBlock", 6, 128, 1, 1, 4],
        ],
    ),
    "mobileformer_96m": dict(
        stem_chs=12,
        token_num=4,
        token_dim=128,
        num_features=1280,
        block_args=[
            ["DnaBlock3", 2, 12, 1, 1, 0],
            ["DnaBlock3", 6, 16, 1, 2, 4],
            ["DnaBlock3", 6, 32, 1, 2, 4],
            ["DnaBlock", 3, 32, 1, 1, 3],
            ["DnaBlock3", 6, 64, 1, 2, 4],
            ["DnaBlock", 4, 64, 1, 1, 4],
            ["DnaBlock", 6, 88, 1, 1, 4],
            ["DnaBlock3", 6, 128, 1, 2, 4],
            ["DnaBlock", 6, 128, 1, 1, 4],
        ],
    ),
    "mobileformer_52m": dict(
        stem_chs=8,
        token_num=3,
        token_dim=128,
        num_features=1024,
        block_args=[
            ["DnaBlock3", 3, 12, 1, 2, 0],
            ["DnaBlock", 3, 12, 1, 1, 3],
            ["DnaBlock3", 6, 24, 1, 2, 4],
            ["DnaBlock", 3, 24, 1, 1, 3],
            ["DnaBlock3", 6, 48, 1, 2, 4],
            ["DnaBlock", 4, 48, 1, 1, 4],
            ["DnaBlock", 6, 64, 1, 1, 4],
            ["DnaBlock3", 6, 96, 1, 2, 4],
            ["DnaBlock", 6, 96, 1, 1, 4],
        ],
    ),
}

# Block class lookup (avoids eval())
_BLOCK_CLS = {
    "DnaBlock": DnaBlock,
    "DnaBlock3": DnaBlock3,
}


class MobileFormer(nn.Module):
    """Mobile-Former: Bridging MobileNet and Transformer.

    When ``num_classes=0`` the classifier head is omitted and only
    ``forward_features()`` is useful (returns ``{"l1", "l2", "l3"}``).
    """

    def __init__(
        self,
        block_args: List[list],
        num_classes: int = 1000,
        img_size: int = 224,
        width_mult: float = 1.0,
        in_chans: int = 3,
        stem_chs: int = 16,
        num_features: int = 1280,
        dw_conv: str = "dw",
        kernel_size: Tuple[int, int] = (3, 3),
        cnn_exp: Tuple[int, int] = (6, 4),
        group_num: int = 1,
        se_flag: list | None = None,
        hyper_token_id: int = 0,
        hyper_reduction_ratio: int = 4,
        token_dim: int = 128,
        token_num: int = 6,
        cls_token_num: int = 1,
        last_act: str = "relu",
        last_exp: int = 6,
        gbr_type: str = "mlp",
        gbr_dynamic: list | None = None,
        gbr_norm: str = "post",
        gbr_ffn: bool = False,
        gbr_before_skip: bool = False,
        gbr_drop: list | None = None,
        mlp_token_exp: int = 4,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        cnn_drop_path_rate: float = 0.0,
        attn_num_heads: int = 2,
        remove_proj_local: bool = True,
    ):
        super().__init__()
        if se_flag is None:
            se_flag = [2, 0, 2, 0]
        if gbr_dynamic is None:
            gbr_dynamic = [False, False, False]
        if gbr_drop is None:
            gbr_drop = [0.0, 0.0]

        cnn_drop_path_rate = drop_path_rate
        mdiv = 8 if width_mult > 1.01 else 4
        self.num_classes = num_classes

        # Global tokens
        self.tokens = nn.Embedding(token_num, token_dim)

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, stem_chs, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_chs),
            nn.ReLU6(inplace=True),
        )
        input_channel = stem_chs

        # Build blocks — keep as a ModuleList so we can iterate manually
        # and capture intermediate features.
        layer_num = len(block_args)
        inp_res = img_size * img_size // 4
        layers: List[nn.Module] = []

        # Track which flat-layer indices correspond to stride-2 transitions
        # so forward_features can tap the right outputs.
        self._stride2_flat_indices: List[int] = []
        self._stage_out_channels_list: List[int] = []
        flat_idx = 0

        for idx, val in enumerate(block_args):
            b, t, c, n, s, t2 = val
            block_cls = _BLOCK_CLS[b]
            t_pair = (t, t2)
            output_channel = (
                _make_divisible(c * width_mult, mdiv)
                if idx > 0
                else _make_divisible(c * width_mult, 4)
            )

            drop_path_prob = drop_path_rate * (idx + 1) / layer_num
            cnn_drop_path_prob = cnn_drop_path_rate * (idx + 1) / layer_num

            layers.append(
                block_cls(
                    input_channel,
                    output_channel,
                    s,
                    t_pair,
                    dw_conv=dw_conv,
                    kernel_size=kernel_size,
                    group_num=group_num,
                    se_flag=se_flag,
                    hyper_token_id=hyper_token_id,
                    hyper_reduction_ratio=hyper_reduction_ratio,
                    token_dim=token_dim,
                    token_num=token_num,
                    inp_res=inp_res,
                    gbr_type=gbr_type,
                    gbr_dynamic=gbr_dynamic,
                    gbr_ffn=gbr_ffn,
                    gbr_before_skip=gbr_before_skip,
                    mlp_token_exp=mlp_token_exp,
                    norm_pos=gbr_norm,
                    drop_path_rate=drop_path_prob,
                    cnn_drop_path_rate=cnn_drop_path_prob,
                    attn_num_heads=attn_num_heads,
                    remove_proj_local=remove_proj_local,
                )
            )

            if s == 2:
                self._stride2_flat_indices.append(flat_idx)
                self._stage_out_channels_list.append(output_channel)

            input_channel = output_channel
            if s == 2:
                inp_res = inp_res // 4
            flat_idx += 1

            for _ in range(1, n):
                layers.append(
                    block_cls(
                        input_channel,
                        output_channel,
                        1,
                        t_pair,
                        dw_conv=dw_conv,
                        kernel_size=kernel_size,
                        group_num=group_num,
                        se_flag=se_flag,
                        hyper_token_id=hyper_token_id,
                        hyper_reduction_ratio=hyper_reduction_ratio,
                        token_dim=token_dim,
                        token_num=token_num,
                        inp_res=inp_res,
                        gbr_type=gbr_type,
                        gbr_dynamic=gbr_dynamic,
                        gbr_ffn=gbr_ffn,
                        gbr_before_skip=gbr_before_skip,
                        mlp_token_exp=mlp_token_exp,
                        norm_pos=gbr_norm,
                        drop_path_rate=drop_path_prob,
                        cnn_drop_path_rate=cnn_drop_path_prob,
                        attn_num_heads=attn_num_heads,
                        remove_proj_local=remove_proj_local,
                    )
                )
                flat_idx += 1

        self.features = nn.ModuleList(layers)

        # Last L2G bridge
        self.local_global = Local2Global(
            input_channel,
            block_type=gbr_type,
            token_dim=token_dim,
            token_num=token_num,
            inp_res=inp_res,
            use_dynamic=gbr_dynamic[0],
            norm_pos=gbr_norm,
            drop_path_rate=drop_path_rate,
            attn_num_heads=attn_num_heads,
        )

        # Classifier (optional)
        if num_classes > 0:
            self.classifier = MergeClassifier(
                input_channel,
                oup=num_features,
                ch_exp=last_exp,
                num_classes=num_classes,
                drop_rate=drop_rate,
                drop_branch=gbr_drop,
                group_num=group_num,
                token_dim=token_dim,
                cls_token_num=cls_token_num,
                last_act=last_act,
                hyper_token_id=hyper_token_id,
                hyper_reduction_ratio=hyper_reduction_ratio,
            )
        else:
            self.classifier = None

        # Determine which stride-2 indices map to l1, l2, l3.
        # We pick the 1st, 2nd, 3rd stride-2 transitions (skip the stem's stride).
        # For 12-block variants: indices 1,3,5 → 56×56, 28×28, 14×14
        # For 9-block variants:  indices 0,2,4 → 56×56, 28×28, 14×14
        # We always want the first 3 stride-2 blocks.
        self._tap_indices: List[int] = self._stride2_flat_indices[:3]
        self.stage_out_channels: List[int] = self._stage_out_channels_list[:3]

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    # ------------------------------------------------------------------
    # Multi-scale feature extraction (for anomaly detection backbones)
    # ------------------------------------------------------------------
    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return multi-scale features ``{"l1", "l2", "l3"}``.

        l1, l2, l3 are tapped right after the 1st, 2nd, 3rd stride-2
        blocks, yielding feature maps at roughly 1/4, 1/8, 1/16 of the
        input spatial resolution.
        """
        bs = x.shape[0]
        z = self.tokens.weight
        tokens = z[None].repeat(bs, 1, 1).clone().permute(1, 0, 2)

        x = self.stem(x)

        tap_set = set(self._tap_indices)
        collected: Dict[str, torch.Tensor] = {}
        tap_counter = 0

        for i, block in enumerate(self.features):
            x, tokens = block((x, tokens))
            if i in tap_set:
                tap_counter += 1
                collected[f"l{tap_counter}"] = x
                if tap_counter == 3:
                    break  # no need to run remaining blocks

        # Safety: if model has fewer than 3 stride-2 blocks, fill remaining
        while tap_counter < 3:
            tap_counter += 1
            collected[f"l{tap_counter}"] = x

        return collected

    # ------------------------------------------------------------------
    # Full forward (classification)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.classifier is None:
            return self.forward_features(x)

        bs = x.shape[0]
        z = self.tokens.weight
        tokens = z[None].repeat(bs, 1, 1).clone().permute(1, 0, 2)

        x = self.stem(x)
        for block in self.features:
            x, tokens = block((x, tokens))
        tokens, _attn = self.local_global((x, tokens))
        y = self.classifier((x, tokens))
        return y


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _build_mobile_former(variant_name: str, num_classes: int = 0, **overrides) -> MobileFormer:
    """Build a MobileFormer variant by name.

    Args:
        variant_name: Key in _VARIANT_CONFIGS (e.g. "mobileformer_294m").
        num_classes: 0 → feature-only (no classifier head).
        **overrides: Extra kwargs forwarded to MobileFormer.
    """
    cfg = _VARIANT_CONFIGS[variant_name]

    kwargs = dict(
        block_args=cfg["block_args"],
        stem_chs=cfg["stem_chs"],
        token_num=cfg["token_num"],
        token_dim=cfg["token_dim"],
        num_features=cfg["num_features"],
        num_classes=num_classes,
        se_flag=[2, 0, 2, 0],
        group_num=1,
        gbr_type="attn",
        gbr_dynamic=[True, False, False],
        gbr_ffn=True,
        **_COMMON_KWARGS,
    )
    kwargs.update(overrides)
    return MobileFormer(**kwargs)


def mobileformer_508m(num_classes: int = 0) -> MobileFormer:
    return _build_mobile_former("mobileformer_508m", num_classes=num_classes)


def mobileformer_294m(num_classes: int = 0) -> MobileFormer:
    return _build_mobile_former("mobileformer_294m", num_classes=num_classes)


def mobileformer_214m(num_classes: int = 0) -> MobileFormer:
    return _build_mobile_former("mobileformer_214m", num_classes=num_classes)


def mobileformer_151m(num_classes: int = 0) -> MobileFormer:
    return _build_mobile_former("mobileformer_151m", num_classes=num_classes)


def mobileformer_96m(num_classes: int = 0) -> MobileFormer:
    return _build_mobile_former("mobileformer_96m", num_classes=num_classes)


def mobileformer_52m(num_classes: int = 0) -> MobileFormer:
    return _build_mobile_former("mobileformer_52m", num_classes=num_classes)
