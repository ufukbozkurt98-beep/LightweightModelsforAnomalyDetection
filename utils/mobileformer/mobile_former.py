from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .dna_blocks import (
    DnaBlock,
    DnaBlock3,
    _make_divisible,
)

# shared config for all variants
_COMMON_KWARGS = dict(
    cnn_drop_path_rate=0.1,
    dw_conv="dw",
    kernel_size=(3, 3),
    cnn_exp=(6, 4),
    hyper_token_id=0,
    hyper_reduction_ratio=4,
    attn_num_heads=2,
    gbr_norm="post",
    mlp_token_exp=4,
    gbr_before_skip=False,
    remove_proj_local=True,
)

# Variant configs
_VARIANT_CONFIGS = {
    "mobileformer_508m": dict(
        stem_chs=24,
        token_num=6,
        token_dim=192,
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
    "mobileformer_52m": dict(
        stem_chs=8,
        token_num=3,
        token_dim=128,
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

# Block class lookup
_BLOCK_CLS = {
    "DnaBlock": DnaBlock,
    "DnaBlock3": DnaBlock3,
}

# Weight filenames in weights/ directory
_WEIGHT_FILES = {
    "mobileformer_508m": "mobileformer_508m.pth",
    "mobileformer_294m": "mobileformer_294m.pth",
    "mobileformer_52m": "mobileformer_52m.pth",
}


def _load_pretrained_weights(model: nn.Module, variant_name: str) -> None:
    """Load pretrained weights from weights/ directory."""
    weight_file = _WEIGHT_FILES.get(variant_name)
    if weight_file is None:
        print(f"No pretrained weights available for {variant_name}")
        return

    weight_path = Path("weights") / weight_file

    if not weight_path.exists():
        print(f"  WARNING: Pretrained weights not found at {weight_path}")
        print(f"  MobileFormer will use random initialization.")
        print(f"  To use pretrained weights, download from official repo:")
        print(f"    https://github.com/AAboys/MobileFormer")
        print(f"  and save as: {weight_path}")
        return

    # Load checkpoint
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Load with strict=False to skip classifier/local_global keys
    result = model.load_state_dict(state_dict, strict=False)
    print(f"  Loading pretrained MobileFormer weights from {weight_path}")
    if result.missing_keys:
        print(f"  Missing keys (expected): {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  Skipped keys (classifier etc.): {result.unexpected_keys}")


class MobileFormer(nn.Module):
    """
    Mobile-Former feature extractor (pretrained, no classifier)
    """

    def __init__(
        self,
        block_args: List[list],
        img_size: int = 224,
        width_mult: float = 1.0,
        in_chans: int = 3,
        stem_chs: int = 16,
        dw_conv: str = "dw",
        kernel_size: Tuple[int, int] = (3, 3),
        cnn_exp: Tuple[int, int] = (6, 4),
        group_num: int = 1,
        se_flag: list | None = None,
        hyper_token_id: int = 0,
        hyper_reduction_ratio: int = 4,
        token_dim: int = 128,
        token_num: int = 6,
        gbr_type: str = "mlp",
        gbr_dynamic: list | None = None,
        gbr_norm: str = "post",
        gbr_ffn: bool = False,
        gbr_before_skip: bool = False,
        mlp_token_exp: int = 4,
        drop_path_rate: float = 0.0,
        cnn_drop_path_rate: float = 0.0,
        attn_num_heads: int = 2,
        remove_proj_local: bool = True,
    ):
        super().__init__()
        # Sets defaults for optional list parameters
        if se_flag is None:
            se_flag = [2, 0, 2, 0]
        if gbr_dynamic is None:
            gbr_dynamic = [False, False, False]

        # Prepares drop-path and divisibility divisor for channel rounding
        cnn_drop_path_rate = drop_path_rate
        mdiv = 8 if width_mult > 1.01 else 4

        # Learnable global tokens
        self.tokens = nn.Embedding(token_num, token_dim)

        # first CNN, it makes the image smaller (stride=2) and creates basic features
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, stem_chs, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_chs),
            nn.ReLU6(inplace=True),
        )
        input_channel = stem_chs

        # Build DNA blocks as ModuleList
        layer_num = len(block_args)
        inp_res = img_size * img_size // 4
        layers: List[nn.Module] = []

        # Track stride-2 block indices for feature extraction
        self._stride2_flat_indices: List[int] = []
        self._stage_out_channels_list: List[int] = []
        flat_idx = 0

        # Read each block config
        for idx, val in enumerate(block_args):
            b, t, c, n, s, t2 = val
            block_cls = _BLOCK_CLS[b]
            t_pair = (t, t2)
            output_channel = (
                _make_divisible(c * width_mult, mdiv)
                if idx > 0
                else _make_divisible(c * width_mult, 4)
            )

            # Linearly increase drop path rate
            drop_path_prob = drop_path_rate * (idx + 1) / layer_num
            cnn_drop_path_prob = cnn_drop_path_rate * (idx + 1) / layer_num

            # Create one DNA block
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

            # if this block downsamples, remember its index and channels (for l1/l2/l3)
            if s == 2:
                self._stride2_flat_indices.append(flat_idx)
                self._stage_out_channels_list.append(output_channel)

            # Update channel count and spatial area after this block
            input_channel = output_channel
            if s == 2:
                inp_res = inp_res // 4
            flat_idx += 1

            # Repeat blocks, stride=1
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

        # Select 3 consecutive stride-2 blocks for l1, l2, l3 feature maps
        # tap_offset=0 (default): first 3 stride-2 blocks → 64×64, 32×32, 16×16
        # tap_offset=1: stride-2 blocks [1:4] → 32×32, 16×16, 8×8 (matches CFlow-AD)
        self._tap_offset = 0
        self._apply_tap_offset(self._tap_offset)

    def set_tap_offset(self, offset: int):
        """Shift which stride-2 blocks are used for feature extraction."""
        self._tap_offset = offset
        self._apply_tap_offset(offset)

    def _apply_tap_offset(self, offset: int):
        end = offset + 3
        if end > len(self._stride2_flat_indices):
            raise ValueError(
                f"tap_offset={offset} requires {end} stride-2 blocks, "
                f"but only {len(self._stride2_flat_indices)} available"
            )
        self._tap_indices: List[int] = self._stride2_flat_indices[offset:end]
        self.stage_out_channels: List[int] = self._stage_out_channels_list[offset:end]

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features
        """

        # Prepare tokens for this batch
        bs = x.shape[0]
        z = self.tokens.weight
        tokens = z[None].repeat(bs, 1, 1).clone().permute(1, 0, 2)

        # take stem to get first CNN feature map
        x = self.stem(x)

        tap_set = set(self._tap_indices) # where to save l1/l2/l3
        collected: Dict[str, torch.Tensor] = {} # store the feature maps
        tap_counter = 0

        # Run blocks and save CNN feature maps at downsample points
        # Stop early after 3 scales to save time
        for i, block in enumerate(self.features):
            x, tokens = block((x, tokens))
            if i in tap_set:
                tap_counter += 1
                collected[f"l{tap_counter}"] = x
                if tap_counter == 3:
                    break  # done, skip remaining blocks

        # If we have fewer than 3 scales, reuse the last feature map
        while tap_counter < 3:
            tap_counter += 1
            collected[f"l{tap_counter}"] = x

        return collected


def _build_mobile_former(variant_name: str, **overrides) -> MobileFormer:
    """Build a pretrained MobileFormer variant by name."""
    cfg = _VARIANT_CONFIGS[variant_name]

    kwargs = dict(
        block_args=cfg["block_args"],
        stem_chs=cfg["stem_chs"],
        token_num=cfg["token_num"],
        token_dim=cfg["token_dim"],
        se_flag=[2, 0, 2, 0],
        group_num=1,
        gbr_type="attn",
        gbr_dynamic=[True, False, False],
        gbr_ffn=True,
        **_COMMON_KWARGS,
    )
    kwargs.update(overrides)
    model = MobileFormer(**kwargs)
    _load_pretrained_weights(model, variant_name)
    return model


def mobileformer_508m() -> MobileFormer:
    """MobileFormer 508M FLOPs"""
    return _build_mobile_former("mobileformer_508m")


def mobileformer_294m() -> MobileFormer:
    """MobileFormer 294M FLOPs"""
    return _build_mobile_former("mobileformer_294m")


def mobileformer_52m() -> MobileFormer:
    """MobileFormer 52M FLOPs"""
    return _build_mobile_former("mobileformer_52m")
