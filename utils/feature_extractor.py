from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class BackboneSpec:
    # Where to get the model from
    source: str
    # Which specific model to use
    name: str
    # Which feature layers to extract
    out_indices: Tuple[int, int, int] = (1, 2, 3)


LIGHTWEIGHT_BACKBONES: Dict[str, BackboneSpec] = {
    # MobileNetV3
    "mobilenetv3_large": BackboneSpec(source="timm", name="mobilenetv3_large_100", out_indices=(1, 2, 3)),
    "mobilenetv3_small": BackboneSpec(source="timm", name="mobilenetv3_small_100", out_indices=(1, 2, 3)),

    # EfficientNet-Lite
    "efficientnet_lite0": BackboneSpec(source="timm", name="tf_efficientnet_lite0", out_indices=(1, 2, 3)),
    "efficientnet_lite1": BackboneSpec(source="timm", name="tf_efficientnet_lite1", out_indices=(1, 2, 3)),

    # MobileViT
    "mobilevit_xxs": BackboneSpec(source="timm", name="mobilevit_xxs", out_indices=(1, 2, 3)),
    "mobilevit_xs": BackboneSpec(source="timm", name="mobilevit_xs", out_indices=(1, 2, 3)),
    "mobilevit_s": BackboneSpec(source="timm", name="mobilevit_s", out_indices=(1, 2, 3)),

    # ShuffleNet
    "shufflenet_g1": BackboneSpec(source="custom_shufflenet", name="shufflenet_g1"),
    "shufflenet_g3": BackboneSpec(source="custom_shufflenet", name="shufflenet_g3"),
    "shufflenet_g8": BackboneSpec(source="custom_shufflenet", name="shufflenet_g8"),

    # MobileFormer (CVPR 2022)
    "mobileformer_508m": BackboneSpec(source="custom_mobileformer", name="mobileformer_508m"),
    "mobileformer_294m": BackboneSpec(source="custom_mobileformer", name="mobileformer_294m"),
    "mobileformer_52m": BackboneSpec(source="custom_mobileformer", name="mobileformer_52m"),
}


class MultiScaleFeatureExtractor(nn.Module):
    # Main feature extractor that wraps any backbone.

    # Initialize the feature extractor.
    def __init__(
        self,
        spec: BackboneSpec,
        *,
        pretrained: bool = True,
        device: Optional[torch.device] = None,
        tap_offset: int = 0,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.pretrained = pretrained

        # For different model other libraries or implementations can be used
        if spec.source == "timm":
            self._impl = _TimmFeaturesOnly(spec.name, out_indices=spec.out_indices, pretrained=pretrained)
        elif spec.source == "custom_shufflenet":
            self._impl = _ShuffleNetFeaturesOnly(spec.name)
        elif spec.source == "custom_mobileformer":
            self._impl = _MobileFormerFeaturesOnly(spec.name, tap_offset=tap_offset)
        else:
            raise ValueError(f"Unknown backbone source: {spec.source}")

        if device is not None:
            self.to(device)

    # Extract features from images.
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._impl(x)

    @property
    def feature_channels(self) -> Dict[str, int]:
        # Get the number of channels for each feature level.
        return getattr(self._impl, "feature_channels", {})


class _TimmFeaturesOnly(nn.Module):
    # Extract features using Timm library
    def __init__(self, model_name: str, *, out_indices: Tuple[int, int, int], pretrained: bool) -> None:
        super().__init__()
        try:
            import timm  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "timm is required for timm-backed extractors. Install with: pip install timm"
            ) from e

        # Create the backbone model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )

        # Will store {"l1": C1, "l2": C2, "l3": C3}
        self.feature_channels = {}

        # Get feature info from timm model
        fi = getattr(self.backbone, "feature_info", None)

        # feature_info.channels() returns channel counts matching the selected out_indices,
        # NOT all feature levels. This is important because iterating `for f in fi`
        # yields ALL feature levels regardless of out_indices.
        if fi is not None:
            try:
                chans = fi.channels()  # only selected out_indices
                self.feature_channels = {f"l{i+1}": int(c) for i, c in enumerate(chans)}
            except Exception:
                pass

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features using the timm backbone.
        feats: List[torch.Tensor] = self.backbone(x)
        if len(feats) != 3:
            raise RuntimeError(
                f"Expected 3 feature levels from timm backbone, got {len(feats)}. "
            )
        return {"l1": feats[0], "l2": feats[1], "l3": feats[2]}


class _ShuffleNetFeaturesOnly(nn.Module):
    """
    Extract multi-scale features using ShuffleNet implementation.
    Returns {"l1", "l2", "l3"} from stage2, stage3, stage4.

    Pretrained ImageNet weights (official, from Megvii/Face++ paper authors, MIT license)
    are loaded automatically if the checkpoint file exists in weights/ directory.

    Download from: https://1drv.ms/f/s!AgaP37NGYuEXhRfQxHRseR7eSxXo
    Source: https://github.com/megvii-model/ShuffleNet-Series (MIT license)

    Also supports third-party checkpoints (jaxony, ericsun99) — format is auto-detected.
    """

    # Map of model name -> (groups, scale, weight filename or None)
    _CONFIGS = {
        "shufflenet_g1":  (1, 1.0, None),  # no pretrained weights available
        "shufflenet_g3":  (3, 1.0, "shufflenet_g3.pth.tar"),
        "shufflenet_g8":  (8, 1.0, "shufflenet_g8.pth.tar"),
    }

    def __init__(self, model_name: str) -> None:
        super().__init__()
        from pathlib import Path
        from utils.shufflenet import ShuffleNet, load_pretrained_shufflenet

        if model_name not in self._CONFIGS:
            raise ValueError(
                f"Unknown ShuffleNet config '{model_name}'. "
                f"Available: {list(self._CONFIGS.keys())}"
            )

        groups, scale, weight_file = self._CONFIGS[model_name]

        # num_classes=0 disables the classifier head since we only need features
        self.backbone = ShuffleNet(groups=groups, num_classes=0, scale=scale)

        # Load pretrained weights if checkpoint file exists
        if weight_file is not None:
            weight_path = Path("weights") / weight_file
            if weight_path.exists():
                print(f"  Loading pretrained ShuffleNet weights from {weight_path}")
                load_pretrained_shufflenet(self.backbone, str(weight_path))
            else:
                print(f"  WARNING: Pretrained weights not found at {weight_path}")
                print(f"  ShuffleNet will use random initialization.")
                print(f"  To use pretrained weights, download from official Megvii repo:")
                print(f"    https://1drv.ms/f/s!AgaP37NGYuEXhRfQxHRseR7eSxXo")
                print(f"  and save the 1.0x g={groups} checkpoint as: {weight_path}")

        # Store feature channel counts for each level
        self.feature_channels = {
            f"l{i+1}": c for i, c in enumerate(self.backbone.stage_out_channels)
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract multi-scale features
        return self.backbone.forward_features(x)


class _MobileFormerFeaturesOnly(nn.Module):
    """
    Extract multi-scale features using MobileFormer (CVPR 2022).
    Returns {"l1", "l2", "l3"} from the first three stride-2 stages.
    Pretrained ImageNet weights from the official repo.
    """

    _CONFIGS = {
        "mobileformer_508m",
        "mobileformer_294m",
        "mobileformer_52m",
    }

    def __init__(self, model_name: str, tap_offset: int = 0) -> None:
        super().__init__()
        from utils.mobileformer.mobile_former import _build_mobile_former

        if model_name not in self._CONFIGS:
            raise ValueError(
                f"Unknown MobileFormer config '{model_name}'. "
                f"Available: {sorted(self._CONFIGS)}"
            )

        # Always loads pretrained weights
        self.backbone = _build_mobile_former(model_name)
        if tap_offset > 0:
            self.backbone.set_tap_offset(tap_offset)

        # Store feature channel counts for each level
        self.feature_channels = {
            f"l{i+1}": c for i, c in enumerate(self.backbone.stage_out_channels)
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.backbone.forward_features(x)


def build_extractor(
    # create a feature extractor.
    backbone_key: str,
    *,
    pretrained: bool = True,
    device: Optional[torch.device] = None,
    out_indices: Optional[Tuple[int, int, int]] = None,
) -> MultiScaleFeatureExtractor:

    if backbone_key not in LIGHTWEIGHT_BACKBONES:
        raise KeyError(
            f"Unknown backbone '{backbone_key}'. Available: {sorted(LIGHTWEIGHT_BACKBONES.keys())}"
        )
    spec = LIGHTWEIGHT_BACKBONES[backbone_key]
    tap_offset = 0
    # Allow method-specific override of out_indices
    # e.g. CFlow-AD original uses (2,3,4) to match paper's feature[-11,-5,-2] layers
    if out_indices is not None:
        if spec.source == "timm":
            from dataclasses import replace
            spec = replace(spec, out_indices=out_indices)
        elif spec.source == "custom_mobileformer":
            # out_indices=(2,3,4) means "use deeper features" → tap_offset=1
            tap_offset = out_indices[0] - 1  # (2,3,4) → offset 1; (1,2,3) → offset 0
    return MultiScaleFeatureExtractor(spec, pretrained=pretrained, device=device, tap_offset=tap_offset)
