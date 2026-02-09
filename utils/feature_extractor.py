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

    # ShuffleNet V1 (custom implementation, not available in timm/torchvision)
    # Three configurations from the original paper (Table 1, Zhang et al. 2017):
    #   g=1: baseline without group conv (smallest channels: 144/288/576)
    #   g=3: paper's default, best accuracy/complexity trade-off (240/480/960)
    #   g=8: most aggressive grouping, highest channels (384/768/1536)
    "shufflenet_g1": BackboneSpec(source="custom_shufflenet", name="shufflenet_g1"),
    "shufflenet_g3": BackboneSpec(source="custom_shufflenet", name="shufflenet_g3"),
    "shufflenet_g8": BackboneSpec(source="custom_shufflenet", name="shufflenet_g8"),

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
    ) -> None:
        super().__init__()
        self.spec = spec
        self.pretrained = pretrained

        # For different model other libraries or implementations can be used
        if spec.source == "timm":
            self._impl = _TimmFeaturesOnly(spec.name, out_indices=spec.out_indices, pretrained=pretrained)
        elif spec.source == "custom_shufflenet":
            self._impl = _ShuffleNetFeaturesOnly(spec.name)
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

        # feature_info contains metadata about each layer
        if fi is not None:
            try:
                # Extract channel counts for each layer
                chans = [f.get("num_chs") for f in fi]
                self.feature_channels = {f"l{i+1}": int(c) for i, c in enumerate(chans) if c is not None}
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
    Extract multi-scale features using our custom ShuffleNet V1 implementation.

    Uses forward_features() to return {"l1", "l2", "l3"} from stage2, stage3, stage4.
    No pretrained weights are available (ShuffleNet V1 is trained from scratch).
    """

    # Map of model name -> (groups, scale)
    # Three configurations from the original ShuffleNet V1 paper (Table 1)
    _CONFIGS = {
        "shufflenet_g1":        (1, 1.0),
        "shufflenet_g3":        (3, 1.0),
        "shufflenet_g8":        (8, 1.0),
    }

    def __init__(self, model_name: str) -> None:
        super().__init__()
        from utils.shufflenet import ShuffleNet

        if model_name not in self._CONFIGS:
            raise ValueError(
                f"Unknown ShuffleNet config '{model_name}'. "
                f"Available: {list(self._CONFIGS.keys())}"
            )

        groups, scale = self._CONFIGS[model_name]

        # num_classes=0 disables the classifier head (we only need features)
        self.backbone = ShuffleNet(groups=groups, num_classes=0, scale=scale)

        # Store feature channel counts for each level
        self.feature_channels = {
            f"l{i+1}": c for i, c in enumerate(self.backbone.stage_out_channels)
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract multi-scale features using forward_features()
        return self.backbone.forward_features(x)


def build_extractor(
    # create a feature extractor.
    backbone_key: str,
    *,
    pretrained: bool = True,
    device: Optional[torch.device] = None,
) -> MultiScaleFeatureExtractor:

    if backbone_key not in LIGHTWEIGHT_BACKBONES:
        raise KeyError(
            f"Unknown backbone '{backbone_key}'. Available: {sorted(LIGHTWEIGHT_BACKBONES.keys())}"
        )
    return MultiScaleFeatureExtractor(LIGHTWEIGHT_BACKBONES[backbone_key], pretrained=pretrained, device=device)
