from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------
# Public configuration
# ---------------------------


@dataclass(frozen=True)
class BackboneSpec:
    """How to build a backbone and which feature levels to return."""

    # One of: "timm", "torchvision", "custom"
    source: str
    # Model identifier for the chosen source (e.g., timm model name)
    name: str
    # For timm features_only: which feature indices to return
    out_indices: Tuple[int, int, int] = (1, 2, 3)


LIGHTWEIGHT_BACKBONES: Dict[str, BackboneSpec] = {
    # MobileNetV3 (timm)
    "mobilenetv3_large": BackboneSpec(source="timm", name="mobilenetv3_large_100", out_indices=(1, 2, 3)),
    "mobilenetv3_small": BackboneSpec(source="timm", name="mobilenetv3_small_100", out_indices=(1, 2, 3)),

    # EfficientNet-Lite (timm uses the TF port naming)
    "efficientnet_lite0": BackboneSpec(source="timm", name="tf_efficientnet_lite0", out_indices=(1, 2, 3)),
    "efficientnet_lite1": BackboneSpec(source="timm", name="tf_efficientnet_lite1", out_indices=(1, 2, 3)),

    # MobileViT (timm)
    "mobilevit_xxs": BackboneSpec(source="timm", name="mobilevit_xxs", out_indices=(1, 2, 3)),
    "mobilevit_xs": BackboneSpec(source="timm", name="mobilevit_xs", out_indices=(1, 2, 3)),
    "mobilevit_s": BackboneSpec(source="timm", name="mobilevit_s", out_indices=(1, 2, 3)),

}

class MultiScaleFeatureExtractor(nn.Module):
    """Backbone wrapper that always returns {"l1","l2","l3"} feature maps."""

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

        if spec.source == "timm":
            self._impl = _TimmFeaturesOnly(spec.name, out_indices=spec.out_indices, pretrained=pretrained)
        else:
            raise ValueError(f"Unknown backbone source: {spec.source}")

        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._impl(x)

    @property
    def feature_channels(self) -> Dict[str, int]:
        """Channel counts for each returned level, if available."""
        return getattr(self._impl, "feature_channels", {})


class _TimmFeaturesOnly(nn.Module):
    def __init__(self, model_name: str, *, out_indices: Tuple[int, int, int], pretrained: bool) -> None:
        super().__init__()
        try:
            import timm  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "timm is required for timm-backed extractors. Install with: pip install timm"
            ) from e

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )

        # If timm exposes feature_info, keep channel metadata for logging.
        self.feature_channels = {}
        fi = getattr(self.backbone, "feature_info", None)
        if fi is not None:
            # feature_info is aligned with returned features (after out_indices)
            try:
                chans = [f.get("num_chs") for f in fi]
                self.feature_channels = {f"l{i+1}": int(c) for i, c in enumerate(chans) if c is not None}
            except Exception:
                pass

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats: List[torch.Tensor] = self.backbone(x)
        if len(feats) != 3:
            raise RuntimeError(
                f"Expected 3 feature levels from timm backbone, got {len(feats)}. "
                f"Try adjusting out_indices."
            )
        return {"l1": feats[0], "l2": feats[1], "l3": feats[2]}


# ---------------------------
# Small helper
# ---------------------------


def build_extractor(
    backbone_key: str,
    *,
    pretrained: bool = True,
    device: Optional[torch.device] = None,
) -> MultiScaleFeatureExtractor:
    """Factory: build an extractor from LIGHTWEIGHT_BACKBONES."""
    if backbone_key not in LIGHTWEIGHT_BACKBONES:
        raise KeyError(
            f"Unknown backbone '{backbone_key}'. Available: {sorted(LIGHTWEIGHT_BACKBONES.keys())}"
        )
    return MultiScaleFeatureExtractor(LIGHTWEIGHT_BACKBONES[backbone_key], pretrained=pretrained, device=device)
