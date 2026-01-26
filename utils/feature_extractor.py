"""Unified multi-scale feature extractors for lightweight backbones.

This file implements Step (1) of the benchmark pipeline:
extract intermediate backbone features in a consistent format so different
anomaly-detection methods (CFLOW / FastFlow / GLASS / STLM / …) can reuse them.

Conventions
-----------
* Input images: torch.Tensor shaped (B, 3, H, W)
* Output features: dict with keys "l1", "l2", "l3" mapped to feature maps
  shaped (B, C, H_i, W_i).

Notes
-----
* For most backbones, we use `timm.create_model(..., features_only=True)`.
  The timm docs explicitly support building feature backbones via
  `features_only=True` and selecting levels with `out_indices`.
* ShuffleNet-V2 is currently not provided by timm in many versions, so this
  module includes a torchvision-based fallback.
* Mobile-Former is *not* consistently available as a timm model. This module
  includes a placeholder loader that you can wire to your Mobile-Former
  implementation/checkpoint.
"""

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

    # ShuffleNet-V2 (torchvision fallback; not consistently in timm)
    "shufflenetv2_x1_0": BackboneSpec(source="torchvision", name="shufflenet_v2_x1_0"),
    "shufflenetv2_x0_5": BackboneSpec(source="torchvision", name="shufflenet_v2_x0_5"),

    # Mobile-Former (custom; wire this to your implementation)
    "mobileformer": BackboneSpec(source="custom", name="mobileformer"),
}


# ---------------------------
# Core extractor wrappers
# ---------------------------


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
        elif spec.source == "torchvision":
            self._impl = _TorchvisionShuffleNet(spec.name, pretrained=pretrained)
        elif spec.source == "custom":
            self._impl = _CustomMobileFormer(pretrained=pretrained)
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


class _TorchvisionShuffleNet(nn.Module):
    """ShuffleNet-V2 feature extraction via forward hooks.

    Why hooks? Torchvision's feature_extraction utility is great, but users often
    have different torchvision versions. Hooks are the most version-tolerant.
    """

    def __init__(self, model_name: str, *, pretrained: bool) -> None:
        super().__init__()
        try:
            import torchvision.models as tvm  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "torchvision is required for ShuffleNet extractors. Install with: pip install torchvision"
            ) from e

        # Build model
        build_fn = getattr(tvm, model_name, None)
        if build_fn is None:
            raise ValueError(f"torchvision.models has no model named '{model_name}'")

        # Newer torchvision uses Weights enums; keep it simple / compatible.
        try:
            self.model = build_fn(weights="DEFAULT" if pretrained else None)
        except Exception:
            self.model = build_fn(pretrained=pretrained)

        self.model.eval()

        # Register hooks at three progressively downsampled stages.
        self._acts: Dict[str, torch.Tensor] = {}
        self._handles = []

        # Common module names in torchvision ShuffleNetV2:
        # conv1 -> maxpool -> stage2 -> stage3 -> stage4 -> conv5
        # We'll tap: stage2, stage3, stage4.
        for lvl, layer_name in enumerate(["stage2", "stage3", "stage4"], start=1):
            layer = getattr(self.model, layer_name, None)
            if layer is None:
                raise RuntimeError(
                    f"Couldn't find '{layer_name}' in ShuffleNet model. "
                    "If your torchvision version differs, update the layer list."
                )
            self._handles.append(layer.register_forward_hook(self._make_hook(f"l{lvl}")))

        # Channels depend on variant; we fill after first forward if you need.
        self.feature_channels: Dict[str, int] = {}

    def _make_hook(self, key: str):
        def _hook(_m, _inp, out):
            self._acts[key] = out
        return _hook

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._acts = {}
        _ = self.model(x)
        feats = {"l1": self._acts["l1"], "l2": self._acts["l2"], "l3": self._acts["l3"]}

        if not self.feature_channels:
            self.feature_channels = {k: int(v.shape[1]) for k, v in feats.items()}
        return feats

    def close(self) -> None:
        for h in self._handles:
            h.remove()


class _CustomMobileFormer(nn.Module):
    """Stub for Mobile-Former.

    Mobile-Former is not consistently available as a timm model.
    Plug your implementation here (or add it as a submodule) and make sure
    `forward()` returns {"l1","l2","l3"} feature maps.
    """

    def __init__(self, *, pretrained: bool) -> None:
        super().__init__()
        self.pretrained = pretrained
        self._impl = None

    def _lazy_init(self) -> None:
        if self._impl is not None:
            return

        # Option A: if you vendored an implementation into your repo.
        #   from models.mobileformer import mobileformer_294m
        #   self._impl = mobileformer_294m(pretrained=self.pretrained)
        #
        # Option B: if you use a pip package.
        #   import mobileformer
        #   self._impl = mobileformer.create_model(...)
        #
        # For now we raise a clear error so the benchmark fails loudly.
        raise ImportError(
            "Mobile-Former backbone is marked as 'custom' in this project. "
            "Please plug in a Mobile-Former implementation inside _CustomMobileFormer._lazy_init()."
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._lazy_init()
        assert self._impl is not None
        return self._impl(x)


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
