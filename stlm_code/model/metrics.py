"""
AUPRO metric for STLM.
Self-contained implementation compatible with torchmetrics.
Based on the original STLM code but without old anomalib dependency.
"""

from typing import Any, Callable, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat
from sklearn.metrics import roc_curve, auc as sk_auc
import scipy.ndimage as ndimage


def connected_components(mask: Tensor) -> Tensor:
    """Compute connected components using scipy (CPU-based, works reliably)."""
    mask_np = mask.cpu().numpy()
    batch_labels = []
    for i in range(mask_np.shape[0]):
        m = mask_np[i]
        if m.ndim == 3:
            m = m.squeeze(0)
        labeled, _ = ndimage.label(m)
        batch_labels.append(labeled)
    result = np.stack(batch_labels, axis=0)
    # Add channel dim back if needed
    if mask.ndim == 4:
        result = result[:, np.newaxis, :, :]
    return torch.tensor(result, device=mask.device, dtype=torch.int64)


class AUPRO(Metric):
    """Area Under the Per-Region Overlap curve."""

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False
    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        fpr_limit: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.register_buffer("fpr_limit", torch.tensor(fpr_limit))

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.target.append(target)
        self.preds.append(preds)

    def _compute(self) -> Tuple[Tensor, Tensor]:
        target = dim_zero_cat(self.target)
        preds = dim_zero_cat(self.preds)

        if target.min() < 0 or target.max() > 1:
            raise ValueError(
                f"Expected target in [0, 1], got [{target.min()}, {target.max()}]"
            )

        target = target.type(torch.float)
        cca = connected_components(target)

        preds_flat = preds.flatten().cpu().numpy()
        cca_flat = cca.flatten().cpu()
        target_flat = target.flatten().cpu().numpy()

        # Compute global FPR to determine output size
        global_fpr, _, _ = roc_curve(target_flat, preds_flat)
        global_fpr = torch.tensor(global_fpr, dtype=torch.float)
        fpr_limit_val = self.fpr_limit.item()
        output_size = int(torch.where(global_fpr <= fpr_limit_val)[0].size(0))

        device = preds.device
        tpr = torch.zeros(output_size, dtype=torch.float)
        fpr = torch.zeros(output_size, dtype=torch.float)
        new_idx = torch.arange(0, output_size, dtype=torch.float)

        labels = cca_flat.unique()[1:]  # 0 is background
        background = cca_flat == 0

        preds_flat_t = torch.tensor(preds_flat, dtype=torch.float)
        target_flat_t = torch.tensor(target_flat, dtype=torch.float)

        for label in labels:
            interp: bool = False
            new_idx[-1] = output_size - 1
            mask = cca_flat == label
            sel = background | mask

            sel_preds = preds_flat_t[sel].numpy()
            sel_target = mask[sel].numpy().astype(np.int32)

            _fpr_np, _tpr_np, _ = roc_curve(sel_target, sel_preds)
            _fpr = torch.tensor(_fpr_np, dtype=torch.float)
            _tpr = torch.tensor(_tpr_np, dtype=torch.float)

            if _fpr[_fpr <= fpr_limit_val].max() == 0:
                _fpr_limit = _fpr[_fpr > fpr_limit_val].min()
            else:
                _fpr_limit = fpr_limit_val

            _fpr_idx = torch.where(_fpr <= _fpr_limit)[0]
            if not torch.allclose(_fpr[_fpr_idx].max(), torch.tensor(fpr_limit_val)):
                _tmp_idx = torch.searchsorted(_fpr, torch.tensor(fpr_limit_val))
                _fpr_idx = torch.cat([_fpr_idx, _tmp_idx.unsqueeze_(0)])
                _slope = 1 - (
                    (_fpr[_tmp_idx] - fpr_limit_val)
                    / (_fpr[_tmp_idx] - _fpr[_tmp_idx - 1])
                )
                interp = True

            _fpr = _fpr[_fpr_idx]
            _tpr = _tpr[_fpr_idx]

            _fpr_idx = _fpr_idx.float()
            _fpr_idx /= _fpr_idx.max()
            _fpr_idx *= new_idx.max()

            if interp:
                new_idx[-1] = _fpr_idx[-2] + ((_fpr_idx[-1] - _fpr_idx[-2]) * _slope)

            _tpr = self.interp1d(_fpr_idx, _tpr, new_idx)
            _fpr = self.interp1d(_fpr_idx, _fpr, new_idx)
            tpr += _tpr
            fpr += _fpr

        tpr /= labels.size(0)
        fpr /= labels.size(0)
        return fpr.to(device), tpr.to(device)

    def compute(self) -> Tensor:
        fpr, tpr = self._compute()
        fpr_np = fpr.cpu().numpy()
        tpr_np = tpr.cpu().numpy()
        # Sort by fpr for proper AUC computation
        sorted_idx = np.argsort(fpr_np)
        aupro_val = sk_auc(fpr_np[sorted_idx], tpr_np[sorted_idx])
        aupro_val = aupro_val / fpr_np.max()
        return torch.tensor(aupro_val, device=fpr.device)

    @staticmethod
    def interp1d(old_x: Tensor, old_y: Tensor, new_x: Tensor) -> Tensor:
        eps = torch.finfo(old_y.dtype).eps
        slope = (old_y[1:] - old_y[:-1]) / (eps + (old_x[1:] - old_x[:-1]))
        idx = torch.searchsorted(old_x, new_x)
        idx -= 1
        idx = torch.clamp(idx, 0, old_x.size(0) - 2)
        y_new = old_y[idx] + slope[idx] * (new_x - old_x[idx])
        return y_new
