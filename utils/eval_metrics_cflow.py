# utils/eval_metrics_cflow.py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc as sk_auc
from skimage.measure import label, regionprops


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def collect_gt_from_loader(loader):
    y_img_list = []
    y_pix_list = []
    has_all_masks = True

    for batch in loader:
        y_img_list.append(_to_numpy(batch["label"]).astype(np.int32))

        if "mask" in batch and batch["mask"] is not None:
            m = _to_numpy(batch["mask"])
            if m.ndim == 4:
                m = m[:, 0]
            y_pix_list.append((m > 0).astype(np.uint8))
        else:
            has_all_masks = False

    y_img = np.concatenate(y_img_list, axis=0)
    y_pix = None if not has_all_masks else np.concatenate(y_pix_list, axis=0)
    return y_img, y_pix


def image_level_auroc(y_img, scores):
    y_img = _to_numpy(y_img).astype(np.int32)
    scores = _to_numpy(scores).astype(np.float32)
    if np.unique(y_img).size < 2:
        raise ValueError("Image AUROC undefined: only one class present.")
    return float(roc_auc_score(y_img, scores))


def pixel_level_auroc(y_pix, maps):
    if y_pix is None:
        raise ValueError("Pixel AUROC requires GT masks.")

    y_pix = _to_numpy(y_pix).astype(np.uint8)
    maps = _to_numpy(maps).astype(np.float32)
    if maps.ndim == 4:
        maps = maps[:, 0]

    if y_pix.shape != maps.shape:
        raise ValueError(f"Shape mismatch: y_pix {y_pix.shape} vs maps {maps.shape}")

    if np.unique(y_pix).size < 2:
        raise ValueError("Pixel AUROC undefined: mask has only one value.")
    return float(roc_auc_score(y_pix.reshape(-1), maps.reshape(-1)))


def _rescale(x):
    x = np.asarray(x, dtype=np.float32)
    mn, mx = x.min(), x.max()
    if mx <= mn:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def aupro(maps, y_pix, expect_fpr=0.3, max_step=1000):
    if y_pix is None:
        raise ValueError("AUPRO requires GT masks.")

    maps = _to_numpy(maps).astype(np.float32)
    if maps.ndim == 4:
        maps = maps[:, 0]  # (N,H,W)

    gt_mask = _to_numpy(y_pix).astype(bool)  # (N,H,W)

    max_th = maps.max()
    min_th = maps.min()
    delta = (max_th - min_th) / max_step

    pros_mean = []
    fprs = []

    binary_score_maps = np.zeros_like(maps, dtype=bool)

    for step in range(max_step):
        thred = max_th - step * delta

        binary_score_maps[maps <= thred] = 0
        binary_score_maps[maps > thred] = 1

        pro = []
        for i in range(len(binary_score_maps)):
            label_map = label(gt_mask[i], connectivity=2)
            props = regionprops(label_map)

            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox
                cropped_pred = binary_score_maps[i][x_min:x_max, y_min:y_max]
                cropped_mask = prop.filled_image
                intersection = np.logical_and(cropped_pred, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)

        pros_mean.append(np.array(pro).mean() if len(pro) > 0 else 0.0)

        gt_masks_neg = ~gt_mask
        fpr = np.logical_and(gt_masks_neg, binary_score_maps).sum() / gt_masks_neg.sum()
        fprs.append(fpr)

    pros_mean = np.asarray(pros_mean, dtype=np.float32)
    fprs = np.asarray(fprs, dtype=np.float32)

    idx = fprs <= expect_fpr
    if not np.any(idx):
        return 0.0

    fprs_sel = fprs[idx]
    pros_sel = pros_mean[idx]

    # CFLOW-AD: fpr [0,0.3] -> [0,1] rescale
    fprs_sel = _rescale(fprs_sel)
    return float(sk_auc(fprs_sel, pros_sel))
