import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc as sk_auc
from skimage.measure import label, regionprops


def _to_numpy(x):
    # Convert to NumPy
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def collect_gt_from_loader(loader):
    # Collect all ground truth labels and masks from the test loader.
    y_img_list = []  # image labels from each batch
    y_pix_list = []  # pixel masks from each batch
    has_all_masks = True # if all batches have masks

    # Loop all batches in test set
    for batch in loader:
        # Collect image labels
        y_img_list.append(_to_numpy(batch["label"]).astype(np.int32))

        # Collect pixel masks
        if "mask" in batch and batch["mask"] is not None:
            m = _to_numpy(batch["mask"])
            if m.ndim == 4:
                m = m[:, 0]
            y_pix_list.append((m > 0).astype(np.uint8))
        else:
            has_all_masks = False

    # Concatenate all batches together
    y_img = np.concatenate(y_img_list, axis=0)
    y_pix = None if not has_all_masks else np.concatenate(y_pix_list, axis=0)
    return y_img, y_pix


def image_level_auroc(y_img, scores):
    #  Compute image-level Area Under ROC Curve.
    y_img = _to_numpy(y_img).astype(np.int32)
    scores = _to_numpy(scores).astype(np.float32)

    # cannot compute auroc with only one class!
    if np.unique(y_img).size < 2:
        raise ValueError("Image AUROC undefined: only one class present.")

    # Compute auroc using sklearn
    return float(roc_auc_score(y_img, scores))


def pixel_level_auroc(y_pix, maps):
    # Compute pixel-level Area Under ROC Curve.

    # check if masks are available
    if y_pix is None:
        raise ValueError("Pixel AUROC requires GT masks.")

    # convert inputs to NumPy arrays
    y_pix = _to_numpy(y_pix).astype(np.uint8)
    maps = _to_numpy(maps).astype(np.float32)

    # handle 4D anomaly maps
    if maps.ndim == 4:
        maps = maps[:, 0]

    # Verify shapes match
    if y_pix.shape != maps.shape:
        raise ValueError(f"Shape mismatch: y_pix {y_pix.shape} vs maps {maps.shape}")

    # cannot compute auroc with only one class!
    if np.unique(y_pix).size < 2:
        raise ValueError("Pixel AUROC undefined: mask has only one value.")

    # flatten and compute auroc
    return float(roc_auc_score(y_pix.reshape(-1), maps.reshape(-1)))


def _rescale(x):
    # Rescale array to range [0, 1]
    x = np.asarray(x, dtype=np.float32)
    mn, mx = x.min(), x.max()
    if mx <= mn:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def aupro(maps, y_pix, expect_fpr=0.3, max_step=1000):
    # Compute Area Under Per-Region Overlap curve.

    # Check if masks are available
    if y_pix is None:
        raise ValueError("AUPRO requires GT masks.")

    # Convert inputs to NumPy
    maps = _to_numpy(maps).astype(np.float32)

    # handle 4D anomaly maps
    if maps.ndim == 4:
        maps = maps[:, 0]

    gt_mask = _to_numpy(y_pix).astype(bool)

    # Set up thresholds
    max_th = maps.max()  # Maximum anomaly score
    min_th = maps.min()  # Minimum anomaly score
    delta = (max_th - min_th) / max_step # Step size between thresholds

    # Initialize storage
    pros_mean = []  # store average PRO for each threshold
    fprs = []   # store fpr for each threshold

    # binary predictions
    binary_score_maps = np.zeros_like(maps, dtype=bool)

    # try each threshold
    for step in range(max_step):
        # current threshold
        thred = max_th - step * delta

        # Create binary predictions with this threshold
        binary_score_maps[maps <= thred] = 0
        binary_score_maps[maps > thred] = 1

        # Compute per-region overlap for pro
        pro = []

        # Loop through each image
        for i in range(len(binary_score_maps)):

            # Find connected defect regions in ground truth
            label_map = label(gt_mask[i], connectivity=2)
            props = regionprops(label_map)

            # compute overlap for each defect region
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox
                cropped_pred = binary_score_maps[i][x_min:x_max, y_min:y_max]
                cropped_mask = prop.filled_image
                intersection = np.logical_and(cropped_pred, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)

        # Average pro across all regions
        pros_mean.append(np.array(pro).mean() if len(pro) > 0 else 0.0)

        # compute false positive Rate (fpr)
        gt_masks_neg = ~gt_mask
        fpr = np.logical_and(gt_masks_neg, binary_score_maps).sum() / gt_masks_neg.sum()
        fprs.append(fpr)

    # compute area under curve
    pros_mean = np.asarray(pros_mean, dtype=np.float32)
    fprs = np.asarray(fprs, dtype=np.float32)

    # Select points (fpr ≤ expected fpr)
    idx = fprs <= expect_fpr
    if not np.any(idx):
        return 0.0

    fprs_sel = fprs[idx]
    pros_sel = pros_mean[idx]

    # Rescale FPR from [0, 0.3] to [0, 1]
    fprs_sel = _rescale(fprs_sel)

    # Compute area under the pro vs fpr curve
    return float(sk_auc(fprs_sel, pros_sel))