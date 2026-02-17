from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List, Tuple

import glob
import os
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# perlin.py is inside the glass_source_code package
from glass_source_code.glass_src.perlin import perlin_mask

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

#Collects all .jpg paths under the images folder of dtd folder
#raises error if nothing is found
def _collect_dtd_paths(dtd_images_root: str) -> List[str]:
    paths = glob.glob(os.path.join(dtd_images_root, "*", "*.jpg"))
    if len(paths) == 0:
        raise FileNotFoundError(
            f"No .jpg files found under {dtd_images_root}. "
            f"Make sure DTD is extracted correctly."
        )
    return paths

#Picks 3 random augmentations from a list of 9 (color jitter, flips, grayscale, autocontrast, equalize, affine)
#Replicates original GLASS's rand_augmenter() method from datasets/mvtec.py
#Returns a transforms.Compose pipeline that resizes, applies the 3 random augmentations, crops, converts to tensor, and normalizes

def _rand_augmenter(resize: int) -> transforms.Compose:
    list_aug = [
        transforms.ColorJitter(contrast=(0.8, 1.2)),
        transforms.ColorJitter(brightness=(0.8, 1.2)),
        transforms.ColorJitter(saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomVerticalFlip(p=1),
        transforms.RandomGrayscale(p=1),
        transforms.RandomAutocontrast(p=1),
        transforms.RandomEqualize(p=1),
        transforms.RandomAffine(degrees=(-45, 45)),
    ]
    idx = np.random.choice(len(list_aug), 3, replace=False)
    return transforms.Compose([
        transforms.Resize((resize, resize)),
        list_aug[idx[0]],
        list_aug[idx[1]],
        list_aug[idx[2]],
        transforms.CenterCrop(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _make_single_aug(img_tensor: torch.Tensor,
                     dtd_paths: List[str],
                     ph: int, pw: int,
                     H: int, W: int,
                     beta_mean: float = 0.5,
                     beta_std: float  = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Produces one augmented image + masks for a single training image.

    Returns:
        aug_tensor : [3, H, W]  – blended image, still ImageNet-normalised
        mask_s     : [ph*pw, 1] – patch-level binary mask (long)
        mask_l_t   : [1, H, W]  – pixel-level binary mask (float)
    """
    # --- pick & augment a DTD texture ---
    dtd_path  = np.random.choice(dtd_paths)
    dtd_pil   = PIL.Image.open(dtd_path).convert("RGB")
    aug_tf    = _rand_augmenter(W)              # W == H == 256 in our setup which is input_size
    dtd_tensor = aug_tf(dtd_pil)               # [3, H, W], ImageNet-normalised

    # --- generating Perlin mask ---
    # perlin_mask expects (C, H, W) shape, feat_size = patch grid side
    # flag=1 returns (mask_s_np [ph,pw], mask_l_np [H,W])
    feat_size = ph                              # ph == pw in our square setup
    img_shape = (3, H, W)
    mask_fg   = torch.ones(H, W)               # no foreground mask → whole image
    mask_s_np, mask_l_np = perlin_mask(img_shape, feat_size, 0, 6, mask_fg, flag=1)

    # converting to tensors
    mask_s_t = torch.from_numpy(mask_s_np).float()       # [ph, pw]
    mask_l_t = torch.from_numpy(mask_l_np).float()       # [H,  W]

    # --- blending texture into image ---
    # un-normalising image to [0,1] to blend
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD ).view(3, 1, 1)

    img_01 = (img_tensor * std + mean).clamp(0.0, 1.0)
    dtd_01 = (dtd_tensor * std + mean).clamp(0.0, 1.0)

    beta = float(np.clip(np.random.normal(beta_mean, beta_std), 0.2, 0.8))

    mask_l_3c = mask_l_t.unsqueeze(0)          # [1, H, W] → broadcasts over 3 channels
    aug_01 = img_01 * (1 - mask_l_3c) \
           + (1 - beta) * dtd_01 * mask_l_3c \
           + beta       * img_01 * mask_l_3c

    aug_tensor = ((aug_01 - mean) / std)       # re-normalise back

    # --- format masks ---
    # mask_s : flatten [ph,pw] → [ph*pw, 1]  (long for GLASS)
    mask_s_out = mask_s_t.flatten().unsqueeze(1).long()   # [P, 1]
    # mask_l : add channel dim
    mask_l_out = mask_l_t.unsqueeze(0)                    # [1, H, W]

    return aug_tensor, mask_s_out, mask_l_out


class GlassLoaderAdapter:
    """
    Adapts the DataLoader batches to what GLASS expects.

    Input batch (from data_loader.py):
      - "image": Tensor [B,3,H,W]
      - "mask":  Tensor [B,1,H,W]  (zeros for good images)
      - "label": Tensor [B]        (0 good, 1 anomaly)

    Output batch (expected by GLASS trainer):
      - "image":      [B,3,H,W]
      - "aug":        [B,3,H,W]   (DTD+Perlin synthetic corruption, TRAIN only)
      - "mask_s":     [B,P,1]     (patch-level binary mask; P = ph*pw)
      - "mask_gt":    [B,1,H,W]   (pixel GT mask)
      - "is_anomaly": [B]
      - "image_path": List[str]
    """

    def __init__(
            self,
            loader:      Iterable[Dict[str, Any]],
            # (ph, pw), taken from glass._embed(). example: (64, 64) - ph: patch grid height pw: patch grid width | total patch positions per image = ph * pw
            patch_grid:  Tuple[int, int],
            is_train:    bool,
            dtd_root:    str  = "",          # path to dtd/images  (required for training)
            image_key:   str  = "image",
            mask_key:    str  = "mask",
            label_key:   str  = "label",
            path_keys:   Tuple[str, ...] = ("image_path", "img_path", "path", "image_paths"),
            beta_mean:   float = 0.5,
            beta_std:    float = 0.1,
    ):
        self.loader     = loader
        self.patch_grid = patch_grid
        self.is_train   = is_train
        self.image_key  = image_key
        self.mask_key   = mask_key
        self.label_key  = label_key
        self.path_keys  = path_keys
        self.beta_mean  = beta_mean
        self.beta_std   = beta_std

        # collect DTD paths only when training
        self.dtd_paths: List[str] = []
        if is_train:
            self.dtd_paths = _collect_dtd_paths(dtd_root)
            print(f"[GlassLoaderAdapter] DTD loaded: {len(self.dtd_paths)} texture images")

    @property
    def dataset(self):
        # In glass.py trainer(), distribution is called by training_data.dataset.distribution
        # Therefore dataset of the loader is needed to be reached as an attribute. @property defines a getter for a dataset that acts like attribute
        return getattr(self.loader, "dataset", None)


    def __len__(self) -> int:
        return len(self.loader) # returns the number of batches in the original loader that is passed to this glass_loader_adapter wrapper


    # To be able to iterate through GlassLoaderAdapter just like the DataLoader (through batch dictionaries)
    # for each batch in the loader, iter takes the batch and coverts the batches in the way that GLASS expects, then yields the converted batch
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        ph, pw = self.patch_grid  # as an example of (64,64) patch grid, each image becomes 1024 patches at the patch level

        for batch in self.loader: # for each batch at the original data loader

            # ----- image processing-----
            # takes image entry from the batch, converts it to tensor if it is not already tensor
            # converts the image tensor's datatype into float for the sake of the future arithmetic operations and mobilenetv3 expects float
            img = batch[self.image_key]
            if not torch.is_tensor(img):
                img = torch.tensor(img)
            img = img.float()
            B, C, H, W = img.shape  # storing batch size and the image dimensions

            # ----- label → is_anomaly -----
            # takes the labels of the images , converts it to tensor if it is not already tensor
            # Flattens the tensor into a 1D vector in case it is not, cause GLASS expects a 1D label vector
            # ensures the values are integer with .long() since the labels are class indicators
            if self.label_key in batch:   # checking if the batch has label attribute just in case
                is_anomaly = batch[self.label_key]
                if not torch.is_tensor(is_anomaly):
                    is_anomaly = torch.tensor(is_anomaly)
                is_anomaly = is_anomaly.view(-1).long()
            else:  # if the batch does not have label attribute, assume it is a good(non-anomalous pic) and create a tensor of B times zeros
                is_anomaly = torch.zeros(B, dtype=torch.long)

            # ----- mask_gt (pixel GT mask) -----
            # Reads the mask entry from the batch, ensure it is a torch tensor with the shape of # [B,1,H,W]
            # Ensures that the mask values are 0.0 or 1.0 depending on their real value which is either 0 or 255
            if self.mask_key in batch and batch[self.mask_key] is not None:
                m = batch[self.mask_key]  # [B,1,H,W]
                if not torch.is_tensor(m):
                    m = torch.tensor(m)
                if m.ndim == 3: # if the mask is in [B,H,W] format, convert it to [B,1,H,W]
                    m = m.unsqueeze(1)  # unsqueeze to add the missing channel info just in case
                mask_bin = (m > 0).float()          # [B,1,H,W]
            else:  # if no mask exists (good img) create mask tensor with all zeros, even though this is done in image_mask_transform.py, putting it for ensuring
                mask_bin = torch.zeros(B, 1, H, W, dtype=torch.float32)

            mask_gt = mask_bin                      # [B,1,H,W]

            # ----- TRAIN: DTD + Perlin augmentation -----
            # To create augmented anomalous version of the images during the training
            if self.is_train:
                aug_list    = []
                mask_s_list = []

                for i in range(B):
                    aug_i, mask_s_i, _ = _make_single_aug(
                        img[i],           # [3,H,W]
                        self.dtd_paths,
                        ph, pw, H, W,
                        self.beta_mean,
                        self.beta_std,
                    )
                    aug_list.append(aug_i)
                    mask_s_list.append(mask_s_i)    # [P,1]

                aug    = torch.stack(aug_list,    dim=0)   # [B,3,H,W]
                mask_s = torch.stack(mask_s_list, dim=0)  # [B,P,1]


            else:
                # ---EVAL (val/test): use GT mask downsampled to patch grid ---
                # artificial noise is not wanted on val/test images
                # mask_bin is the pixel-level GT mask, turning it into patch-grid mask that GLASS would accept
                aug        = img
                mask_small = F.interpolate(mask_bin, size=(ph, pw), mode="nearest") # [B,1,ph,pw]
                mask_s     = (mask_small.flatten(2).transpose(1, 2) > 0).long()  # [B,P,1]

            # ----- image paths -----
            # extracting file paths for each image in the batch
            # reading the file path strings from the existing batch dictionary and putting them into a standardized field
            # parenting every yielded batch has image bath that holds list of length b
            image_path: List[str] = []
            for k in self.path_keys:
                if k in batch:  # if the name is found
                    v = batch[k]
                    # if v is already a list, convert each element into a string
                    # if v is a single value, repeat it B times so the image_path will have still the length B
                    image_path = [str(x) for x in v] if isinstance(v, list) else [str(v)] * B
                    break
            if not image_path:
                image_path = [""] * B

            # yield returns one batch to the caller and the next time, it continues from the next batch
            yield {
                "image":      img,
                "aug":        aug,
                "mask_s":     mask_s,      # [B, ph*pw, 1]
                "mask_gt":    mask_gt,     # [B, 1, H, W]
                "is_anomaly": is_anomaly,
                "image_path": image_path,
            }
