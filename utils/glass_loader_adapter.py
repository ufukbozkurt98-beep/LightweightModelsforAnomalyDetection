# utilsc
from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F


class GlassLoaderAdapter:
    """
    Adapts the DataLoader batches to what GLASS expects.

    Normal input batch organisation that comes from data_loader.py:
      - "image": Tensor [B,3,H,W]
      - "mask":  Tensor [B,1,H,W] (zeros for good images)
      - "label": Tensor [B]       (0 good, 1 anomaly)


    Output (what GLASS trainer uses):
      - "image": [B,3,H,W]
      - "aug":   [B,3,H,W]            (synthetic corruption for TRAIN)
      - "mask_s":[B,P,1]              (patch mask matching embeddings; P=ph*pw)
      - "mask_gt":[B,H,W]             (pixel GT mask)
      - "is_anomaly":[B]              (0/1)
      - "image_path": List[str]
    """

    def __init__(
            self,
            loader: Iterable[Dict[str, Any]],
            # (ph, pw), taken from glass._embed(). example: (64, 64) - ph: patch grid height pw: patch grid width | total patch positions per image = ph * pw
            patch_grid: Tuple[int, int],
            is_train: bool,
            image_key: str = "image",
            mask_key: str = "mask",
            label_key: str = "label",
            path_keys: Tuple[str, ...] = ("image_path", "img_path", "path", "image_paths"),
            # synthetic anomaly params (TRAIN only)
            rect_min: int = 6,  # rectangle size in PATCH units
            rect_max: int = 14,
            noise_std: float = 0.2,  # pixel noise strength
            synth_prob: float = 1.0,  # probability to apply synthetic corruption
    ):
        self.loader = loader
        self.patch_grid = patch_grid
        self.is_train = is_train
        self.image_key = image_key
        self.mask_key = mask_key
        self.label_key = label_key
        self.path_keys = path_keys

        self.rect_min = rect_min
        self.rect_max = rect_max
        self.noise_std = noise_std
        self.synth_prob = synth_prob

    @property
    def dataset(self):
        # In glass.py trainer(), distribution is called by training_data.dataset.distribution
        # Therefore dataset of the loader is needed to be reached as an attribute. @property defines a getter for a dataset that acts like attribute
        return getattr(self.loader, "dataset", None)  # if the loader has .dataset, return it

    def __len__(self) -> int:
        return len(self.loader)  # returns the number of batches in the original loader that is passed to this glass_loader_adapter wrapper

    # To be able to iterate through GlassLoaderAdapter just like the DataLoader (through batch dictionaries)
    # for each batch in the loader, iter takes the batch and coverts the batches in the way that GLASS expects, then yields the converted batch
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        ph, pw = self.patch_grid  # as an example of (64,64) patch grid, each image becomes 1024 patches at the patch level

        for batch in self.loader:  # for each batch at the original data loader
            # ----- image processing-----
            # takes image entry from the batch, converts it to tensor if it is not already tensor
            # converts the image tensor's datatype into float for the sake of the future arithmetic operations and mobilenetv3 expects float
            img = batch[self.image_key]  # [B,3,H,W]
            if not torch.is_tensor(img):
                img = torch.tensor(img)
            img = img.float()
            B, C, H, W = img.shape  # storing batch size and the image dimensions

            # ----- label -> is_anomaly -----
            # takes the labels of the images , converts it to tensor if it is not already tensor
            # Flattens the tensor into a 1D vector in case it is not, cause GLASS expects a 1D label vector
            # ensures the values are integer with .long() since the labels are class indicators
            if self.label_key in batch:  # checking if the batch has label attribute just in case
                is_anomaly = batch[self.label_key]
                if not torch.is_tensor(is_anomaly):
                    is_anomaly = torch.tensor(is_anomaly)
                is_anomaly = is_anomaly.view(-1).long()
            else:  # if the batch does not have label attribute, assume it is a good(non-anomalous pic) and create a tensor of B times zeros
                is_anomaly = torch.zeros((B,), dtype=torch.long)

            # ----- mask_gt (pixel mask) -----
            # Reads the mask entry from the batch, ensure it is a torch tensor with the shape of # [B,1,H,W]
            # Ensures that the mask values are 0.0 or 1.0 depending on their real value which is either 0 or 255
            if self.mask_key in batch and batch[self.mask_key] is not None:
                m = batch[self.mask_key]  # [B,1,H,W]
                if not torch.is_tensor(m):
                    m = torch.tensor(m)
                if m.ndim == 3:  # if the mask is in [B,H,W] format, convert it to [B,1,H,W]
                    m = m.unsqueeze(1)  # unsqueeze to add the missing channel info just in case
                mask_bin = (m > 0).float()  # [B,1,H,W]
            else:  # if no mask exists (good img) create mask tensor with all zeros, even though this is done in image_mask_transform.py, putting it for ensuring
                mask_bin = torch.zeros((B, 1, H, W), dtype=torch.float32, device=img.device)

            mask_gt = mask_bin  # [B,1,H,W]

            # ----- TRAIN: build synthetic aug + patch mask -----
            #To create augmented anomalous version of the images during the training
            if self.is_train and (torch.rand(1).item() < self.synth_prob):  # apply synthetic corruption with the
                # probability of synth_prob. torch.rand(1).item() picks a number between [0,1).

                # creating an empty patch-level mask (tensor of zeros)
                # for each image in the batch 0 means normal region and 1 means the patch is patch is synthetic anomaly region
                mask_patch = torch.zeros((B, ph, pw), dtype=torch.long, device=img.device)

                # choosing random rectangles per image in patch space by creating B random integers, one per image.
                # Each value is between rect_min and rect_max. that specifies the rectangle's height and weight
                rect_h = torch.randint(self.rect_min, self.rect_max + 1, (B,), device=img.device)
                rect_w = torch.randint(self.rect_min, self.rect_max + 1, (B,), device=img.device)

                # picking where the rectangle starts vertically horizontally
                # using max(1, ph(or pw) - self.rect_max) to not exceed ph height and pw weight
                top = torch.randint(0, max(1, ph - self.rect_max), (B,), device=img.device)
                left = torch.randint(0, max(1, pw - self.rect_max), (B,), device=img.device)

                # for each image in the batch; take height, weight, top and left from the tensor
                # convert them to a Python number with .item(), take the min so it never exceeds patch-grid height, patch-grid weight and fits vertically and horizontally
                # converting the final results to int to make sure that the slicing is feasible
                for i in range(B):
                    h_i = int(min(rect_h[i].item(), ph))
                    w_i = int(min(rect_w[i].item(), pw))
                    t_i = int(min(top[i].item(), ph - h_i))
                    l_i = int(min(left[i].item(), pw - w_i))
                    mask_patch[i, t_i:t_i + h_i, l_i:l_i + w_i] = 1  # for the image i, setting the rectangle region to 1

                # converting patch mask into pixel mask to be able to apply noise to the corresponding region in the image
                # adding channel info with unsquuze and converting the datatype to float since interpolation works with float
                mask_patch_1 = mask_patch.unsqueeze(1).float()  # [B,1,ph,pw]
                # upsampling from patch-grid size to image size (mode="nearest" since we want binary values in mask)
                # mask_pix shows where exatly to add the noise in the original training image
                mask_pix = F.interpolate(mask_patch_1, size=(H, W), mode="nearest")  # [B,1,H,W]

                # --create augmented image by adding noise only in masked region--

                # creating the copy of the training image (later this will be the anomalous version of the image)
                aug = img.clone()

                # ImageNet mean/std (because the pipeline normalizes images)
                mean = torch.tensor([0.485, 0.456, 0.406], device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=img.device, dtype=img.dtype).view(1, 3, 1, 1)

                # unnormalize the image -> [0,1]
                img_01 = (img * std + mean).clamp(0.0, 1.0)

                # add noise in [0,1] space (only inside pixel mask)
                noise = torch.randn_like(img_01) * self.noise_std
                # applying noise to the only masked region
                aug_01 = torch.where(mask_pix.bool(), (img_01 + noise).clamp(0.0, 1.0), img_01)

                # renormalize back (so aug has same distribution as img)
                aug = (aug_01 - mean) / std

                # final mask_s must match embedding patch count: [B,P,1], what GLASS would expect
                #transpose is for swapping the last two dimensions: [B,1,P] → [B,P,1]
                mask_s = mask_patch_1.flatten(2).transpose(1, 2).long()  # [B,P,1]

            else:
                # ---EVAL (val/test): use GT mask downsampled to patch grid ---
                # artificial noise is not wanted on val/test images
                # mask_bin is the pixel-level GT mask, turning it into patch-grid mask that GLASS would accept
                aug = img
                mask_small = F.interpolate(mask_bin, size=(ph, pw), mode="nearest")  # [B,1,ph,pw]
                mask_s = (mask_small.flatten(2).transpose(1, 2) > 0).long()  # [B,P,1]


            # extracting file paths for each image in the batch
            # reading the file path strings from the existing batch dictionary and putting them into a standardized field
            # parenting every yielded batch has image bath that holds list of length b
            image_path: List[str] = []
            for k in self.path_keys:
                if k in batch:  # if the name is found
                    v = batch[k]
                    if isinstance(v, list):  # if v is already a list, convert each element into a string
                        image_path = [str(x) for x in v]
                    else:  # if v is a single value, repeat it B times so the image_path will have still the length B
                        image_path = [str(v)] * B
                    break
            if not image_path:
                image_path = [""] * B

            # yield returns one batch to the caller and the next time, it continues from the next batch
            yield {
                "image": img,
                "aug": aug,
                "mask_s": mask_s,  # [B, ph*pw, 1]
                "mask_gt": mask_gt,  # [B,1,H,W]
                "is_anomaly": is_anomaly,
                "image_path": image_path,
            }
