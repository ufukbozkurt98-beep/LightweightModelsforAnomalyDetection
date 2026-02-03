import torch
from typing import Optional

from PIL import Image
import torchvision.transforms.functional as TF

from utils.constants import IMAGENET_MEAN, IMAGENET_STD
from utils.transform_config import resize_keep_aspect, pad_to_square, TransformConfig

import random
from torchvision.transforms import InterpolationMode


class ImageMaskTransform:
    def __init__(self, cfg: TransformConfig):
        self.cfg = cfg  #storing configuration inside the object

    # transformation pipeline
    # returning processed tensors for single image
    def __call__(self, img: Image.Image, mask: Optional[Image.Image]):
        # apply resizing/padding to the image so that it becomes 256x256
        img = resize_keep_aspect(img, self.cfg.input_size,
                                 # Using 'bi linear' interpolation, taking a weighted average of the 4 nearest pixels
                                 # (surrounding 2x2 neighbor of the pixel)
                                 Image.BILINEAR)
        img = pad_to_square(img, self.cfg.input_size, fill=0)

        if mask is not None:  #apply the same procedure to the mask if exists (mask exists when the image is anomalous)
            mask = resize_keep_aspect(mask, self.cfg.input_size,
                                      # Using 'nearest' for masks, so we don’t create blurry “gray” edges.
                                      Image.NEAREST)
            mask = pad_to_square(mask, self.cfg.input_size, fill=0)

        # AUGMENTATION: only for rotate_deg > 0
        if self.cfg.rotate_deg and self.cfg.rotate_deg > 0:
            angle = random.uniform(-self.cfg.rotate_deg, self.cfg.rotate_deg)

            img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR, fill=0)

            if mask is not None:
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)

        x = TF.to_tensor(img)  # Converting PIL image to tensor
        if self.cfg.normalize:  #if normalize is enabled, apply normalization
            x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)

        # if the image is normal, creating a mask with all 0's with the goals of simplicity and consistency in batching
        if mask is None:
            m = torch.zeros((1, self.cfg.input_size, self.cfg.input_size),
                            dtype=torch.float32)  #m will be all 0.0 values with the shape of [1,256,256] in this case.
        else:
            m = TF.to_tensor(
                mask)  #if a mask exist for the image (in the case of anomalous images), convert mask into tensor
            m = (m > 0.5).float()  #ensuring that the tensor values of the mask is boolean

        return x, m
