from dataclasses import dataclass
from PIL import Image
import torchvision.transforms.functional as TF
from configs.config import (
    IMAGE_INPUT_SIZE
)


# holding the settings
@dataclass
class TransformConfig:
    input_size: int = IMAGE_INPUT_SIZE  # model will receive images of size 256×256
    normalize: bool = True  # apply ImageNet normalization by default
    rotate_deg: float = 5.0  # for train augmentation


# resizing the image without breaking its ratio as an example, if the image is 700x700, scale = 0.3657.
# Therefore, the new_h and new_w will be equal to round(700×0.3657) = 256
def resize_keep_aspect(img: Image.Image, target: int, interpolation) -> Image.Image:
    w, h = img.size
    scale = target / float(max(w, h))  # how much to multiply both sides so the longer edge becomes the target
    new_w = max(1,
                int(round(w * scale)))  # using max(1,) to never get dimension 0 error in case of an unusual edge case
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), interpolation)  # interpolation is 'nearest' for masks, 'bi linear' for the images


# if the img is already 256x256 after resizing, pad_w and pad_h will be equal to 0, therefore there will be no
# padding performed
def pad_to_square(img: Image.Image, target: int, fill: int = 0) -> Image.Image:
    w, h = img.size
    pad_w = max(0, target - w)
    pad_h = max(0, target - h)
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    return TF.pad(img, [left, top, right, bottom], fill=fill)
