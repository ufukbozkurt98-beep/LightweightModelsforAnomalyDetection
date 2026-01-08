import torch
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset

from utils.image_mask_transform import ImageMaskTransform
from utils.transform_config import TransformConfig
from utils.helper_methods import load_split_json, get_mask_path


class MVTecADDataset(Dataset):
    """
    mode:
      - train/val: normal only (from split JSON)
      - test: good + anomalies (from folder structure), mask for anomalies if available
    """

    def __init__(self, mvtec_root: Path, category: str, mode: str, split_json: Optional[Path],
                 tcfg: TransformConfig):
        self.root = Path(mvtec_root)
        self.category = category
        self.mode = mode
        self.split_json = split_json
        self.tf = ImageMaskTransform(tcfg)

        self.samples = self._build_samples()

    def _build_samples(self) -> List[Dict]:
        cat_dir = self.root / self.category

        if self.mode in ("train", "val"):
            # if there is no split json given, we cannot know which images belong to train/val
            if self.split_json is None:
                raise ValueError("split_json is required for train/val")
            split = load_split_json(self.split_json)  # get the split dictionary from json
            paths = split["train"] if self.mode == "train" else split["val"]

            # these are relative paths saved by Task 1
            return [{"img": self.root / p, "label": 0, "defect": "good", "mask": None} for p in paths]

        if self.mode == "test":
            test_dir = cat_dir / "test"
            gt_dir = cat_dir / "ground_truth"
            out = []
            for defect_dir in sorted([d for d in test_dir.iterdir() if d.is_dir()]):
                # Get the label inside the test folder,
                # as an example: [broken_small, broken_large, contamination, good] for the 'bottle'
                defect_type = defect_dir.name
                # iterate through the list of paths that are every .png files in the folder in a sorted way
                for img_path in sorted(defect_dir.glob("*.png")):
                    # append image and its information to the out[] according its label
                    if defect_type == "good":
                        out.append({"img": img_path, "label": 0, "defect": "good", "mask": None})
                    else:
                        mask_path = get_mask_path(gt_dir, defect_type, img_path)
                        out.append({"img": img_path, "label": 1, "defect": defect_type, "mask": mask_path})
            return out

        raise ValueError(f"Unknown mode: {self.mode}")  # throw error if the mode is not entered train, val or test

    # Returns the number of images for the respected mode
    def __len__(self):
        return len(self.samples)

    # finds the path of the item, opens the image(also its mask if it exists),
    # transforms them into tensors, returns as dictionary
    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]

        img = Image.open(s["img"]).convert("RGB")
        # for 'good' images, mask will stay as none, for anomalous images, mask_img becomes a mask image
        mask_img = None
        if s["mask"] is not None:
            mask_img = Image.open(s["mask"]).convert("L")

        x, m = self.tf(img, mask_img)  # calling ImageMaskTransform pipeline
        # x: image tensor [3, 256, 256] (normalized)
        # m: mask tensor [1, 256, 256] (zeros if mask_img is None)

        # returning one sample as dictionary
        return {
            "image": x,
            "mask": m,
            "label": torch.tensor(s["label"], dtype=torch.long),
            "category": self.category,
            "defect_type": s["defect"],
            "path": str(s["img"]),
        }
