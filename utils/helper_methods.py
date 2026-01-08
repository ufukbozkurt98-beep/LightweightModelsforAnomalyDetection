import json
from pathlib import Path
from typing import Dict, Optional


# to read the train/val split from json and convert it to python dictionary
def load_split_json(split_path: Path) -> Dict:
    return json.loads(split_path.read_text())


# construct mask path for the ground truth image, if it truly exists, return it
def get_mask_path(gt_dir: Path, defect_type: str, img_path: Path) -> Optional[Path]:
    # .stem removes the extension such as .png, so that we can add _mask.png
    p = gt_dir / defect_type / f"{img_path.stem}_mask.png"
    return p if p.exists() else None
