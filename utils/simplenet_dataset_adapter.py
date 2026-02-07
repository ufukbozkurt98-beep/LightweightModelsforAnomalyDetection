from __future__ import annotations

from typing import Any, Dict
import torch
from torch.utils.data import Dataset

#Simplenet expects is_anomaly and image_path in the batch with proper names and type
class SimpleNetDatasetAdapter(Dataset): #creating a dataset

    def __init__(self, base: Dataset):
        self.base = base #passing the existing dataset

    def __len__(self) -> int:
        return len(self.base) # the adapter will have the  same length as the base

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        d = self.base[idx]
        out = dict(d)  # ensuring to not modify the existing dictionary

        # making sure that the label is an integer and not a tensor with an unwanted shape as (1,)
        lbl = out.get("label", 0)
        lbl_int = int(lbl.item()) if hasattr(lbl, "item") else int(lbl)

        # converting it to a scalar tensor with dtype=long
        out["is_anomaly"] = torch.tensor(lbl_int, dtype=torch.long)
        # adding the existing path
        out["image_path"] = out.get("path", "")
        return out
