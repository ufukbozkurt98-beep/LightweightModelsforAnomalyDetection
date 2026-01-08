from pathlib import Path
import torch
from torch.utils.data import DataLoader

from utils.transform_config import TransformConfig
from utils.mvtecAD_dataset import MVTecADDataset

from configs.config import (
    IMAGE_INPUT_SIZE, BATCH_SIZE, NUM_WORKERS
)


def make_loader_mvtec_ad(mvtec_root: Path, category: str, mode: str, split_json: Path,
                         input_size: int = IMAGE_INPUT_SIZE, batch_size: int = BATCH_SIZE,
                         num_workers: int = NUM_WORKERS,
                         normalize: bool = True):
    tcfg = TransformConfig(input_size=input_size, normalize=normalize)  # creating a config object for transforms
    # creating dataset object to store parameters and build self.samples by scanning json (train/val) or folders (test)
    ds = MVTecADDataset(mvtec_root, category, mode, split_json, tcfg)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True if mode == "train" else False,
        # val/test modes need to be stable&repeatable so no shuffle for them
        num_workers=num_workers,
        # pin_memory=True  # for speed transfer to GPU
        pin_memory=torch.cuda.is_available()  # enable pinned memory only for CUDA
    )
