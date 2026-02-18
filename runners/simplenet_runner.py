from pathlib import Path

from utils.simplenet_dataset_adapter import SimpleNetDatasetAdapter

from configs.config import (
    MVTEC_ROOT, REPORTS_DIR, CATEGORY, VAL_RATIO, SEED, IMAGE_INPUT_SIZE, BATCH_SIZE, SPLIT_JSON, TAR_PATH
)

from utils.mvtec_extract import ensure_extracted
from utils.data_check_and_split import scan_and_split
from utils.data_loader import make_loader_mvtec_ad

import torch
import torchvision.models as tvm

from utils.feature_extractor import build_extractor


from configs.config import BACKBONE_KEY

from simplenet_code.simplenet_author.simplenet import SimpleNet
from torch.utils.data import DataLoader

def run_simplenet(train_loader, val_loader, test_loader):
    # for simplenet
    train_ds = SimpleNetDatasetAdapter(train_loader.dataset)
    val_ds = SimpleNetDatasetAdapter(val_loader.dataset)
    test_ds = SimpleNetDatasetAdapter(test_loader.dataset)

    # recreate loaders with SAME settings
    train_loader = DataLoader(
        train_ds,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=train_loader.num_workers,
        pin_memory=getattr(train_loader, "pin_memory", False),
        drop_last=getattr(train_loader, "drop_last", False),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=val_loader.batch_size,
        shuffle=False,
        num_workers=val_loader.num_workers,
        pin_memory=getattr(val_loader, "pin_memory", False),
        drop_last=getattr(val_loader, "drop_last", False),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=test_loader.batch_size,
        shuffle=False,
        num_workers=test_loader.num_workers,
        pin_memory=getattr(test_loader, "pin_memory", False),
        drop_last=getattr(test_loader, "drop_last", False),
    )

    b = next(iter(test_loader))
    print("TEST  shapes:", b["image"].shape, b["mask"].shape, "labels:", b["label"].unique().tolist())
    print("TEST defect types sample:", b["defect_type"][:4])

    # -------- lightweight stuff ------
    # get a batch to do mask check
    b = next(iter(test_loader))
    mask_sums = b["mask"].sum(dim=(1, 2, 3))
    print("Mask sums (per sample):", mask_sums.tolist())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = build_extractor(BACKBONE_KEY, device=device)
    extractor.eval()  # for deterministic behavior

    # getting a batch for feature extraction check
    b = next(iter(train_loader))
    with torch.no_grad():
        feats = extractor(b["image"].to(device))
    print({k: tuple(v.shape) for k, v in feats.items()})

    b = next(iter(test_loader))
    print(b.keys())
    print("is_anomaly:", b["is_anomaly"][:4], "image_path sample:", b["image_path"][:2])

    # -------------------------------------------

    """

    sn = SimpleNet(device)
    sn.load(
        backbone=extractor,
        layers_to_extract_from=["l1", "l2", "l3"],
        device=device,
        input_shape=[3, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE],
        pretrain_embed_dimension=1536,
        target_embed_dimension=1536,
    )

    b = next(iter(train_loader))
    with torch.no_grad():
        emb = sn._embed(b["image"].to(device), evaluation=True)
    print("SimpleNet embed OK, got type:", type(emb))

    """

    sn = SimpleNet(device)
    sn.load(
        backbone=extractor,
        layers_to_extract_from=["l1", "l2", "l3"],
        #layers_to_extract_from=["l2", "l3"],
        device=device,
        input_shape=[3, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE],
        pretrain_embed_dimension=128,
        target_embed_dimension=128,
        patchsize=3,
        patchstride=1,
        meta_epochs=40,
        gan_epochs=4,
        aed_meta_epochs=1,
        dsc_layers=2,
        dsc_hidden=1024,
        train_backbone=False,
        noise_std=0.5,
        dsc_lr=0.0001,       
        lr=0.0001,           
        dsc_margin=0.5,      
    )

    sn.set_model_dir(str(REPORTS_DIR / "simplenet_runs"), CATEGORY)

    # Train and evaluate
    best = sn.train(train_loader, test_loader)
    print("Best record:", best)
