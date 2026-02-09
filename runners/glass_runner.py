from glass_src.glass import GLASS  # import the GLASS object from the glass.py file of glass_src package
from utils.glass_backbone_adapter import GlassBackboneAdapter
from utils.glass_loader_adapter import GlassLoaderAdapter
#--------------------------------
from pathlib import Path  # to enable to use path objects and /|\ handling

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

def run_glass(train_loader, val_loader, test_loader):
    # getting the first batch from the loaders and printing the sizes and the labels
    b = next(iter(train_loader))
    print("TRAIN shapes:", b["image"].shape, b["mask"].shape, "labels:", b["label"].unique().tolist())

    b = next(iter(val_loader))
    print("VALIDATION shapes:", b["image"].shape, b["mask"].shape, "labels:", b["label"].unique().tolist())

    b = next(iter(test_loader))
    print("TEST  shapes:", b["image"].shape, b["mask"].shape, "labels:", b["label"].unique().tolist())
    print("TEST defect types sample:", b["defect_type"][:4])  # printing the defect types

    # check if the test masks have positive number of pixel sums to ensure we have non-empty masks in test
    mask_sums = b["mask"].sum(dim=(1, 2, 3))
    print("Mask sums (per sample):", mask_sums.tolist())

    # ----------------- lightweight backbone -----------------
    # run the model in GPU if available, otherwise run it on CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # lightweight backbone feature extractor building
    extractor = build_extractor(BACKBONE_KEY, device=device)

    # checking if the extractor's output is in correct shape
    b = next(iter(train_loader))
    with torch.no_grad():  # don't track gradients since this is not a training but an output check
        feats = extractor(b["image"].to(device))  # feats is a dictionary of feature maps
    print({k: tuple(v.shape) for k, v in feats.items()})
    # shape of the mobilenetv3 output : {'l1': tensor, 'l2': tensor, 'l3': tensor}
    # for example: {'l1': (16, 16, 64, 64), 'l2': (16, 24, 32, 32), 'l3': (16, 48, 16, 16)}

    # ----------------- GLASS -----------------
    # creating the adaptor object and storing the extractor inside, moves the adaptor to the chosen device
    # eval() for making the backbone forward deterministic by disabling dropout
    backbone = GlassBackboneAdapter(extractor).to(device).eval()

    # reaching to the submodules inside backbone (l1, l2, l3) to check if we have the correct layer names for GLASS
    print("Adapter hookable layers:",
          [name for name, _ in backbone.named_modules() if name in ["l1", "l2", "l3"]])

    # moving the image batch to the same device as the backbone and performing a forward pass
    # adaptor stores features into for example backbone.l1.last but only after one forward pass is performed
    with torch.no_grad():  # don't track gradients since this is not a training but an output check
        _ = backbone(b["image"].to(
            device))  # the output is not important, we would like to store the feature map and  perofrm check

    # checking if the shapes of the cached feature maps are correct
    print("Adapter forward OK. l1/l2/l3 cached shapes:",
          tuple(backbone.l1.last.shape), tuple(backbone.l2.last.shape), tuple(backbone.l3.last.shape))

    # ---- GLASS: load with the adapter backbone ----
    glass = GLASS(device)  # GLASS instance creation. Device is stores inside so that the GLASS will know where to run

    glass.load(
        backbone=backbone,
        layers_to_extract_from=["l2", "l3"],  # or ["l1","l2","l3"] which layers to capture the output from
        #layers_to_extract_from=["l2", "l1", "l3"],
        device=device,
        input_shape=[3, 256, 256],  # matching with the dataloader size
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        patchstride=1,
        meta_epochs=200,
        # eval_epochs=10 ** 9,  # disables val AUROC/PRO
        eval_epochs=20,  # GLASS runs predict() on every x epochs
        step=10,
        train_backbone=False,  # we want the lightweight backbone to stay as frozen
        pre_proj=0
    )

    print("GLASS load OK")
    glass.eval()  # putting glass into evaluation mode since we are not starting to training yet

    with torch.no_grad():  # don't track gradients since this is not a training but a test run
        # conforming the backbone and the patch embedding works
        emb, patch_shapes = glass._embed(b["image"].to(device), evaluation=True)
    print("Embed OK. emb shape:", tuple(emb.shape), "patch_shapes:", patch_shapes)

    # GLASS needs output folders (ckpt_dir/tb_dir)
    glass.set_model_dir("data/glass_runs", CATEGORY)

    # avoiding the distribution==1 mode because of the risk of early training stopping
    # no Excel file needed in this way since we use our own loading pipeline
    # svd = 0 in this mode which is more reliable for now since we change backbone to lightweight
    train_loader.dataset.distribution = 2
    val_loader.dataset.distribution = 2
    test_loader.dataset.distribution = 2

    patch_grid = tuple(patch_shapes[0])  # ensuring mask_s shape matches how _embed() patches. This goes into the GlassLoaderAdapter as (ph, pw)

    # loaders cannot directly be passed through the GLASS. We need a wrapper.
    train_g = GlassLoaderAdapter(train_loader, patch_grid=patch_grid, is_train=True)
    val_g = GlassLoaderAdapter(val_loader, patch_grid=patch_grid, is_train=False)
    test_g = GlassLoaderAdapter(test_loader, patch_grid=patch_grid, is_train=False)

    print("Training...")
    # glass.trainer(train_g, val_g, name=CATEGORY) #since in our current pipeline, val has only good images, the AUROC were not going to be meaningful
    # That's why we send test_g instead of val_g to the training
    # But still, the test images ARE NOT being used for compute gradients, but to choose what is best epoch, so the model selection
    # We have done it this way since the authors of the glass have chosen a similar way as well for checkpoint selection, but we can alter this later
    glass.trainer(train_g, test_g, name=CATEGORY)

    # counting and printing how many "good" and how many "anomalous" samples each block has
    from collections import Counter
    cnt = Counter()
    for b in test_loader:
        cnt.update(b["label"].tolist())
    print("test label counts:", cnt)

    print("Testing...")
    i_auroc, i_ap, p_auroc, p_ap, pro, epoch = glass.tester(test_g, name=CATEGORY)

    print(f"[TEST @ epoch {epoch}] "
          f"Image-AUROC={i_auroc * 100:.2f}  "
          f"Pixel-AUROC={p_auroc * 100:.2f}  "
          f"PRO={pro * 100:.2f}")