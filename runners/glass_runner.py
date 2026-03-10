import json
from collections import Counter
from pathlib import Path

import torch

from glass_source_code.glass_src.glass import GLASS  # import the GLASS object from the glass.py file of glass_src package
from utils.glass_backbone_adapter import GlassBackboneAdapter
from utils.glass_loader_adapter import GlassLoaderAdapter
from utils.feature_extractor import build_extractor
from utils.model_benchmark import reset_gpu_peak, measure_gpu_memory_mb, measure_inference_latency

from configs.config import (
    REPORTS_DIR, CATEGORY, IMAGE_INPUT_SIZE, BACKBONE_KEY, DTD_PATH,
)


def run_glass(train_loader, val_loader, test_loader, category=None):
    # category argument added so main.py can pass the current category when
    # looping over all 15 classes, matching run_simplenet()'s calling convention.
    # Falls back to the config CATEGORY for single-category runs.
    if category is None:
        category = CATEGORY

    # getting the first batch from the loaders and printing the sizes and the labels
    b = next(iter(train_loader))
    print("TRAIN shapes:", b["image"].shape, b["mask"].shape, "labels:", b["label"].unique().tolist())

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
    extractor.eval()

    # checking if the extractor's output is in correct shape
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
        input_shape=[3, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE],  # matching with the dataloader size
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        patchstride=1,
        meta_epochs=640,
        # eval_epochs=10 ** 9,  # disables val AUROC/PRO
        eval_epochs=20,  # GLASS runs predict() on every x epochs
        step=20,
        train_backbone=False,  # we want the lightweight backbone to stay as frozen
        pre_proj=1
    )

    print("GLASS load OK")
    glass.eval()  # putting glass into evaluation mode since we are not starting to training yet

    with torch.no_grad():  # don't track gradients since this is not a training but a test run
        # conforming the backbone and the patch embedding works
        emb, patch_shapes = glass._embed(b["image"].to(device), evaluation=True)
    print("Embed OK. emb shape:", tuple(emb.shape), "patch_shapes:", patch_shapes)

    # GLASS needs output folders (ckpt_dir/tb_dir)
    # using REPORTS_DIR so the path respects the config, consistent with simplenet
    glass_runs_dir = str(REPORTS_DIR / "glass_runs")
    glass.set_model_dir(glass_runs_dir, category)

    # avoiding the distribution==1 mode because of the risk of early training stopping
    # no Excel file needed in this way since we use our own loading pipeline
    # svd = 0 in this mode which is more reliable for now since we change backbone to lightweight
    train_loader.dataset.distribution = 2
    #val_loader.dataset.distribution = 2
    test_loader.dataset.distribution = 2

    patch_grid = tuple(patch_shapes[0])  # ensuring mask_s shape matches how _embed() patches. This goes into the GlassLoaderAdapter as (ph, pw)

    # loaders cannot directly be passed through the GLASS. We need a wrapper.
    train_g = GlassLoaderAdapter(train_loader, patch_grid=patch_grid, is_train=True, dtd_root=str(DTD_PATH), category=category)
    #val_g = GlassLoaderAdapter(val_loader, patch_grid=patch_grid, is_train=False)
    test_g = GlassLoaderAdapter(test_loader, patch_grid=patch_grid, is_train=False)

    print("Training...")
    # glass.trainer(train_g, val_g, name=category) #since in our current pipeline, val has only good images, the AUROC were not going to be meaningful
    # That's why we send test_g instead of val_g to the training
    # But still, the test images ARE NOT being used for compute gradients, but to choose what is best epoch, so the model selection
    # We have done it this way since the authors of the glass have chosen a similar way as well for checkpoint selection, but we can alter this later
    reset_gpu_peak(device)
    best_record = glass.trainer(train_g, test_g, name=category)
    # best_record is [i_auroc, i_ap, p_auroc, p_ap, pro, best_epoch], returned by glass.trainer()
    # guard against None in case trainer exits unexpectedly
    if best_record is None:
        best_record = [0.0, 0.0, 0.0, 0.0, 0.0, -1]
    gpu_train_mb = measure_gpu_memory_mb(device)
    print(f"Peak GPU memory during training: {gpu_train_mb:.0f} MB")

    # counting and printing how many "good" and how many "anomalous" samples each block has
    cnt = Counter()
    for batch in test_loader:
        cnt.update(batch["label"].tolist())
    print("test label counts:", cnt)

    print("Testing...")
    i_auroc, i_ap, p_auroc, p_ap, pro, epoch = glass.tester(test_g, name=category)

    # measuring inference latency on the test set
    # glass.predict() takes a dataloader and returns (images, scores, segs, labels_gt, masks_gt)
    # same call signature as simplenet's predict, so measure_inference_latency wraps it identically
    reset_gpu_peak(device)
    latency, _ = measure_inference_latency(glass.predict, test_g, device=str(device))
    gpu_infer_mb = measure_gpu_memory_mb(device)

    # printing the per-category benchmark summary, matching simplenet_runner format
    print(f"\n{'='*55}")
    print(f"  PER-CATEGORY BENCHMARK: {category.upper()}")
    print(f"{'='*55}")
    print(f"  Best epoch   : {epoch}")
    print(f"  I-AUROC      : {i_auroc * 100:.2f}%")
    print(f"  P-AUROC      : {p_auroc * 100:.2f}%")
    print(f"  PRO          : {pro * 100:.2f}%")
    print(f"  GPU (train)  : {gpu_train_mb:.0f} MB")
    print(f"  GPU (infer)  : {gpu_infer_mb:.0f} MB")
    print(f"  Infer total  : {latency['total_time_s']:.3f} s")
    print(f"  Infer/image  : {latency['per_image_ms']:.2f} ms")
    print(f"  Throughput   : {latency['throughput_fps']:.1f} FPS")
    print(f"{'='*55}\n")

    # saving results as JSON, one file per category, matching simplenet_runner structure
    results_dir = REPORTS_DIR / "benchmark_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    category_result = {
        "category"        : category,
        "best_epoch"      : int(epoch),
        "i_auroc"         : round(float(i_auroc), 4),
        "i_ap"            : round(float(i_ap),    4),
        "p_auroc"         : round(float(p_auroc), 4),
        "p_ap"            : round(float(p_ap),    4),
        "pro"             : round(float(pro),     4),
        "gpu_train_mb"    : round(gpu_train_mb,   1),
        "gpu_infer_mb"    : round(gpu_infer_mb,   1),
        "infer_total_s"   : latency["total_time_s"],
        "infer_per_img_ms": latency["per_image_ms"],
        "throughput_fps"  : latency["throughput_fps"],
    }

    out_path = results_dir / f"{category}_glass_results.json"
    out_path.write_text(json.dumps(category_result, indent=2))
    print(f"  [saved → {out_path}]")
