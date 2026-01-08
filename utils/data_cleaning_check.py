import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError



# This function gets a folder path and returns a list of paths that are every .png files in the folder in a sorted way
# Sorting is to ensure that we don't get different file ordering at each run
def list_pngs(folder: Path) -> List[Path]:
    if folder.is_dir():
        return sorted(folder.glob("*.png"))
    return []


# This function opens an image, converts it to the mode(RGB for images or L for masks in this case) it is supposed to
# be to ensure consistency among images, returns its size It is used to ensure there isn't any corrupted image file
# in the dataset
def try_open_image(path: Path, mode: str) -> Optional[Tuple[int, int]]:
    try:
        with Image.open(path) as im:
            im = im.convert(mode)
            return im.size  #size returns in (W, H) format
    except (UnidentifiedImageError, OSError):
        return None


# In MVTec AD, ground truth masks are named in the following format: imgNum_mask.png
# For example -> test/contamination/000.png -> ground_truth/contamination/000_mask.png
# This function creates the expected mask path for the given image, check if it exists.
# It is used to check if every anomalous image has ground truth, also used as getter to transform masks in loader.
def get_mask_path(gt_dir: Path, defect_type: str, img_path: Path) -> Optional[Path]:
    p = gt_dir / defect_type / f"{img_path.stem}_mask.png"
    if p.exists():
        return p
    else:
        return None


def scan_category(cat_dir: Path) -> Dict:  # Cat_dir represents the category directory, as an example: .../bottle
    # Creating the path objects of the category directory content's expected paths for the future checks
    train_good = cat_dir / "train" / "good"  #.../bottle/train/good
    test_dir = cat_dir / "test"  #.../bottle/test
    gt_dir = cat_dir / "ground_truth"  #.../bottle/ground_truth

    # Checking if the expected paths actually exists
    for p in [train_good, test_dir, gt_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Missing folder: {p}")

    # Recording the perfromed checks in a dictionary for each category
    report = {
        "category": cat_dir.name,  #cat_dir.name gets the folder name, 'bottle' as an example
        "counts": {"train_good": 0, "test_good": 0, "test_anomaly": 0},
        # Storing the number of images for train and test
        "issues": {
            "unreadable_images": [],  #recording the img files that cannot be read/corrupted
            "missing_masks": [],  #recording anomalous images that do not have masks
            "unreadable_masks": [],  #recording corrupted masks
            "mask_size_mismatch": [],  #recording the masks that has diffrent size than the image
        },
    }

    # train/good
    train_imgs = list_pngs(train_good)  #train_imgs stores the .../category/train/good images as a list
    report["counts"]["train_good"] = len(train_imgs)  #records the number of training images in the category
    # If any training image is corrupted(unable to be opened), it gets recorded in the report
    for p in train_imgs:
        if try_open_image(p, "RGB") is None:
            report["issues"]["unreadable_images"].append(
                p.as_posix())  #posix converts the path to string for it to be added to the report dictionary

    # test (good + anomalies)
    for defect_dir in sorted([d for d in test_dir.iterdir() if
                              d.is_dir()]):  # Iterating through each folder inside test/ in sorted way. Example: [.../category/test/broken_large, .../category/test/broken_small, .../category/test/contamination, .../category/test/good]

        defect_type = defect_dir.name  # Gets the name(label) of the defect(defect type or 'good' in our case) for the later counts

        for img_path in list_pngs(defect_dir):  #iterate through all images in the respected label folder of test

            # If any ... /category/test/"type" image is corrupted(unable to be opened), it gets recorded in the report
            img_size = try_open_image(img_path, "RGB")
            if img_size is None:
                report["issues"]["unreadable_images"].append(img_path.as_posix())
                continue

            # Counting the non-anomalous images in test folder of the category
            if defect_type == "good":
                report["counts"]["test_good"] += 1
                continue  # If the label of the image is good, do not perform the rest of the mask checks and continue

            report["counts"]["test_anomaly"] += 1  # Counting anomalous images in test folder of the category

            mask_path = get_mask_path(gt_dir, defect_type, img_path)  #get the mask path of the image if it exists

            # The case where mask for the anomalous image does not exist
            if mask_path is None:
                # record the expected path so it's clear what's missing
                expected = (gt_dir / defect_type / f"{img_path.stem}_mask.png").as_posix()
                report["issues"]["missing_masks"].append(expected)
                continue

            # The case where the mask for the anomalous image is corrupted/unable to be pened
            mask_size = try_open_image(mask_path, "L")
            if mask_size is None:
                report["issues"]["unreadable_masks"].append(mask_path.as_posix())
                continue

            # The case where the mask size does not match the anomalous image size
            if mask_size != img_size:
                report["issues"]["mask_size_mismatch"].append({
                    "image": img_path.as_posix(),
                    "mask": mask_path.as_posix(),
                    "img_size": img_size,
                    "mask_size": mask_size
                })

    return report


def make_train_val_split(train_good_dir: Path, val_ratio: float, seed: int, root: Path):
    rng = random.Random(seed)  #always producing the same random order when the same seed is used
    imgs = list_pngs(train_good_dir)  #list of all training images in the category (a sorted list of Paths)

    #separating the root from the image path to be able to work outside of drive
    #Example: /content/drive/MyDrive/datasets/mvtec_ad/bottle/train/good/000.png  to  bottle/train/good/000.png
    #record these nre relative paths in a list
    rel_paths = [p.relative_to(root).as_posix() for p in imgs]
    rng.shuffle(rel_paths)  #reproducable random shuffle of the relative image paths

    n_val = int(
        len(rel_paths) * val_ratio)  #determining the number of the validation images according to the validation_ratio that is defined at the beginning
    val = sorted(rel_paths[:n_val])  #take the first n_val items from the shuffeld relative path list as validation
    train = sorted(rel_paths[n_val:])  #rest of the list stays as training images
    return train, val  #return both train and val image(path) lists


def run_task1(mvtec_root: Path, out_dir: Path, category: Optional[str], val_ratio: float, seed: int):
    if category is None:
        categories = [d for d in sorted(mvtec_root.iterdir()) if (
                d / "train").exists()]  #if category is not given, iterate and append all the category paths in MVTec AD dataset into the categories list
    else:
        categories = [
            mvtec_root / category]  #if not, get the category specified. As an example, the list will be looking like: [Path(".../mvtec_ad/bottle")]

    all_reports = {}  #empty dictionary that will be used to collect the reports for all categories in one place

    for cat_dir in categories:
        if not (
                cat_dir / "train").exists():  #skip the checks and splits in the case of it is not a category folder with training images
            continue

        report = scan_category(
            cat_dir)  #perform all the checks for the respected category, returns the report dictionary
        all_reports[
            cat_dir.name] = report  #add the resulted dictionary in the all_reports dictionary to collect all category results in one place

        #perfroming the validation set split
        train_good_dir = cat_dir / "train" / "good"
        train_list, val_list = make_train_val_split(train_good_dir, val_ratio, seed, mvtec_root)

        #create the split dictionary that holds all the seperation info
        split = {
            "category": cat_dir.name,
            "seed": seed,
            "val_ratio": val_ratio,
            "train": train_list,
            "val": val_list,
        }

        #creating output paths for split and report json files that will be created for the category
        split_path = out_dir / f"mvtec_{cat_dir.name}_split.json"
        report_path = out_dir / f"mvtec_{cat_dir.name}_report.json"

        #saving split and report dictionaries as json files
        split_path.write_text(json.dumps(split, indent=2))
        report_path.write_text(json.dumps(report, indent=2))

        #count the issues and print a one line status check for a quick control
        issue_counts = {k: len(v) for k, v in report["issues"].items()}
        print(
            f"✅ {cat_dir.name}: train={len(train_list)} val={len(val_list)} | counts={report['counts']} | issues={issue_counts}")

    # Save one big json file that contains all reports and print where the summary is saved
    (out_dir / "mvtec_all_reports.json").write_text(json.dumps(all_reports, indent=2))
    print(f"\nSaved summary: {out_dir / 'mvtec_all_reports.json'}")
