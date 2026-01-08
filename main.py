# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from pathlib import Path

from utils.mvtec_extract import ensure_extracted
from utils.data_cleaning_check import run_task1

TAR_PATH = "./data/mvtec_anomaly_detection.tar.xz"
MVTec_ROOT = "./data/MVTec-AD"

data_root = ensure_extracted(TAR_PATH, MVTec_ROOT)

out_dir = Path("./data/reports")
out_dir.mkdir(parents=True, exist_ok=True)

run_task1(
    mvtec_root=Path(data_root),
    out_dir=out_dir,
    category= 'bottle',   # None =  all categories
    val_ratio=0.2,
    seed=42
)
