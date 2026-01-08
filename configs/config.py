# configs/config.py
from pathlib import Path

MVTEC_ROOT = Path("./data/MVTec-AD")
REPORTS_DIR = Path("./data/reports")
CATEGORY = "bottle"
VAL_RATIO = 0.2
SEED = 42
BATCH_SIZE = 16
IMAGE_INPUT_SIZE = 256
SPLIT_JSON = REPORTS_DIR / f"mvtec_{CATEGORY}_split.json"
NUM_WORKERS = 2
TAR_PATH = Path("./data/mvtec_anomaly_detection.tar.xz")

