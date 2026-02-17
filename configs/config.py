# configs/config.py
from pathlib import Path

MVTEC_ROOT = Path("./data/MVTec-AD")
REPORTS_DIR = Path("./data/reports")
CATEGORY = "bottle"
VAL_RATIO = 0.0
SEED = 42
BATCH_SIZE = 4
IMAGE_INPUT_SIZE = 256
SPLIT_JSON = REPORTS_DIR / f"mvtec_{CATEGORY}_split.json" # path object pointing to split file of the category
NUM_WORKERS = 2
TAR_PATH = Path("./data/mvtec_anomaly_detection.tar.xz")
#BACKBONE_KEY = "mobilenetv3_small"
BACKBONE_KEY = "efficientnet_lite0"
#METHOD = "glass"
METHOD = "simplenet"


