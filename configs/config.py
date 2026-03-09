# configs/config.py
from pathlib import Path
import os

MVTEC_ROOT = Path("./data/MVTec-AD")
REPORTS_DIR = Path("./data/reports")
DTD_PATH = Path(os.environ.get("DTD_PATH", "./data/dtd/images"))
CATEGORY = "wood"
VAL_RATIO = 0.0
SEED = 42
BATCH_SIZE = 8
IMAGE_INPUT_SIZE = 256
SPLIT_JSON = REPORTS_DIR / f"mvtec_{CATEGORY}_split.json" # path object pointing to split file of the category
NUM_WORKERS = 2
TAR_PATH = Path("./data/mvtec_anomaly_detection.tar.xz")
BACKBONE_KEY = "mobilenetv3_large"
#BACKBONE_KEY = "efficientnet_lite1"
#BACKBONE_KEY = "shufflenetv2_x1_0"
METHOD = "glass"
#METHOD = "simplenet"

# Set to True to run all 15 MVTec-AD categories in a loop.
# When True, CATEGORY above is ignored.
RUN_ALL = True


