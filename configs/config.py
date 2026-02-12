# configs/config.py
from pathlib import Path

MVTEC_ROOT = Path("./data/MVTec-AD")
REPORTS_DIR = Path("./data/reports")
CATEGORY = "cable"
VAL_RATIO = 0.2
VAL_RATIO_CFLOW= 0.0
SEED = 42
BATCH_SIZE = 32
IMAGE_INPUT_SIZE = 256
SPLIT_JSON = REPORTS_DIR / f"mvtec_{CATEGORY}_split.json" # path object pointing to split file of the category
NUM_WORKERS = 2
TAR_PATH = Path("./data/mvtec_anomaly_detection.tar.xz")
BACKBONE_KEY = "mobilenetv3_small"
#BACKBONE_KEY = "efficientnet_lite0"
#BACKBONE_KEY = "shufflenet_g1"  # ShuffleNet V1 g=1: baseline (144/288/576 ch)
#BACKBONE_KEY = "shufflenet_g3"  # ShuffleNet V1 g=3: paper's default (240/480/960 ch)
#BACKBONE_KEY = "shufflenet_g8"  # ShuffleNet V1 g=8: most aggressive (384/768/1536 ch)
#BACKBONE_KEY = "mobileformer_508m"  # MobileFormer 508M FLOPs (14.0M params)
#BACKBONE_KEY = "mobileformer_294m"  # MobileFormer 294M FLOPs (11.4M params)
#BACKBONE_KEY = "mobileformer_96m"   # MobileFormer 96M FLOPs (4.6M params)
BACKBONE_KEY = "mobileformer_52m"   # MobileFormer 52M FLOPs (3.5M params)
#METHOD = "cflow"
#METHOD = "simplenet"
METHOD = "cflow"