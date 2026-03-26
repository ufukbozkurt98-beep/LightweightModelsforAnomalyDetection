"""
Run FastFlow on all 15 MVTec-AD categories for 5 lightweight backbones.
Results are saved per backbone, with resume support (skips already-done categories).

Usage:
    python run_fastflow_all.py
"""

import configs.config as cfg

# Override method
cfg.METHOD = "fastflow"

BACKBONES = [
    "mobilenetv3_large",
    "efficientnet_lite1",
    "mobilevit_xs",
    "mobileformer_294m",
    "shufflenet_g8",
]

if __name__ == "__main__":
    from main import main

    for backbone in BACKBONES:
        print(f"\n{'*'*80}")
        print(f"  FASTFLOW  —  BACKBONE: {backbone}")
        print(f"{'*'*80}\n")

        # Patch config for this backbone
        cfg.BACKBONE_KEY = backbone

        main()

        print(f"\n  ✓ Done: {backbone}\n")
