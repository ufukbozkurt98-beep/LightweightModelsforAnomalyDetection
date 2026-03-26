"""
Run FastFlow on all 15 MVTec-AD categories for 5 lightweight backbones.
After all backbones are done, prints all 5 summary tables together.

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
    from main import main, print_summary_table

    all_backbone_results = {}

    for backbone in BACKBONES:
        print(f"\n{'*'*80}")
        print(f"  FASTFLOW  —  BACKBONE: {backbone}")
        print(f"{'*'*80}\n")

        # Patch config for this backbone
        cfg.BACKBONE_KEY = backbone

        results = main()
        all_backbone_results[backbone] = results

    # Print all 5 summary tables together at the end
    print(f"\n{'#'*100}")
    print(f"  ALL FASTFLOW RESULTS")
    print(f"{'#'*100}")
    for backbone, results in all_backbone_results.items():
        if results:
            print_summary_table(results, "fastflow", backbone)
