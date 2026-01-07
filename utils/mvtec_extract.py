import tarfile
from pathlib import Path


def ensure_extracted(tar_path: str, target_root: str) -> str:
    """
    tar_path: ./data/mvtec_anomaly_detection.tar.xz
    target_root: ./data/MVTec-AD
    - If target root is not empty, it doesn't extract again
    - If it is empty, it extracts
    """
    tar_path = Path(tar_path)
    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    # If folder is not empty, Don't extract
    if any(target_root.iterdir()):
        print(f"It is already exists, skip extraction: {target_root}")
        return str(target_root)

    print(f"Extraction process is started: {tar_path} -> {target_root}")
    with tarfile.open(tar_path, mode="r:*") as tar:
        tar.extractall(path=str(target_root))

    print(f"Extraction process is completed: {target_root}")
    return str(target_root)
