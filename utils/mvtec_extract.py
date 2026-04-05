import tarfile
import zipfile
from pathlib import Path


def ensure_extracted(target_path: str, target_root: str) -> str:
    """
    - If target root is not empty, it doesn't extract again
    - If it is empty, it extracts
    - Supports .tar.xz, .tar.gz, .tar, .zip formats
    """
    target_path = Path(target_path)
    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    # If folder is not empty, Don't extract
    if any(target_root.iterdir()):
        print(f"It is already exists, skip extraction: {target_root}")
        return str(target_root)

    # Check if archive file exists
    if not target_path.exists():
        raise FileNotFoundError(f"Archive not found: {target_path}")

    print(f"Extraction process is started: {target_path} -> {target_root}")

    if zipfile.is_zipfile(target_path):
        with zipfile.ZipFile(target_path, 'r') as zf:
            zf.extractall(path=str(target_root))
    elif tarfile.is_tarfile(target_path):
        with tarfile.open(target_path, mode="r:*") as tar:
            tar.extractall(path=str(target_root))
    else:
        raise ValueError(f"Unsupported archive format: {target_path}")

    print(f"Extraction process is completed: {target_root}")
    return str(target_root)
