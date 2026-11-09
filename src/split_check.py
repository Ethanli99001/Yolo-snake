"""Check that train / val / test splits do not share images (data leakage)."""
import hashlib
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_images(folder: Path):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"split folder not found: {folder}")
    return [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]


def find_leaks(splits: dict):
    """
    splits: {"train": path, "val": path, "test": path}
    Returns a list of (hash, [(split, file), ...]) for images found in more than one split.
    Detects copies even if they were renamed.
    """
    seen = {}  # hash -> list of (split, path)
    for name, folder in splits.items():
        for img in list_images(folder):
            seen.setdefault(file_hash(img), []).append((name, img))

    leaks = []
    for h, items in seen.items():
        if len({split for split, _ in items}) > 1:
            leaks.append((h, items))
    return leaks
