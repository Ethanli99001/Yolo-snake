import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import io
from pathlib import Path
import pytest
from PIL import Image
from split_check import find_leaks, file_hash, list_images

# Change this to your dataset folder (YOLO layout: images/train, images/val, images/test)
DATASET_DIR = Path(__file__).resolve().parent.parent / "data" / "dataset" / "images"


def save_img(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color).save(path, "PNG")


@pytest.fixture
def splits(tmp_path):
    return {s: tmp_path / s for s in ("train", "val", "test")}


# --- Unit tests on fake data ---
def test_no_leak_when_splits_are_distinct(splits):
    save_img(splits["train"] / "1.png", (255, 0, 0))
    save_img(splits["val"] / "2.png", (0, 255, 0))
    save_img(splits["test"] / "3.png", (0, 0, 255))
    assert find_leaks(splits) == []


def test_same_image_in_train_and_test_is_leak(splits):
    save_img(splits["train"] / "1.png", (255, 0, 0))
    save_img(splits["val"] / "2.png", (0, 255, 0))
    save_img(splits["test"] / "1.png", (255, 0, 0))
    leaks = find_leaks(splits)
    assert len(leaks) == 1
    assert {s for s, _ in leaks[0][1]} == {"train", "test"}


def test_renamed_copy_is_still_detected(splits):
    # Same pixels, different filename -> hash catches it
    save_img(splits["train"] / "cottonmouth_12.png", (10, 20, 30))
    save_img(splits["val"] / "img_999.png", (10, 20, 30))
    save_img(splits["test"] / "x.png", (1, 2, 3))
    assert len(find_leaks(splits)) == 1


def test_duplicate_inside_one_split_is_not_leak(splits):
    save_img(splits["train"] / "a.png", (5, 5, 5))
    save_img(splits["train"] / "b.png", (5, 5, 5))
    save_img(splits["val"] / "c.png", (6, 6, 6))
    save_img(splits["test"] / "d.png", (7, 7, 7))
    assert find_leaks(splits) == []


def test_species_subfolders_are_searched(splits):
    save_img(splits["train"] / "Cottonmouth" / "1.png", (9, 9, 9))
    save_img(splits["val"] / "Cottonmouth" / "1.png", (9, 9, 9))
    save_img(splits["test"] / "Other" / "1.png", (8, 8, 8))
    assert len(find_leaks(splits)) == 1


def test_non_image_files_ignored(splits):
    for s in splits.values():
        s.mkdir(parents=True)
        (s / "labels.csv").write_text("same content")
    assert find_leaks(splits) == []


def test_missing_split_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_leaks({"train": tmp_path / "nope"})


def test_file_hash_is_stable(tmp_path):
    save_img(tmp_path / "a.png", (1, 1, 1))
    assert file_hash(tmp_path / "a.png") == file_hash(tmp_path / "a.png")


# --- Check the real dataset (skipped until the split exists) ---
@pytest.mark.skipif(not DATASET_DIR.is_dir(), reason=f"dataset not found at {DATASET_DIR}")
def test_real_dataset_has_no_leaks():
    real = {s: DATASET_DIR / s for s in ("train", "val", "test") if (DATASET_DIR / s).is_dir()}
    leaks = find_leaks(real)
    msg = "\n".join(
        " <-> ".join(f"{s}:{p.name}" for s, p in items) for _, items in leaks[:20]
    )
    assert not leaks, f"{len(leaks)} images appear in more than one split:\n{msg}"
