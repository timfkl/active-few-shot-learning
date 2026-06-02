import argparse
import random
from pathlib import Path


def find_case_ids(data_dir, image_suffix="_img.nii", mask_suffix="_mask.nii"):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    image_ids = {
        path.name[: -len(image_suffix)]
        for path in data_dir.glob(f"*{image_suffix}")
    }
    mask_ids = {
        path.name[: -len(mask_suffix)]
        for path in data_dir.glob(f"*{mask_suffix}")
    }

    case_ids = sorted(image_ids & mask_ids)
    if not case_ids:
        raise FileNotFoundError(
            f"No matching *{image_suffix} and *{mask_suffix} pairs found in {data_dir}"
        )

    return case_ids


def split_case_ids(case_ids, train_ratio, val_ratio, test_ratio, seed=42):
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("The sum of train, val, and test ratios must be greater than 0")

    train_ratio = train_ratio / total_ratio
    val_ratio = val_ratio / total_ratio
    test_ratio = test_ratio / total_ratio

    shuffled = list(case_ids)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    train_ids = shuffled[:train_count]
    val_ids = shuffled[train_count : train_count + val_count]
    test_ids = shuffled[train_count + val_count :]

    if test_count != len(test_ids):
        raise RuntimeError("Unexpected split size mismatch")

    return train_ids, val_ids, test_ids


def write_case_ids(case_ids, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(case_ids) + "\n", encoding="utf-8")
