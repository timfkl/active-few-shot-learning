import argparse
import random
from pathlib import Path


def find_case_ids(data_dir, image_suffix="_img.nii", mask_suffix="_mask.nii"):
    """
    Scans a directory for matching image and mask pairs and extracts their case IDs.
    
    Args:
        data_dir (str or Path): Directory to scan.
        image_suffix (str): File suffix for images.
        mask_suffix (str): File suffix for masks.
        
    Returns:
        list: A sorted list of case ID strings (e.g., ['case_000', 'case_001']).
    """
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

    # Set intersection ensures we only return IDs that have BOTH an image and a mask
    case_ids = sorted(image_ids & mask_ids)
    if not case_ids:
        raise FileNotFoundError(
            f"No matching *{image_suffix} and *{mask_suffix} pairs found in {data_dir}"
        )

    return case_ids


def split_case_ids(case_ids, train_ratio, val_ratio, test_ratio, seed=42):
    """
    Randomly partitions a list of case IDs into train, validation, and test sets.
    
    Args:
        case_ids (list): List of case ID strings.
        train_ratio (float): Relative proportion for the training set.
        val_ratio (float): Relative proportion for the validation set.
        test_ratio (float): Relative proportion for the testing set.
        seed (int): Random seed for reproducibility across runs.
        
    Returns:
        tuple: (train_ids, val_ids, test_ids) as lists.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("The sum of train, val, and test ratios must be greater than 0")

    # Normalize ratios in case they don't exactly sum up to 1.0
    train_ratio = train_ratio / total_ratio
    val_ratio = val_ratio / total_ratio
    test_ratio = test_ratio / total_ratio

    shuffled = list(case_ids)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    # Slice the shuffled list into the 3 sets
    train_ids = shuffled[:train_count]
    val_ids = shuffled[train_count : train_count + val_count]
    test_ids = shuffled[train_count + val_count :]

    if test_count != len(test_ids):
        raise RuntimeError("Unexpected split size mismatch")

    return train_ids, val_ids, test_ids


def write_case_ids(case_ids, output_path):
    """
    Writes a list of case IDs to a plain text file, one ID per line.
    
    Args:
        case_ids (list): List of case IDs.
        output_path (str or Path): Target path for the output text file (e.g., 'train.txt').
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(case_ids) + "\n", encoding="utf-8")
