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


def main():
    parser = argparse.ArgumentParser(
        description="Split paired NIfTI image/mask case IDs into train, val, and test files."
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing *_img.nii and *_mask.nii files.")
    parser.add_argument("--output-dir", default=".", help="Directory where split txt files are saved.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits.")
    parser.add_argument("--image-suffix", default="_img.nii", help="Image filename suffix.")
    parser.add_argument("--mask-suffix", default="_mask.nii", help="Mask filename suffix.")
    parser.add_argument("--dry-run", action="store_true", help="Print split sizes without writing files.")
    args = parser.parse_args()

    case_ids = find_case_ids(
        args.data_dir,
        image_suffix=args.image_suffix,
        mask_suffix=args.mask_suffix,
    )
    train_ids, val_ids, test_ids = split_case_ids(
        case_ids,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(f"Total cases: {len(case_ids)}")
    print(f"Train: {len(train_ids)}")
    print(f"Val:   {len(val_ids)}")
    print(f"Test:  {len(test_ids)}")

    if args.dry_run:
        return

    output_dir = Path(args.output_dir)
    write_case_ids(train_ids, output_dir / "train.txt")
    write_case_ids(val_ids, output_dir / "val.txt")
    write_case_ids(test_ids, output_dir / "test.txt")
    print(f"Saved splits to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
