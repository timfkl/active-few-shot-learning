import os
from pathlib import Path
import numpy as np
import logging

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "torch is required for this module. Install it with: pip install torch"
    ) from exc

from config import (
    RESIZED_DATA_DIR,
    IMAGE_SUFFIX,
    MASK_SUFFIX,
    NORMALIZE,
    BATCH_SIZE,
    NUM_WORKERS,
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


class NiftiDataset(Dataset):
    """
    PyTorch Dataset for loading 3D NIfTI volumes that have been pre-processed
    (e.g., resized to a uniform shape).

    Reads a list of case IDs from a text file and loads the corresponding
    image/mask pairs from the data directory.
    """
    def __init__(
        self,
        data_dir,
        split_file,
        image_suffix=IMAGE_SUFFIX,
        mask_suffix=MASK_SUFFIX,
        normalize=NORMALIZE,
    ):
        self.data_dir = Path(data_dir)
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix
        self.normalize = normalize
        self.samples = self._find_pairs(split_file)

        if not self.samples:
            raise FileNotFoundError(
                f"No valid samples found for split file '{split_file}' in directory '{data_dir}'"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            import nibabel as nib
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "nibabel is required to load NIfTI files. Install it with: "
                "python -m pip install nibabel"
            ) from exc

        sample = self.samples[idx]

        image = nib.load(sample["image_path"]).get_fdata(dtype=np.float32)
        mask = np.asanyarray(nib.load(sample["mask_path"]).dataobj).astype(np.int64)

        if image.shape != mask.shape:
            raise ValueError(f"Image and mask shapes do not match for {sample['case_id']}")

        if self.normalize:
            std = image.std()
            if std > 1e-8:
                image = (image - image.mean()) / std
            else:
                image = image - image.mean()

        # Add channel dimension: [C, H, W, D]
        # The model.py script expects permutation to [C, D, H, W] later
        return {
            "image": torch.from_numpy(image[None, ...].copy()).float(),
            "mask": torch.from_numpy(mask.copy()).long(),
            "case_id": sample["case_id"],
        }

    def _find_pairs(self, split_file):
        image_paths = {
            path.name[: -len(self.image_suffix)]: path
            for path in self.data_dir.glob(f"*{self.image_suffix}")
        }
        mask_paths = {
            path.name[: -len(self.mask_suffix)]: path
            for path in self.data_dir.glob(f"*{self.mask_suffix}")
        }

        path = Path(split_file)
        if not path.is_absolute() and not path.exists():
            path = self.data_dir / path

        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        case_ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]

        pairs = []
        missing_cases = []
        for case_id in case_ids:
            if case_id in image_paths and case_id in mask_paths:
                pairs.append(
                    {
                        "case_id": case_id,
                        "image_path": image_paths[case_id],
                        "mask_path": mask_paths[case_id],
                    }
                )
            else:
                missing_cases.append(case_id)

        if missing_cases:
            logging.warning(f"Missing image/mask pairs for {len(missing_cases)} cases listed in {split_file}. Examples: {missing_cases[:5]}")

        return pairs


def nifti_collate_fn(samples):
    """
    Custom collate function to combine a list of samples from NiftiDataset
    into a single batch dictionary.
    """
    if not samples:
        return {}
    return {
        "image": torch.stack([sample["image"] for sample in samples]),
        "mask": torch.stack([sample["mask"] for sample in samples]),
        "case_id": [sample["case_id"] for sample in samples],
    }


def build_dataloader(
    split_file,
    data_dir=RESIZED_DATA_DIR,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    **kwargs,
):
    """
    Builds a standard, high-performance PyTorch DataLoader for 3D NIfTI volumes.
    This is the recommended, general-purpose function for loading data.

    Args:
        data_dir (str): Path to the directory containing pre-processed NIfTI files.
        split_file (str): Path to the .txt file containing case IDs for this split.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data at every epoch.
        num_workers (int): How many subprocesses to use for data loading.
                           0 means that the data will be loaded in the main process.
        **kwargs: Additional arguments to pass to the NiftiDataset constructor.
    """
    dataset = NiftiDataset(data_dir=data_dir, split_file=split_file, **kwargs)

    # Pin memory only if a GPU is available for performance optimization
    pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=nifti_collate_fn,
        pin_memory=pin_memory,
    )
