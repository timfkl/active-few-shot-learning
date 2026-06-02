import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


class SegmentationDataset:
    """
    Pure Python Dataset for 3D medical image segmentation.
    Reads case IDs from a split file and loads corresponding images and masks.
    """
    def __init__(self, data_dir, split_file, image_suffix="_img.nii", label_suffix="_mask.nii", normalize=True):
        self.data_dir = Path(data_dir)
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix
        self.normalize = normalize

        with open(split_file, 'r') as f:
            case_ids = [line.strip() for line in f if line.strip()]

        self.samples = []
        for case_id in case_ids:
            image_path = self.data_dir / f"{case_id}{self.image_suffix}"
            label_path = self.data_dir / f"{case_id}{self.label_suffix}"
            if image_path.exists() and label_path.exists():
                self.samples.append({
                    "image_path": image_path,
                    "label_path": label_path,
                    "case_id": case_id
                })
        
        if not self.samples:
            raise FileNotFoundError(f"No valid samples found for split file {split_file} in {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = nib.load(sample["image_path"]).get_fdata(dtype=np.float32)
        label = np.asanyarray(nib.load(sample["label_path"]).dataobj).astype(np.int64)

        if image.shape != label.shape:
            raise ValueError(f"Image and label shapes do not match: {sample['image_path']}, {sample['label_path']}")

        if self.normalize:
            std = image.std()
            if std > 1e-8:
                image = (image - image.mean()) / std
            else:
                image = image - image.mean()

        # Add channel dimension to image: [1, H, W, D]
        image = image[None, ...].astype(np.float32)

        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(label),
            "case_id": sample["case_id"]
        }


class CustomDataLoader:
    """
    Pure Python DataLoader to batch and shuffle dataset elements.
    Does not inherit from torch.utils.data.DataLoader.
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        # Accept and ignore unused kwargs (like num_workers) to keep existing tests from failing
        self.kwargs = kwargs

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        # Iterate through the indices to create true, non-overlapping batches (proper epoch)
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            
            batch_images = []
            batch_masks = []
            batch_case_ids = []
            
            for idx in batch_indices:
                item = self.dataset[idx]
                batch_images.append(item["image"])
                batch_masks.append(item["mask"])
                batch_case_ids.append(item["case_id"])
                
            yield {
                "image": torch.stack(batch_images),
                "mask": torch.stack(batch_masks),
                "case_id": batch_case_ids
            }


def build_dataloader(data_dir, split_file, batch_size, shuffle=True, **kwargs):
    dataset = SegmentationDataset(data_dir=data_dir, split_file=split_file)
    return CustomDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
