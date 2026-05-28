from pathlib import Path

try:
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
    DataLoader = None

    class Dataset:
        """Fallback so this file can be imported before PyTorch is installed."""


class NiiPatchDataset(Dataset):
    """Dataset that returns fixed-size 3D NIfTI image/mask patches.

    Returned item:
        image: Tensor [1, x, y, depth]
        mask: Tensor [x, y, depth]
        filename: case name
        image_path: source image path
        mask_path: source mask path
    """

    def __init__(
        self,
        x,
        y,
        depth,
        filename=None,
        image_path="data",
        mask_path="data",
        image_suffix="_img.nii",
        mask_suffix="_mask.nii",
        normalize=True,
        crop_mode="center",
        transform=None,
    ):
        self.patch_size = (int(x), int(y), int(depth))
        self.filename = filename
        self.image_path = Path(image_path)
        self.mask_path = Path(mask_path)
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix
        self.normalize = normalize
        self.crop_mode = crop_mode
        self.transform = transform

        if any(size <= 0 for size in self.patch_size):
            raise ValueError("x, y, and depth must be positive integers")
        if self.crop_mode not in {"center", "random"}:
            raise ValueError("crop_mode must be 'center' or 'random'")

        self.samples = self._build_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        self._check_dependencies()
        import nibabel as nib
        import numpy as np
        import torch

        sample = self.samples[index]
        image = nib.load(sample["image_path"]).get_fdata(dtype=np.float32)
        mask = np.asanyarray(nib.load(sample["mask_path"]).dataobj).astype(np.int64)

        if image.shape != mask.shape:
            raise ValueError(
                f"Image and mask shapes do not match for {sample['filename']}: "
                f"{image.shape} != {mask.shape}"
            )
        if image.ndim != 3:
            raise ValueError(f"Expected a 3D NIfTI volume, got shape {image.shape}")

        image, mask = self._pad_to_patch_size(image, mask)
        image, mask = self._crop_to_patch_size(image, mask)

        if self.normalize:
            image = self._zscore_normalize(image)

        item = {
            "image": torch.from_numpy(np.ascontiguousarray(image[None, ...])).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)).long(),
            "filename": sample["filename"],
            "image_path": str(sample["image_path"]),
            "mask_path": str(sample["mask_path"]),
        }

        if self.transform is not None:
            item = self.transform(item)

        return item

    def _build_samples(self):
        if self.image_path.is_file() and self.mask_path.is_file():
            filename = self.filename or self._case_id_from_image(self.image_path)
            return [
                {
                    "filename": filename,
                    "image_path": self.image_path,
                    "mask_path": self.mask_path,
                }
            ]

        if not self.image_path.exists():
            raise FileNotFoundError(f"Image path not found: {self.image_path}")
        if not self.mask_path.exists():
            raise FileNotFoundError(f"Mask path not found: {self.mask_path}")

        image_files = {
            self._case_id_from_image(path): path
            for path in self.image_path.glob(f"*{self.image_suffix}")
        }
        mask_files = {
            self._case_id_from_mask(path): path
            for path in self.mask_path.glob(f"*{self.mask_suffix}")
        }

        case_ids = self._resolve_filenames(image_files, mask_files)
        samples = [
            {
                "filename": case_id,
                "image_path": image_files[case_id],
                "mask_path": mask_files[case_id],
            }
            for case_id in case_ids
        ]

        if not samples:
            raise FileNotFoundError("No matching NIfTI image/mask pairs were found")

        return samples

    def _resolve_filenames(self, image_files, mask_files):
        if self.filename is None:
            return sorted(image_files.keys() & mask_files.keys())

        filenames = self._read_filename_argument(self.filename)
        missing = [
            name
            for name in filenames
            if name not in image_files or name not in mask_files
        ]
        if missing:
            raise FileNotFoundError(f"Missing image or mask for: {missing[:10]}")

        return filenames

    @staticmethod
    def _read_filename_argument(filename):
        if isinstance(filename, (list, tuple, set)):
            return [str(name).strip() for name in filename]

        filename = Path(filename) if not isinstance(filename, Path) else filename
        if filename.exists() and filename.is_file():
            with filename.open("r", encoding="utf-8") as file:
                return [line.strip() for line in file if line.strip()]

        return [str(filename).strip()]

    def _case_id_from_image(self, path):
        name = Path(path).name
        return name[: -len(self.image_suffix)] if name.endswith(self.image_suffix) else Path(path).stem

    def _case_id_from_mask(self, path):
        name = Path(path).name
        return name[: -len(self.mask_suffix)] if name.endswith(self.mask_suffix) else Path(path).stem

    def _pad_to_patch_size(self, image, mask):
        import numpy as np

        pad_width = []
        for current_size, target_size in zip(image.shape, self.patch_size):
            total_pad = max(target_size - current_size, 0)
            before = total_pad // 2
            after = total_pad - before
            pad_width.append((before, after))

        if any(before or after for before, after in pad_width):
            image = np.pad(image, pad_width, mode="constant", constant_values=0)
            mask = np.pad(mask, pad_width, mode="constant", constant_values=0)

        return image, mask

    def _crop_to_patch_size(self, image, mask):
        import numpy as np

        starts = []
        for current_size, target_size in zip(image.shape, self.patch_size):
            max_start = current_size - target_size
            if self.crop_mode == "random" and max_start > 0:
                starts.append(np.random.randint(0, max_start + 1))
            else:
                starts.append(max_start // 2)

        slices = tuple(
            slice(start, start + target_size)
            for start, target_size in zip(starts, self.patch_size)
        )
        return image[slices], mask[slices]

    @staticmethod
    def _zscore_normalize(image):
        import numpy as np

        image = image.astype(np.float32, copy=False)
        std = image.std()
        if std == 0:
            return image - image.mean()
        return (image - image.mean()) / (std + 1e-8)

    @staticmethod
    def _check_dependencies():
        missing = []
        for package in ("nibabel", "numpy", "torch"):
            try:
                __import__(package)
            except ModuleNotFoundError:
                missing.append(package)

        if missing:
            raise ModuleNotFoundError(
                "Missing required package(s): "
                + ", ".join(missing)
                + ". Install them with: pip install nibabel numpy torch"
            )


def build_dataloader(
    x,
    y,
    depth,
    filename=None,
    image_path="data",
    mask_path="data",
    batch_size=1,
    shuffle=True,
    num_workers=0,
    normalize=True,
    crop_mode="center",
    transform=None,
):
    """Create a PyTorch DataLoader yielding batches of shape [B, 1, x, y, depth]."""

    if DataLoader is None:
        raise ModuleNotFoundError("Missing required package: torch. Install it with: pip install torch")

    dataset = NiiPatchDataset(
        x=x,
        y=y,
        depth=depth,
        filename=filename,
        image_path=image_path,
        mask_path=mask_path,
        normalize=normalize,
        crop_mode=crop_mode,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    dataset = NiiPatchDataset(128, 128, 64, image_path="data", mask_path="data")
    print(f"Number of samples: {len(dataset)}")
    print(dataset.samples[0])
