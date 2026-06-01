from pathlib import Path

import numpy as np
import torch


class Nifti3dDataset:
    def __init__(
        self,
        data_dir="data-resize",
        split_file=None,
        image_suffix="_img.nii",
        mask_suffix="_mask.nii",
        normalize=True,
    ):
        self.data_dir = Path(data_dir)
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix
        self.normalize = normalize
        self.samples = self._find_pairs(split_file)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        try:
            import nibabel as nib
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "nibabel is required to load NIfTI files. Install it with: "
                "python -m pip install nibabel"
            ) from exc

        sample = self.samples[index]

        image = nib.load(sample["image_path"]).get_fdata(dtype=np.float32)
        mask = np.asanyarray(nib.load(sample["mask_path"]).dataobj).astype(np.int64)

        if image.shape != mask.shape:
            raise ValueError(f"Image and mask shapes do not match for {sample['case_id']}")

        if self.normalize:
            std = image.std()
            if std == 0:
                image = image - image.mean()
            else:
                image = (image - image.mean()) / (std + 1e-8)

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

        if split_file:
            path = Path(split_file)
            if path.exists():
                case_ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
            else:
                case_ids = [str(split_file).strip()]
        else:
            case_ids = sorted(image_paths.keys() & mask_paths.keys())
        pairs = []

        for case_id in case_ids:
            if case_id not in image_paths or case_id not in mask_paths:
                raise FileNotFoundError(f"Missing image or mask for case {case_id}")
            pairs.append(
                {
                    "case_id": case_id,
                    "image_path": image_paths[case_id],
                    "mask_path": mask_paths[case_id],
                }
            )

        return pairs

def make_batch(samples):
    return {
        "image": torch.stack([sample["image"] for sample in samples]),
        "mask": torch.stack([sample["mask"] for sample in samples]),
        "case_id": [sample["case_id"] for sample in samples],
    }


def build_3d_dataloader(data_dir="data-resize", split_file=None, batch_size=1, shuffle=True):
    dataset = Nifti3dDataset(data_dir=data_dir, split_file=split_file)

    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        samples = [dataset[int(index)] for index in batch_indices]
        yield make_batch(samples)


def build_episode_loader(data_dir="data-resize", split_file=None, n_support=1, n_query=1, episodes=100):
    dataset = Nifti3dDataset(data_dir=data_dir, split_file=split_file)

    for _ in range(episodes):
        indices = np.random.choice(len(dataset), size=n_support + n_query, replace=False)
        support = [dataset[int(index)] for index in indices[:n_support]]
        query = [dataset[int(index)] for index in indices[n_support:]]

        yield {
            "support": make_batch(support),
            "query": make_batch(query),
        }


build_dataloader = build_3d_dataloader
