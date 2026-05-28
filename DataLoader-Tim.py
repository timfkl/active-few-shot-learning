import os
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# For NIfTI file loading
import nibabel as nib
from torch.utils.data import Dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# --- Data Extraction ---
def extract_data(file_path='data.zip', extract_dir='.'):
    """Unzips the data.zip file into the specified directory."""
    if os.path.exists(file_path):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        logging.info(f"Extracted {file_path}")

class VolumeDataset(Dataset):
    """
    PyTorch Dataset for loading 3D NIfTI volumes (.nii or .nii.gz) and their corresponding masks.
    Each sample is a tuple (image, mask), both as torch tensors.
    
    WARNING for RL/Large Datasets:
    Dynamically loading NIfTI files from disk via `nibabel` on every fetch is extremely 
    CPU-intensive. If used in a Reinforcement Learning environment where episodes reset frequently, 
    this will bottleneck training. Future development should migrate this to an HDF5-based dataset.

    Assumes mask files have the same filename and are in a separate directory (mask_dir), or optionally with a suffix.
    Includes logging and optional visualization.
    """
    def __init__(self, data_dir, mask_dir=None, transform=None, mask_transform=None, img_suffix='_img', mask_suffix='_mask', verbose=False, show_sample=False, show_sample_idx=0, show_slice=None):
        """
        Args:
            data_dir (str): Path to the directory containing NIfTI image files.
            mask_dir (str, optional): Path to the directory containing NIfTI mask files. Defaults to data_dir if None.
            transform (callable, optional): Transform to be applied on the image.
            mask_transform (callable, optional): Transform to be applied on the mask.
            img_suffix (str): Identifier in the filename for images.
            mask_suffix (str): Identifier in the filename for masks.
            verbose (bool): If True, logs intermittent loading info.
            show_sample (bool): If True, displays a slice of a sample image and mask.
            show_sample_idx (int): Index of the sample to display.
            show_slice (int or None): Slice index to display (defaults to middle slice if None).
        """
        self.data_dir = data_dir
        self.mask_dir = mask_dir if mask_dir is not None else data_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.verbose = verbose
        # List all files that match the image suffix
        self.file_list = [f for f in os.listdir(data_dir) if (f.endswith('.nii') or f.endswith('.nii.gz')) and self.img_suffix in f]
        if self.verbose:
            logging.info(f"Found {len(self.file_list)} image files in {data_dir}")

        # Optionally show a sample image and mask
        if show_sample and len(self.file_list) > 0:
            idx = min(show_sample_idx, len(self.file_list) - 1)
            img, mask = self.__getitem__(idx)
            # Pick a slice to show (middle if not specified)
            d = img.shape[1]  # [C, D, H, W]
            slice_idx = show_slice if show_slice is not None else d // 2
            img_slice = img[0, slice_idx, :, :].cpu().numpy()
            mask_slice = mask[0, slice_idx, :, :].cpu().numpy()
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            axs[0].imshow(img_slice, cmap='gray')
            axs[0].set_title(f'Image slice {slice_idx}')
            axs[1].imshow(mask_slice, cmap='Reds', alpha=0.5)
            axs[1].set_title(f'Mask slice {slice_idx}')
            plt.tight_layout()
            plt.show()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Get the file name for this index
        file_name = self.file_list[idx]
        img_path = os.path.join(self.data_dir, file_name)

        # Determine mask filename
        mask_name = file_name.replace(self.img_suffix, self.mask_suffix)
        mask_path = os.path.join(self.mask_dir, mask_name)

        if self.verbose:
            logging.info(f"Loading image: {img_path}")
            logging.info(f"Loading mask: {mask_path}")

        # Load the 3D image and mask using nibabel
        img_nii = nib.load(img_path)
        mask_nii = nib.load(mask_path)
        img = img_nii.get_fdata()
        mask = mask_nii.get_fdata()

        # Ensure channel-first: [1, D, H, W]
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)
        if mask.ndim == 3:
            mask = np.expand_dims(mask, axis=0)

        # Convert to torch tensor (float32 for image, long/int for mask)
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask)
        # If mask is not integer, convert to long
        if not torch.is_floating_point(mask):
            mask = mask.long()
        else:
            mask = mask.float()

        # Apply any transforms
        if self.transform:
            img = self.transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return img, mask

# Example usage (to be used in your RL script or main training loop):
# dataset = VolumeDataset(data_dir='unzipped_data_folder')
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

# Note: You can add label loading, augmentation, or other logic as needed later.


# --- In-Memory 3D Dataset Class ---


class InMemoryVolumeDataset(Dataset):
    """
    Loads all 3D NIfTI images and masks into memory as tensors for fast access.
    Each sample is a tuple (image, mask).
    Use the classmethod from_splits() to get train, val, test datasets in one call.
    """
    def __init__(self, image_tensor, mask_tensor, transform=None, mask_transform=None):
        """
        Args:
            image_tensor (torch.Tensor): Tensor of shape [N, C, D, H, W].
            mask_tensor (torch.Tensor): Tensor of shape [N, C, D, H, W] (or [N, 1, D, H, W]).
            transform (callable, optional): Transform for images.
            mask_transform (callable, optional): Transform for masks.
        """
        self.image_tensor = image_tensor
        self.mask_tensor = mask_tensor
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return self.image_tensor.shape[0]

    def __getitem__(self, idx):
        img = self.image_tensor[idx]
        mask = self.mask_tensor[idx]
        if self.transform:
            img = self.transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return img, mask

    @classmethod
    def from_splits(cls, data_dir, mask_dir=None, val_ratio=0.1, test_ratio=0.1, transform=None, mask_transform=None, img_suffix='_img', mask_suffix='_mask', random_seed=42):
        """
        Loads all NIfTI image and mask files, splits into train/val/test, and returns three InMemoryVolumeDataset objects.
        Args:
            data_dir (str): Directory with NIfTI image files.
            mask_dir (str, optional): Directory with NIfTI mask files. Defaults to data_dir if None.
            val_ratio (float): Fraction for validation.
            test_ratio (float): Fraction for testing.
            transform (callable, optional): Transform for images.
            mask_transform (callable, optional): Transform for masks.
            img_suffix (str): Identifier in the filename for images.
            mask_suffix (str): Identifier in the filename for masks.
            random_seed (int): Seed for reproducibility.
        Returns:
            train_set, val_set, test_set: Three InMemoryVolumeDataset objects.
            
        Note: 
            Future development should replace this method with a pre-processing script 
            that chunks the data into an `.h5` file rather than holding lists of arrays in RAM.
        """
        mask_dir = mask_dir if mask_dir is not None else data_dir
        # List all image files
        file_list = [f for f in os.listdir(data_dir) if (f.endswith('.nii') or f.endswith('.nii.gz')) and img_suffix in f]
        file_list.sort()

        # Shuffle files for random split
        rng = np.random.RandomState(random_seed)
        rng.shuffle(file_list)

        n_total = len(file_list)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_val - n_test

        train_files = file_list[:n_train]
        val_files = file_list[n_train:n_train + n_val]
        test_files = file_list[n_train + n_val:]

        def load_vols(file_list):
            imgs = []
            masks = []
            for fname in file_list:
                img_path = os.path.join(data_dir, fname)
                # Determine mask filename
                mask_name = fname.replace(img_suffix, mask_suffix)
                mask_path = os.path.join(mask_dir, mask_name)

                img_nii = nib.load(img_path)
                mask_nii = nib.load(mask_path)
                img = img_nii.get_fdata()
                mask = mask_nii.get_fdata()
                if img.ndim == 3:
                    img = np.expand_dims(img, axis=0)
                if mask.ndim == 3:
                    mask = np.expand_dims(mask, axis=0)
                imgs.append(img)
                masks.append(mask)
            if imgs:
                img_tensor = torch.from_numpy(np.stack(imgs)).float()
                mask_tensor = torch.from_numpy(np.stack(masks))
                # If mask is not integer, convert to long
                if not torch.is_floating_point(mask_tensor):
                    mask_tensor = mask_tensor.long()
                else:
                    mask_tensor = mask_tensor.float()
                return img_tensor, mask_tensor
            else:
                return torch.empty((0, 1, 1, 1, 1)), torch.empty((0, 1, 1, 1, 1))

        train_imgs, train_masks = load_vols(train_files)
        val_imgs, val_masks = load_vols(val_files)
        test_imgs, test_masks = load_vols(test_files)

        train_set = cls(train_imgs, train_masks, transform=transform, mask_transform=mask_transform)
        val_set = cls(val_imgs, val_masks, transform=transform, mask_transform=mask_transform)
        test_set = cls(test_imgs, test_masks, transform=transform, mask_transform=mask_transform)
        return train_set, val_set, test_set


if __name__ == "__main__":
    # NOTE: Run dataset conversion or tests here in future development
    logging.info("DataLoader module loaded. Note: Native NIfTI loading is highly unoptimized for large RL training runs.")
    pass
