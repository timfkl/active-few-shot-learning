import nibabel as nib
import numpy as np

from pathlib import Path
from tqdm import tqdm
from skimage.transform import resize

def resize_3d_image_mask(image, mask, dimensions=(256, 256, 32), interpolation_order=1):
    """Resize a 3D image and its corresponding mask to the same target shape.

    Args:
        image: 3D image array with shape (x, y, depth).
        mask: 3D mask array with shape (x, y, depth).
        dimensions: Target output shape.
        interpolation_order: skimage interpolation order. The same
            resize grid and interpolation order are used for image and mask.

    Returns:
        resized_image, resized_mask
    """
    image = np.asarray(image)
    mask = np.asarray(mask)
    target_shape = tuple(int(size) for size in dimensions)

    if image.ndim != 3 or mask.ndim != 3:
        raise ValueError(f"Expected 3D image and mask, got {image.shape} and {mask.shape}")
    if image.shape != mask.shape:
        raise ValueError(f"Image and mask shapes must match, got {image.shape} and {mask.shape}")
    if len(target_shape) != 3 or any(size <= 0 for size in target_shape):
        raise ValueError("dimensions must contain three positive integers")

    # Resize the continuous image values (usually with linear interpolation, order=1)
    resized_image = resize(
        image,
        target_shape,
        order=interpolation_order,
        preserve_range=True, # Keeps values in their original range instead of 0-1
        anti_aliasing=True,  # Smooths image to prevent artifacts when downsampling
    )
    # Resize the categorical mask using the same grid
    resized_mask = resize(
        mask,
        target_shape,
        order=0,  # Nearest-neighbor is required for categorical data
        preserve_range=True,
        anti_aliasing=False,  # No smoothing for discrete labels
    )

    # Ensure the mask remains discrete integer class labels, not floats, after resizing
    if np.issubdtype(mask.dtype, np.integer):
        resized_mask = np.rint(resized_mask).astype(mask.dtype)

    return resized_image, resized_mask


def resize_dataset(
    input_dir="data",
    output_dir="data-resize",
    dimensions=(256, 256, 32),
    image_suffix="_img.nii",
    mask_suffix="_mask.nii",
):
    """
    Resizes all matching NIfTI image and mask pairs in a directory.

    Args:
        input_dir (str or Path): Directory containing original NIfTI files.
        output_dir (str or Path): Directory to save the resized NIfTI files.
        dimensions (tuple): Target shape (x, y, depth) for the volumes.
        image_suffix (str): Suffix identifying the image files.
        mask_suffix (str): Suffix identifying the mask files.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    # Create output directory if it doesn't already exist
    output_dir.mkdir(exist_ok=True)

    # Discover all image and mask paths, mapped by their shared base name
    image_paths = {
        path.name[: -len(image_suffix)]: path
        for path in input_dir.glob(f"*{image_suffix}")
    }
    mask_paths = {
        path.name[: -len(mask_suffix)]: path
        for path in input_dir.glob(f"*{mask_suffix}")
    }
    
    # Intersect keys to ensure we only process complete pairs
    pair_keys = sorted(image_paths.keys() & mask_paths.keys())

    if not pair_keys:
        raise FileNotFoundError(f"No matching image/mask pairs found in {input_dir}")

    for pair_key in tqdm(pair_keys, desc="Resizing dataset"):
        image_nii = nib.load(image_paths[pair_key])
        mask_nii = nib.load(mask_paths[pair_key])

        image = image_nii.get_fdata(dtype=np.float32)
        mask = np.asanyarray(mask_nii.dataobj)
        resized_image, resized_mask = resize_3d_image_mask(image, mask, dimensions)

        # Repackage the resized numpy arrays back into NIfTI format 
        # keeping the original affine transformation and header metadata
        resized_image_nii = nib.Nifti1Image(
            resized_image.astype(np.float32),
            image_nii.affine,
            header=image_nii.header,
        )
        resized_mask_nii = nib.Nifti1Image(
            resized_mask.astype(mask.dtype),
            mask_nii.affine,
            header=mask_nii.header,
        )

        nib.save(resized_image_nii, output_dir / image_paths[pair_key].name)
        nib.save(resized_mask_nii, output_dir / mask_paths[pair_key].name)

    print(f"Saved resized dataset to {output_dir}")