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

    resized_image = resize(
        image,
        target_shape,
        order=interpolation_order,
        preserve_range=True,
        anti_aliasing=True,
    )
    resized_mask = resize(
        mask,
        target_shape,
        order=interpolation_order,
        preserve_range=True,
        anti_aliasing=True,
    )

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

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    image_paths = {
        path.name[: -len(image_suffix)]: path
        for path in input_dir.glob(f"*{image_suffix}")
    }
    mask_paths = {
        path.name[: -len(mask_suffix)]: path
        for path in input_dir.glob(f"*{mask_suffix}")
    }
    pair_keys = sorted(image_paths.keys() & mask_paths.keys())

    if not pair_keys:
        raise FileNotFoundError(f"No matching image/mask pairs found in {input_dir}")

    for pair_key in tqdm(pair_keys, desc="Resizing dataset"):
        image_nii = nib.load(image_paths[pair_key])
        mask_nii = nib.load(mask_paths[pair_key])

        image = image_nii.get_fdata(dtype=np.float32)
        mask = np.asanyarray(mask_nii.dataobj)
        resized_image, resized_mask = resize_3d_image_mask(image, mask, dimensions)

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
