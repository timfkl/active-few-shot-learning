from pathlib import Path

import nibabel as nib
import numpy as np


def find_image_mask_pairs(images_dir, masks_dir, image_suffix="_img.nii", mask_suffix="_mask.nii"):
    """
    Finds matching 3D image and mask files from specified directories.

    Args:
        images_dir (str or Path): Directory containing the NIfTI images.
        masks_dir (str or Path): Directory containing the corresponding NIfTI masks.
        image_suffix (str): The suffix identifying an image file.
        mask_suffix (str): The suffix identifying a mask file.

    Returns:
        list of tuples: A list where each element is a tuple containing 
                        (path_to_image, path_to_mask).
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    # Pair files by the shared part of the filename before the suffix.
    # Example: case_001_img.nii matches case_001_mask.nii.
    image_paths = {
        path.name[: -len(image_suffix)]: path
        for path in images_dir.glob(f"*{image_suffix}")
    }
    mask_paths = {
        path.name[: -len(mask_suffix)]: path
        for path in masks_dir.glob(f"*{mask_suffix}")
    }

    # Intersect keys to guarantee we only return complete pairs
    pairs = [
        (image_paths[key], mask_paths[key])
        for key in sorted(image_paths.keys() & mask_paths.keys())
    ]
    
    if not pairs:
        raise FileNotFoundError(f"No image/mask pairs found in {images_dir} and {masks_dir}")

    return pairs


def load_nifti_pair(image_path, mask_path, normalize=True):
    """
    Loads a single NIfTI image and its corresponding mask from disk.

    Args:
        image_path (Path): Path to the NIfTI image.
        mask_path (Path): Path to the NIfTI mask.
        normalize (bool): Whether to scale image intensities to the range [0, 1].

    Returns:
        tuple: (image_array, mask_array) as numpy arrays.
               The image will have an added channel dimension: [1, H, W, D].
               The mask will remain: [H, W, D].
    """
    image = nib.load(image_path).get_fdata(dtype=np.float32)
    mask = np.asanyarray(nib.load(mask_path).dataobj).astype(np.int64)

    if image.shape != mask.shape:
        raise ValueError(f"Image and mask shapes do not match: {image_path}, {mask_path}")

    # Scale image intensities to [0, 1].
    if normalize:
        min_value = image.min()
        max_value = image.max()
        if max_value > min_value:
            image = (image - min_value) / (max_value - min_value)
        else:
            image = image - min_value

    # Deep learning models typically expect a channel dimension for the image input.
    # This transforms shapes from (H, W, D) -> (1, H, W, D)
    image = image[None, ...].astype(np.float32)
    
    # Removed the channel dimension from the mask as class IDs usually don't need one.

    return image, mask


def build_nifti_batch_generator(
    batch_size=1,
    images_dir="data-resize",
    masks_dir="data-resize",
    normalize=True,
    image_suffix="_img.nii",
    mask_suffix="_mask.nii",
    task_number=1,
):
    """
    Simple infinite generator for yielding 3D segmentation batches.
    Uses random sampling with replacement to generate batches.

    Args:
        batch_size (int): Number of samples per batch.
        images_dir (str): Directory containing the images.
        masks_dir (str): Directory containing the masks.
        normalize (bool): If True, normalizes the images.
        image_suffix (str): File suffix for images.
        mask_suffix (str): File suffix for masks.
        task_number (int, optional): The specific task (class ID) to generate masks for.
                                     Masks will be binarized for this specific task.

    Yields:
        tuple: (batch_images, batch_masks) as numpy arrays.
               Image shapes will be (B, 1, H, W, D).
               Mask shapes will be (B, H, W, D).
    """
    pairs = find_image_mask_pairs(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_suffix=image_suffix,
        mask_suffix=mask_suffix,
    )
    
    # Load the first pair temporarily to determine the exact array shapes
    first_image, first_mask = load_nifti_pair(*pairs[0], normalize=normalize)

    # Infinite loop to keep generating batches for training
    while True:
        # Pre-allocate memory for the batch to speed up array operations
        batch_images = np.zeros((batch_size, *first_image.shape), dtype=first_image.dtype)
        batch_masks = np.zeros((batch_size, *first_mask.shape), dtype=first_mask.dtype)

        for batch_index in range(batch_size):
            # Random sampling with replacement for each slot in the batch
            index = np.random.randint(len(pairs))
            image_path, mask_path = pairs[index]
            
            image, mask = load_nifti_pair(
                image_path,
                mask_path,
                normalize=normalize,
            )
            
            if task_number is not None:
                mask = (mask == task_number).astype(mask.dtype)

            batch_images[batch_index] = image
            batch_masks[batch_index] = mask

        yield batch_images, batch_masks
