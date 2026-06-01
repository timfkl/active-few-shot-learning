import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import argparse

from config import DATA_DIR, IMAGE_SUFFIX, MASK_SUFFIX, RESIZE_DIMS, RESIZED_DATA_DIR


def resize_3d_image_mask(image, mask, dimensions=RESIZE_DIMS, interpolation_order=1):
    """Resize a 3D image and its corresponding mask to the same target shape.

    Args:
        image: 3D image array with shape (x, y, depth).
        mask: 3D mask array with shape (x, y, depth).
        dimensions: Target output shape, default (256, 256, 32).
        interpolation_order: skimage interpolation order. The same
            resize grid and interpolation order are used for image and mask.

    Returns:
        resized_image, resized_mask
    """
    import numpy as np
    from skimage.transform import resize

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
    input_dir=DATA_DIR,
    output_dir=RESIZED_DATA_DIR,
    dimensions=RESIZE_DIMS,
    image_suffix=IMAGE_SUFFIX,
    mask_suffix=MASK_SUFFIX,
):
    import nibabel as nib
    import numpy as np
    from tqdm import tqdm

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
    case_ids = sorted(image_paths.keys() & mask_paths.keys())

    if not case_ids:
        raise FileNotFoundError(f"No matching image/mask pairs found in {input_dir}")

    for case_id in tqdm(case_ids, desc="Resizing dataset"):
        image_nii = nib.load(image_paths[case_id])
        mask_nii = nib.load(mask_paths[case_id])

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

        nib.save(resized_image_nii, output_dir / image_paths[case_id].name)
        nib.save(resized_mask_nii, output_dir / mask_paths[case_id].name)

    print(f"Saved resized dataset to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Resize paired 3D NIfTI images and masks.")
    parser.add_argument("--input-dir", default=DATA_DIR, help="Folder containing *_img.nii and *_mask.nii files.")
    parser.add_argument("--output-dir", default=RESIZED_DATA_DIR, help="Folder to save resized files.")
    parser.add_argument("--x", type=int, default=RESIZE_DIMS[0], help="Target x dimension.")
    parser.add_argument("--y", type=int, default=RESIZE_DIMS[1], help="Target y dimension.")
    parser.add_argument("--depth", type=int, default=RESIZE_DIMS[2], help="Target depth dimension.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    resize_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dimensions=(args.x, args.y, args.depth),
    )
