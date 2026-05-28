def resize_image_mask(image, mask, dimensions=(256, 256, 32), interpolation_order=1):
    """Resize a 3D image and its corresponding mask to the same target shape.

    Args:
        image: 3D image array with shape (x, y, depth).
        mask: 3D mask array with shape (x, y, depth).
        dimensions: Target output shape, default (256, 256, 32).
        interpolation_order: SciPy spline interpolation order. Use the same
            value for image and mask to keep them aligned after resizing.

    Returns:
        resized_image, resized_mask
    """

    import numpy as np
    from scipy.ndimage import zoom

    image = np.asarray(image)
    mask = np.asarray(mask)
    target_shape = tuple(int(size) for size in dimensions)

    if image.ndim != 3 or mask.ndim != 3:
        raise ValueError(f"Expected 3D image and mask, got {image.shape} and {mask.shape}")
    if image.shape != mask.shape:
        raise ValueError(f"Image and mask shapes must match, got {image.shape} and {mask.shape}")
    if len(target_shape) != 3 or any(size <= 0 for size in target_shape):
        raise ValueError("dimensions must contain three positive integers")

    zoom_factors = tuple(
        target_size / current_size
        for target_size, current_size in zip(target_shape, image.shape)
    )

    resized_image = zoom(image, zoom_factors, order=interpolation_order)
    resized_mask = zoom(mask, zoom_factors, order=interpolation_order)

    if np.issubdtype(mask.dtype, np.integer):
        resized_mask = np.rint(resized_mask).astype(mask.dtype)

    return resized_image, resized_mask
