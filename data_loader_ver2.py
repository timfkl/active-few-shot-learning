from pathlib import Path

import nibabel as nib
import numpy as np


def find_image_label_pairs(images_dir, labels_dir, image_suffix="_img.nii", label_suffix="_mask.nii"):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    image_paths = {
        path.name[: -len(image_suffix)]: path
        for path in images_dir.glob(f"*{image_suffix}")
    }
    label_paths = {
        path.name[: -len(label_suffix)]: path
        for path in labels_dir.glob(f"*{label_suffix}")
    }

    pairs = [
        (image_paths[key], label_paths[key])
        for key in sorted(image_paths.keys() & label_paths.keys())
    ]
    if not pairs:
        raise FileNotFoundError(f"No image/label pairs found in {images_dir} and {labels_dir}")

    return pairs


def load_image_label(image_path, label_path, normalize=True):
    image = nib.load(image_path).get_fdata(dtype=np.float32)
    label = np.asanyarray(nib.load(label_path).dataobj).astype(np.int64)

    if image.shape != label.shape:
        raise ValueError(f"Image and label shapes do not match: {image_path}, {label_path}")

    if normalize:
        std = image.std()
        if std > 1e-8:
            image = (image - image.mean()) / std
        else:
            image = image - image.mean()

    image = image[None, ...].astype(np.float32)
    label = label[None, ...]

    return image, label


def my_3d_data_loader(
    batch_size=1,
    images_dir="data-resize/iamages",
    labels_dir="data-resize/labels",
    normalize=True,
    image_suffix="_img.nii",
    label_suffix="_mask.nii",
):
    pairs = find_image_label_pairs(
        images_dir=images_dir,
        labels_dir=labels_dir,
        image_suffix=image_suffix,
        label_suffix=label_suffix,
    )

    while True:
        batch_images = []
        batch_labels = []

        for _ in range(batch_size):
            index = np.random.randint(len(pairs))
            image_path, label_path = pairs[index]
            image, label = load_image_label(
                image_path,
                label_path,
                normalize=normalize,
            )

            batch_images.append(image)
            batch_labels.append(label)

        yield np.stack(batch_images), np.stack(batch_labels)
