
"""Shared image preprocessing for DATA7703 malaria classification."""

import numpy as np
from PIL import Image


def load_and_pad(path: str, size: int = 128) -> np.ndarray:
    """
    Load an RGB image, resize with preserved aspect ratio,
    and centre it on a black square canvas.

    Parameters
    ----------
    path : str
        Path to the source image.
    size : int
        Target width and height in pixels.

    Returns
    -------
    np.ndarray
        Float32 RGB array with shape (size, size, 3)
        and pixel values in [0, 1].
    """
    if not isinstance(size, (int, np.integer)) or size <= 0:
        raise ValueError("size must be a positive integer")

    with Image.open(path) as source:
        img = source.convert("RGB")

    width, height = img.size
    scale = size / max(width, height)

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    img = img.resize(
        (new_width, new_height),
        Image.Resampling.BILINEAR
    )

    canvas = Image.new("RGB", (size, size), (0, 0, 0))

    x = (size - new_width) // 2
    y = (size - new_height) // 2

    canvas.paste(img, (x, y))

    return np.asarray(canvas, dtype=np.float32) / 255.0


def normalize_image(image: np.ndarray, method: str = "zero_one") -> np.ndarray:
    """
    Normalise an RGB image produced by load_and_pad().

    Parameters
    ----------
    image : np.ndarray
        Float32 RGB array with values in [0, 1].
    method : str
        "zero_one": preserve [0, 1] scaling.
        "imagenet": apply ImageNet channel mean/std.
        "none": return the 0-255 numeric range.

    Returns
    -------
    np.ndarray
        Float32 RGB array with unchanged dimensions.
    """
    image = np.asarray(image, dtype=np.float32)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3)")

    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Input pixel values must be finite and within [0, 1]")

    if method == "zero_one":
        return image.copy()

    if method == "imagenet":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (image - mean) / std

    if method == "none":
        return image * 255.0

    raise ValueError("Unknown normalisation method")


def scale_pixels(images: np.ndarray) -> np.ndarray:
    """
    Scale raw image arrays from [0, 255] to [0, 1].

    Use this for raw pixel arrays, not arrays already
    returned by load_and_pad().
    """
    images = np.asarray(images, dtype=np.float32)

    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 255:
        raise ValueError("Expected finite pixel values in [0, 255]")

    return images / 255.0
