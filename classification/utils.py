import numpy as np
from typing import Tuple


def center_crop(image: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
    """
    Crop the center of the image to the specified crop size.

    :param image: Input image (numpy array).
    :param crop_size: Desired size of the crop (height, width).
    :return: Cropped image (numpy array).
    """
    h, w = image.shape
    crop_h, crop_w = crop_size

    # Calculate cropping box
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2

    return image[start_h:start_h + crop_h, start_w:start_w + crop_w]