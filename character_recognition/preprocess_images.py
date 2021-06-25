import os
import sys
from typing import Union

import numpy as np
from PIL import Image

DEFAULT_INPUT_DIRECTORY: str = "data/dataset"
OUTPUT_HEIGHT: int = 64
"""
The height every output image will have.
"""
OUTPUT_WIDTH: int = 64
"""
The width every output image will have.
"""
EXTRA_PADDING: int = 16
"""
The total extra layer of padding to apply around each image in the horizontal and vertical axes. A layer of half this 
value is added on each of the 4 sides of every image as whitespace.

This does not affect the output width or height, as it is subtracted from those values before scaling the images.
"""


def count_leading_sequential_occurrences(array: np.ndarray, target: int) -> int:
    """
    Gets the number of leading sequential occurrences of a value in a Numpy array.

    For example, given an array [0, 0, 1, 0, 0, 0, 0] with a target of "0", this method would return 4.

    :param array: The array to search through
    :param target: The target value to look for
    :return: The number of leading sequential occurrences of the target value in the array.
    """
    out: int = 0
    for idx, value in enumerate(array):
        if value == target:
            out += 1
        else:
            break
    return out


def trim_image(image: np.ndarray, white_value: int = 255) -> np.ndarray:
    """
    Trims a greyscale image by removing whitespace along the borders.

    :param image: The image to trim. This must be a 2d array.
    :param white_value: The white value to trim.
    :return: The trimmed image.
    """
    row_max: int = image.shape[0]
    col_max: int = image.shape[1]

    max_row_value: int = col_max * white_value
    max_col_value: int = row_max * white_value

    row_sums: np.ndarray = np.sum(image, axis=1)
    col_sums: np.ndarray = np.sum(image, axis=0)

    row_min: int = count_leading_sequential_occurrences(row_sums, max_row_value)
    row_max -= count_leading_sequential_occurrences(np.flip(row_sums), max_row_value)
    col_min: int = count_leading_sequential_occurrences(col_sums, max_col_value)
    col_max -= count_leading_sequential_occurrences(np.flip(col_sums), max_col_value)

    return image[row_min:row_max, col_min:col_max]


def pad_image(image: np.ndarray, height: int, width: int, fill_value: int = 255) -> np.ndarray:
    """
    Pads an image to the desired dimensions.

    The original image will be centered within the padding.

    :param image: The image to pad.
    :param height: The height to pad up to.
    :param width: The width to pad up to.
    :param fill_value: The value to use for padding empty spots.
    :return: A numpy array of size (height, width).
    """
    if image.shape[0] > height or image.shape[1] > width:
        raise ValueError("Image with dimensions {} exceeds padding dimensions ({}, {})"
                         .format(image.shape, height, width))
    if image.shape[0] == height and image.shape[1] == width:
        return image

    output: np.ndarray = np.full((height, width), fill_value, dtype=image.dtype)

    height_min: int = int((height - image.shape[0]) / 2)
    height_max: int = height_min + image.shape[0]

    width_min: int = int((width - image.shape[1]) / 2)
    width_max: int = width_min + image.shape[1]

    output[height_min:height_max, width_min:width_max] = image

    return output


def scale_image(image_array: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Scales an image to the desired dimensions.

    The scaling operation preserves the aspect ratio and will ensure that the output image does not exceed the provided
    height/width values.

    :param image_array: A 2d uint8 numpy array representing an image.
    :param height: The desired height to scale to.
    :param width: The desired width to scale to.
    :return: A 2d uint8 numpy array of height * width dimensions.
    """
    factor_height: float = height / image_array.shape[0]
    factor_width: float = width / image_array.shape[1]
    factor: float = min(factor_height, factor_width)

    image: Image.Image = Image.fromarray(image_array, mode="L")

    new_height: int = int(image_array.shape[0] * factor)
    new_width: int = int(image_array.shape[1] * factor)

    assert new_width <= width and new_height <= height

    image: Image.Image = image.resize((new_width, new_height))
    # noinspection PyTypeChecker
    return np.asarray(image, dtype=np.uint8)


def __normalize_array(image_data: np.ndarray) -> np.ndarray:
    min_val: int = np.min(image_data)
    if min_val < 0:
        image_data -= min_val

    max_val: int = np.max(image_data)
    div: int = max_val if max_val != 0 else 1
    multiplier: float = 255 / div

    image_data = np.multiply(image_data, multiplier, casting="unsafe")
    return image_data.astype(dtype=np.uint8, casting="unsafe")


def preprocess_image(image_data: Union[np.ndarray, Image.Image], is_inverted: bool, output_height: int = OUTPUT_HEIGHT,
                     output_width: int = OUTPUT_WIDTH, extra_padding: int = EXTRA_PADDING) -> np.ndarray:
    if isinstance(image_data, Image.Image):
        image: Image.Image = image_data
        if image.mode != "L":
            image: Image.Image = image.convert(mode="L")
    elif isinstance(image_data, np.ndarray):
        image: Image = Image.fromarray(__normalize_array(image_data), mode="L")
    else:
        raise ValueError("Type {} is not a supported image type! Please use only Pil.Image.Image or np.ndarray!"
                         .format(type(image_data)))

    image_np: np.ndarray = np.asarray(image, dtype=np.uint8)
    if is_inverted:
        image_np: np.ndarray = 255 - image_np

    if np.max(image_np) != np.min(image_np):
        image_np: np.ndarray = trim_image(image_np)
    image_np: np.ndarray = scale_image(image_np, output_height - extra_padding, output_width - extra_padding)
    image_np: np.ndarray = pad_image(image_np, output_height, output_width)

    return image_np


def preprocess_files(data_dir: str, output_height: int, output_width: int, extra_padding: int) -> None:
    """
    Preprocesses all image files in a given directory.

    This process first removes all whitespace on the outer edges of the image.
    The images are then scaled such that at least 1 dimension (i.e. either width or height) is of its
    desired size - extra_padding) with the other dimension equal to or less than its desired size - extra_padding.
    After the images have been scaled to the appropriate dimensions, they are padded to the desired output height and
    width. The padding is applied around the image, to ensure it is centered.

    Note that this process overwrites existing files.

    :param data_dir: The directory containing image files.
    :param output_height: The desired height of the output files.
    :param output_width: The desired width of the output files.
    :param extra_padding: The additional layer of padding to use. This is subtracted from the output height and width
    when rescaling the image, but not from the final padding. As such, it is always guaranteed that a layer of half
    this value is filled with whitespace around the image.
    """
    for path, dirs, files in os.walk(data_dir):
        for file in files:
            fullname: str = os.path.join(path, file)
            image: Image.Image = Image.open(fullname)
            preprocessed_image_arr: np.ndarray = preprocess_image(image, output_height, output_width, extra_padding)
            preprocessed_image: Image.Image = Image.fromarray(preprocessed_image_arr, mode="L")
            preprocessed_image.save(fullname)


if __name__ == "__main__":
    assert EXTRA_PADDING < OUTPUT_WIDTH and EXTRA_PADDING < OUTPUT_HEIGHT
    input_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_DIRECTORY
    preprocess_files(input_dir, OUTPUT_HEIGHT, OUTPUT_WIDTH, EXTRA_PADDING)
