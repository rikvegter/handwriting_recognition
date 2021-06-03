import os
import random
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import cv2

INPUT_DIRECTORY: str = "dataset"
AUGMENTED_DIRECTORY_SUFFIX: str = "_augmented"

SHEAR_FACTOR: float = 0.3
SHIFT_FACTOR: float = 0.2

SHEAR_COUNT: int = 0
SHIFT_COUNT: int = 0
ENABLE_EROSION: bool = True
ENABLE_DILATION: bool = True


class AugmentationMethod(ABC):
    def __init__(self, image: np.ndarray, output_file_base: str, output_file_extension: str, index: int):
        self.image: np.ndarray = image
        self.output_file: str = self.__get_indexed_output_file(output_file_base, output_file_extension, index)
        self.rows: int = image.shape[0]
        self.cols: int = image.shape[1]

    def run(self):
        self.apply_augmentation()
        self.save_image()

    def __get_indexed_output_file(self, output_file_base: str, output_file_extension: str, index: int) -> str:
        return "{}_{}{}".format(output_file_base, index, output_file_extension)

    def save_image(self):
        cv2.imwrite(self.output_file, self.image)

    @abstractmethod
    def apply_augmentation(self):
        pass

    def _crop_image(self):
        new_rows: int = self.image.shape[0]
        new_cols: int = self.image.shape[1]
        if new_rows == self.rows and new_cols == self.cols:
            return

        self.image: np.ndarray = self.image[:self.rows, :self.cols]


class AugmentationMethodNull(AugmentationMethod, ABC):

    def __init__(self, image: np.ndarray, output_file_base: str, output_file_extension: str, index: int):
        super().__init__(image, output_file_base, output_file_extension, index)

    def apply_augmentation(self):
        pass


class AugmentationMethodShear(AugmentationMethod, ABC):

    def __init__(self, image: np.ndarray, output_file_base: str, output_file_extension: str, index: int,
                 shearing_factor_range: float):
        super().__init__(image, output_file_base, output_file_extension, index)
        self.shearing_factor_range: float = shearing_factor_range
        self.__create_shearing_factor()

    def __create_shearing_factor(self):
        self.shearing_factor: float = random.uniform(-self.shearing_factor_range, self.shearing_factor_range)
        self.shearing_matrix = np.float32([[1, self.shearing_factor, 0],
                                           [0, 1, 0],
                                           [0, 0, 1]])

    def apply_augmentation(self):
        size_factor: float = 1 + abs(self.shearing_factor)

        new_rows: int = int(self.rows * size_factor)
        new_cols: int = int(self.cols * size_factor)
        self.image: np.ndarray = cv2.warpPerspective(self.image, self.shearing_matrix,
                                                     (new_rows, new_cols), borderValue=255,
                                                     borderMode=cv2.BORDER_CONSTANT)
        self._crop_image()


class AugmentationMethodShift(AugmentationMethod, ABC):

    def __init__(self, image: np.ndarray, output_file_base: str, output_file_extension: str, index: int,
                 shift_factor_range: float):
        super().__init__(image, output_file_base, output_file_extension, index)
        self.shift_factor_x: float = random.uniform(-shift_factor_range, shift_factor_range)
        self.shift_factor_y: float = random.uniform(-shift_factor_range, shift_factor_range)

    def apply_augmentation(self):
        row_shift: int = int(self.rows * self.shift_factor_y)
        col_shift: int = int(self.cols * self.shift_factor_x)

        new_image: np.ndarray = np.full((self.rows, self.cols), 255, dtype=self.image.dtype)

        min_row: int = max(0, -row_shift)
        max_row: int = min(self.rows, self.rows - row_shift)
        min_col: int = max(0, -col_shift)
        max_col: int = min(self.cols, self.cols - col_shift)

        new_rows: int = max_row - min_row
        new_cols: int = max_col - min_col

        if row_shift >= 0:
            target_offset_row: int = self.rows - new_rows
        else:
            target_offset_row: int = 0
        if col_shift >= 0:
            target_offset_col: int = self.cols - new_cols
        else:
            target_offset_col: int = 0

        new_image[target_offset_row:target_offset_row + new_rows, target_offset_col:target_offset_col + new_cols] = \
            self.image[min_row:max_row, min_col:max_col]

        self.image: np.ndarray = new_image


class AugmentationMethodDilation(AugmentationMethod, ABC):

    def __init__(self, image: np.ndarray, output_file_base: str, output_file_extension: str, index: int,
                 kernel_size: int = 3):
        super().__init__(image, output_file_base, output_file_extension, index)
        self.kernel: np.ndarray = np.ones((kernel_size, kernel_size), np.uint8)

    def apply_augmentation(self):
        self.image: np.ndarray = cv2.dilate(self.image, self.kernel, iterations=1)


class AugmentationMethodErosion(AugmentationMethod, ABC):

    def __init__(self, image, output_file_base: str, output_file_extension: str, index: int, kernel_size: int = 3):
        super().__init__(image, output_file_base, output_file_extension, index)
        self.kernel: np.ndarray = np.ones((kernel_size, kernel_size), np.uint8)

    def apply_augmentation(self):
        self.image: np.ndarray = cv2.erode(self.image, self.kernel, iterations=1)


def apply_data_augmentation_to_file(input_dir: str, file: str, output_dir: str, augmentation_methods: List) -> None:
    ensure_directory_exists(output_dir)
    image: np.ndarray = cv2.imread(input_dir + "/" + file, cv2.IMREAD_GRAYSCALE)
    output_file_base, output_file_extension = os.path.splitext(output_dir + "/" + file)

    for idx, augmentation_method in enumerate(augmentation_methods):
        augmentation_method(image, output_file_base, output_file_extension, idx).run()


def ensure_directory_exists(dirname: str) -> None:
    assert not os.path.isfile(dirname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def apply_data_augmentation_to_subdir(data_dir: str, augmented_directory_suffix: str,
                                      augmentation_methods: List) -> None:
    augmented_dir = data_dir + augmented_directory_suffix
    ensure_directory_exists(augmented_dir)

    for path, dirs, files in os.walk(data_dir):
        for file in files:
            character_name = path.replace(data_dir + "/", "")
            apply_data_augmentation_to_file(path, file, augmented_dir + "/" + character_name, augmentation_methods)


def apply_data_augmentation(data_dir: str, augmented_directory_suffix: str, augmentation_methods: List) -> None:
    for path, dirs, files in os.walk(data_dir):
        for dirname in dirs:
            if augmented_directory_suffix in dirname:
                continue
            full_path = os.path.join(path, dirname)
            apply_data_augmentation_to_subdir(full_path, augmented_directory_suffix, augmentation_methods)
        break


if __name__ == "__main__":
    augmentation_methods_list = [
        lambda image, output_file_base, output_file_extension, index:
        AugmentationMethodNull(image, output_file_base, output_file_extension, index),
    ]
    if ENABLE_DILATION:
        augmentation_methods_list.append(lambda image, output_file_base, output_file_extension, index:
                                         AugmentationMethodDilation(image, output_file_base, output_file_extension,
                                                                    index))
    if ENABLE_EROSION:
        augmentation_methods_list.append(lambda image, output_file_base, output_file_extension, index:
                                         AugmentationMethodErosion(image, output_file_base, output_file_extension,
                                                                   index))

    for _ in range(SHEAR_COUNT):
        augmentation_methods_list.append(lambda image, output_file_base, output_file_extension, index:
                                         AugmentationMethodShear(image, output_file_base, output_file_extension, index,
                                                                 SHEAR_FACTOR))

    for _ in range(SHIFT_COUNT):
        augmentation_methods_list.append(lambda image, output_file_base, output_file_extension, index:
                                         AugmentationMethodShift(image, output_file_base, output_file_extension, index,
                                                                 SHIFT_FACTOR))

    apply_data_augmentation(INPUT_DIRECTORY, AUGMENTED_DIRECTORY_SUFFIX, augmentation_methods_list)
