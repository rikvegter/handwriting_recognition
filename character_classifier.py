import glob
import os.path
import time
from typing import Union, List

import numpy as np
import tensorflow as tf
from PIL import Image

import character_recognizer
import preprocess_images

DEFAULT_MODEL: str = "classification_model/"
"""
The default model to use.

An example model can be found here: https://drive.google.com/file/d/1utDs6NSGYzpgvAll2yvAOp1SF-z6ydza/view?usp=sharing
"""

LABELS: List[str] = ["Tsadi-final", "Kaf-final", "Mem", "Nun-final", "Gimel", "Tet", "Qof", "He", "Pe", "Lamed", "Resh",
                     "Alef", "Shin", "Taw", "Nun-medial", "Kaf", "Tsadi-medial", "Zayin", "Mem-medial", "Dalet",
                     "Pe-final", "Waw", "Samekh", "Yod", "Het", "Bet", "Ayin"]
"""
List of all the labels.

The classifier outputs indices corresponding to the characters in this list.
"""


class CharacterClassifier:
    """
    Represents a class that can be used to classify images of ancient hebrew characters.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL):
        """
        :param model_path: The path to the directory containing the weights of the model to use for classification.
        Defaults to #DEFAULT_MODEL.
        """
        self.model = self.__load_model(model_path)

    def __load_model(self, model_path: str) -> tf.keras.models.Model:
        assert os.path.isdir(model_path)
        model: tf.keras.models.Model = character_recognizer.get_model(LABELS)
        model.load_weights(model_path).expect_partial()
        return model

    def __classify_image(self, image: np.ndarray) -> int:
        assert len(image.shape) >= 2
        image_arr = np.expand_dims(image, axis=0)
        predictions: np.ndarray = self.model.predict(image_arr)
        return np.argmax(predictions, axis=1)[0]

    def __classify_images(self, images: np.ndarray) -> np.ndarray:
        predictions: np.ndarray = self.model.predict_on_batch(images)
        return np.argmax(predictions, axis=1)

    def __prepare_image(self, image_data: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        if not isinstance(image_data, (np.ndarray, Image.Image, str)):
            raise ValueError("Type {} is not a valid image type! Only numpy arrays and PIL Images are allowed!"
                             .format(type(image_data)))

        if isinstance(image_data, str):
            image_data: Union[np.ndarray, Image.Image] = Image.open(image_data)

        image: np.ndarray = preprocess_images.preprocess_image(image_data)
        image: np.ndarray = 255 - image
        image: np.ndarray = np.repeat(image[..., np.newaxis], 3, -1)
        return image.astype(dtype=np.uint8)

    def __generate_image_holder(self, image_count: int) -> np.ndarray:
        return np.full((image_count, preprocess_images.OUTPUT_HEIGHT, preprocess_images.OUTPUT_WIDTH, 3), 255,
                       dtype=np.uint8)

    def classify_image(self, image_data: Union[np.ndarray, Image.Image, str]) -> int:
        """
        Classifies an image.

        :param image_data: The data representing an image. This can either be a Numpy array (3d for rgb, or 2d for
        greyscale images), a PIL image, or a string containing the path to an image.
        :return: The index of the classified character in the #LABELS list.
        """
        image = self.__prepare_image(image_data)
        return self.__classify_image(image)

    def classify_images(self, image_data: Union[List[Union[np.ndarray, Image.Image, str]], np.ndarray]) -> np.ndarray:
        """
        Classifies multiple images.

        Note that all images will be loaded and then classified in one go. If you are low on memory, it might be
        advisable to use #classify_image instead, so the images are processed and classified one-by-one. However,
        classifying images one-by-one is about 3x slower.

        :param image_data: The data representing multiple images. This can either be a Numpy array (4d for rgb, or 3d
        for greyscale images), or a list of PIL images, numpy arrays (3d for rgb, or 2d for greyscale), or strings
        containing the paths to a images.
        Note that it's safe to mix and match image sizes in the arrays/lists as well as to mix types in the list.
        :return: An array of indices of the classified character in the #LABELS list.
        """
        if isinstance(image_data, np.ndarray):
            if len(image_data.shape) < 3:
                raise ValueError("Failed to process array of shape: {}. Images need at least 2 dimensions!"
                                 .format(image_data.shape))
            elif len(image_data.shape) > 4:
                raise ValueError("Failed to process array of shape: {}. Images cannot have more than 3 dimensions!"
                                 .format(image_data.shape))
            elif len(image_data.shape) == 3 and image_data.shape[2] == 3:
                raise ValueError("Failed to process array of shape: {}. RGB images need at least 3 dimensions!"
                                 .format(image_data.shape))
            image_count = image_data.shape[0]
        elif isinstance(image_data, List):
            image_count = len(image_data)
        else:
            raise ValueError("Type {} cannot be processed! Only Numpy arrays and lists are allowed!"
                             .format(type(image_data)))
        if image_count == 0:
            return np.empty(0, dtype=int)
        images = self.__generate_image_holder(image_count)
        for idx in range(image_count):
            images[idx, :, :, :] = self.__prepare_image(image_data[idx])
        return self.__classify_images(images)
