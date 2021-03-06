from typing import List, Union

import numpy as np
from PIL import Image

from character_recognition.character_classifier import CharacterClassifier


class WordClassifier:
    def __init__(self, character_classifier: Union[str, CharacterClassifier]):
        """
        :param character_classifier: The path to a directory containing a trained model or a `CharacterClassifier` object.
        """
        if isinstance(character_classifier, CharacterClassifier):
            self.character_classifier = character_classifier
        elif isinstance(character_classifier, str):
            self.character_classifier = CharacterClassifier(character_classifier)
        else:
            raise ValueError("Unexpected input for character classifier!")

    def classify_word(self, word: Union[List[Union[np.ndarray, Image.Image, str]], np.ndarray],
                      is_inverted: bool = False) -> List[int]:
        """
        Classifies a single Word.

        :param word: The images that represent a single word together. Each image can either be a Numpy array
        (4d for rgb, or 3d for greyscale images), or a list of PIL images, numpy arrays (3d for rgb, or 2d for
        greyscale), or strings containing the paths to a images.
        Note that it's safe to mix and match image sizes in the arrays/lists as well as to mix types in the list.
        :param is_inverted: Whether or not the images are provided in inverted form. When this is set to False, it is
        expected that the background is white and that the ink is black. When true, it is assumed to be the other way
        round.
        :return: A numpy array containing the predicted character indices.
        """
        # noinspection PyTypeChecker
        result: List[int] = self.character_classifier.classify_images(word, is_inverted).tolist()
        return result

    def classify_words(self, words: List[Union[List[Union[np.ndarray, Image.Image, str]], np.ndarray]],
                       is_inverted: bool = False) -> List[List[int]]:
        """
        Classifies a list of Words.

        :param words: A list of words. Each word consists of one or more images. Each image can either be a Numpy array
        (4d for rgb, or 3d for greyscale images), or a list of PIL images, numpy arrays (3d for rgb, or 2d for
        greyscale), or strings containing the paths to a images.
        Note that it's safe to mix and match image sizes in the arrays/lists as well as to mix types in the list.
        :param is_inverted: Whether or not the images are provided in inverted form. When this is set to False, it is
        expected that the background is white and that the ink is black. When true, it is assumed to be the other way
        round.
        :return: A list of numpy arrays containing the predicted character indices.
        """
        ret: List[List[int]] = []
        for word in words:
            ret.append(self.classify_word(word, is_inverted))
        return ret

    def classify_lines(self, lines: List[List[Union[List[Union[np.ndarray, Image.Image, str]], np.ndarray]]],
                       is_inverted: bool = False) -> List[List[List[int]]]:
        """
        Classifies one or more lines with one or more words each.

        See the classify_words method for more information.
        :return: The classified characters per word per line.
        """
        ret: List[List[List[int]]] = []
        for line in lines:
            ret.append(self.classify_words(line, is_inverted))
        return ret
