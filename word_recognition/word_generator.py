import os
import random
from typing import List

from character_recognition.character_recognizer import LABELS


class Word:
    """
    Represents a word as a list of character images and a list of indices corresponding to characters in the LABELS
    list in the character_recognizer class.
    """

    def __init__(self, files: List[str], indices: List[int]):
        self.files: List[str] = files
        """
        The list of paths pointing to character images.
        """
        self.indices: List[int] = indices
        """
        The list of indices for the characters corresponding to characters in the LABELS list in the
        character_recognizer class.
        """


class WordGenerator:
    """
    Represents a class that generated a list of Word objects from a file of indices referring to the list of labels
    as used by the character_recognizer class.
    """
    __indexed_words: List[List[int]] = []

    def __init__(self, input_file: str, dataset_dir: str):
        """
        Constructs a new WordGenerator.

        :param input_file: The file to read the words from. Every line in this file should contain one word and every
        word should consist of one or more integers corresponding to an index in the LABELS list in the
        character_recognizer class (i.e. range of [0 27]).
        :param dataset_dir: The directory to select character images from.
        """
        self.__input_file = input_file
        self.__dataset_dir = dataset_dir
        self.__parse_input()

    def __parse_input(self) -> None:
        with open(self.__input_file) as file:
            for line in file.readlines():
                self.__indexed_words.append([int(idx) for idx in line.split()])

    def __get_path_for_index(self, idx: int) -> str:
        char_name = LABELS[idx]
        search_dir = "{}/{}".format(self.__dataset_dir, char_name)
        random_file = random.choice([f for f in os.listdir(search_dir) if os.path.isfile(os.path.join(search_dir, f))])
        return os.path.abspath(search_dir + "/" + random_file)

    def generate_files(self) -> List[Word]:
        """
        Generates a list of Words from the indices read from the file. The Words are supplied with randomly selected
        character images from the provided dataset folder. As such, repeated calls of this method will result in
        different output.

        :return: A list of Words.
        """
        ret: List[Word] = []
        for indexed_word in self.__indexed_words:
            current_word: List[str] = []
            for idx in indexed_word:
                current_word.append(self.__get_path_for_index(idx))
            ret.append(Word(current_word, indexed_word))
        return ret
