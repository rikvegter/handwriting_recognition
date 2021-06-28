from typing import List

import numpy as np

UNICODE_VALUES: List[int] = [1488, 1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1499, 1498, 1500, 1502,
                             1501, 1504, 1503, 1505, 1506, 1508, 1507, 1510, 1509, 1511, 1512, 1513, 1514]
UNICODE_CHARS: List[str] = [chr(val) for val in UNICODE_VALUES]


class StepInfoPrinter:
    """A utility class to allow for easily printing messages to the terminal
    indicating progress in some process.
    """

    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self.__cur_step = 1
        self.__p = int(np.floor(np.log10(max_steps))) + 1

    def print(self, message: str, step: bool = True, end: str = ''):
        """Prints some info along with which step number the program is at.

        Args:
            message (str):
                The info message to print.

            step (bool):
                Whether to print the current step number.
                Defaults to true.

            end (str, optional):
                The character to end the line with. If this is the last message
                being printed, end this with the line feed ('\n') character.
                Defaults to ''.
        """
        if step:
            print(f"\x1b[1K\r[{self.__cur_step:{self.__p}}/{self.max_steps}] {message}", end=end)
            self.__cur_step += 1
        else:
            print(f"\x1b[1K\r{message}", end=end)

    def print_done(self):
        print(f"\x1b[1K\r[{self.max_steps}/{self.max_steps}] Done.")


def decode_word(word: List[int]) -> str:
    """
    Decodes a word, represented by a list of character labels into a single string with unicode characters.

    :param word: The list of characters. Each character is represented by an index.
    :return: The decoded string.
    """
    return "".join(UNICODE_CHARS[label] for label in word)


def decode_words(words: List[List[int]]) -> List[str]:
    """
    Decodes a list of word. See `decode_word` for more information.
    """
    return [decode_word(word) for word in words]
