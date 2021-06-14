from typing import List, Tuple

import numpy as np

from character_recognition.character_classifier import CharacterClassifier
from word_recognition.word_generator import WordGenerator, Word

DEFAULT_INPUT_FILES: List[str] = ["../util/124-Fg004_characters.docx.txt", "../25-Fg001_characters.docx.txt"]
DEFAULT_DATASET_DIR: str = "../character_recognition/data/dataset/0_augmented"
RUNS = 10


def classify_word(classifier: CharacterClassifier, word: Word) -> np.ndarray:
    """
    Classifies a single Word.

    :param classifier: The classifier to use.
    :param word: The Word to classify.
    :return: A numpy array containing the predicted character indices.
    """
    assert classifier is not None
    return classifier.classify_images(word.files)


def classify_words(classifier: CharacterClassifier, words: List[Word]) -> List[np.ndarray]:
    """
    Classifies a list of Words.

    :param classifier: The classifier to use.
    :param words: The list of words to classify.
    :return: A list of numpy arrays containing the predicted character indices.
    """
    ret: List[np.ndarray] = []
    for word in words:
        ret.append(classify_word(classifier, word))
    return ret


def get_classification_accuracy(classifier: CharacterClassifier, words: List[Word]) -> Tuple[float, float]:
    """
    Classifies a list of Words and retrieves the accuracy.

    :param classifier: The classifier to use.
    :param words: The list of words to classify.
    :return: The accuracy based on the number of correctly classified characters and the percentage of words that were
    classified correctly without any mistakes.
    """
    predicted: List[np.ndarray] = classify_words(classifier, words)
    tot_correct = 0
    tot_incorrect = 0
    correct_words = 0
    incorrect_words = 0

    for word_idx in range(len(words)):
        current_indices = words[word_idx].indices
        current_prediction = predicted[word_idx]

        current_correct = 0
        current_incorrect = 0
        for score_idx in range(len(current_indices)):
            if current_indices[score_idx] == current_prediction[score_idx]:
                current_correct += 1
            else:
                current_incorrect += 1
        tot_correct += current_correct
        tot_incorrect += current_incorrect
        if current_incorrect == 0:
            correct_words += 1
            status = "  CORRECT"
        else:
            incorrect_words += 1
            status = "INCORRECT"
        print("{}: {} / {}".format(status, current_indices, current_prediction))

    character_accuracy = 100 * tot_correct / (tot_correct + tot_incorrect)
    word_accuracy = 100 * correct_words / (correct_words + incorrect_words)

    return character_accuracy, word_accuracy


if __name__ == "__main__":
    words: List[Word] = []

    for file in DEFAULT_INPUT_FILES:
        word_generator = WordGenerator(file, DEFAULT_DATASET_DIR)
        for idx in range(RUNS):
            words.extend(word_generator.generate_files())
    classifier: CharacterClassifier = CharacterClassifier("../character_recognition/data/models/model_0")

    char_acc, word_acc = get_classification_accuracy(classifier, words)
    print("Classification accuracy: {:.2f}%, word accuracy: {:.2f}%".format(char_acc, word_acc))
