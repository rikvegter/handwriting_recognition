from dataclasses import dataclass

from simple_parsing import field


@dataclass
class ClassifierOptions:
    """Parameters for the different classification steps
    """

    classifier: str = field(default="character_recognition/data/classification_model/")

    ngram_file: str = field(default="util/ngrams_frequencies_withNames.xlsx.txt")
