import os.path
from typing import List

import numpy as np
from simple_parsing import ArgumentParser

import utils
from options import GeneralOptions
from pathlib import Path
from segmentation.character import CharacterSegmenter
from segmentation.line import LineSegmenter
from segmentation.options import SegmentationOptions
from word_recognition.ngram_processor import NGramProcessor
from word_recognition.options import ClassifierOptions
from word_recognition.word_classifier import WordClassifier


def run_for_file(args, file_name: str, word_classifier: WordClassifier, ngp: NGramProcessor):
    """
    Runs the classification pipeline for a single file.

    :param args: The input arguments.
    :param file_name: The name (not path!) of the image to classify.
    :param word_classifier: The classifier to use for classification.
    :param ngp: The NGramProcessor to use for post-processing the character classifications.
    """
    general_options: GeneralOptions = args.general
    segment_options: SegmentationOptions = args.segmentation

    input_image_path: str = general_options.input_path + "/" + file_name

    # Segment lines
    line_segmenter = LineSegmenter(general_options=general_options,
                                   input_path=input_image_path,
                                   segment_options=segment_options.line)
    n_lines, char_height, stroke_width, labeled_lines = line_segmenter.shred()

    if general_options.stop_after == 1:
        print("Stopping after line segmentation")
        exit()

    # Segment characters
    segmenter = CharacterSegmenter(general_options=general_options,
                                   segment_options=segment_options.character,
                                   n_lines=n_lines,
                                   labeled_lines=labeled_lines,
                                   char_height=char_height,
                                   stroke_width=stroke_width)
    segmented_image: List[List[List[np.ndarray]]] = segmenter.segment()
    segmented_image.reverse()  # Reverse the lines so it goes from top to bottom

    if general_options.stop_after == 2:
        print("Stopping after character segmentation")
        exit()

    # Reduce Line[Words[Char[]]] to Line[Char[]]
    segmented_image_lines: List[List[np.ndarray]] = [[char for word in line for char in word] for line in segmented_image]
    classified_lines: List[List[int]] = word_classifier.classify_words(segmented_image_lines, True)

    # Use bi-grams to improve accuracy
    classified_lines: List[List[int]] = ngp.predict_multiple(classified_lines)

    # Get the final transcribed lines as strings of unicode characters
    transcribed_output = "\n".join(utils.decode_words(classified_lines))
    if general_options.output_path is not None:
        output_file = general_options.output_path + "/" + Path(file_name).stem + "_characters.txt"
        with open(output_file, encoding="utf-8", mode="w+") as file:
            file.write(transcribed_output)
    else:
        print(transcribed_output)

    # Classify style
    # TODO

    if general_options.stop_after == 4:
        print("Stopping after style classification",)
        exit()


def main(args):
    general_options: GeneralOptions = args.general
    classifier_options: ClassifierOptions = args.classifier

    word_classifier = WordClassifier(character_classifier=classifier_options.classifier)
    ngp = NGramProcessor(classifier_options.ngram_file, ngram_length=2)

    os.makedirs(general_options.output_path, exist_ok=True)

    if general_options.single:
        file = general_options.input_path
        print("Processing file: {}".format(file))
        run_for_file(args, file, word_classifier, ngp)
    else:
        assert os.path.isdir(general_options.input_path)

        _, _, filenames = next(os.walk(general_options.input_path))
        for file in filenames:
            print("Processing file: {}".format(file))
            run_for_file(args, file, word_classifier, ngp)


if __name__ == "__main__":
    # CLI argument parsing
    parser = ArgumentParser()

    # Add general options
    parser.add_arguments(GeneralOptions, dest="general")

    # Add segmentation options
    parser.add_arguments(SegmentationOptions, dest="segmentation")

    # Add classification options
    parser.add_arguments(ClassifierOptions, dest="classifier")

    main(parser.parse_args())
