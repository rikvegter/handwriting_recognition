from typing import List

import numpy as np
from simple_parsing import ArgumentParser

import utils
from options import GeneralOptions
from segmentation.character import CharacterSegmenter
from segmentation.line import LineSegmenter
from segmentation.options import SegmentationOptions
from word_recognition.ngram_processor import NGramProcessor
from word_recognition.word_classifier import WordClassifier


def main(args):
    general_options: GeneralOptions = args.general
    segment_options: SegmentationOptions = args.segmentation

    # Segment lines
    line_segmenter = LineSegmenter(general_options=general_options,
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

    # Classify characters
    word_classifier = WordClassifier(character_classifier="character_recognition/data/classification_model/")
    # Reduce Line[Words[Char[]]] to Line[Char[]]
    segmented_image_lines: List[List[np.ndarray]] = [[char for word in line for char in word] for line in segmented_image]
    classified_lines: List[List[int]] = word_classifier.classify_words(segmented_image_lines, True)

    # Use bi-grams to improve accuracy
    ngp = NGramProcessor("util/ngrams_frequencies_withNames.xlsx.txt", ngram_length=2)
    classified_lines: List[List[int]] = ngp.predict_multiple(classified_lines)

    # Get the final transcribed lines as strings of unicode characters
    transcribed_lines: List[str] = utils.decode_words(classified_lines)

    print("transcribed_lines: \n=========================\n{}\n=========================\n"
          .format("\n".join(transcribed_lines)))

    # Classify style
    # TODO

    if general_options.stop_after == 4:
        print("Stopping after style classification",)
        exit()


if __name__ == "__main__":
    # CLI argument parsing
    parser = ArgumentParser()

    # Add general options
    parser.add_arguments(GeneralOptions, dest="general")

    # Add segmentation options
    parser.add_arguments(SegmentationOptions, dest="segmentation")

    args = parser.parse_args()

    main(args)
