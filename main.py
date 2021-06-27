from typing import List
import numpy as np
from simple_parsing import ArgumentParser
from feature_character_recognition.feature_extractor import FeatureExtractor
import utils
from options import GeneralOptions
from segmentation.character import CharacterSegmenter
from segmentation.line import LineSegmenter
from segmentation.options import SegmentationOptions
import pandas as pd
import skimage.transform as st
import skimage
import pickle as pk
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_levenshtein_distance(real: List[int], pred: List[int]) -> int:
    real_letters: str = "".join([ALT_LABELS[char] for char in real])
    pred_letters: str = "".join([ALT_LABELS[char] for char in pred])
    print("P: {}".format(pred_letters))
    print("R: {}".format(real_letters))
    return Levenshtein.distance(real_letters, pred_letters)


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
    segmented_image.reverse()

    if general_options.stop_after == 2:
        print("Stopping after character segmentation")
        exit()

    #Extract features
    feature_extractor = FeatureExtractor()
    total_count = 0
    for line in segmented_image:
        for word in line:
            for char in word:
                #Resize character
                char = st.resize(char, (80, 80))
                char = np.logical_not(char).astype(int)
                #Reshape the image
                #img = Image.fromarray(char, 'RGB')
                plt.imsave('chars/' + str(total_count) + '.PNG', char, cmap = cm.gray)

                #Extract featires
                df = feature_extractor.extract_features(char)
                #Apply PCA
                pca = pk.load(open('feature_character_recognition/models/pca_duplicated.pkl', 'rb'))
                pca_data = pca.transform(df)

                #Load the classifier
                model = pk.load(open('feature_character_recognition/models/svm_duplicated.pkl', 'rb'))
                #Predict the letter
                predicted_letter = model.predict(pca_data)
                print(total_count)
                total_count += 1
                print(predicted_letter)


    # Classify characters


    if general_options.stop_after == 3:
        print("Stopping after character recognition")
        exit()

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
