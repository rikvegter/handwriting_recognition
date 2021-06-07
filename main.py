from simple_parsing import ArgumentParser

from options import GeneralOptions
from segmentation.character import (CharacterSegmentationMethod,
                                    CharacterSegmenter)
from segmentation.line import LineSegmenter
from segmentation.options import SegmentationOptions


def main(args):
    general_options: GeneralOptions = args.general_options
    segment_options: SegmentationOptions = args.segmentation_options

    if general_options.input_path:
        # Segment lines
        line_segmenter = LineSegmenter(
            general_options=general_options, 
            segment_options=segment_options.line
        )
        n_lines, labeled_lines = line_segmenter.shred()

        # Segment characters
        segmenter = CharacterSegmenter(
            general_options=general_options,
            segment_options=segment_options.character, 
            n_lines=n_lines,
            labeled_lines=labeled_lines
        )
        segmenter.segment(CharacterSegmentationMethod.CONNECTED_COMPONENTS)

        # Classify characters
        # TODO

        # Classify style
        # TODO
    else:
        print("Please provide an input image.")
        parser.print_help()
        exit()
    pass


if __name__ == "__main__":
    # CLI argument parsing
    parser = ArgumentParser()

    # Add general options
    parser.add_arguments(GeneralOptions, dest="general_options")

    # Add segmentation options
    parser.add_arguments(SegmentationOptions, dest="segmentation_options")

    args = parser.parse_args()

    main(args)
