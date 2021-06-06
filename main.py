from shredder import Shredder
from segmenter import Segmenter, SegmentationMethod
import argparse
import os

def main(args):
    if args.image_path:
        shredder = Shredder(args.image_path, args.debug, args.output_path)
        n_lines, labeled_lines = shredder.shred()

        segmenter = Segmenter(n_lines, labeled_lines, args.debug, args.output_path)
        segmenter.segment(SegmentationMethod.CONNECTED_COMPONENTS)
    else:
        print("Please provide an input image.")
        parser.print_help()
        exit()
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i",
                        "--image_path",
                        type=str,
                        help="path of the image to use")
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="./",
        help="path to save output images to (mostly for debugging)")
    parser.add_argument(
        "-d",
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=
        "Save intermediary images for debugging purposes and show progress",
    )
    args = parser.parse_args()

    main(args)