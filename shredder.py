import argparse
from PIL import Image, ImageOps
import numpy as np
from scipy import ndimage


class Shredder:
    """Implementation of "Handwritten Text Line Segmentation by Shredding Text
    into its Lines" by Anguelos Nicolaou and Basilis Gatos
    """

    def __init__(self, image_path: str) -> None:
        # Open the image, convert it to a numpy array and make sure it is binary
        self.image = self.prepare_image(image_path)

        # Label the connected components
        self.components, self.n_components = ndimage.label(self.image)

    def prepare_image(self, image_path: str):
        # Prepare the image
        image = Image.open(image_path)
        # make sure we're in grayscale
        image = ImageOps.grayscale(image)
        # Convert to numpy array
        np_image = np.asarray(image)
        # Convert to binary; 255 is white, 0 is black.
        # We want ones where the image is black
        np_image_binarized = np.where(np_image < 127, 1, 0)

        return np_image_binarized

    def find_letter_height(self) -> float:
        """Finds the average letter height based on the average height of the
        connected components

        Returns:
            float: the average letter height in pixels
        """
        slices = ndimage.find_objects(self.components)
        heights = np.zeros(self.n_components)
        for i, s in enumerate(slices):
            height, _ = self.components[s].shape
            heights[i] = height

        return np.mean(heights)

    def blur_image(self):
        lh = self.find_letter_height()
        # define size of the blur window
        bw = np.round(lh * 8).astype(int)
        bh = np.round(lh * 0.8).astype(int)

        # this is slightly different from the paper but performs way better
        blurred_image = ndimage.uniform_filter(self.image.astype(np.float64), size=(bh, bw))

        return blurred_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--image_path", type=str, help="path of the image to use")
    args = parser.parse_args()

    if args.image_path:
        shredder = Shredder(args.image_path)
        lh = shredder.find_letter_height()
        print(lh)
        shredder.blur_image()