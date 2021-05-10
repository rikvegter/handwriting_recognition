import argparse
from PIL import Image, ImageOps
import numpy as np
from numpy.core.fromnumeric import trace
from scipy import ndimage


class Shredder:
    """Implementation of "Handwritten Text Line Segmentation by Shredding Text
    into its Lines" by Anguelos Nicolaou and Basilis Gatos (2009; doi:10/b5wsx6)
    """

    def __init__(self, image_path: str, save_debug_imgs: bool) -> None:
        # Set debug settings
        self.save_debug_imgs = save_debug_imgs
        
        # Open the image, convert it to a numpy array and make sure it is binary
        self.image = self.prepare_image(image_path)

        # Label the connected components
        self.components, self.n_components = ndimage.label(self.image)

        # Find the letter height and define the blurring window
        self.letter_height = self.find_letter_height()
        self.blur_width = (self.letter_height * 4.0).astype(int)
        self.blur_height = (self.letter_height * 1.0).astype(int)

        # Blur the image
        self.blurred_image = self.blur_image()

    def prepare_image(self, image_path: str) -> np.ndarray:
        # Prepare the image
        image = Image.open(image_path)
        # make sure we're in grayscale
        image = ImageOps.grayscale(image)
        # Convert to numpy array
        np_image = np.asarray(image)
        # Convert to binary; 255 is white, 0 is black.
        # We want ones where the image is black
        np_image_binarized = np.where(np_image < 127, 1, 0)

        if self.save_debug_imgs:
            output = Image.fromarray(
                (np_image_binarized * 255).astype(np.uint8))
            output.save("binarized_image.png")

        return ndimage.rotate(np_image_binarized, 180)

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

    def blur_image(self) -> np.ndarray:
        """Blurs the image with the given blur window.

        Returns:
            np.ndarray: The blurred image
        """

        # First dilate horizontally and erode vertically
        # ! different from the paper!

        # erosion_struct = np.zeros((3,3))
        # erosion_struct[:, 1] = True
        # eroded_image = ndimage.binary_erosion(self.image, structure=erosion_struct, iterations=10)

        # dilation_struct = np.zeros((3,3))
        # dilation_struct[1] = True
        # dilated_image = ndimage.binary_dilation(eroded_image, structure=dilation_struct, iterations=40)

        if self.save_debug_imgs:
            output = Image.fromarray((self.image.astype(np.uint8) * 255).astype(np.uint8))
            output.save("binarized_dilated.png")

        # Use a uniform filter as a fast blur operator
        # ! different from the paper!
        blurred_image = ndimage.uniform_filter(
            self.image.astype(np.float64), size=(self.blur_height, self.blur_width)
        )

        if self.save_debug_imgs:
            # increase range so blurred image is visible
            output = np.interp(blurred_image, (blurred_image.min(), blurred_image.max()), (0, 255))
            output = Image.fromarray(blurred_image.astype(np.uint8))
            output.save("binarized_blurred_image.png")

        return blurred_image

    def generate_tracers(self):
        """Generates the tracers between the text lines. This is a vectorized
        operation.
        """

        # k cannot be higher than the max y, x corresponds to image x
        max_y, max_x = self.blurred_image.shape

        offset = self.blur_height // 2

        # Precompute tracers. Results in a (x × k) array, where k indicates the
        # kth tracer
        tracers = np.empty_like(self.image)

        # initialize all x == 0 to y
        tracers[:, 0] = np.arange(max_y)

        for x in range(max_x):
            if x > 0:
                prev = tracers[:, x - 1]

                # calculate lhs y and bound it in [0, max_y)
                lhs_y = np.clip(prev + offset, 0, max_y - 1)
                lhs = self.blurred_image[lhs_y, x]

                # # calculate rhs y and bound it in [0, max_y)
                rhs_y = np.clip(prev - offset, 0, max_y - 1)
                rhs = self.blurred_image[rhs_y, x]

                tracers[:, x] = np.where(
                    lhs > rhs, prev - 1, np.where(lhs < rhs, prev + 1, prev)
                )

        # Prepare image to save traces to
        tracer_image = np.empty_like(self.image)

        # Calculate tracers (vectorized)
        for x in range(max_x):
            col_k_values = tracers[:, x]
            col_y_values = np.arange(0, max_y)
            tracer_image[:, x] = np.where(
                np.isin(col_y_values, col_k_values), 0, 1)

        # If necessary, save intermediary debug images
        if self.save_debug_imgs:
            # array of k values
            output_1 = Image.fromarray(
                np.interp(tracers, (tracers.min(), tracers.max()), (0, 255)).astype(
                    np.uint8
                )
            )
            output_1.save("tracer_helper.png")

            # tracers
            output_2 = Image.fromarray(
                (
                    np.interp(
                        tracer_image, (tracer_image.min(),
                                       tracer_image.max()), (0, 255)
                    )
                ).astype(np.uint8)
            )
            output_2.save("tracers.png")


if __name__ == "__main__":

    #TODO###################################################################################
    #TODO: Undo rotation when processing output! (also check if it even makes a difference)#
    #TODO###################################################################################

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--image_path", type=str,
                        help="path of the image to use")
    parser.add_argument(
        "-d",
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save intermediary images for debugging purposes",
    )
    args = parser.parse_args()

    if args.image_path:
        shredder = Shredder(args.image_path, args.debug)
        shredder.generate_tracers()
    else:
        print("Please provide an input image.")
        parser.print_help()
        exit()
