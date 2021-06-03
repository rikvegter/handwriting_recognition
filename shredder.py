import argparse
import os
from typing import Tuple
from PIL import Image, ImageOps
import numpy as np
from numpy.core.fromnumeric import ndim, trace
from numpy.lib.arraysetops import unique
from scipy import ndimage
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class Shredder:
    """Implementation of "Handwritten Text Line Segmentation by Shredding Text
    into its Lines" by Anguelos Nicolaou and Basilis Gatos (2009; doi:10/b5wsx6)
    """
    def __init__(self,
                 image_path: str,
                 debug: bool = False,
                 output_path: str = "./") -> None:
        """Initialize the line shredder

        Args: image_path (str): the path of the image to segment lines of debug
            (bool, optional): Whether to enable debug mode. Defaults to False.
            output_path (str, optional): Where to save debug images to. Defaults
            to "./".
        """
        # Set debug settings
        self.debug = debug
        self.__print_debug("Setting up...")
        self.output_path = output_path
        self.im_counter = 0  # for labeling image order
        if debug:
            os.makedirs(self.output_path, exist_ok=True)

        # Open the image, convert it to a numpy array and make sure it is binary
        self.image = self.__prepare_image(image_path)

        # 2.1 Preprocessing

        # Label the connected components
        self.__print_debug("Labeling connected components...")
        self.components, self.n_components = ndimage.label(self.image)

        # Find the letter height and define the blurring window
        self.__print_debug("Finding letter height...")
        self.letter_height = self.__find_letter_height()
        self.blur_width = (self.letter_height * 4.0).astype(int)
        self.blur_height = (self.letter_height * 1.0).astype(int)

        # Blur the image (B(x, y))
        self.__print_debug("Blurring image...")
        self.blurred_image = self.__blur_image()

    def shred(self) -> Tuple[int, np.ndarray]:
        """Labels all connected components in an image as belonging to a line.

        Returns:
            Tuple[int, np.ndarray]: A labeled image and the number of labels.
        """

        # 2.2.1 Tracing line areas (LA(x, y))
        self.__print_debug("Generating white path traces...")
        line_areas = self.__generate_traces()

        # 2.2.2 Labeling line areas (LLA(x, y))
        self.__print_debug("Labeling line areas...")
        n_lines, labeled_line_areas = self.__get_lla(line_areas)

        # 2.2.3 Tracing line centers (LC(x, y))
        self.__print_debug("Generating black path traces...")
        line_centers = self.__generate_traces(invert=True)

        # 2.2.4 Labeling line centers
        self.__print_debug("Labeling line centers...")
        labeled_line_centers = self.__get_llc(labeled_line_areas, line_centers,
                                              n_lines)

        # 2.3.1 Assigning to line centers
        self.__print_debug("Separating lines, pass 1...")
        result_line_centers = self.__separate_lines(n_lines, labeled_line_centers)

        # 2.3.2 Assigning to line areas
        self.__print_debug("Separating lines, pass 2...")
        result_line_areas = self.__separate_lines(n_lines, labeled_line_areas)

        # The original paper adds the results from 2.3.2 directly to RES(x,y),
        # but for us it is more convenient to do the adding here
        intermediate_result = result_line_areas + result_line_centers

        # 2.3.3 Assigning remaining pixels
        result_final = self.__assign_remaining(n_lines, intermediate_result, labeled_line_areas)

        return n_lines, result_final

    def __prepare_image(self, image_path: str) -> np.ndarray:
        # Prepare the image
        image = Image.open(image_path)
        # make sure we're in grayscale
        image = ImageOps.grayscale(image)
        # Convert to numpy array
        np_image = np.asarray(image)
        # Convert to binary; 255 is white, 0 is black.
        # We want ones where the image is black
        np_image_binarized = np.where(np_image < 127, 1, 0)

        if self.debug:
            output = Image.fromarray(
                (np_image_binarized * 255).astype(np.uint8))
            output.save(
                os.path.join(self.output_path,
                             f"{self.im_counter}_binarized_image.png"))
            self.im_counter += 1

        # rotate the image by 180 degrees by mirroring horizontally and
        # vertically
        rotated = np.flip(np_image_binarized)

        return rotated  # np_image_binarized

    def __find_letter_height(self) -> float:
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

    def __blur_image(self) -> np.ndarray:
        """Blurs the image with the given blur window.

        Returns:
            np.ndarray: The blurred image
        """

        # Use a uniform filter as a fast blur operator
        # This is different from the paper!
        blurred_image = ndimage.uniform_filter(self.image.astype(np.float64),
                                               size=(self.blur_height,
                                                     self.blur_width))

        if self.debug:
            # increase range so blurred image is visible
            output = np.interp(blurred_image,
                               (blurred_image.min(), blurred_image.max()),
                               (0, 255))
            output = Image.fromarray(output.astype(np.uint8))
            output.save(
                os.path.join(self.output_path,
                             f"{self.im_counter}_binarized_image_blurred.png"))
            self.im_counter += 1

        return blurred_image

    def __generate_traces(self, invert: bool = False) -> np.ndarray:
        """Generates a set of traces.

        Args: invert (bool, optional): Whether to invert the image before
            generating the tracers. Defaults to False.

        Returns: np.ndarray: A binary image containing the tracers.
        """

        # k cannot be higher than the max y, x corresponds to image x
        max_y, max_x = self.blurred_image.shape

        # used for calculating the tracers
        offset = self.blur_height // 2

        # Blurred image, possibly inverted
        if invert:
            source_image = 255 - self.blurred_image
        else:
            source_image = self.blurred_image

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
                lhs = source_image[lhs_y, x]

                # # calculate rhs y and bound it in [0, max_y)
                rhs_y = np.clip(prev - offset, 0, max_y - 1)
                rhs = source_image[rhs_y, x]

                tracers[:, x] = np.where(lhs > rhs, prev - 1,
                                         np.where(lhs < rhs, prev + 1, prev))

        # Prepare image to save traces to
        traces = np.empty_like(self.image)

        # Calculate tracers (vectorized)
        for x in range(max_x):
            col_k_values = tracers[:, x]
            col_y_values = np.arange(0, max_y)
            if invert:
                traces[:, x] = np.where(np.isin(col_y_values, col_k_values), 1,
                                        0)
            else:
                traces[:, x] = np.where(np.isin(col_y_values, col_k_values), 0,
                                        1)

        # If necessary, save intermediary debug images
        if self.debug:
            # array of k values
            output_1 = Image.fromarray(
                np.interp(tracers, (tracers.min(), tracers.max()),
                          (0, 255)).astype(np.uint8))

            if invert:
                is_inverted = "_inverted"
            else:
                is_inverted = ""

            output_1.save(
                os.path.join(
                    self.output_path,
                    f"{self.im_counter}_tracer_helper{is_inverted}.png"))
            self.im_counter += 1

            # tracers
            output_2 = Image.fromarray(
                (np.interp(traces, (traces.min(), traces.max()),
                           (0, 255))).astype(np.uint8))
            output_2.save(
                os.path.join(self.output_path,
                             f"{self.im_counter}_tracers{is_inverted}.png"))
            self.im_counter += 1

        # Return traces
        return traces

    def __get_lla(self, line_areas: np.ndarray) -> Tuple[int, np.ndarray]:
        """Assigns labels to lines

        Args: line_areas (np.ndarray): The line areas resulting from tracing the
            interline spacing

        Returns: Tuple[int, np.ndarray]: The number of lines and the labeled
            image.
        """

        # find connected components
        labels, n_lines = ndimage.label(line_areas)

        # set minimum pixel count
        min_pix_count = self.letter_height**2

        # find the pixel counts of every labeled image part
        label_sizes = ndimage.labeled_comprehension(line_areas, labels,
                                                    np.arange(1, n_lines + 1),
                                                    np.size, np.uint16, 0,
                                                    False)

        # Mark labels to keep
        # labels should start at 1
        labels_to_keep = np.argwhere(
            label_sizes >= min_pix_count).flatten() + 1

        # Set everything we do not want to keep to 0 and relabel so it is labels are monotonously increasing
        output_labels = np.zeros_like(self.image)

        for i, l in enumerate(labels_to_keep):
            output_labels += np.where(labels == l, i + 1, 0)

        if self.debug:
            output_labels = np.ma.masked_where(output_labels == 0,
                                               output_labels)
            cm = plt.get_cmap('turbo', lut=len(labels_to_keep)).copy()
            cm.set_bad(color='black')
            colored_image = cm(output_labels)
            Image.fromarray(
                (colored_image[:, :, :3] * 255).astype(np.uint8)).save(
                    os.path.join(self.output_path,
                                 f"{self.im_counter}_line_labels.png"))
            self.im_counter += 1

        return len(labels_to_keep), output_labels

    def __get_llc(self, labeled_line_areas: np.ndarray,
                  line_centers: np.ndarray, n_labels: int) -> np.ndarray:

        llc = labeled_line_areas * line_centers

        if self.debug:
            output_labels = np.ma.masked_where(llc == 0, llc)
            cm = plt.get_cmap('turbo', lut=n_labels).copy()
            cm.set_bad(color='black')
            colored_image = cm(output_labels)
            Image.fromarray(
                (colored_image[:, :, :3] * 255).astype(np.uint8)).save(
                    os.path.join(self.output_path,
                                 f"{self.im_counter}_line_center_labels.png"))
            self.im_counter += 1

        return llc

    def __separate_lines(self, n_lines: int,
                         labeled_line_centers: np.ndarray) -> np.ndarray:
        """Separates the lines based on the calculated line centers

        Args: n_lines (int): The number of lines that have been found
            labeled_line_centers (np.ndarray): The centers of the text lines

        Returns: np.ndarray: All connected components intersecting the center
            line, labeled with the center line they are intersecting
        """

        # prepare result
        res = np.zeros_like(self.image)

        for label in range(1, n_lines + 1):
            # Find the locations of just the current line
            this_line = np.where(labeled_line_centers == label,
                                 labeled_line_centers, 0)
            # Find the connected components that intersect with the current line
            components_on_line = np.where(self.components * this_line != 0,
                                          self.components, 0)
            # Get the unique components
            unique_components = np.unique(components_on_line)
            # Filter out zeroes
            unique_components = unique_components[unique_components != 0]
            # Label components on the line
            res = np.where(np.isin(self.components, unique_components), label,
                           res)
            # Remove from original set of labeled components LIN(x, y)
            self.components = np.where(np.isin(self.components, unique_components), 0, self.components)

        if self.debug:
            # Print a colored image to represent component labeling
            output_labels = np.ma.masked_where(res == 0, res)
            cm = plt.get_cmap('turbo', lut=n_lines).copy()
            cm.set_bad(color='black')
            colored_image = cm(output_labels)
            Image.fromarray(
                (colored_image[:, :, :3] * 255).astype(np.uint8)).save(
                    os.path.join(
                        self.output_path,
                        f"{self.im_counter}_letter_labels_step_1.png"))
            self.im_counter += 1

        return res

    def __assign_remaining(self, n_lines: int, intermediate_result: np.ndarray, labeled_line_areas: np.ndarray):
        """After steps 2.3.1 and 2.3.2, there might be some unlabeled
        components. This step takes care of them.

        Args: intermediate_result (np.ndarray): The result of the earlier
            labeling steps.
        """

        # We take labeled_line_areas iff LIN(x, y) != AND RES(x, y) == 0.
        # | RES(x, y)   if    LIN(x, y) == 0  OR  RES(x, y) != 0
        # | LLA(x, y)   otherwise
        # The way the paper writes this is a bit convoluted but is equal to the 
        # above.
        final_image = np.where(self.components == 0,
            # LIN(x, y) == 0; RES(x, y) = whatever
            intermediate_result,
            np.where(intermediate_result == 0,
                # LIN(x, y) != 0; RES(x, y) == 0
                labeled_line_areas,
                # LIN(x, y) != 0; RES(x, y) != 0
                intermediate_result
            )
        )

        # rotate back so output corresponds to input
        final_image = np.flip(final_image)

        # reverse line labels so first line has label 1 instead of the highest label
        final_image = np.where(final_image != 0, n_lines - final_image + 1, final_image)

        if self.debug:
            # Print a colored image to represent component labeling
            output_labels = np.ma.masked_where(final_image == 0, final_image)
            cm = plt.get_cmap('turbo', lut=n_lines).copy()
            cm.set_bad(color='black')
            colored_image = cm(output_labels)
            Image.fromarray(
                (colored_image[:, :, :3] * 255).astype(np.uint8)).save(
                    os.path.join(
                        self.output_path,
                        f"{self.im_counter}_letter_labels_final.png"))
            self.im_counter += 1

        return final_image

    def __print_debug(self, message: str):
        """Helper function that prints a status message to the terminal.
        Overwrites the current line, so that all messages are printed on the
        same line.

        Args: message (str): The debug message to print
        """
        if self.debug: print(f"\x1b[1K\r{message}", end='')


if __name__ == "__main__":

    #TODO###################################################################################
    #TODO: Undo rotation when processing output! (also check if it even makes a difference)#
    #TODO###################################################################################

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

    if args.image_path:
        shredder = Shredder(args.image_path, args.debug, args.output_path)
        shredder.shred()
    else:
        print("Please provide an input image.")
        parser.print_help()
        exit()
